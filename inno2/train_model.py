#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创新点2 - Step 2: 跨模态可见光→红外生成与残差检测模型

核心思想：
1. 内容-风格解耦：分离形状/结构（模态不变）与辐射特性（模态特定）
2. 跨模态对齐：域对抗 + 对比学习
3. 三阶段训练：重构 → 对齐 → 异常检测

模型架构：
- E_c: 内容编码器（提取模态不变特征）
- E_v: 可见光风格编码器
- E_ir: 红外风格编码器  
- G_ir: 红外解码器（通过AdaIN注入风格）
- D_domain: 域判别器（用于对抗训练）
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from typing import Tuple, Dict, List, Optional
import argparse
from tqdm import tqdm
import wandb


# ===========================
# 1. 数据集定义
# ===========================

class PairedVIRDataset(Dataset):
    """成对可见光-红外数据集"""
    
    def __init__(self, csv_path: str, transform=None, normalize_ir: bool = True):
        """
        Args:
            csv_path: CSV文件路径
            transform: 数据增强（配对一致）
            normalize_ir: 是否归一化红外强度
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.normalize_ir = normalize_ir
        # 可见光标准化（ImageNet统计）
        self.vis_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.vis_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # 标签编码
        self.label_to_idx = {label: idx for idx, label in 
                            enumerate(sorted(self.df['label_en'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"加载数据集: {len(self.df)} 个样本")
        print(f"类别: {list(self.label_to_idx.keys())}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载可见光图像
        visible_path = row['visible_path']
        visible = Image.open(visible_path).convert('RGB')
        
        # 加载红外图像
        infrared_path = row['infrared_path']
        infrared = Image.open(infrared_path)
        
        # 红外图像处理：保持为单通道'L'
        if infrared.mode != 'L':
            infrared = infrared.convert('L')
        
        # 配对一致的几何变换
        if self.transform:
            # 使用相同的随机种子确保配对一致
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            visible = self.transform(visible)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            # 红外图像不做颜色扰动
            infrared = self.transform(infrared)
        else:
            visible = transforms.ToTensor()(visible)
            infrared = transforms.ToTensor()(infrared)
        
        # 红外强度归一化
        if self.normalize_ir:
            # 取消按样本min-max，直接保持ToTensor的[0,1]标度
            infrared = infrared.clamp(0,1)

        # 仅对可见光做标准化（保持配准一致的几何增强）
        # 注意：这里将常量移至与数据相同的设备（在DataLoader CPU侧为CPU张量即可）
        visible = (visible - self.vis_mean) / (self.vis_std + 1e-8)
        
        # 标签
        label = self.label_to_idx[row['label_en']]
        
        # 环境参数
        metadata = {
            'distance': float(row['Distance']) if pd.notna(row['Distance']) else 0.0,
            'humidity': float(row['Humidity']) if pd.notna(row['Humidity']) else 0.0,
            'temperature': float(row['Temperature']) if pd.notna(row['Temperature']) else 0.0,
        }
        
        return {
            'visible': visible,
            'infrared': infrared,
            'label': label,
            'metadata': metadata
        }


# ===========================
# 2. 模型组件
# ===========================

class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层（用于域对抗）"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class AdaptiveInstanceNorm2d(nn.Module):
    """自适应实例归一化（AdaIN，残差式调制以防塌缩）"""
    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, style):
        """style: [B, 2*C] (前C=gamma, 后C=beta)"""
        normalized = self.norm(x)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return (1 + gamma) * normalized + beta


class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization (简化版)，以结构特征为像素级条件。"""
    def __init__(self, norm_channels: int, cond_channels: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
        # 门控，确保初始为近似恒等映射
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden, norm_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden, norm_channels, kernel_size=3, padding=1)
        # 初始化为恒等变换：gamma≈0, beta≈0，避免早期不稳定
        nn.init.zeros_(self.mlp_gamma.weight); nn.init.zeros_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.weight); nn.init.zeros_(self.mlp_beta.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.mlp_shared(cond)
        gamma = self.mlp_gamma(h)
        beta = self.mlp_beta(h)
        x_norm = self.norm(x)
        x_spade = (1 + gamma) * x_norm + beta
        gate = torch.tanh(self.gate)  # [-1,1]，初始≈0，逐步放开
        return x + gate * (x_spade - x)


class PhysicsFiLM(nn.Module):
    """基于环境物理先验的FiLM调制（标量级）。"""
    def __init__(self, num_features: int, env_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(env_dim, num_features * 2)
        )
        # 恒等初始化：gamma≈0, beta≈0
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, env_vec: torch.Tensor) -> torch.Tensor:
        # env_vec: [B, E]
        gamma_beta = self.fc(env_vec)  # [B, 2*C]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class CrossAttention2D(nn.Module):
    """跨注意力：用style tokens调制空间特征。"""
    def __init__(self, channels: int, token_dim: int, num_tokens: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Linear(token_dim, channels)
        self.v_proj = nn.Linear(token_dim, channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_tokens = num_tokens
        # 学习门控，初始为0，避免早期注意力扰动过大
        self.attn_gate = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], tokens: [B, T, D]
        b, c, h, w = x.shape
        q = self.q_proj(x).view(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        k = self.k_proj(tokens)  # [B, T, C]
        v = self.v_proj(tokens)  # [B, T, C]
        
        # 多头分组
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # [B, N, C] -> [B, heads, N, C//heads]
            return t.view(b, -1, self.num_heads, c // self.num_heads).transpose(1, 2)
        qh = split_heads(q)
        kh = split_heads(k)
        vh = split_heads(v)
        
        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [B, heads, HW, T]
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, vh)  # [B, heads, HW, C//heads]
        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = out.transpose(1, 2).view(b, c, h, w)
        out = self.out_proj(out)
        gate = torch.tanh(self.attn_gate)  # 限制到[-1,1]
        return x + gate * out



class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


def icnr_init(conv_weight: torch.Tensor, scale: int = 2, init=nn.init.kaiming_normal_) -> None:
    """ICNR init for PixelShuffle to reduce checkerboard artifacts."""
    out_channels, in_channels, kH, kW = conv_weight.shape
    r2 = scale ** 2
    if out_channels % r2 != 0:
        return
    new_out = out_channels // r2
    kernel = torch.zeros(new_out, in_channels, kH, kW, device=conv_weight.device)
    init(kernel)
    kernel = kernel.repeat_interleave(r2, dim=0)
    with torch.no_grad():
        conv_weight.copy_(kernel)

'''class UpShuffle(nn.Module):
    """卷积 + PixelShuffle(2x) + 细化卷积"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_expand = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.refine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_expand(x)
        x = self.ps(x)
        x = self.refine(x)
        return x'''

# --- Sharp upsampling: Conv -> PixelShuffle -> Conv ---
class UpShuffle(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # ICNR init for first conv
        icnr_init(self.up[0].weight, scale=scale)
        if self.up[0].bias is not None:
            nn.init.zeros_(self.up[0].bias)
    def forward(self, x):
        return self.up(x)



class ContentEncoder(nn.Module):
    """内容编码器E_c：提取模态不变的结构特征（提供跳连特征）"""
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_blocks: int = 4):
        super().__init__()
        
        # 下采样（分stage，便于导出跳连）
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels * 4) for _ in range(num_blocks)]
        )
    
    def forward(self, x):
        c1 = self.enc1(x)                  # [B, 64, H/2, W/2]
        c2 = self.enc2(c1)                 # [B, 128, H/4, W/4]
        c3 = self.enc3(c2)                 # [B, 256, H/8, W/8]
        c3 = self.residual_blocks(c3)      # 语义增强
        return c1, c2, c3


class StyleEncoder(nn.Module):
    """风格编码器E_v/E_ir：提取模态特定的辐射特征"""
    
    def __init__(self, in_channels: int = 3, style_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, style_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InfraredDecoder(nn.Module):
    """红外解码器G_ir：SPADE(结构) + PhysicsFiLM(环境) + Cross-Attention(跨谱)（双线性上采样，无深监督）"""
    
    def __init__(self, content_channels: int = 256, style_dim: int = 256, out_channels: int = 3,
                 env_dim: int = 5, num_tokens: int = 8, token_dim: int = 64):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.style_to_tokens = nn.Linear(style_dim, num_tokens * token_dim)
        
        # 双线性上采样路径（更稳定，PSNR 友好）
        self.dec1 = nn.Sequential(
            nn.Conv2d(content_channels, content_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip_reduce1 = nn.Sequential(
            nn.Conv2d(content_channels // 2 + content_channels // 2, content_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.refine1 = nn.Sequential(
            ResidualBlock(content_channels // 2),
            nn.Conv2d(content_channels // 2, content_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.spade1 = SPADE(content_channels // 2, content_channels // 2)
        self.film1 = PhysicsFiLM(content_channels // 2, env_dim)
        self.attn1 = CrossAttention2D(content_channels // 2, token_dim, num_tokens)
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(content_channels // 2, content_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip_reduce2 = nn.Sequential(
            nn.Conv2d(content_channels // 4 + content_channels // 4, content_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            ResidualBlock(content_channels // 4),
            nn.Conv2d(content_channels // 4, content_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.spade2 = SPADE(content_channels // 4, content_channels // 4)
        self.film2 = PhysicsFiLM(content_channels // 4, env_dim)
        self.attn2 = CrossAttention2D(content_channels // 4, token_dim, num_tokens)
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(content_channels // 4, content_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.refine3 = nn.Sequential(
            ResidualBlock(content_channels // 4),
            nn.Conv2d(content_channels // 4, content_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.spade3 = SPADE(content_channels // 4, content_channels // 4)
        self.film3 = PhysicsFiLM(content_channels // 4, env_dim)
        self.attn3 = CrossAttention2D(content_channels // 4, token_dim, num_tokens)
        self.out_conv = nn.Sequential(
            nn.Conv2d(content_channels // 4, out_channels, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, content_skips, style, env: torch.Tensor = None):
        c1, c2, c3 = content_skips
        tokens = self.style_to_tokens(style).view(style.size(0), self.num_tokens, self.token_dim)
        
        x = self.dec1(c3)
        x = torch.cat([x, c2], dim=1)
        x = self.skip_reduce1(x)
        x = self.refine1(x)
        x = self.spade1(x, c2)
        if env is not None:
            x = self.film1(x, env)
        x = self.attn1(x, tokens)
        
        x = self.dec2(x)
        x = torch.cat([x, c1], dim=1)
        x = self.skip_reduce2(x)
        x = self.refine2(x)
        x = self.spade2(x, c1)
        if env is not None:
            x = self.film2(x, env)
        x = self.attn2(x, tokens)
        
        x = self.dec3(x)
        x = self.refine3(x)
        cond3 = F.interpolate(c1, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.spade3(x, cond3)
        if env is not None:
            x = self.film3(x, env)
        x = self.attn3(x, tokens)
        
        x = self.out_conv(x)
        return x


class DomainDiscriminator(nn.Module):
    """域判别器：用于域对抗训练"""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # 可见光域 vs 红外域
        )
    
    def forward(self, x):
        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ===========================
# 3. 完整模型
# ===========================

class CrossModalGenerationModel(nn.Module):
    """跨模态生成模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        # 与数据集中对可见光的标准化保持一致，用于IR在编码器前做域对齐
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # 编码器
        self.content_encoder = ContentEncoder(
            in_channels=3,
            base_channels=config.get('base_channels', 64),
            num_blocks=config.get('num_blocks', 4)
        )
        
        self.visible_style_encoder = StyleEncoder(
            in_channels=3,
            style_dim=config.get('style_dim', 256)
        )
        
        self.infrared_style_encoder = StyleEncoder(
            in_channels=3,
            style_dim=config.get('style_dim', 256)
        )
        
        # 解码器
        self.infrared_decoder = InfraredDecoder(
            content_channels=config.get('base_channels', 64) * 4,
            style_dim=config.get('style_dim', 256),
            out_channels=1
        )
        
        # 域判别器
        self.domain_discriminator = DomainDiscriminator(
            feature_dim=config.get('base_channels', 64) * 4
        )
    
    def forward(self, visible, infrared=None, alpha=1.0, env: Optional[torch.Tensor] = None):
        """
        Args:
            visible: 可见光图像 [B, 3, H, W]
            infrared: 红外图像 [B, 3, H, W] (训练时提供)
            alpha: GRL强度参数
        
        Returns:
            Dict with keys: generated_ir, content_v, content_ir, style_v, style_ir, domain_pred_v, domain_pred_ir
        """
        # 提取可见光的内容和风格（含跳连）
        c1_v, c2_v, c3_v = self.content_encoder(visible)
        style_v = self.visible_style_encoder(visible)
        
        outputs = {
            'content_v': c3_v,
            'content_v_skips': (c1_v, c2_v, c3_v),
            'style_v': style_v
        }
        
        # 如果提供红外图像，提取其内容和风格
        if infrared is not None:
            # 内容/风格编码器期望3通道，单通道IR在进入编码器前做通道重复
            infrared_rgb = infrared.repeat(1, 3, 1, 1) if infrared.size(1) == 1 else infrared
            # 使用与可见光相同的ImageNet标准化，以对齐编码器输入分布
            infrared_rgb = (infrared_rgb - self.imagenet_mean) / (self.imagenet_std + 1e-8)
            c1_ir, c2_ir, c3_ir = self.content_encoder(infrared_rgb)
            style_ir = self.infrared_style_encoder(infrared_rgb)
            
            outputs['content_ir'] = c3_ir
            outputs['content_ir_skips'] = (c1_ir, c2_ir, c3_ir)
            outputs['style_ir'] = style_ir
            
            # 域对抗：判别内容特征来自哪个域
            content_v_reversed = GradientReversalLayer.apply(c3_v, alpha)
            content_ir_reversed = GradientReversalLayer.apply(c3_ir, alpha)
            
            outputs['domain_pred_v'] = self.domain_discriminator(content_v_reversed)
            outputs['domain_pred_ir'] = self.domain_discriminator(content_ir_reversed)
            
            # 生成红外图像：使用可见光内容 + 真实红外风格 + 物理先验
            generated_ir = self.infrared_decoder((c1_v, c2_v, c3_v), style_ir, env=env)
        else:
            # 推理模式：仅从可见光生成（使用可见光风格作为近似）
            generated_ir = self.infrared_decoder((c1_v, c2_v, c3_v), style_v, env=env)
        
        # 统一在此处写入生成结果
        outputs['generated_ir'] = generated_ir
        
        return outputs


# ===========================
# 4. 损失函数
# ===========================

class MS_SSIM_Loss(nn.Module):
    """多尺度结构相似性损失（简化实现）"""
    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels
    def forward(self, pred, target):
        mu_pred = F.avg_pool2d(pred, 3, 1, 1)
        mu_target = F.avg_pool2d(target, 3, 1, 1)
        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target
        sigma_pred_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred_target
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        return 1 - ssim_map.mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier L1（更平滑的L1，利于细节与稳健性）"""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class FrequencyLoss(nn.Module):
    """频域幅值差异（强调中高频，缓解模糊）"""
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 转灰度以降低维度影响
        if x.size(1) == 3:
            xg = 0.2989 * x[:,0:1] + 0.5870 * x[:,1:2] + 0.1140 * x[:,2:3]
            yg = 0.2989 * y[:,0:1] + 0.5870 * y[:,1:2] + 0.1140 * y[:,2:3]
        else:
            xg, yg = x, y
        X = torch.fft.rfft2(xg, norm='ortho')
        Y = torch.fft.rfft2(yg, norm='ortho')
        Ax = torch.log1p(torch.abs(X))
        Ay = torch.log1p(torch.abs(Y))
        return torch.mean(torch.abs(Ax - Ay))


class TotalVariationLoss(nn.Module):
    """全变分正则化（按像素数归一，避免量纲过大）"""
    def forward(self, x):
        # x: [B, C, H, W]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        norm = x.size(0) * x.size(1) * x.size(2) * x.size(3)
        return (h_tv + w_tv) / (norm + 1e-8)


class SobelGradLoss(nn.Module):
    """Sobel梯度一致性：约束边缘与中高频结构"""
    def __init__(self):
        super().__init__()
        # Sobel核（3x3），分别用于x/y方向
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('kx', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('ky', sobel_y.view(1, 1, 3, 3))
        self.eps = 1e-6

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] in [0,1], 取灰度后做梯度
        if x.size(1) == 3:
            # RGB转灰度（固定系数）
            r, g, b = x[:, 0:1], x[:, 1:1+1], x[:, 2:2+1]
            x_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            x_gray = x
        kx = self.kx.to(device=x_gray.device, dtype=x_gray.dtype)
        ky = self.ky.to(device=x_gray.device, dtype=x_gray.dtype)
        gx = F.conv2d(x_gray, kx, padding=1)
        gy = F.conv2d(x_gray, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gp = self._grad_mag(pred)
        gt = self._grad_mag(target)
        return torch.mean(torch.abs(gp - gt))


class LaplacianLoss(nn.Module):
    """拉普拉斯高频一致性损失，促进边缘/细节恢复"""
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0, 1, 0],
                          [1,-4, 1],
                          [0, 1, 0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('k', k)
    def forward(self, x, y):
        # 兼容1/3通道：对每个通道独立卷积（groups=channels）
        k = self.k.to(device=x.device, dtype=x.dtype)
        c_x = x.size(1)
        c_y = y.size(1)
        kx = k.repeat(c_x, 1, 1, 1)
        ky = k.repeat(c_y, 1, 1, 1)
        lx = F.conv2d(x, kx, padding=1, groups=c_x)
        ly = F.conv2d(y, ky, padding=1, groups=c_y)
        return torch.mean(torch.abs(lx - ly))


def edge_weighted_l1(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """按目标Sobel梯度加权的L1，强化边缘位置的像素一致性。"""
    # 生成目标的梯度权重
    if target.size(1) == 3:
        t_gray = 0.2989 * target[:,0:1] + 0.5870 * target[:,1:2] + 0.1140 * target[:,2:3]
    else:
        t_gray = target
    # 复用上面的Sobel实现
    sobel = SobelGradLoss()
    wt = sobel._grad_mag(t_gray)
    wt = wt / (wt.mean() + 1e-8)
    wt = torch.clamp(wt, 0.0, 3.0) ** beta
    return torch.mean(wt * torch.abs(pred - target))


class PerceptualLoss(nn.Module):
    """感知损失（使用VGG特征）"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        # 使用预训练的VGG16
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
            self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).to(device)
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()
        except:
            print("警告: 无法加载VGG16，感知损失将被禁用")
            self.feature_extractor = None
    
    def forward(self, pred, target):
        if self.feature_extractor is None:
            return torch.tensor(0.0).to(pred.device)
        
        # 归一化到ImageNet标准
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # 提取特征
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        # 计算L2距离
        loss = F.mse_loss(pred_features, target_features)
        
        return loss


class InfoNCE_Loss(nn.Module):
    """跨模态对比学习损失（InfoNCE）"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, content_v, content_ir):
        """
        Args:
            content_v: 可见光内容特征 [B, C, H, W]
            content_ir: 红外内容特征 [B, C, H, W]
        """
        # 全局池化
        content_v_pooled = F.adaptive_avg_pool2d(content_v, 1).view(content_v.size(0), -1)
        content_ir_pooled = F.adaptive_avg_pool2d(content_ir, 1).view(content_ir.size(0), -1)
        
        # L2归一化
        content_v_norm = F.normalize(content_v_pooled, dim=1)
        content_ir_norm = F.normalize(content_ir_pooled, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(content_v_norm, content_ir_norm.t()) / self.temperature
        
        # 对角线为正样本对
        batch_size = content_v.size(0)
        labels = torch.arange(batch_size).to(content_v.device)
        
        loss = F.cross_entropy(similarity, labels)
        
        return loss


def compute_mmd_loss(x, y):
    """最大均值差异（MMD）损失"""
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())
    
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    
    K_xx = torch.exp(-0.5 * (rx.t() + rx - 2 * xx))
    K_yy = torch.exp(-0.5 * (ry.t() + ry - 2 * yy))
    K_xy = torch.exp(-0.5 * (rx.t() + ry - 2 * xy))
    
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


# ===========================
# 5. 训练器
# ===========================

class Trainer:
    """三阶段训练器"""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer_g = torch.optim.Adam(
            list(model.content_encoder.parameters()) +
            list(model.visible_style_encoder.parameters()) +
            list(model.infrared_style_encoder.parameters()) +
            list(model.infrared_decoder.parameters()),
            lr=config.get('lr_g', 1e-4),
            betas=(0.5, 0.999)
        )
        
        self.optimizer_d = torch.optim.Adam(
            model.domain_discriminator.parameters(),
            lr=config.get('lr_d', 1e-4),
            betas=(0.5, 0.999)
        )
        
        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.charb_loss = CharbonnierLoss()
        self.l2_loss = nn.MSELoss()
        self.ms_ssim_loss = MS_SSIM_Loss()
        self.tv_loss = TotalVariationLoss()
        self.perceptual_loss = PerceptualLoss(device=device)
        self.grad_loss = SobelGradLoss().to(self.device)
        self.lap_loss = LaplacianLoss()
        self.charb_loss = CharbonnierLoss(eps=1e-3)
        self.freq_loss = FrequencyLoss()
        self.infonce_loss = InfoNCE_Loss(temperature=config.get('nce_temp', 0.07))
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=config.get('epochs', 100)
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=config.get('epochs', 100)
        )
    
    @staticmethod
    def _metadata_to_tensor(metadata_batch, device):
        """
        将批量metadata(list[dict]或dict of lists)转为 [B, E] 张量。
        顺序: distance, humidity, temperature, weather, windspeed
        简单缩放到大致[0,1]范围。
        """
        # 兼容DataLoader的默认collate：通常是list[dict]
        if isinstance(metadata_batch, list):
            distances = [m.get('distance', 0.0) for m in metadata_batch]
            humidities = [m.get('humidity', 0.0) for m in metadata_batch]
            temperatures = [m.get('temperature', 0.0) for m in metadata_batch]
            weathers = [m.get('weather', 0.0) for m in metadata_batch]
            winds = [m.get('windspeed', 0.0) for m in metadata_batch]
        elif isinstance(metadata_batch, dict):
            distances = metadata_batch.get('distance', [])
            humidities = metadata_batch.get('humidity', [])
            temperatures = metadata_batch.get('temperature', [])
            weathers = metadata_batch.get('weather', [])
            winds = metadata_batch.get('windspeed', [])
        else:
            # 不可识别，退化为全零
            b = metadata_batch.size(0) if hasattr(metadata_batch, 'size') else 1
            return torch.zeros(b, 5, device=device, dtype=torch.float32)
        
        import numpy as np
        def to_tensor(xs, scale):
            arr = np.array(xs, dtype=np.float32) / scale
            return torch.from_numpy(arr)
        
        d = to_tensor(distances, 100.0)
        h = to_tensor(humidities, 100.0)
        t = to_tensor(temperatures, 50.0)
        w = to_tensor(weathers, 10.0) if len(weathers) else torch.zeros_like(d)
        s = to_tensor(winds, 30.0) if len(winds) else torch.zeros_like(d)
        env = torch.stack([d, h, t, w, s], dim=1).to(device)
        return env
    
    def compute_phase_a_loss(self, outputs, targets):
        """Phase A: 掩膜加权重构损失 + 感知损失"""
        generated_ir = outputs['generated_ir']
        target_ir = targets['infrared']
        
        # 主像素项：Charbonnier + L2（1:1 起步，由 lambda_l1 / lambda_l2 控制）
        '''charb = self.charb_loss(generated_ir, target_ir)
        l2 = self.l2_loss(generated_ir, target_ir)
        ms_ssim = self.ms_ssim_loss(generated_ir, target_ir)

        
        # TV正则化
        tv = self.tv_loss(generated_ir)
        
        # 感知损失（VGG特征）；当权重为0时跳过计算以节省开销
        if self.config.get('lambda_perceptual', 1.0) > 0:
            perceptual = self.perceptual_loss(
                generated_ir if generated_ir.size(1)==3 else generated_ir.repeat(1,3,1,1),
                target_ir if target_ir.size(1)==3 else target_ir.repeat(1,3,1,1)
            )
        else:
            perceptual = torch.tensor(0.0, device=generated_ir.device)
        
        # 梯度一致性（Sobel）
        grad = self.grad_loss(generated_ir, target_ir)
        # 高频一致性（Laplacian）
        lap = self.lap_loss(generated_ir, target_ir)
        # 频域一致性（小权重）
        freq = self.freq_loss(generated_ir, target_ir)
        # 边缘加权像素一致
        ewl1 = edge_weighted_l1(generated_ir, target_ir, beta=1.5)

        # 感知项的数值通常在 8-12 左右，为避免主导总损失，这里做固定缩放
        perceptual_scaled = 0.05 * perceptual
        # 总损失（量纲更平衡）
        l1 = charb
        loss = (self.config.get('lambda_l1', 1.0) * charb +
                self.config.get('lambda_l2', 1.0) * l2 +
                self.config.get('lambda_ssim', 0.2) * ms_ssim +
                self.config.get('lambda_tv', 0.01) * tv +
                self.config.get('lambda_perceptual', 0.5) * perceptual_scaled +
                self.config.get('lambda_grad', 1.0) * grad +
                self.config.get('lambda_lap', 0.4) * lap +
                self.config.get('lambda_fft', 0.05) * freq +
                self.config.get('lambda_edge', 0.3) * ewl1)
        
        return loss, {
            'l1': l1.item(), 
            'l2': l2.item(), 
            'ms_ssim': ms_ssim.item(), 
            'tv': tv.item(),
            'perceptual': perceptual.item(),
            'grad': grad.item(),
            'lap': lap.item(),
            'freq': freq.item(),
            'ewl1': ewl1.item()
        }'''

        # 主像素项：Charbonnier + L2
        charb = self.charb_loss(generated_ir, target_ir)
        l2 = self.l2_loss(generated_ir, target_ir)
        ms_ssim = self.ms_ssim_loss(generated_ir, target_ir)

        # TV正则化
        tv = self.tv_loss(generated_ir)
        
        # 感知损失（VGG特征）；当权重为0时跳过计算以节省开销
        if self.config.get('lambda_perceptual', 0.0) > 0:
            perceptual = self.perceptual_loss(
                generated_ir if generated_ir.size(1)==3 else generated_ir.repeat(1,3,1,1),
                target_ir if target_ir.size(1)==3 else target_ir.repeat(1,3,1,1)
            )
        else:
            perceptual = torch.tensor(0.0, device=generated_ir.device)
        
        # 梯度/高频/边缘项：当作轻量正则，不再主导
        grad = self.grad_loss(generated_ir, target_ir)
        lap = self.lap_loss(generated_ir, target_ir)
        freq = self.freq_loss(generated_ir, target_ir)
        ewl1 = edge_weighted_l1(generated_ir, target_ir, beta=1.5)

        # 感知项缩放，避免直接把总 loss 顶爆
        perceptual_scaled = 0.05 * perceptual

        # 总损失：L2 为核心，Charbonnier 可选
        loss = (
            self.config.get('lambda_l2', 2.0) * l2 +
            self.config.get('lambda_l1', 0.0) * charb +
            self.config.get('lambda_ssim', 0.4) * ms_ssim +
            self.config.get('lambda_tv', 0.01) * tv +
            self.config.get('lambda_perceptual', 0.0) * perceptual_scaled +
            self.config.get('lambda_grad', 0.5) * grad +
            self.config.get('lambda_lap', 0.3) * lap +
            self.config.get('lambda_fft', 0.0) * freq +
            self.config.get('lambda_edge', 0.3) * ewl1
        )

        # 这里的 l1 只是用来在日志里看 Charbonnier 数值
        l1 = charb
        return loss, {
            'l1': l1.item(), 
            'l2': l2.item(), 
            'ms_ssim': ms_ssim.item(), 
            'tv': tv.item(),
            'perceptual': perceptual.item(),
            'grad': grad.item(),
            'lap': lap.item(),
            'freq': freq.item(),
            'ewl1': ewl1.item()
        }

        
    
    '''def compute_phase_b_loss(self, outputs, targets, alpha, warmup_factor: float = 1.0):
        """Phase B: 重构 + 对齐"""
        # Phase A损失
        recon_loss, recon_metrics = self.compute_phase_a_loss(outputs, targets)
        
        # 跨模态对比学习
        content_v = outputs['content_v']
        content_ir = outputs['content_ir']
        nce_loss = self.infonce_loss(content_v, content_ir)
        
        # 域对抗损失（生成器步：重新前向且冻结判别器参数，防止参数版本冲突）
        content_v_rev = GradientReversalLayer.apply(content_v, alpha)
        content_ir_rev = GradientReversalLayer.apply(content_ir, alpha)
        requires = []
        for p in self.model.domain_discriminator.parameters():
            requires.append(p.requires_grad)
            p.requires_grad = False
        domain_pred_v = self.model.domain_discriminator(content_v_rev)
        domain_pred_ir = self.model.domain_discriminator(content_ir_rev)
        for p, req in zip(self.model.domain_discriminator.parameters(), requires):
            p.requires_grad = req
        
        batch_size = content_v.size(0)
        domain_labels_v = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        domain_labels_ir = torch.ones(batch_size, dtype=torch.long).to(self.device)
        
        # 生成器希望判别器混淆
        domain_loss_g = (self.ce_loss(domain_pred_v, domain_labels_ir) +
                        self.ce_loss(domain_pred_ir, domain_labels_v)) / 2
        
        # 统计匹配（MMD）
        content_v_pooled = F.adaptive_avg_pool2d(content_v, 1).view(batch_size, -1)
        content_ir_pooled = F.adaptive_avg_pool2d(content_ir, 1).view(batch_size, -1)
        mmd_loss = compute_mmd_loss(content_v_pooled, content_ir_pooled)
        
        # 总损失
        #loss = (recon_loss +
                #warmup_factor * self.config.get('lambda_nce', 0.1) * nce_loss +
                #warmup_factor * self.config.get('lambda_domain', 0.1) * domain_loss_g +
                #warmup_factor * self.config.get('lambda_mmd', 0.01) * mmd_loss)

        recon_weight_b = self.config.get('lambda_recon_b', 2.0)
        loss = (recon_weight_b * recon_loss +
                warmup_factor * self.config.get('lambda_nce', 0.02) * nce_loss +
                warmup_factor * self.config.get('lambda_domain', 0.02) * domain_loss_g +
                warmup_factor * self.config.get('lambda_mmd', 0.005) * mmd_loss)

        
        metrics = {
            **recon_metrics,
            'nce': nce_loss.item(),
            'domain_g': domain_loss_g.item(),
            'mmd': mmd_loss.item()
        }
        
        return loss, metrics'''

    def compute_phase_b_loss(self, outputs, targets, alpha=None, warmup_factor: float = 1.0):
        """
        Phase B：细节增强重构阶段
        - 不再做对比学习 / 域对抗 / MMD
        - 只优化 generated_ir 与 target_ir 的差异，但更强调结构和高频
        """
        generated_ir = outputs['generated_ir']
        target_ir = targets['infrared']

        # 像素项：Charbonnier + L2 + MS-SSIM
        '''charb = self.charb_loss(generated_ir, target_ir)
        l2 = self.l2_loss(generated_ir, target_ir)
        ms_ssim = self.ms_ssim_loss(generated_ir, target_ir)

        # TV 正则
        tv = self.tv_loss(generated_ir)

        # 感知损失（可选，默认关）
        if self.config.get('lambda_perceptual_b', 0.0) > 0:
            perceptual = self.perceptual_loss(
                generated_ir if generated_ir.size(1) == 3 else generated_ir.repeat(1, 3, 1, 1),
                target_ir if target_ir.size(1) == 3 else target_ir.repeat(1, 3, 1, 1)
            )
        else:
            perceptual = torch.tensor(0.0, device=generated_ir.device)

        # 梯度 / 高频 / 边缘
        grad = self.grad_loss(generated_ir, target_ir)
        lap = self.lap_loss(generated_ir, target_ir)
        freq = self.freq_loss(generated_ir, target_ir)
        ewl1 = edge_weighted_l1(generated_ir, target_ir, beta=1.5)

        # 感知项缩放，避免爆掉
        perceptual_scaled = 0.05 * perceptual

        # Phase B 专用权重（如果没单独给，就回退到 Phase A 的配置）
        loss = (
            # Charbonnier 不主导，主要看 L2 和结构
            self.config.get('lambda_l1', 0.0) * charb +
            self.config.get('lambda_l2_b', self.config.get('lambda_l2', 2.0)) * l2 +
            self.config.get('lambda_ssim_b', self.config.get('lambda_ssim', 0.4)) * ms_ssim +
            self.config.get('lambda_tv_b', self.config.get('lambda_tv', 0.01)) * tv +
            self.config.get('lambda_perceptual_b', 0.0) * perceptual_scaled +
            self.config.get('lambda_grad_b', self.config.get('lambda_grad', 0.5)) * grad +
            self.config.get('lambda_lap_b', self.config.get('lambda_lap', 0.3)) * lap +
            self.config.get('lambda_fft_b', self.config.get('lambda_fft', 0.0)) * freq +
            self.config.get('lambda_edge_b', self.config.get('lambda_edge', 0.3)) * ewl1
        )

        return loss, {
            'l1': charb.item(),
            'l2': l2.item(),
            'ms_ssim': ms_ssim.item(),
            'tv': tv.item(),
            'perceptual': perceptual.item(),
            'grad': grad.item(),
            'lap': lap.item(),
            'freq': freq.item(),
            'ewl1': ewl1.item(),
        }'''

        # ========= 只用 L2 做优化 =========
        l2 = self.l2_loss(generated_ir, target_ir)

        # 下面这些只是为了你在 log 里还能看到数值变化，不参与反向传播
        charb = self.charb_loss(generated_ir, target_ir)
        ms_ssim = self.ms_ssim_loss(generated_ir, target_ir)
        tv = self.tv_loss(generated_ir)

        # 纯 PSNR 模式：loss 就是 L2（乘一个权重）
        loss = self.config.get('lambda_l2_b', 1.0) * l2

        return loss, {
            'l1': charb.item(),      # 这里的 l1 其实是 Charbonnier
            'l2': l2.item(),
            'ms_ssim': ms_ssim.item(),
            'tv': tv.item(),
            'perceptual': 0.0,
            'grad': 0.0,
            'lap': 0.0,
            'freq': 0.0,
            'ewl1': 0.0,
        }


    
    def train_discriminator(self, outputs):
        """训练域判别器"""
        # 使用未经过GRL的内容特征，并与生成器图断开，避免干扰
        content_v = outputs['content_v'].detach()
        content_ir = outputs['content_ir'].detach()
        domain_pred_v = self.model.domain_discriminator(content_v)
        domain_pred_ir = self.model.domain_discriminator(content_ir)
        
        batch_size = content_v.size(0)
        domain_labels_v = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        domain_labels_ir = torch.ones(batch_size, dtype=torch.long).to(self.device)
        
        # 判别器希望正确分类
        loss_d = (self.ce_loss(domain_pred_v, domain_labels_v) +
                 self.ce_loss(domain_pred_ir, domain_labels_ir)) / 2
        
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()
        
        return loss_d.item()
    
    def train_epoch(self, dataloader, phase='A', epoch=0):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        metrics_sum = {}
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Phase {phase}]')
        
        for batch_idx, batch in enumerate(progress_bar):
            visible = batch['visible'].to(self.device)
            infrared = batch['infrared'].to(self.device)
            env = self._metadata_to_tensor(batch.get('metadata', []), self.device)
            
            # 计算GRL的alpha（逐渐增大）
            p = (epoch + batch_idx / len(dataloader)) / self.config.get('epochs', 100)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            self.progress = p
            
            # 前向传播
            outputs = self.model(visible, infrared, alpha=alpha, env=env)
            
            # 准备已移到device的batch
            batch_gpu = {
                'visible': visible,
                'infrared': infrared,
                'env': env,
                'label': batch['label'].to(self.device) if 'label' in batch else None
            }
            
            # 计算损失
            if phase == 'A':
                loss, metrics = self.compute_phase_a_loss(outputs, batch_gpu)

            elif phase == 'B':
                loss, metrics = self.compute_phase_b_loss(outputs, batch_gpu)

            '''elif phase == 'B':
                # 先训练判别器
                if batch_idx % self.config.get('d_steps', 1) == 0:
                    d_loss = self.train_discriminator(outputs)
                    metrics = {'d_loss': d_loss}
                
                # 再训练生成器
                # 线性 warmup: 在前 warmup_pct * epochs 内从 0 → 1
                warmup_pct = float(self.config.get('phase_b_warmup', 0.2))
                epochs_b = max(1, int(self.config.get('epochs_phase_b', 1)))
                progress_b = (epoch - 1 + batch_idx / max(1, len(dataloader))) / epochs_b
                warmup_factor = 1.0 if warmup_pct <= 0 else min(1.0, progress_b / warmup_pct)
                loss, metrics = self.compute_phase_b_loss(outputs, batch_gpu, alpha, warmup_factor=warmup_factor)'''



            
            # 反向传播
            self.optimizer_g.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer_g.step()
            
            # 统计
            total_loss += loss.item()
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 平均指标
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
        
        # 更新学习率
        self.scheduler_g.step()
        if phase == 'B':
            self.scheduler_d.step()
        
        return avg_loss, avg_metrics
    
    def evaluate(self, dataloader, phase='A'):
        """评估模型"""
        self.model.eval()
        
        total_loss = 0
        metrics_sum = {}
        
        # 额外评估指标
        psnr_list = []
        ssim_list = []
        
        all_generated = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='评估中'):
                visible = batch['visible'].to(self.device)
                infrared = batch['infrared'].to(self.device)
                env = self._metadata_to_tensor(batch.get('metadata', []), self.device)
                
                # 前向传播
                outputs = self.model(visible, infrared, alpha=1.0, env=env)
                
                # 准备batch
                batch_gpu = {
                    'visible': visible,
                    'infrared': infrared,
                    'env': env,
                }
                
                # 计算损失
                if phase == 'A':
                    loss, metrics = self.compute_phase_a_loss(outputs, batch_gpu)
                elif phase == 'B':
                    loss, metrics = self.compute_phase_b_loss(outputs, batch_gpu, alpha=1.0)
                
                total_loss += loss.item()
                for k, v in metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
                
                # 计算PSNR和SSIM
                generated_ir = outputs['generated_ir']
                target_ir = batch_gpu['infrared']
                
                # 转为numpy计算PSNR/SSIM（评估前clamp到[0,1]）
                gen_np = torch.clamp(generated_ir, 0, 1).cpu().numpy()
                tgt_np = torch.clamp(target_ir, 0, 1).cpu().numpy()
                
                for i in range(gen_np.shape[0]):
                    # PSNR
                    mse = np.mean((gen_np[i] - tgt_np[i]) ** 2)
                    if mse > 0:
                        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                        psnr_list.append(psnr)
                    
                    # SSIM (简化版)
                    # 实际应用中建议使用skimage.metrics.structural_similarity
                
                # 保存前几个样本用于可视化
                if len(all_generated) < 5:
                    all_generated.append(generated_ir[:1].cpu())
                    all_targets.append(target_ir[:1].cpu())
        
        # 平均指标
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
        avg_psnr = np.mean(psnr_list) if psnr_list else 0
        
        # 添加PSNR到指标
        avg_metrics['psnr'] = avg_psnr
        
        return avg_loss, avg_metrics, all_generated, all_targets
    
    def save_checkpoint(self, epoch, phase, save_path, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")


# ===========================
# 6. 主训练流程
# ===========================

def main():
    parser = argparse.ArgumentParser(description='创新点2 - 跨模态生成与残差检测')
    parser.add_argument('--resume', type=str, default='/mnt/e/code/project/inno2/checkpoints/phase_a_best.pth',help='path to a phase A checkpoint to resume generator weights from')

    
    # 数据参数
    parser.add_argument('--csv_path', type=str, default='/mnt/e/code/project/inno2/inno2.csv')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 模型参数
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--style_dim', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=4)
    
    # 训练参数
    parser.add_argument('--epochs_phase_a', type=int, default=50, help='Phase A epochs')
    parser.add_argument('--epochs_phase_b', type=int, default=0, help='Phase B epochs')
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    
    # 损失权重
    '''parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_l2', type=float, default=1.0)
    parser.add_argument('--lambda_ssim', type=float, default=0.2)
    parser.add_argument('--lambda_tv', type=float, default=0.01)
    parser.add_argument('--lambda_perceptual', type=float, default=0.5)
    parser.add_argument('--lambda_lap', type=float, default=0.4)
    parser.add_argument('--lambda_edge', type=float, default=0.3)
    parser.add_argument('--lambda_grad', type=float, default=1.0)
    parser.add_argument('--lambda_nce', type=float, default=0.1)
    parser.add_argument('--lambda_domain', type=float, default=0.1)
    parser.add_argument('--lambda_mmd', type=float, default=0.01)
    parser.add_argument('--lambda_fft', type=float, default=0.05)
    parser.add_argument('--lambda_charb', type=float, default=0.0)'''

    # 损失权重（已调成“PSNR 优先”的默认配置）
    # Phase A 主要目标：减小 L2 (MSE) 以提升 PSNR，结构项只做轻度约束
    parser.add_argument('--lambda_l1', type=float, default=0.0)   # 不再让 Charbonnier 主导
    parser.add_argument('--lambda_l2', type=float, default=2.0)   # 提高 L2 权重，直推 PSNR
    parser.add_argument('--lambda_ssim', type=float, default=0.4) # 适度保留结构
    parser.add_argument('--lambda_tv', type=float, default=0.01)  # 轻度平滑
    parser.add_argument('--lambda_perceptual', type=float, default=0.0)  # 先关闭感知损失，防止牺牲 PSNR
    parser.add_argument('--lambda_lap', type=float, default=0.3)  # 高频/边缘做轻正则
    parser.add_argument('--lambda_edge', type=float, default=0.3)
    parser.add_argument('--lambda_grad', type=float, default=0.5)
    parser.add_argument('--lambda_fft', type=float, default=0.0)  # 暂停频域损失（对 PSNR 不一定友好）
    parser.add_argument('--lambda_charb', type=float, default=0.0)

    # Phase B 对齐项默认关小，后面需要时再慢慢开
    parser.add_argument('--lambda_nce', type=float, default=0.02)
    parser.add_argument('--lambda_domain', type=float, default=0.02)
    parser.add_argument('--lambda_mmd', type=float, default=0.005)

    #parser.add_argument('--lambda_recon_b', type=float, default=0.2,
                    #help='Phase B 权重线性热身占比，0.2表示前20%进度把权重从0→设定值')

    # Phase B 专用重构权重：更强调结构与高频
    parser.add_argument('--lambda_l2_b', type=float, default=1.5)
    parser.add_argument('--lambda_ssim_b', type=float, default=0.6)
    parser.add_argument('--lambda_tv_b', type=float, default=0.005)
    parser.add_argument('--lambda_perceptual_b', type=float, default=0.1)
    parser.add_argument('--lambda_grad_b', type=float, default=1.0)
    parser.add_argument('--lambda_lap_b', type=float, default=0.8)
    parser.add_argument('--lambda_edge_b', type=float, default=0.8)
    parser.add_argument('--lambda_fft_b', type=float, default=0.0)

    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='/mnt/e/code/project/inno2/checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use W&B for logging')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化W&B
    if args.use_wandb:
        wandb.init(project='inno2-cross-modal', config=vars(args))
    
    print("=" * 80)
    print("创新点2 - 跨模态可见光→红外生成与残差检测")
    print("=" * 80)
    
    # 数据增强（配对一致）
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    
    # 加载数据集
    print("\n[1/3] 加载数据集...")
    dataset = PairedVIRDataset(args.csv_path, transform=transform, normalize_ir=False)
    
    # 划分训练集和验证集
    import random
    # 固定随机种子，保证每次训练/验证划分一致
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    g = torch.Generator().manual_seed(seed)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    print("\n[2/3] 创建模型...")
    config = vars(args)
    model = CrossModalGenerationModel(config)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(model, config, args.device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        state = ckpt.get('model_state_dict', ckpt)
        trainer.model.load_state_dict(state, strict=True)
        print(f"Resumed generator weights from {args.resume} (epoch={ckpt.get('epoch')}, phase={ckpt.get('phase')})")
    
    # Phase A: 重构训练
    print("\n[3/3] 开始训练...")
    print("\n" + "=" * 80)
    print("Phase A: 掩膜加权重构（L1/L2 + MS-SSIM + TV）")
    print("=" * 80)
    
    best_val_loss_a = float('inf')
    best_psnr_a = -1e9
    
    for epoch in range(1, args.epochs_phase_a + 1):
        # 训练
        loss, metrics = trainer.train_epoch(train_loader, phase='A', epoch=epoch)
        
        print(f"Epoch {epoch}/{args.epochs_phase_a} - Train Loss: {loss:.4f}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # 验证评估（每5个epoch评估一次）
        if epoch % 1 == 0 or epoch == args.epochs_phase_a:
            print(f"\n  [验证评估 Epoch {epoch}]")
            val_loss, val_metrics, gen_samples, tgt_samples = trainer.evaluate(val_loader, phase='A')
            print(f"  Val Loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                print(f"    {k}: {v:.4f}")
            
            # 保存可视化样本（每20个epoch或最后一个epoch）
            if epoch % 20 == 0 or epoch == args.epochs_phase_a:
                print(f"  保存可视化样本...")
                vis_dir = os.path.join(args.save_dir, 'visualization')
                os.makedirs(vis_dir, exist_ok=True)
                
                import matplotlib
                matplotlib.use('Agg')  # 使用非交互式后端，避免多线程问题
                import matplotlib.pyplot as plt
                if gen_samples and tgt_samples:
                    for idx, (gen, tgt) in enumerate(zip(gen_samples[:3], tgt_samples[:3])):
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                        g = gen[0].cpu().numpy()
                        t = tgt[0].cpu().numpy()
                        gen_img = g[0] if g.shape[0] == 1 else np.transpose(g, (1, 2, 0))
                        tgt_img = t[0] if t.shape[0] == 1 else np.transpose(t, (1, 2, 0))
                        residual = np.abs(gen_img - tgt_img)

                        gen_img = np.clip(gen_img, 0, 1)
                        tgt_img = np.clip(tgt_img, 0, 1)
                        residual = np.clip(residual, 0, 1)

                        axes[0].imshow(tgt_img, cmap='gray' if tgt_img.ndim == 2 else None)
                        axes[0].set_title(f'真实红外 (Epoch {epoch})', fontsize=14)
                        axes[0].axis('off')
                        
                        axes[1].imshow(gen_img, cmap='gray' if gen_img.ndim == 2 else None)
                        axes[1].set_title(f'生成红外 (Epoch {epoch})', fontsize=14)
                        axes[1].axis('off')
                        
                        axes[2].imshow(residual.squeeze() if residual.ndim == 3 else residual, cmap='hot')
                        axes[2].set_title(f'残差图 (Epoch {epoch})', fontsize=14)
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        save_path = os.path.join(vis_dir, f'epoch_{epoch}_sample_{idx+1}.png')
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                    print(f"    ✓ 已保存 {min(3, len(gen_samples))} 个样本到: {vis_dir}")
            
            # 检查是否是最佳模型
            is_best = val_metrics.get('psnr', 0) > best_psnr_a
            if is_best:
                best_psnr_a = val_metrics.get('psnr', 0)
                print(f"  ✓ 新的最佳模型！(best_psnr: {best_psnr_a:.4f} dB)")
            
            if args.use_wandb:
                wandb.log({
                    'phase': 'A', 
                    'epoch': epoch, 
                    'train_loss': loss, 
                    'val_loss': val_loss,
                    **{f'train_{k}': v for k, v in metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
        else:
            is_best = False
            if args.use_wandb:
                wandb.log({'phase': 'A', 'epoch': epoch, 'train_loss': loss, **metrics})
        
        # 保存最佳模型（不保存中间检查点）
        if is_best:
            save_path = os.path.join(args.save_dir, 'phase_a_best.pth')
            trainer.save_checkpoint(epoch, 'A', save_path, is_best=False)
    
    # 保存Phase A最后一个epoch
    if args.epochs_phase_a > 0:
        last_a_path = os.path.join(args.save_dir, 'phase_a_last.pth')
        trainer.save_checkpoint(args.epochs_phase_a, 'A', last_a_path)
        print(f"✓ Phase A 最后模型已保存: {last_a_path}")

    # --- 在进入 Phase B 前，强制从 Phase A 的 best 恢复 ---
    best_a_path = os.path.join(args.save_dir, 'phase_a_best.pth')
    if os.path.isfile(best_a_path):
        print(f"从 Phase A 最优模型恢复: {best_a_path}")
        ckpt = torch.load(best_a_path, map_location=args.device)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("警告：未找到 phase_a_best.pth，将从 Phase A 最后一轮权重继续")


    # ====== 关键改动：Phase B 只微调解码器和风格编码器，冻结内容编码器 ======
    # 这样可以保持 Phase A 已经学好的“结构/几何”表示，只针对红外强度做 PSNR 微调
    for p in trainer.model.content_encoder.parameters():
        p.requires_grad = False
    # 如果你想更激进一点，也可以把可见光风格编码器冻结，只调 IR 风格 + 解码器：
    # for p in trainer.model.visible_style_encoder.parameters():
    #     p.requires_grad = False
    # 域判别器在我们简化后的 Phase B 里已经不用了，不需要管它

    # 域判别器如果 Phase B 不再用 GAN，可以一并冻结
    for p in trainer.model.domain_discriminator.parameters():
        p.requires_grad = False


    
    # Phase B: 特征对齐训练
    if args.epochs_phase_b > 0:
        print("\n" + "=" * 80)
        print("Phase B: 特征对齐（对比学习 + 域对抗 + MMD）")
        print("=" * 80)
        
        best_val_loss_b = float('inf')
        best_psnr_b = -1e9

        for epoch in range(1, args.epochs_phase_b + 1):
            # 训练
            loss, metrics = trainer.train_epoch(train_loader, phase='B', epoch=epoch)
            
            print(f"Epoch {epoch}/{args.epochs_phase_b} - Train Loss: {loss:.4f}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # 验证评估（每5个epoch评估一次）
            if epoch % 1 == 0 or epoch == args.epochs_phase_b:
                print(f"\n  [验证评估 Epoch {epoch}]")
                val_loss, val_metrics, _, _ = trainer.evaluate(val_loader, phase='B')
                print(f"  Val Loss: {val_loss:.4f}")
                for k, v in val_metrics.items():
                    print(f"    {k}: {v:.4f}")
                # 检查是否是最佳模型
                #is_best = val_loss < best_val_loss_b
                is_best = val_metrics.get('psnr', 0) > best_psnr_b
                if is_best:
                    #best_val_loss_b = val_loss
                    best_psnr_b = val_metrics.get('psnr', 0)
                    print(f"  ✓ 新的最佳模型！(best_psnr: {best_psnr_b:.4f} dB)")
                
                if args.use_wandb:
                    wandb.log({
                        'phase': 'B', 
                        'epoch': epoch, 
                        'train_loss': loss, 
                        'val_loss': val_loss,
                        **{f'train_{k}': v for k, v in metrics.items()},
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    })
            else:
                is_best = False
                if args.use_wandb:
                    wandb.log({'phase': 'B', 'epoch': epoch, 'train_loss': loss, **metrics})
            
            # 保存最佳模型（不保存中间检查点）
            if is_best:
                save_path = os.path.join(args.save_dir, 'phase_b_best.pth')
                trainer.save_checkpoint(epoch, 'B', save_path, is_best=False)
        
        # 保存Phase B最后一个epoch
        last_b_path = os.path.join(args.save_dir, 'phase_b_last.pth')
        trainer.save_checkpoint(args.epochs_phase_b, 'B', last_b_path)
        print(f"✓ Phase B 最后模型已保存: {last_b_path}")
    
    # 保存最终模型：仅当运行了Phase B时另存final_model，否则直接使用Phase A的最后模型
    if args.epochs_phase_b > 0:
        final_path = os.path.join(args.save_dir, 'final_model.pth')
        final_phase = 'B'
        trainer.save_checkpoint(args.epochs_phase_a + args.epochs_phase_b, final_phase, final_path)
    else:
        final_phase = 'A'
        final_path = os.path.join(args.save_dir, 'phase_a_last.pth')
        print("\n未运行 Phase B：跳过保存 final_model.pth，使用 Phase A 最后模型进行评估。")
    
    print("\n" + "=" * 80)
    print("训练完成！开始最终评估...")
    print("=" * 80)
    
    # 最终评估
    print("\n[最终评估] 在验证集上评估模型性能")
    val_loss, val_metrics, generated_samples, target_samples = trainer.evaluate(
        val_loader, phase=final_phase
    )
    
    print(f"\n最终验证结果:")
    print(f"  Loss: {val_loss:.4f}")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 保存可视化样本
    if generated_samples and target_samples:
        print(f"\n保存可视化样本...")
        vis_dir = os.path.join(args.save_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        
        import matplotlib.pyplot as plt
        
        for idx, (gen, tgt) in enumerate(zip(generated_samples[:3], target_samples[:3])):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 转为numpy并适配单通道
            g = gen[0].numpy()
            t = tgt[0].numpy()
            gen_img = g[0] if g.shape[0] == 1 else np.transpose(g, (1, 2, 0))
            tgt_img = t[0] if t.shape[0] == 1 else np.transpose(t, (1, 2, 0))
            residual = np.abs(gen_img - tgt_img)

            # 归一化到[0,1]
            gen_img = np.clip(gen_img, 0, 1)
            tgt_img = np.clip(tgt_img, 0, 1)
            residual = np.clip(residual, 0, 1)

            axes[0].imshow(tgt_img, cmap='gray' if tgt_img.ndim == 2 else None)
            axes[0].set_title('真实红外 (Target)')
            axes[0].axis('off')
            
            axes[1].imshow(gen_img, cmap='gray' if gen_img.ndim == 2 else None)
            axes[1].set_title('生成红外 (Generated)')
            axes[1].axis('off')
            
            axes[2].imshow(residual.squeeze() if residual.ndim == 3 else residual, cmap='hot')
            axes[2].set_title('残差图 (Residual)')
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f'sample_{idx+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 保存样本 {idx+1}: {save_path}")
    
    print("\n" + "=" * 80)
    print("全部完成！")
    print("=" * 80)
    print(f"\n模型文件:")
    if args.epochs_phase_b > 0:
        print(f"  - 最终模型: {final_path}")
        print(f"  - 最佳模型: {final_path.replace('.pth', '_best.pth')}")
    else:
        print(f"  - Phase A 最后模型: {final_path}")
    print(f"\n可视化结果:")
    print(f"  - 位置: {os.path.join(args.save_dir, 'visualization')}")
    print(f"\n最终性能:")
    print(f"  - Val Loss: {val_loss:.4f}")
    print(f"  - PSNR: {val_metrics.get('psnr', 0):.2f} dB")
    
    # Phase C说明
    print("\n" + "-" * 80)
    print("下一步（可选）:")
    print("  Phase C - 异常检测训练:")
    print("    python train_anomaly_detector.py \\")
    print(f"        --checkpoint {final_path} \\")
    print("        --method density")
    print("-" * 80)


if __name__ == '__main__':
    main()

