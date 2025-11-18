#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创新点2 - 推理与残差异常检测

用法：
    python inference.py --checkpoint checkpoints/final_model.pth \
                       --visible path/to/visible.jpg \
                       --infrared path/to/infrared.jpg \
                       --output_dir results/
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from train_model import CrossModalGenerationModel
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False



class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 设备
        """
        self.device = device
        
        # 加载检查点
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 创建模型
        config = checkpoint['config']
        self.model = CrossModalGenerationModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ 模型加载成功 (Epoch {checkpoint['epoch']}, Phase {checkpoint['phase']})")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def load_image(self, image_path: str, is_infrared: bool = False):
        """
        加载并预处理图像
        
        Args:
            image_path: 图像路径
            is_infrared: 是否为红外图像
        
        Returns:
            预处理后的张量 [1, 3, H, W]
        """
        img = Image.open(image_path)
        
        if is_infrared:
            if img.mode != 'L':
                img = img.convert('L')
            img = img.convert('RGB')
        else:
            img = img.convert('RGB')
        
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor
    
    @torch.no_grad()
    def generate_infrared(self, visible_image: torch.Tensor):
        """
        从可见光生成红外图像
        
        Args:
            visible_image: 可见光图像 [1, 3, H, W]
        
        Returns:
            生成的红外图像 [1, 3, H, W]
        """
        visible_image = visible_image.to(self.device)
        outputs = self.model(visible_image, infrared=None)
        generated_ir = outputs['generated_ir']
        return generated_ir
    
    @torch.no_grad()
    def compute_residual(self, visible_image: torch.Tensor, infrared_image: torch.Tensor):
        """
        计算残差
        
        Args:
            visible_image: 可见光图像 [1, 3, H, W]
            infrared_image: 真实红外图像 [1, 3, H, W]
        
        Returns:
            Dict with keys: generated_ir, residual, anomaly_score
        """
        visible_image = visible_image.to(self.device)
        infrared_image = infrared_image.to(self.device)
        
        # 生成红外
        outputs = self.model(visible_image, infrared=None)
        generated_ir = outputs['generated_ir']
        
        # 计算残差
        residual = torch.abs(infrared_image - generated_ir)
        
        # 异常分数（多种计算方式）
        # 1. 像素级平均
        pixel_score = residual.mean().item()
        
        # 2. 最大残差（检测局部异常）
        max_score = residual.max().item()
        
        # 3. 前景区域加权（假设中心为设备区域）
        h, w = residual.shape[2:]
        center_mask = torch.zeros_like(residual)
        center_mask[:, :, h//4:3*h//4, w//4:3*w//4] = 1.0
        weighted_score = (residual * center_mask).sum() / center_mask.sum()
        weighted_score = weighted_score.item()
        
        # 4. 感知损失（特征级残差）
        content_v = outputs['content_v']
        with torch.no_grad():
            outputs_ir = self.model(infrared_image, infrared=None)
        content_ir = outputs_ir['content_v']
        feature_residual = F.mse_loss(content_v, content_ir).item()
        
        return {
            'generated_ir': generated_ir,
            'residual': residual,
            'scores': {
                'pixel_mean': pixel_score,
                'pixel_max': max_score,
                'weighted': weighted_score,
                'feature': feature_residual
            }
        }
    
    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将张量转换为图像
        
        Args:
            tensor: [1, 3, H, W] or [3, H, W]
        
        Returns:
            [H, W, 3] uint8 array
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        # 处理单通道情况，避免 PIL 无法识别 (H, W, 1) 形状
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(2)
        
        return img
    
    def create_heatmap(self, residual: torch.Tensor, colormap: str = 'jet') -> np.ndarray:
        """
        创建残差热力图
        
        Args:
            residual: 残差张量 [1, 3, H, W]
            colormap: 颜色映射
        
        Returns:
            热力图 [H, W, 3]
        """
        # 转为灰度
        residual_gray = residual.mean(dim=1).squeeze(0).cpu().numpy()
        
        # 归一化到[0, 1]
        residual_norm = (residual_gray - residual_gray.min()) / (residual_gray.max() - residual_gray.min() + 1e-8)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap((residual_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def visualize_results(self, visible_path: str, infrared_path: str, 
                         generated_ir: torch.Tensor, residual: torch.Tensor,
                         scores: dict, output_path: str):
        """
        可视化结果
        
        Args:
            visible_path: 可见光图像路径
            infrared_path: 红外图像路径
            generated_ir: 生成的红外
            residual: 残差
            scores: 异常分数
            output_path: 输出路径
        """
        # 加载原始图像
        visible_img = np.array(Image.open(visible_path).convert('RGB'))
        infrared_img = np.array(Image.open(infrared_path).convert('RGB'))
        
        # 转换张量
        generated_ir_img = self.tensor_to_image(generated_ir)
        
        # 创建热力图
        heatmap = self.create_heatmap(residual)
        # 调整热力图尺寸以匹配原始红外图像，确保叠加时尺寸一致
        if heatmap.shape[:2] != infrared_img.shape[:2]:
            heatmap = cv2.resize(
                heatmap,
                (infrared_img.shape[1], infrared_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行
        axes[0, 0].imshow(visible_img)
        axes[0, 0].set_title('可见光图像 (输入)', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(infrared_img)
        axes[0, 1].set_title('真实红外图像', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(generated_ir_img)
        axes[0, 2].set_title('生成红外图像', fontsize=12)
        axes[0, 2].axis('off')
        
        # 第二行
        axes[1, 0].imshow(heatmap)
        axes[1, 0].set_title('残差热力图', fontsize=12)
        axes[1, 0].axis('off')
        
        # 残差叠加在真实红外上
        overlay = cv2.addWeighted(infrared_img, 0.6, heatmap, 0.4, 0)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('残差叠加', fontsize=12)
        axes[1, 1].axis('off')
        
        # 异常分数显示
        axes[1, 2].axis('off')
        score_text = "异常分数:\n\n"
        score_text += f"像素平均: {scores['pixel_mean']:.4f}\n"
        score_text += f"像素最大: {scores['pixel_max']:.4f}\n"
        score_text += f"加权分数: {scores['weighted']:.4f}\n"
        score_text += f"特征残差: {scores['feature']:.4f}\n"
        
        # 判断是否异常（简单阈值）
        threshold = 0.1  # 可调整
        is_anomaly = scores['weighted'] > threshold
        status = "异常" if is_anomaly else "正常"
        score_text += f"\n状态: {status}"
        
        axes[1, 2].text(0.1, 0.5, score_text, fontsize=14,
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='创新点2 - 推理与残差检测')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--visible', type=str, required=True, help='可见光图像路径')
    parser.add_argument('--infrared', type=str, default=None, help='红外图像路径（可选，用于残差计算）')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("创新点2 - 推理与残差异常检测")
    print("=" * 80)
    
    # 创建推理引擎
    engine = InferenceEngine(args.checkpoint, args.device)
    
    # 加载可见光图像
    print(f"\n加载可见光图像: {args.visible}")
    visible_tensor = engine.load_image(args.visible, is_infrared=False)
    
    # 模式1: 仅生成红外
    if args.infrared is None:
        print("\n[模式1] 仅生成红外图像")
        generated_ir = engine.generate_infrared(visible_tensor)
        
        # 保存生成的红外
        output_path = os.path.join(args.output_dir, 'generated_infrared.jpg')
        generated_ir_img = engine.tensor_to_image(generated_ir)
        Image.fromarray(generated_ir_img).save(output_path)
        print(f"✓ 生成红外已保存: {output_path}")
    
    # 模式2: 残差检测
    else:
        print(f"\n加载红外图像: {args.infrared}")
        infrared_tensor = engine.load_image(args.infrared, is_infrared=True)
        
        print("\n[模式2] 残差异常检测")
        results = engine.compute_residual(visible_tensor, infrared_tensor)
        
        print("\n异常分数:")
        for name, score in results['scores'].items():
            print(f"  {name}: {score:.4f}")
        
        # 保存生成的红外
        output_ir_path = os.path.join(args.output_dir, 'generated_infrared.jpg')
        generated_ir_img = engine.tensor_to_image(results['generated_ir'])
        Image.fromarray(generated_ir_img).save(output_ir_path)
        print(f"\n✓ 生成红外已保存: {output_ir_path}")
        
        # 保存残差热力图
        heatmap_path = os.path.join(args.output_dir, 'residual_heatmap.jpg')
        heatmap = engine.create_heatmap(results['residual'])
        Image.fromarray(heatmap).save(heatmap_path)
        print(f"✓ 残差热力图已保存: {heatmap_path}")
        
        # 保存可视化
        viz_path = os.path.join(args.output_dir, 'visualization.png')
        engine.visualize_results(
            args.visible,
            args.infrared,
            results['generated_ir'],
            results['residual'],
            results['scores'],
            viz_path
        )
        
        # 保存异常分数到文本文件
        score_txt_path = os.path.join(args.output_dir, 'anomaly_scores.txt')
        with open(score_txt_path, 'w') as f:
            f.write("异常分数报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"可见光图像: {args.visible}\n")
            f.write(f"红外图像: {args.infrared}\n\n")
            f.write("分数详情:\n")
            for name, score in results['scores'].items():
                f.write(f"  {name}: {score:.6f}\n")
            
            # 简单判断
            threshold = 0.1
            is_anomaly = results['scores']['weighted'] > threshold
            f.write(f"\n判断结果 (阈值={threshold}): ")
            f.write("异常" if is_anomaly else "正常")
        
        print(f"✓ 分数报告已保存: {score_txt_path}")
    
    print("\n" + "=" * 80)
    print("推理完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

