#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创新点2 - Phase C: 异常检测训练

在Phase B完成后，使用正常样本训练异常检测器。
支持两种方案：
1. 密度估计（高斯分布拟合）
2. Teacher-Student架构
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from train_model import CrossModalGenerationModel, PairedVIRDataset
from torchvision import transforms


# ===========================
# 方案1: 密度估计
# ===========================

class GaussianDensityEstimator:
    """高斯密度估计器"""
    
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.inv_sigma = None
    
    def fit(self, residuals: np.ndarray):
        """
        拟合高斯分布
        
        Args:
            residuals: 残差数组 [N, C, H, W] 或 [N, D]
        """
        # 展平
        if len(residuals.shape) > 2:
            N = residuals.shape[0]
            residuals = residuals.reshape(N, -1)
        
        # 计算均值和协方差
        self.mu = np.mean(residuals, axis=0)
        self.sigma = np.cov(residuals, rowvar=False)
        
        # 添加正则化防止奇异
        self.sigma += np.eye(self.sigma.shape[0]) * 1e-6
        
        # 计算逆矩阵
        self.inv_sigma = np.linalg.inv(self.sigma)
        
        print(f"✓ 高斯分布拟合完成")
        print(f"  均值范围: [{self.mu.min():.4f}, {self.mu.max():.4f}]")
        print(f"  标准差范围: [{np.sqrt(np.diag(self.sigma)).min():.4f}, {np.sqrt(np.diag(self.sigma)).max():.4f}]")
    
    def score(self, residual: np.ndarray) -> float:
        """
        计算马氏距离（异常分数）
        
        Args:
            residual: 单个残差 [C, H, W] 或 [D]
        
        Returns:
            异常分数（越大越异常）
        """
        # 展平
        if len(residual.shape) > 1:
            residual = residual.flatten()
        
        # 马氏距离
        diff = residual - self.mu
        score = np.sqrt(diff.T @ self.inv_sigma @ diff)
        
        return float(score)
    
    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'mu': self.mu,
                'sigma': self.sigma,
                'inv_sigma': self.inv_sigma
            }, f)
        print(f"✓ 密度估计器已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.mu = data['mu']
        self.sigma = data['sigma']
        self.inv_sigma = data['inv_sigma']
        print(f"✓ 密度估计器已加载: {path}")


# ===========================
# 方案2: Teacher-Student
# ===========================

class StudentNetwork(nn.Module):
    """学生网络：预测Teacher的内容特征"""
    
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, feature_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TeacherStudentDetector:
    """Teacher-Student异常检测器"""
    
    def __init__(self, teacher_model: CrossModalGenerationModel, device: str = 'cuda'):
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.student = StudentNetwork().to(device)
        self.device = device
        
        # 记录正常分布
        self.normal_mu = None
        self.normal_sigma = None
    
    def train_student(self, dataloader: DataLoader, epochs: int = 20, lr: float = 1e-3):
        """训练学生网络"""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        
        print(f"\n训练学生网络 ({epochs} epochs)...")
        
        for epoch in range(1, epochs + 1):
            self.student.train()
            total_loss = 0
            
            for batch in tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}'):
                visible = batch['visible'].to(self.device)
                
                # Teacher特征（冻结）
                with torch.no_grad():
                    teacher_content = self.teacher.content_encoder(visible)
                    teacher_feat = F.adaptive_avg_pool2d(teacher_content, 1).view(visible.size(0), -1)
                
                # Student预测
                student_feat = self.student(visible)
                
                # 损失
                loss = F.mse_loss(student_feat, teacher_feat)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch} - Loss: {avg_loss:.6f}")
        
        print("✓ 学生网络训练完成")
    
    def compute_normal_distribution(self, dataloader: DataLoader):
        """计算正常样本的特征分布"""
        self.student.eval()
        
        print("\n计算正常特征分布...")
        
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='提取特征'):
                visible = batch['visible'].to(self.device)
                
                # Teacher特征
                teacher_content = self.teacher.content_encoder(visible)
                teacher_feat = F.adaptive_avg_pool2d(teacher_content, 1).view(visible.size(0), -1)
                
                all_features.append(teacher_feat.cpu().numpy())
        
        # 合并所有特征
        all_features = np.concatenate(all_features, axis=0)
        
        # 计算均值和协方差
        self.normal_mu = np.mean(all_features, axis=0)
        self.normal_sigma = np.cov(all_features, rowvar=False)
        
        # 添加正则化
        self.normal_sigma += np.eye(self.normal_sigma.shape[0]) * 1e-6
        
        print(f"✓ 正常分布计算完成")
        print(f"  特征维度: {self.normal_mu.shape[0]}")
    
    def score(self, visible_image: torch.Tensor) -> float:
        """
        计算异常分数
        
        Args:
            visible_image: 可见光图像 [1, 3, H, W]
        
        Returns:
            异常分数
        """
        self.student.eval()
        
        with torch.no_grad():
            visible_image = visible_image.to(self.device)
            
            # Teacher特征
            teacher_content = self.teacher.content_encoder(visible_image)
            teacher_feat = F.adaptive_avg_pool2d(teacher_content, 1).view(1, -1)
            teacher_feat = teacher_feat.cpu().numpy()[0]
            
            # 马氏距离
            diff = teacher_feat - self.normal_mu
            inv_sigma = np.linalg.inv(self.normal_sigma)
            score = np.sqrt(diff.T @ inv_sigma @ diff)
        
        return float(score)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'normal_mu': self.normal_mu,
            'normal_sigma': self.normal_sigma
        }, path)
        print(f"✓ Teacher-Student检测器已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.normal_mu = checkpoint['normal_mu']
        self.normal_sigma = checkpoint['normal_sigma']
        print(f"✓ Teacher-Student检测器已加载: {path}")


# ===========================
# 主训练流程
# ===========================

def extract_residuals(model, dataloader, device, residual_size=32):
    """提取所有正常样本的残差"""
    model.eval()
    
    all_residuals = []
    
    print("\n提取正常样本残差...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='处理样本'):
            visible = batch['visible'].to(device)
            infrared = batch['infrared'].to(device)
            
            # 生成红外
            outputs = model(visible, infrared=None)
            generated_ir = outputs['generated_ir']
            
            # 计算残差
            residual = torch.abs(infrared - generated_ir)
            
            if residual_size is not None:
                residual = F.interpolate(
                    residual,
                    size=(residual_size, residual_size),
                    mode='bilinear',
                    align_corners=False
                )
            
            all_residuals.append(residual.cpu().numpy().astype(np.float32))
    
    # 合并
    all_residuals = np.concatenate(all_residuals, axis=0)
    
    print(f"✓ 提取完成: {all_residuals.shape[0]} 个样本")
    
    return all_residuals


def main():
    parser = argparse.ArgumentParser(description='Phase C - 异常检测训练')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Phase B模型检查点')
    parser.add_argument('--csv_path', type=str, default='/mnt/e/code/project/inno2/inno2.csv')
    parser.add_argument('--method', type=str, default='density', choices=['density', 'teacher_student'],
                       help='异常检测方法')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='/mnt/e/code/project/inno2/anomaly_models')
    parser.add_argument('--residual_size', type=int, default=32,
                        help='残差图下采样尺寸（边长），避免协方差矩阵过大')
    
    # Teacher-Student特定参数
    parser.add_argument('--student_epochs', type=int, default=20)
    parser.add_argument('--student_lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Phase C - 异常检测训练")
    print(f"方法: {args.method}")
    print("=" * 80)
    
    # 加载Phase B模型
    print("\n[1/3] 加载Phase B模型...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']
    model = CrossModalGenerationModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    print(f"✓ 模型加载成功 (Epoch {checkpoint['epoch']}, Phase {checkpoint['phase']})")
    
    # 加载数据（仅正常样本）
    print("\n[2/3] 加载数据...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = PairedVIRDataset(args.csv_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"✓ 数据集加载: {len(dataset)} 个正常样本")
    
    # 训练异常检测器
    print("\n[3/3] 训练异常检测器...")
    
    if args.method == 'density':
        # 方案1: 密度估计
        print("\n使用高斯密度估计方法...")
        
        # 提取残差
        residuals = extract_residuals(model, dataloader, args.device, residual_size=args.residual_size)
        
        # 拟合高斯分布
        estimator = GaussianDensityEstimator()
        estimator.fit(residuals)
        
        # 保存
        save_path = os.path.join(args.output_dir, 'density_estimator.pkl')
        estimator.save(save_path)
        
        # 测试
        print("\n测试异常分数（前5个样本）:")
        for i in range(min(5, len(residuals))):
            score = estimator.score(residuals[i])
            print(f"  样本 {i}: {score:.4f}")
    
    elif args.method == 'teacher_student':
        # 方案2: Teacher-Student
        print("\n使用Teacher-Student方法...")
        
        detector = TeacherStudentDetector(model, args.device)
        
        # 训练学生网络
        detector.train_student(dataloader, epochs=args.student_epochs, lr=args.student_lr)
        
        # 计算正常分布
        detector.compute_normal_distribution(dataloader)
        
        # 保存
        save_path = os.path.join(args.output_dir, 'teacher_student_detector.pth')
        detector.save(save_path)
        
        # 测试
        print("\n测试异常分数（前5个样本）:")
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            visible = batch['visible'][:1]  # 取第一个
            score = detector.score(visible)
            print(f"  样本 {i}: {score:.4f}")
    
    print("\n" + "=" * 80)
    print("Phase C 训练完成！")
    print(f"模型已保存到: {args.output_dir}")
    print("=" * 80)
    
    print("\n使用方法:")
    print("  在inference.py中加载训练好的异常检测器")
    print("  对新样本计算异常分数，超过阈值即判定为异常")


if __name__ == '__main__':
    main()

