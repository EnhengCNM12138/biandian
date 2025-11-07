#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创新点2 工具函数
"""

import os
import numpy as np
import torch
import cv2
from typing import Tuple, Optional
import yaml


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_foreground_mask(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """
    创建前景掩膜
    
    Args:
        image: 输入图像 [H, W, 3] 或 [H, W]
        method: 'otsu' 或 'threshold'
    
    Returns:
        掩膜 [H, W] (0-1)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if method == 'otsu':
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        threshold = np.mean(gray) + np.std(gray)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask.astype(np.float32) / 255.0


def align_image_pair(visible: np.ndarray, infrared: np.ndarray, 
                     max_features: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    对齐可见光和红外图像对
    
    Args:
        visible: 可见光图像 [H, W, 3]
        infrared: 红外图像 [H, W, 3] 或 [H, W]
    
    Returns:
        (对齐后的可见光, 对齐后的红外)
    """
    # 转为灰度
    if len(visible.shape) == 3:
        visible_gray = cv2.cvtColor(visible, cv2.COLOR_RGB2GRAY)
    else:
        visible_gray = visible
    
    if len(infrared.shape) == 3:
        infrared_gray = cv2.cvtColor(infrared, cv2.COLOR_RGB2GRAY)
    else:
        infrared_gray = infrared
    
    # SIFT特征检测
    sift = cv2.SIFT_create(max_features)
    kp1, des1 = sift.detectAndCompute(visible_gray, None)
    kp2, des2 = sift.detectAndCompute(infrared_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("警告: 特征点不足，跳过配准")
        return visible, infrared
    
    # BFMatcher匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        print("警告: 好的匹配点不足，跳过配准")
        return visible, infrared
    
    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 估计仿射变换矩阵
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if M is None:
        print("警告: 无法估计变换矩阵，跳过配准")
        return visible, infrared
    
    # 应用变换
    h, w = infrared.shape[:2]
    visible_aligned = cv2.warpAffine(visible, M, (w, h))
    
    print(f"✓ 图像配准成功 (使用 {len(good_matches)} 个匹配点)")
    
    return visible_aligned, infrared


def normalize_infrared(infrared: np.ndarray, method: str = 'minmax',
                      temp_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    红外图像归一化
    
    Args:
        infrared: 红外图像
        method: 'minmax' 或 'temperature'
        temp_range: 温度范围 (T_min, T_max)，仅在method='temperature'时使用
    
    Returns:
        归一化后的图像
    """
    if method == 'minmax':
        ir_min = infrared.min()
        ir_max = infrared.max()
        normalized = (infrared - ir_min) / (ir_max - ir_min + 1e-8)
    
    elif method == 'temperature' and temp_range is not None:
        T_min, T_max = temp_range
        normalized = (infrared - T_min) / (T_max - T_min)
        normalized = np.clip(normalized, 0, 1)
    
    else:
        # 默认使用minmax
        ir_min = infrared.min()
        ir_max = infrared.max()
        normalized = (infrared - ir_min) / (ir_max - ir_min + 1e-8)
    
    return normalized


def compute_image_statistics(image: np.ndarray) -> dict:
    """
    计算图像统计信息
    
    Args:
        image: 输入图像
    
    Returns:
        统计字典
    """
    stats = {
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'median': float(np.median(image))
    }
    
    return stats


def visualize_feature_maps(features: torch.Tensor, num_maps: int = 16, 
                          save_path: Optional[str] = None):
    """
    可视化特征图
    
    Args:
        features: 特征张量 [B, C, H, W]
        num_maps: 显示的特征图数量
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    features = features[0].detach().cpu().numpy()  # 取第一个样本
    num_maps = min(num_maps, features.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_maps):
        feat_map = features[i]
        axes[i].imshow(feat_map, cmap='viridis')
        axes[i].set_title(f'Feature {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✓ 特征图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0 if img1.max() <= 1 else 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算SSIM"""
    from skimage.metrics import structural_similarity
    
    # 确保在[0, 1]范围
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0
    
    # 如果是彩色图像，转为灰度
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    
    ssim_value = structural_similarity(img1, img2, data_range=1.0)
    return float(ssim_value)


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: dict, filename: str, is_best: bool = False):
    """保存检查点"""
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace('.pth', '_best.pth')
        torch.save(state, best_filename)


def load_checkpoint(filename: str, model, optimizer=None):
    """加载检查点"""
    if os.path.isfile(filename):
        print(f"加载检查点 '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✓ 加载成功 (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"找不到检查点 '{filename}'")
        return None


if __name__ == '__main__':
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试随机种子
    set_seed(42)
    print("✓ 随机种子设置成功")
    
    # 测试图像统计
    test_img = np.random.rand(256, 256, 3)
    stats = compute_image_statistics(test_img)
    print(f"✓ 图像统计: {stats}")
    
    # 测试PSNR/SSIM
    img1 = np.random.rand(256, 256)
    img2 = img1 + np.random.randn(256, 256) * 0.1
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    print(f"✓ PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    
    print("\n所有测试通过！")

