#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创新点2 - Step 1: 数据提取与构建CSV
从红外图像数据集中提取成对的可见光-红外图像，生成训练CSV文件
"""

import os
import json
import pandas as pd
import zipfile
import rarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import shutil


# 标签映射：中文到英文
LABEL_MAPPING = {
    "绝缘子": "Insulator",
    "电压互感器": "VT",
    "电流互感器": "CT",
    "避雷器": "Arrester",
    "断路器": "Breaker",
    "隔离开关": "IS"
}


def extract_rar(rar_path: str, extract_to: str) -> str:
    """
    解压RAR文件
    
    Args:
        rar_path: RAR文件路径
        extract_to: 解压目标路径
    
    Returns:
        解压后的路径
    """
    print(f"正在解压 {rar_path} 到 {extract_to}...")
    
    if not os.path.exists(rar_path):
        raise FileNotFoundError(f"RAR文件不存在: {rar_path}")
    
    # 创建解压目标目录
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(extract_to)
        print(f"解压完成！")
        return extract_to
    except Exception as e:
        print(f"解压失败: {e}")
        print("尝试使用系统unrar命令...")
        import subprocess
        try:
            subprocess.run(['unrar', 'x', '-y', rar_path, extract_to], check=True)
            print("解压完成！")
            return extract_to
        except Exception as e2:
            raise RuntimeError(f"无法解压RAR文件: {e2}")


def extract_label_from_filename(filename: str) -> Optional[str]:
    """
    从文件名中提取设备类别标签
    
    文件名格式: ...XXXX-[A|B|C|其他]-本体...
    提取XXXX作为类别，并映射到英文标签
    
    Args:
        filename: 文件名
    
    Returns:
        英文标签或None
    """
    # 尝试匹配模式: -XXXX-[A|B|C|其他]-本体
    # 或更宽松的模式: -XXXX-
    
    # 首先尝试找到所有连字符分隔的部分
    parts = filename.split('-')
    
    # 遍历所有部分，查找匹配的设备类别
    for i, part in enumerate(parts):
        # 检查是否是六类设备之一
        for chinese_label, english_label in LABEL_MAPPING.items():
            if chinese_label in part:
                return english_label
    
    # 如果没有直接匹配，尝试更宽松的匹配
    for chinese_label, english_label in LABEL_MAPPING.items():
        if chinese_label in filename:
            return english_label
    
    return None


def load_dat_file(dat_path: str) -> Optional[Dict]:
    """
    加载.dat文件（JSON格式）
    
    Args:
        dat_path: dat文件路径
    
    Returns:
        解析后的字典或None
    """
    try:
        with open(dat_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"警告: 无法读取 {dat_path}: {e}")
        return None


def process_directory(base_dir: str, target_folder: str = "2020-06-16-艾山变巡检任务") -> List[Dict]:
    """
    处理数据目录，提取成对的可见光-红外图像
    
    Args:
        base_dir: 基础目录路径
        target_folder: 目标文件夹名称
    
    Returns:
        处理后的数据列表
    """
    # 查找"原始图"文件夹
    original_img_dir = None
    for root, dirs, files in os.walk(base_dir):
        if "原始图" in dirs:
            original_img_dir = os.path.join(root, "原始图")
            break
    
    if original_img_dir is None:
        raise FileNotFoundError(f"在 {base_dir} 中找不到'原始图'文件夹")
    
    print(f"找到原始图文件夹: {original_img_dir}")
    
    # 定位目标文件夹
    target_dir = os.path.join(original_img_dir, target_folder)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"目标文件夹不存在: {target_dir}")
    
    print(f"处理文件夹: {target_dir}")
    
    # 收集所有.dat文件
    dat_files = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.dat'):
                dat_files.append(os.path.join(root, file))
    
    print(f"找到 {len(dat_files)} 个.dat文件")
    
    # 处理每个.dat文件
    valid_samples = []
    skipped_count = 0
    no_label_count = 0
    
    for dat_path in dat_files:
        dat_data = load_dat_file(dat_path)
        if dat_data is None:
            skipped_count += 1
            continue
        
        # 筛选条件1: UserDefectType == 1 (正常样本)
        if dat_data.get('UserDefectType') != 1:
            skipped_count += 1
            continue
        
        # 筛选条件2: 同时存在DCPath和IRPath
        dc_path = dat_data.get('DCPath')
        ir_path = dat_data.get('IRPath')
        
        if not dc_path or not ir_path:
            skipped_count += 1
            continue
        
        # 构建完整路径
        dat_dir = os.path.dirname(dat_path)
        full_dc_path = os.path.join(dat_dir, dc_path)
        full_ir_path = os.path.join(dat_dir, ir_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_dc_path) or not os.path.exists(full_ir_path):
            skipped_count += 1
            continue
        
        # 提取标签
        label_en = extract_label_from_filename(dc_path)
        if label_en is None:
            # 尝试从IR路径提取
            label_en = extract_label_from_filename(ir_path)
        
        if label_en is None:
            no_label_count += 1
            skipped_count += 1
            continue
        
        # 提取环境参数
        sample = {
            'visible_path': full_dc_path,
            'infrared_path': full_ir_path,
            'label_en': label_en,
            'Distance': dat_data.get('Distance', ''),
            'Humidity': dat_data.get('Humidity', ''),
            'Temperature': dat_data.get('Temperature', ''),
            'Weather': dat_data.get('Weather', ''),
            'WindSpeed': dat_data.get('WindSpeed', '')
        }
        
        valid_samples.append(sample)
    
    print(f"成功处理: {len(valid_samples)} 个样本")
    print(f"跳过样本: {skipped_count} 个")
    print(f"  - 其中无法提取标签: {no_label_count} 个")
    
    # 统计各类别数量
    if valid_samples:
        label_counts = pd.Series([s['label_en'] for s in valid_samples]).value_counts()
        print("\n各类别样本数量:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
    
    return valid_samples


def main():
    """主函数"""
    # 配置路径
    rar_path = "/mnt/e/code/project/Dataset-total/红外图像.rar"
    extract_dir = "/mnt/e/code/project/Dataset-total/红外图像_extracted"
    output_csv = "/mnt/e/code/project/inno2/inno2.csv"
    
    print("=" * 80)
    print("创新点2 - Step 1: 数据提取与构建CSV")
    print("=" * 80)
    
    # Step 1: 检查是否已解压
    if not os.path.exists(extract_dir):
        print("\n[1/3] 解压数据集...")
        extract_rar(rar_path, extract_dir)
    else:
        print(f"\n[1/3] 数据集已解压: {extract_dir}")
    
    # Step 2: 处理数据
    print("\n[2/3] 处理数据并提取成对样本...")
    try:
        samples = process_directory(extract_dir, "2020-06-16-艾山变巡检任务")
    except Exception as e:
        print(f"错误: {e}")
        print("\n尝试查找可用的文件夹...")
        # 列出可用文件夹
        original_img_dir = None
        for root, dirs, files in os.walk(extract_dir):
            if "原始图" in dirs:
                original_img_dir = os.path.join(root, "原始图")
                break
        
        if original_img_dir and os.path.exists(original_img_dir):
            available_folders = [d for d in os.listdir(original_img_dir) 
                               if os.path.isdir(os.path.join(original_img_dir, d))]
            print(f"可用文件夹: {available_folders}")
        raise
    
    # Step 3: 保存CSV
    print("\n[3/3] 保存CSV文件...")
    if samples:
        df = pd.DataFrame(samples)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✓ CSV文件已保存: {output_csv}")
        print(f"  总样本数: {len(df)}")
        print(f"  字段: {list(df.columns)}")
    else:
        print("警告: 没有找到有效样本！")
    
    print("\n" + "=" * 80)
    print("数据准备完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

