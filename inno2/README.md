# 创新点2：跨模态可见光→红外生成与残差检测

## 概述

本模块实现了基于内容-风格解耦的跨模态生成模型，用于从可见光图像生成对应的红外图像，并通过残差分析实现故障检测。

### 核心思想

1. **内容-风格解耦**：将图像分解为模态不变的结构特征（内容）和模态特定的辐射特性（风格）
2. **跨模态对齐**：通过域对抗学习和对比学习实现特征空间对齐
3. **残差检测**：正常工况下生成的红外图像与真实红外的残差可用于异常检测

### 模型架构

```
可见光图像 V          红外图像 IR
    |                    |
    v                    v
[E_c] -------------- [E_c]  ← 内容编码器（共享）
    |                |
    |                |--- 域对抗 + 对比学习
    v                v
 content_v       content_ir
    |                
    |            [E_ir]  ← 红外风格编码器
    |                |
    |                v
    |            style_ir
    |                |
    +-------> [G_ir] <---  ← 红外解码器（AdaIN）
                |
                v
            IR̂ (生成的红外)
                |
                v
            残差 = |IR - IR̂|
```

## 使用方法

### 快速导航

**基础流程（3步，推荐）**：
1. 数据准备 → 2. 模型训练 → 3. 异常检测 ✓

**高级流程（+1步，可选）**：
基础流程 + Phase C（训练专门的异常检测器）

---

### 1. 数据准备

```bash
python prepare_data.py
```

这将：
- 解压 `/mnt/e/code/project/Dataset-total/红外图像.rar`
- 提取 "2020-06-16-艾山变巡检任务" 文件夹中的成对图像
- 筛选正常样本（UserDefectType == 1）
- 生成 `inno2.csv` 文件

输出CSV格式：
```
visible_path, infrared_path, label_en, Distance, Humidity, Temperature, Weather, WindSpeed
```

### 2. 模型训练

#### Phase A: 重构训练（50 epochs）

```bash
python train_model.py \
    --csv_path inno2.csv \
    --epochs_phase_a 50 \
    --epochs_phase_b 0 \
    --batch_size 8 \
    --img_size 256
```

目标：学习基本的跨模态生成能力
损失：L1 + L2 + MS-SSIM + TV正则

#### Phase B: 特征对齐（50 epochs）

```bash
python train_model.py \
    --csv_path inno2.csv \
    --epochs_phase_a 0 \
    --epochs_phase_b 50 \
    --batch_size 8 \
    --img_size 256 \
    --lambda_nce 0.1 \
    --lambda_domain 0.1 \
    --lambda_mmd 0.01
```

目标：增强跨模态特征对齐
额外损失：
- InfoNCE（对比学习）
- 域对抗（GRL）
- MMD（统计匹配）

#### 完整训练（推荐）

```bash
python train_model.py \
    --csv_path inno2.csv \
    --epochs_phase_a 50 \
    --epochs_phase_b 50 \
    --batch_size 8 \
    --img_size 256 \
    --device cuda \
    --use_wandb
```

### 3. 推理与异常检测

#### 基础异常检测（推荐）

```bash
# 使用 Phase B 模型直接检测
python inference.py \
    --checkpoint checkpoints/final_model.pth \
    --visible path/to/visible.jpg \
    --infrared path/to/infrared.jpg \
    --output_dir results/
```

输出：
- `generated_infrared.jpg` - 生成的红外图像
- `residual_heatmap.jpg` - 残差热力图 ⭐
- `visualization.png` - 完整可视化对比
- `anomaly_scores.txt` - 异常分数报告

**这已经足够进行异常检测！** 残差热力图会清晰显示异常区域。

#### 高级异常检测（可选）

如果需要更精确的异常分数，可以先训练 Phase C：

```bash
# 先训练异常检测器（仅需一次）
python train_anomaly_detector.py \
    --checkpoint checkpoints/final_model.pth \
    --csv_path inno2.csv \
    --method density  # 或 teacher_student

# 然后使用（与基础检测相同）
python inference.py \
    --checkpoint checkpoints/final_model.pth \
    --visible path/to/visible.jpg \
    --infrared path/to/infrared.jpg \
    --output_dir results/
```

## 关键特性

### 1. 内容-风格解耦

- **内容编码器 E_c**：提取设备形状、结构等模态不变特征
- **风格编码器 E_v/E_ir**：提取可见光/红外的辐射特性
- **解码器 G_ir**：通过 AdaIN 注入风格，生成红外图像

### 2. 跨模态对齐

- **对比学习（InfoNCE）**：拉近成对样本的内容特征，推开非成对样本
- **域对抗（GRL）**：让判别器无法区分内容特征来自哪个模态
- **统计匹配（MMD/CORAL）**：对齐特征分布的统计量

### 3. 残差异常检测

正常工况下训练的模型能生成"正常态"红外图像。故障样本的特征：

- **热斑**：局部过热 → 残差高值区域
- **热桥**：异常热传导 → 残差模式异常
- **绝缘衰退**：温度分布异常 → 残差空间分布异常

检测方式：
```python
residual = |IR_real - IR_generated|
anomaly_score = mean(residual * mask)  # 聚焦前景区域
```

## 数据增强策略

### 配对一致性

几何变换（旋转、翻转、缩放）必须同时作用于V和IR：

```python
seed = random.randint(0, 2**31)
torch.manual_seed(seed)
visible = transform(visible)
torch.manual_seed(seed)  # 使用相同种子
infrared = transform(infrared)
```

### 颜色扰动限制

- 仅对可见光图像做颜色扰动
- 幅度要小（brightness=0.1），避免破坏跨模态对应
- 红外图像不做颜色扰动

## 技术细节

### 1. 几何配准

如果V和IR存在轻微偏移，建议预处理时进行配准：
```python
# 使用SIFT特征点匹配 + 仿射变换
matcher = cv2.SIFT_create()
kp1, des1 = matcher.detectAndCompute(visible, None)
kp2, des2 = matcher.detectAndCompute(infrared, None)
# ... 匹配 + 估计变换矩阵
```

### 2. 红外强度归一化

```python
# 方法1：Min-Max归一化
ir_norm = (ir - ir.min()) / (ir.max() - ir.min())

# 方法2：基于温度范围的归一化（如果有温度标定）
T_min, T_max = 273.15, 373.15  # 0°C - 100°C
ir_norm = (ir - T_min) / (T_max - T_min)
```

### 3. 掩膜加权

为了突出设备前景区域，可以使用掩膜加权损失：

```python
# 简单阈值法生成掩膜
mask = (ir > threshold).float()

# 加权L1损失
weighted_l1 = (mask * |pred - target|).sum() / mask.sum()
```

## Phase C: 异常检测（可选高级功能）

**重要说明**：Phase C 是**可选的**！`inference.py` 已经可以通过残差进行异常检测。

### 为什么 Phase C 是可选的？

**基础检测（inference.py）已足够**：
```python
# inference.py 中的异常检测
residual = |IR_real - IR_generated|
anomaly_score = mean(residual)  # 简单但有效
```

**Phase C 提供更精细的检测**：
- 更准确的异常分数（基于统计建模）
- 更好的阈值自适应能力
- 适用于需要精细调优的场景

### 何时使用 Phase C？

✓ **使用基础检测（inference.py）如果**：
- 你只需要快速判断是否异常
- 残差热力图已经能清晰显示故障
- 不需要精确的异常分数

✓ **使用 Phase C 如果**：
- 需要精确的异常分数排序
- 需要自动化阈值选择
- 有大量数据需要精细调优

Phase C在Phase B完成后进行，专注于提升异常检测精度：

### 方案1：密度估计

对正常样本的残差建模：
```python
# 收集正常样本的残差
residuals = []
for sample in normal_samples:
    ir_gen = model(sample.visible)
    residual = sample.infrared - ir_gen
    residuals.append(residual)

# 拟合高斯分布
mu, sigma = fit_gaussian(residuals)

# 异常检测
def detect_anomaly(visible, infrared):
    ir_gen = model(visible)
    residual = infrared - ir_gen
    score = mahalanobis_distance(residual, mu, sigma)
    return score
```

### 方案2：Teacher-Student

```python
# Teacher: 冻结的生成器
teacher = model.eval()

# Student: 学习预测Teacher的特征
student = StudentNetwork()

# 训练Student预测正常特征
for sample in normal_samples:
    with torch.no_grad():
        teacher_feat = teacher.content_encoder(sample.visible)
    student_feat = student(sample.visible)
    loss = MSE(student_feat, teacher_feat)

# 异常检测：偏离正常特征
def detect_anomaly(visible):
    with torch.no_grad():
        teacher_feat = teacher.content_encoder(visible)
    student_feat = student(visible)
    score = |teacher_feat - student_feat|
    return score
```

## 实验建议

### 超参数调优

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `lr_g` | 1e-4 | 5e-5 ~ 2e-4 | 生成器学习率 |
| `lambda_nce` | 0.1 | 0.05 ~ 0.2 | 对比学习权重 |
| `lambda_domain` | 0.1 | 0.05 ~ 0.2 | 域对抗权重 |
| `lambda_tv` | 0.1 | 0.01 ~ 0.5 | TV正则权重 |
| `batch_size` | 8 | 4 ~ 16 | 根据GPU显存调整 |

### 训练监控

关键指标：
- **L1/L2 Loss**：应持续下降，Phase A结束时 < 0.1
- **MS-SSIM**：结构相似性，越小越好，< 0.2为佳
- **NCE Loss**：对比学习损失，Phase B应下降
- **Domain Loss (D)**：判别器损失，应稳定在0.6-0.7（混淆状态）

### 常见问题

1. **生成图像模糊**
   - 增大 `lambda_ssim`
   - 减小 `lambda_tv`
   - 检查数据配准质量

2. **训练不稳定**
   - 降低学习率
   - 增加梯度裁剪 `clip_grad_norm`
   - 先训练Phase A更长时间

3. **域对抗失效**
   - 调整GRL的alpha增长速度
   - 平衡判别器和生成器的更新频率
   - 检查 `lambda_domain` 权重

## 文件结构

```
inno2/
├── 核心脚本（必需）
│   ├── prepare_data.py          # Step 1: 数据准备
│   ├── train_model.py           # Step 2: 模型训练（Phase A & B）
│   └── inference.py             # Step 3: 推理与异常检测 ⭐
│
├── 高级功能（可选）
│   ├── train_anomaly_detector.py # Phase C: 训练专门的异常检测器（可选）
│   ├── utils.py                  # 工具函数（配准、归一化等）
│   └── run_full_pipeline.sh      # 一键运行脚本
│
├── 配置与文档
│   ├── requirements.txt         # 依赖包
│   ├── config.yaml              # 配置文件
│   └── README.md               # 本文件
│
└── 生成文件（运行后）
    ├── inno2.csv               # 数据索引
    ├── checkpoints/            # 模型检查点
    │   ├── phase_a_epoch_10.pth
    │   ├── phase_b_epoch_10.pth
    │   └── final_model.pth
    └── anomaly_models/         # 异常检测器（Phase C后，可选）
        ├── density_estimator.pkl
        └── teacher_student_detector.pth
```

### 说明

**基础流程（推荐）**：
1. `prepare_data.py` → 准备数据
2. `train_model.py` → 训练生成模型
3. `inference.py` → 直接使用残差检测异常 ✓

**高级流程（可选）**：
1-2. 同上
3. `train_anomaly_detector.py` → 训练专门的异常检测器（更精确）
4. `inference.py` → 使用高级检测器

## 引用与致谢

本模块基于以下研究思路：
- 内容-风格解耦：MUNIT, DRIT
- 跨模态对齐：CMC (Contrastive Multiview Coding)
- 域对抗：DANN (Domain-Adversarial Neural Networks)
- 残差检测：AnoGAN, f-AnoGAN

## 许可

本代码仅供学术研究使用。

