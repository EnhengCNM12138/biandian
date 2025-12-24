# 创新点2：跨模态生成模型结构与流程概览

以下示意严格按照时间顺序展开，覆盖 Phase A/B 前向网络细节、训练损失组合以及 Phase C 推理流程。所有图均为 ASCII，以便快速查阅与维护。

---

## 1. Phase A/B 前向网络结构（按步骤展开）

```
Step 0  Inputs
────────────────────────────────────────────────────────
Iv   ∈ ℝ^{3×H×W}  → 可见光图像
Iir  ∈ ℝ^{1×H×W}  → 红外图像（训练期可用，推理可选）
env  ∈ ℝ^{5}      → [distance, humidity, temperature, weather, windspeed]

Step 1  预处理
────────────────────────────────────────────────────────
Iv  ──► ToTensor → ImageNet 规范化 ──────────┐
                                             │
Iir ──► repeat(3通道) → ToTensor → 规范化 ──►│─ Iir_rgb（仅训练）
                                             │
                                             └─ 输出 Iv_norm / Iir_rgb / env

Step 2  内容编码器 E_c（参数共享，Conv-Down + Residual）
────────────────────────────────────────────────────────
Iv_norm ─► Conv7×7/stride2 → 64 → IN → ReLU ─────────┐  → c1_v (64, H/2)
          Conv3×3/stride2 → 128 → IN → ReLU ────────┐│  → c2_v (128, H/4)
          Conv3×3/stride2 → 256 → IN → ReLU ───────┐││
          ResidualBlock ×4 (Conv3×3 + IN + ReLU) ─┘││  → c3_v (256, H/8)
                                                    ││
Iir_rgb ─► 同一 E_c（共享权重） ─────────────────────┘│  → c1_ir/c2_ir/c3_ir（训练期）
                                                    │
                                                    └─ 提供跳连特征与域对齐特征

Step 3  风格编码器
────────────────────────────────────────────────────────
E_v(Iv_norm):
  Conv7×7/stride2 → ReLU → Conv3×3/stride2 → ReLU
  → Conv3×3/stride2 → ReLU → AdaptiveAvgPool2d(1)
  → FC(256) = style_v

E_ir(Iir_rgb)（仅训练）:
  结构同 E_v → FC(256) = style_ir

Step 4  风格向量 → Tokens
────────────────────────────────────────────────────────
style_v/style_ir ─► Linear(256 → 8×64) → reshape → tokens[8,64]
  • 训练期：首选 style_ir tokens（目标域风格）
  • 推理期：仅保留 style_v tokens（CrossAttention 退化）

Step 5  环境调制向量
────────────────────────────────────────────────────────
env ─► Linear(5 → 2×C_stage) → γ_env, β_env
  • 为 G_ir 中的 PhysicsFiLM 提供每阶段的通道级缩放/平移

Step 6  红外解码器 G_ir（逐级上采样，融合条件）
────────────────────────────────────────────────────────
Input: c3_v (256, H/8), tokens, env γ/β

Stage 3 (H/8 → H/4, 通道 128)
  a. Conv3×3 → 128 → ReLU
  b. Upsample ×2（bilinear）
  c. 拼接 c2_v (128) → Conv3×3 → 128 → ReLU
  d. SPADE(c2_v) 注入结构
  e. PhysicsFiLM(env γ/β) 调制通道
  f. CrossAttention(tokens)
        - Q：当前特征
        - K/V：tokens（style_ir or style_v）

Stage 2 (H/4 → H/2, 通道 64)
  与 Stage 3 相同流程，使用 c1_v 作为 SPADE 条件

Stage 1 (H/2 → H, 通道 64)
  a. Conv3×3 → 64 → ReLU
  b. Upsample ×2
  c. 拼接 F.interpolate(c1_v, scale=2)
  d. SPADE(Interp c1_v) + PhysicsFiLM(env) + CrossAttention(tokens)

Output
  Conv7×7 → Sigmoid → Ĩir ∈ ℝ^{1×H×W}

Step 7  域对齐（可选）
────────────────────────────────────────────────────────
c3_v, c3_ir ─► GRL(α) ─► DomainDiscriminator(FC256→256→128→2)
           └► InfoNCE / MMD（基于 GAP 后向量） → 跨模态一致性
```

---

## 2. 训练流程：Phase A 与 Phase B 串联

```
Forward（两阶段共享）
───────────────────────────────────────────────────────────────────────
(Iv, Iir, env)
   │
   ├─► Step1 预处理 → Iv_norm, Iir_rgb, env
   ├─► Step2 内容编码器 → c1_v/c2_v/c3_v (+ c*_ir)
   ├─► Step3 风格编码器 → style_v/style_ir
   ├─► Step4 tokens 生成
   ├─► Step5 env → γ_env/β_env
   └─► Step6 解码器 G_ir → Ĩir

Phase A（基础重构）
───────────────────────────────────────────────────────────────────────
  重构损失：   L2 (主导), Charbonnier(可选)
  结构损失：   MS-SSIM
  正则项：     TV
  高频项：     Grad, Lap, Edge-Weighted L1, FFT
  感知项：     VGG 特征差（默认关闭，按需开启）
→ 训练目标：获得稳定、高 PSNR 的跨模态生成能力，输出 Phase A 最优模型

Phase B（细节/对齐增强，继承 Phase A 权重）
───────────────────────────────────────────────────────────────────────
  • 常规做法：冻结 E_c（保持结构编码）与域判别器，仅调节风格编码器 + 解码器
  • Forward 流程保持一致
  • 提升细节相关项权重：Grad、Lap、Edge、FFT、感知等
  • 适度保留 L2 / MS-SSIM（防止发散）
  • 按需启用 InfoNCE / MMD / 域对抗（强化跨模态一致性）
→ 训练目标：在不牺牲 Phase A 性能的前提下增强纹理与高频表现，得到最终生成器


# 创新点2：跨模态生成模型结构与流程概览

                                       环境信息 env (5维)
                                         │
                                         ▼
                             ┌──────────────────────────┐
                             │   PhysicsFiLM Embedding   │
                             │  Linear(5→2C) → γ/β(调制)  │
                             └───────────┬──────────────┘
                                         │
                                         │（注入解码器各级Norm/SPADE进行物理调制）
                                         ▼


输入：Iv（可见光）                                   输入：Iir（红外）〔仅训练期用于风格对齐〕
       │                                                        │
       ▼                                                        ▼
┌──────────────────────────┐                        ┌──────────────────────────┐
│   可见光内容编码器 E_c    │                        │   红外内容编码器 E_c [T] │
│ Conv7×7 s2 → c1_v (H/2)  │                        │ repeat→3ch→Conv→c*_ir   │
│ Conv3×3 s2 → c2_v (H/4)  │                        │ 仅用于训练期对齐，不进解码 │
│ Conv3×3 s2 → c3_v (H/8)  │                        └──────────────────────────┘
└─────────┬────────────────┘
          │
          │  (c1_v, c2_v, c3_v 是唯一的结构特征来源)
          │


风格编码路径（用于 CrossAttention）
────────────────────────────────────────────────
            Iv                                       Iir [T]
            │                                          │
            ▼                                          ▼
┌──────────────────────────┐               ┌──────────────────────────┐
│   可见光风格编码器 E_v    │               │   红外风格编码器 E_ir[T] │
│ Conv→Conv→Conv→GAP→FC256 │               │ 同结构，仅训练期用        │
│        → style_v(256)    │               │        → style_ir(256)   │
└──────┬───────────────────┘               └──────────┬───────────────┘
       │                                               │
       │ 推理期只使用 style_v                          │ 训练期用于风格一致性约束
       └───────────────┬──────────────────────────────┘
                       ▼
           ┌───────────────────────────────────┐
           │ style_to_tokens: Linear → 8×64tok │
           └───────────────────┬───────────────┘
                               │ tokens（作为 CrossAttention 的 K/V）
                               ▼



                          （解码器内部调用注意力）
                           ┌──────────────────────────┐
                           │   CrossAttention2D        │
                           │  Q：来自解码器当前特征     │
                           │  K,V：来自 style tokens   │
                           └─────────────┬────────────┘
                                         │ 注意力调制输出
                                         ▼



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【完整的红外解码器 G_ir（唯一重建管道，从 c3_v→Stage3→Stage2→Stage1→Ĩir）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

c3_v  ────────────────────────────────────────────────────────────────┐
                                                                       │
                                                                       ▼
                     ┌─────────────────────────────────────────────┐
                     │ Stage3（起始：256ch, H/8 分辨率）             │
                     │ ─ Conv3×3 → ReLU                            │
                     │ ─ Upsample ×2                               │
                     │ ─ SPADE(F3, c3_v)〔内容调制〕                │
                     │ ─ + PhysicsFiLM(γ/β)〔环境调制〕             │
                     │ ─ + CrossAttention(tokens)〔风格调制〕       │
                     └───────────────┬─────────────────────────────┘
                                     │  F3_out（H/4）
                                     ▼

c2_v  ───────────────────────────────┐
                                     │ 跳连（skip connection）
                                     ▼
                     ┌─────────────────────────────────────────────┐
                     │ Stage2（H/4）                               │
                     │ ─ Upsample ×2                               │
                     │ ─ SPADE(F2, c2_v)                           │
                     │ ─ + PhysicsFiLM(γ/β)                        │
                     │ ─ + CrossAttention(tokens)                  │
                     └───────────────┬─────────────────────────────┘
                                     │  F2_out（H/2）
                                     ▼

c1_v  ───────────────────────────────┐
                                     │ 跳连
                                     ▼
                     ┌─────────────────────────────────────────────┐
                     │ Stage1（H/2 → H）                           │
                     │ ─ Upsample ×2                               │
                     │ ─ SPADE(F1, c1_v)                           │
                     │ ─ + PhysicsFiLM(γ/β)                        │
                     │ ─ + CrossAttention(tokens)                  │
                     └───────────────┬─────────────────────────────┘
                                     │
                                     ▼

                         Conv7×7 → Sigmoid → 输出 Ĩir（预测红外图）


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【推理流程总结（Inference）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入:
  ├─ Iv (可见光 3×H×W)
  ├─ env (可选，若无则 PhysicsFiLM 默认恒等)
  └─ Iir (真实红外，可选，仅用于残差/评分)

           ┌────────────────────────────────────────────────────────┐
           │   冻结的生成器 (Phase B 最优权重)                     │
           │   - 使用 E_c, E_v, G_ir, PhysicsFiLM                   │
           │   - 不再调用 E_ir（CrossAttention 仅用 style_v）      │
           └─────────────┬──────────────────────────────────────────┘
                         │
                         ▼
                     Ĩir = G_iv(Iv, env)
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        若有真实 Iir：         若仅可见光：
        ────────────          ─────────
        R = |Ĩir - Iir|        可直接输出 Ĩir
        • 残差热力图
        • 残差叠加图
        • 图像保存
              │
              ▼
    多指标异常分数 (inference.py)
        - pixel_mean / pixel_max
        - weighted_mean（中心加权）
        - feature_score（特征层残差）
        - 可叠加高级分支：高斯密度估计、Teacher-Student（train_anomaly_detector.py）
              │
              ▼
    阈值判断 → 正常 / 异常 + 可视化结果

说明：
  • 推理阶段不再计算域对抗、InfoNCE 等对齐损失。
  • CrossAttention 仅依赖 E_v 得到的 style_v tokens；若 env 缺省则 PhysicsFiLM ≈ 恒等。
  • Phase C 的扩展（train_anomaly_detector.py）可基于残差特征进一步建模异常分布。

