#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase C 指标评估脚本（在没有真实异常标签时，基于“正常配对 + 合成异常”计算 AUC / F1）

输入：
- inno2.csv：第一列=可见光路径，第二列=红外路径（可有/无表头都支持）
- final_model.pth：Phase B 模型，用于生成红外与算 residual
- density_estimator.pkl：高级 density 模型（可选）
- teacher_student_detector.pth：高级 teacher-student 模型（可选）

输出：
- 控制台打印：三种方法各自 AUC / Best-F1 / best-threshold
- 保存 eval_scores.csv：每张样本的三种分数与 label
"""

import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.metrics import roc_auc_score, f1_score

# 你工程里的类
from inference import InferenceEngine
from train_anomaly_detector import GaussianDensityEstimator, TeacherStudentDetector


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pairs(csv_path: str):
    """
    兼容两种 CSV：
    1) 无表头：第0列=visible，第1列=infrared
    2) 有表头：前两列仍按顺序取
    """
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        # 尝试当作有表头
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError(f"CSV 至少需要两列：visible_path, infrared_path。当前列数={df.shape[1]}")
        vis = df.iloc[:, 0].astype(str).tolist()
        ir = df.iloc[:, 1].astype(str).tolist()
    else:
        vis = df.iloc[:, 0].astype(str).tolist()
        ir = df.iloc[:, 1].astype(str).tolist()

    # 过滤不存在的文件
    pairs = []
    for v, t in zip(vis, ir):
        v = v.strip()
        t = t.strip()
        # 兼容用户漏写 leading slash 的情况
        if v.startswith("mnt/"):
            v = "/" + v
        if t.startswith("mnt/"):
            t = "/" + t
        if os.path.exists(v) and os.path.exists(t):
            pairs.append((v, t))
    if len(pairs) == 0:
        raise FileNotFoundError("CSV 中未找到任何有效的可见光/红外文件路径（检查路径是否带 /mnt/...）")
    return pairs


def cutpaste_visible(img: Image.Image, rng: np.random.RandomState):
    """
    更强的 CutPaste 合成异常：
    - patch 面积更大（15%~35%）
    - patch 做亮度/对比度/饱和度扰动
    - 支持“擦除型”异常（patch 变暗/变亮）与“贴片型”异常
    """
    img = img.convert("RGB")
    w, h = img.size

    # patch 尺寸：更大
    pw = int(rng.uniform(0.15, 0.35) * w)
    ph = int(rng.uniform(0.15, 0.35) * h)
    pw = max(10, min(pw, w - 1))
    ph = max(10, min(ph, h - 1))

    x1 = rng.randint(0, w - pw)
    y1 = rng.randint(0, h - ph)
    patch = img.crop((x1, y1, x1 + pw, y1 + ph))

    # patch 做颜色/亮度扰动（关键：让它像“缺陷”而不是简单复制）
    patch = ImageEnhance.Brightness(patch).enhance(float(rng.uniform(0.6, 1.6)))
    patch = ImageEnhance.Contrast(patch).enhance(float(rng.uniform(0.6, 1.8)))
    patch = ImageEnhance.Color(patch).enhance(float(rng.uniform(0.4, 1.8)))
    patch = ImageEnhance.Sharpness(patch).enhance(float(rng.uniform(0.6, 2.0)))

    out = img.copy()

    mode = rng.choice(["paste", "erase"])  # 贴片型 / 擦除型

    if mode == "paste":
        # 贴到另一个随机位置（尽量远离原位置，增加异常明显度）
        x2 = rng.randint(0, w - pw)
        y2 = rng.randint(0, h - ph)
        out.paste(patch, (x2, y2))
    else:
        # 擦除型：把区域变暗或变亮（模拟污渍/烧蚀/反光）
        x2 = rng.randint(0, w - pw)
        y2 = rng.randint(0, h - ph)
        region = out.crop((x2, y2, x2 + pw, y2 + ph))
        factor = float(rng.uniform(0.3, 0.7)) if rng.rand() < 0.5 else float(rng.uniform(1.3, 2.0))
        region = ImageEnhance.Brightness(region).enhance(factor)
        out.paste(region, (x2, y2))

    return out



def inject_hotspot_ir(ir_tensor: torch.Tensor, rng: np.random.RandomState):
    """
    在红外张量上注入一个“热斑”伪异常（让 residual 增大）
    ir_tensor: [1,3,H,W]  (你的 load_image 对红外会转 L->RGB，所以是3通道)
    """
    x = ir_tensor.clone()
    _, c, h, w = x.shape

    # 选一个圆形热点区域
    cx = int(rng.uniform(0.2, 0.8) * w)
    cy = int(rng.uniform(0.2, 0.8) * h)
    r = int(rng.uniform(0.06, 0.14) * min(h, w))

    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)
    mask = mask.to(x.device).float()[None, None, :, :]  # [1,1,H,W]

    amp = float(rng.uniform(0.25, 0.70))  # 注入幅度（0~1空间）
    x = torch.clamp(x + amp * mask, 0.0, 1.0)
    return x


def best_f1_by_sweep(y_true, scores, n_steps: int = 400):
    """用分位数扫描阈值，取 best F1"""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    qs = np.linspace(0.0, 1.0, n_steps)
    best_f1, best_t = -1.0, None
    for q in qs:
        t = float(np.quantile(scores, q))
        y_pred = (scores > t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="inno2.csv：第1列visible，第2列infrared")
    ap.add_argument("--checkpoint", required=True, help="Phase B final_model.pth")
    ap.add_argument("--density_path", default=None, help="density_estimator.pkl（可选）")
    ap.add_argument("--ts_path", default=None, help="teacher_student_detector.pth（可选）")

    ap.add_argument("--num_normals", type=int, default=300, help="抽多少对正常做评估（<=总数）")
    ap.add_argument("--anomaly_mode", type=str, default="mixed",
                    choices=["mismatch_ir", "hotspot_ir", "visible_cutpaste", "mixed"],
                    help="构造伪异常的方式：错配红外/红外热斑/可见光cutpaste/混合")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_csv", type=str, default="eval_scores.csv")

    args = ap.parse_args()
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    pairs = load_pairs(args.csv_path)
    print(f"[Info] 有效正常配对数量: {len(pairs)}")

    n = min(args.num_normals, len(pairs))
    idx = rng.choice(len(pairs), size=n, replace=False)
    normals = [pairs[i] for i in idx]

    # 初始化推理引擎
    engine = InferenceEngine(args.checkpoint, args.device)

    # 加载高级模型
    density = None
    if args.density_path:
        density = GaussianDensityEstimator()
        density.load(args.density_path)

    ts = None
    if args.ts_path:
        ts = TeacherStudentDetector(engine.model, args.device)
        ts.load(args.ts_path)

    # 准备异常集合（与正常等量）
    anomalies = []
    if args.anomaly_mode in ("mismatch_ir", "mixed"):
        # 错配红外：可见光来自 A，红外来自 B（B != A）
        ir_pool = [ir for _, ir in normals]
        rng.shuffle(ir_pool)
        for (v, _), ir2 in zip(normals, ir_pool):
            anomalies.append(("mismatch_ir", v, ir2))

    if args.anomaly_mode in ("hotspot_ir", "mixed"):
        # 热斑注入：用同一对 (v, ir) 但对 ir_tensor 注入热点
        for v, ir in normals:
            anomalies.append(("hotspot_ir", v, ir))

    if args.anomaly_mode in ("visible_cutpaste", "mixed"):
        # 可见光 CutPaste：只影响 TS（对 residual/density 也会影响，因为生成IR会变）
        for v, ir in normals:
            anomalies.append(("visible_cutpaste", v, ir))

    # 只取与 normals 等量
    anomalies = anomalies[:len(normals)]

    # 评估
    rows = []

    def score_one(visible_path, infrared_path_or_tensor, tag, label):
        # 读可见光
        vis = engine.load_image(visible_path, is_infrared=False).to(args.device)

        # Teacher-Student 分数（只依赖可见光）
        ts_score = None
        if ts is not None:
            ts_score = float(ts.score(vis))

        # 基础 residual / density 需要红外
        base_weighted = None
        base_feature = None
        density_score = None

        if infrared_path_or_tensor is not None:
            if isinstance(infrared_path_or_tensor, str):
                ir = engine.load_image(infrared_path_or_tensor, is_infrared=True).to(args.device)
            else:
                ir = infrared_path_or_tensor.to(args.device)

            res = engine.compute_residual(vis, ir)
            base_weighted = float(res["scores"]["weighted"])
            base_feature = float(res["scores"]["feature"])

            if density is not None:
                residual = res["residual"]
                # 推理时对齐训练：单通道 + 32x32
                if residual.size(1) != 1:
                    residual = residual.mean(dim=1, keepdim=True)
                residual32 = F.interpolate(residual, size=(32, 32), mode="bilinear", align_corners=False)
                vec = residual32.squeeze(0).squeeze(0).cpu().numpy()  # [32,32]
                density_score = float(density.score(vec))

        rows.append({
            "label": int(label),
            "tag": tag,
            "visible_path": visible_path,
            "infrared_path": infrared_path_or_tensor if isinstance(infrared_path_or_tensor, str) else "(tensor_hotspot)",
            "base_weighted": base_weighted,
            "base_feature": base_feature,
            "density": density_score,
            "teacher_student": ts_score,
        })

    # 正常样本
    for v, ir in normals:
        score_one(v, ir, "normal", 0)

    # 异常样本
    for mode, v, ir in anomalies:
        if mode == "hotspot_ir":
            ir_t = engine.load_image(ir, is_infrared=True).to(args.device)
            ir_t = inject_hotspot_ir(ir_t, rng)
            score_one(v, ir_t, "anom_hotspot_ir", 1)
        elif mode == "mismatch_ir":
            score_one(v, ir, "anom_mismatch_ir", 1)
        elif mode == "visible_cutpaste":
            # 把可见光做 CutPaste 并临时保存到内存（通过 PIL->tensor）
            img = Image.open(v).convert("RGB")
            img2 = cutpaste_visible(img, rng)
            # 走 engine 的 transform：复用 inference 里同样的预处理
            vis_t = engine.transform(img2).unsqueeze(0).to(args.device)

            # TS 分数可直接用 vis_t
            ts_score = float(ts.score(vis_t)) if ts is not None else None

            # residual/density：仍需要红外（用原配对红外）
            ir_t = engine.load_image(ir, is_infrared=True).to(args.device)
            res = engine.compute_residual(vis_t, ir_t)
            base_weighted = float(res["scores"]["weighted"])
            base_feature = float(res["scores"]["feature"])

            density_score = None
            if density is not None:
                residual = res["residual"]
                if residual.size(1) != 1:
                    residual = residual.mean(dim=1, keepdim=True)
                residual32 = F.interpolate(residual, size=(32, 32), mode="bilinear", align_corners=False)
                vec = residual32.squeeze(0).squeeze(0).cpu().numpy()
                density_score = float(density.score(vec))

            rows.append({
                "label": 1,
                "tag": "anom_visible_cutpaste",
                "visible_path": v + " (cutpaste)",
                "infrared_path": ir,
                "base_weighted": base_weighted,
                "base_feature": base_feature,
                "density": density_score,
                "teacher_student": ts_score,
            })
        else:
            raise ValueError(mode)

    out = pd.DataFrame(rows)
    out.to_csv(args.save_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 已保存打分明细: {args.save_csv}")

    # 计算 AUC / F1（对每个可用分数列）
    def report(metric_name: str):
        s = out[metric_name].dropna()
        if len(s) != len(out):
            # 有些方法可能缺值（比如没提供 density_path/ts_path）
            valid = out.dropna(subset=[metric_name])
        else:
            valid = out

        y = valid["label"].astype(int).to_numpy()
        scores = valid[metric_name].astype(float).to_numpy()

        # AUC 需要同时有 0/1
        if len(np.unique(y)) < 2:
            print(f"[{metric_name}] 无法计算 AUC（评估集中没有同时包含正常/异常）")
            return

        auc = roc_auc_score(y, scores)
        best_f1, best_t = best_f1_by_sweep(y, scores)

        print(f"\n===== {metric_name} =====")
        print(f"AUC      : {auc:.6f}")
        print(f"Best F1   : {best_f1:.6f}")
        print(f"Best Th   : {best_t:.6f}")

    report("base_weighted")
    report("base_feature")

    if args.density_path:
        report("density")
    else:
        print("\n[density] 未提供 --density_path，跳过")

    if args.ts_path:
        report("teacher_student")
    else:
        print("\n[teacher_student] 未提供 --ts_path，跳过")


if __name__ == "__main__":
    main()
