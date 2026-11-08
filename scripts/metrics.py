#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指标库（论文②）：统一评测指标，纯 numpy/scipy 实现，独立于具体模型。

约定：
- 均值轮廓 (mean profile) = 基因维向量（log-normalized 空间）；
- 样本矩阵 = cells × genes；
- DE 基因索引一律由 de_top_genes() 产生，保证所有模型同一把尺子。

指标（启动指令 §4.2）：
  DE 基因 Pearson / top-20 DE 的 MSE / R² / E-distance / Jaccard / 簇纯度
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


def de_top_genes(ctrl_expr, pert_expr, n_top=20, method="mean_diff"):
    """ctrl_expr, pert_expr: cells×genes（log-normalized）。返回 top-n DE 基因索引。

    method:
      mean_diff — 按 |均值差| 排序（GEARS 论文口径）
      welch     — 按 Welch t 统计量绝对值排序
    """
    ctrl_expr = np.asarray(ctrl_expr, dtype=np.float64)
    pert_expr = np.asarray(pert_expr, dtype=np.float64)
    if method == "mean_diff":
        stat = np.abs(pert_expr.mean(axis=0) - ctrl_expr.mean(axis=0))
    elif method == "welch":
        st, _ = stats.ttest_ind(pert_expr, ctrl_expr, axis=0, equal_var=False)
        stat = np.nan_to_num(np.abs(st), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError(f"unknown method: {method}")
    n_top = min(n_top, stat.shape[0])
    return np.argsort(-stat)[:n_top]


def pearson_deg(pred_mean, true_mean, de_idx):
    """DE 基因上的均值 Pearson 相关。pred_mean/true_mean: (n_genes,)。"""
    x = np.asarray(pred_mean, dtype=np.float64)[de_idx]
    y = np.asarray(true_mean, dtype=np.float64)[de_idx]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(stats.pearsonr(x, y)[0])


def mse_top20(pred_mean, true_mean, de_idx):
    """top-DE 基因上的均值 MSE。"""
    x = np.asarray(pred_mean, dtype=np.float64)[de_idx]
    y = np.asarray(true_mean, dtype=np.float64)[de_idx]
    return float(np.mean((x - y) ** 2))


def r2_deg(pred_mean, true_mean, de_idx):
    """DE 基因上的 R²（对真实均值）。"""
    x = np.asarray(pred_mean, dtype=np.float64)[de_idx]
    y = np.asarray(true_mean, dtype=np.float64)[de_idx]
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def e_distance(pred_samples, true_samples, max_n=2000, seed=149):
    """多元 Energy distance：E||X-Y|| − ½E||X-X′|| − ½E||Y-Y′||（下界 0）。

    pred_samples/true_samples: cells×genes；超出 max_n 随机子采样（seed 固定保证可复现）。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(pred_samples, dtype=np.float64)
    y = np.asarray(true_samples, dtype=np.float64)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]
    dxy = cdist(x, y).mean()
    dxx = cdist(x, x).mean()
    dyy = cdist(y, y).mean()
    return float(max(dxy - 0.5 * dxx - 0.5 * dyy, 0.0))


def jaccard_topk(pred_stat, true_stat, k=20):
    """预测/真实扰动效应统计量（基因维向量）各自取 top-k 基因集合的 Jaccard。"""
    pred_stat = np.asarray(pred_stat, dtype=np.float64)
    true_stat = np.asarray(true_stat, dtype=np.float64)
    k = min(k, pred_stat.shape[0])
    sp = set(np.argsort(-np.abs(pred_stat))[:k].tolist())
    st = set(np.argsort(-np.abs(true_stat))[:k].tolist())
    return len(sp & st) / len(sp | st)


def cluster_purity(true_labels, pred_labels):
    """簇纯度：Σ_cluster max_class_count / N。"""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    total = 0
    for c in np.unique(pred_labels):
        mask = pred_labels == c
        if mask.any():
            classes, counts = np.unique(true_labels[mask], return_counts=True)
            total += counts.max()
    return float(total / len(pred_labels))


def eval_all(pred_mean, true_mean, de_idx, pred_samples=None, true_samples=None):
    """一步汇总均值型指标；样本型（E-distance）在提供样本矩阵时才计算。"""
    out = {
        "pearson_deg": pearson_deg(pred_mean, true_mean, de_idx),
        "mse_deg": mse_top20(pred_mean, true_mean, de_idx),
        "r2_deg": r2_deg(pred_mean, true_mean, de_idx),
    }
    if pred_samples is not None and true_samples is not None:
        out["e_distance"] = e_distance(pred_samples, true_samples)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(149)
    g, n = 100, 120
    ctrl = rng.normal(0, 1, (n, g))
    eff = np.zeros(g)
    eff[:10] = rng.normal(3, 0.5, 10)  # 前 10 个基因是真实 DE
    pert = ctrl + eff + rng.normal(0, 0.2, (n, g))
    de = de_top_genes(ctrl, pert, n_top=20)
    pred_perfect = pert.mean(axis=0)
    pred_ctrl = ctrl.mean(axis=0)
    print("DE idx[:5] =", de[:5])
    print("perfect :", eval_all(pred_perfect, pert.mean(axis=0), de, pert, pert))
    print("ctrl-mimic:", eval_all(pred_ctrl, pert.mean(axis=0), de, ctrl, pert))
    print("jaccard perfect:", jaccard_topk(pred_perfect - ctrl.mean(axis=0), pert.mean(axis=0) - ctrl.mean(axis=0)))
    print("jaccard ctrl   :", jaccard_topk(np.zeros(g), pert.mean(axis=0) - ctrl.mean(axis=0)))