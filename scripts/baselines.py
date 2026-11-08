#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baselines.py：三基线（均值直通 / 线性回归 / 随机对照）+ 统一指标评测（论文②可信度支柱）。

设计（docs/统一评测接口设计.md §3）：
  输入：data/processed/<ds>/<ds>.h5ad + splits/split_T3-<split>_s<seed>.json（按扰动三元组划分）
  对每个 test 条件 c：
    true_mean[c] = 该条件细胞在 log-normalized 空间的均值轮廓
    de_idx[c]    = de_top_genes(控制组子采样, 条件细胞, 20)（所有方法同一把尺）
  预测：
    mean_passthrough : 控制组均值轮廓（样本集=控制组细胞）
    linear           : Ridge(one-hot(训练条件)) 拟合 → 未见条件外推=全局均值轮廓
                       （线性模型对 unseen 扰动无法外推——这正是与深度模型对比的公开争论点）
    random           : 测试条件间置换真实均值（seed 固定），样本集=供体条件细胞
  指标：pearson_deg / mse_deg / r2_deg / jaccard_topk（效应=扰动均值-控制均值）；
        e_distance 仅在样本集可得时计算（子采样≤500，seed 固定）。

用法:
  python baselines.py --dataset norman --seed 149 [--max-test 0]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
RUNS = ROOT / "results" / "runs"


def delta_r2(pred_mean, true_mean, ctrl_mean, de_idx):
    """delta 中心化 R²：在 DE 基因上对 (扰动-控制) 差值算 R²（2026-11-02 审查附录）。"""
    x = (np.asarray(pred_mean, dtype=np.float64) - np.asarray(ctrl_mean, dtype=np.float64))[de_idx]
    y = (np.asarray(true_mean, dtype=np.float64) - np.asarray(ctrl_mean, dtype=np.float64))[de_idx]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((y - x) ** 2).sum()) / ss_tot


def subsample_rows(X, idx, max_n, rng):
    if len(idx) > max_n:
        idx = rng.choice(idx, max_n, replace=False)
    sub = X[idx]
    return sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=149)
    ap.add_argument("--max-test", type=int, default=0, help="0=全部 test 条件；>0 截断（调试用）")
    ap.add_argument("--gpu", action="store_true", help="占 GPU 登记（基线默认 CPU 不占）")
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    import anndata as ad
    import pandas as pd
    from sklearn.linear_model import Ridge
    from scipy import sparse

    import metrics as M
    import runlib

    run = runlib.begin(model="baselines", dataset=a.dataset, level="T3-simulation", seed=a.seed,
                       task=f"论文② 基线三件套 {a.dataset} T3 s{a.seed}", hours=1, gpu=a.gpu,
                       inputs=[str(ROOT / "data" / "processed" / a.dataset / f"{a.dataset}.h5ad"),
                               str(ROOT / "data" / "processed" / a.dataset / "splits" / f"split_T3-simulation_s{a.seed}.json")],
                       extra={"baselines": ["mean_passthrough", "linear", "random"]})
    run.log("加载契约数据 ...")
    adata = ad.read_h5ad(ROOT / "data" / "processed" / a.dataset / f"{a.dataset}.h5ad")
    split = json.loads((ROOT / "data" / "processed" / a.dataset / "splits" / f"split_T3-simulation_s{a.seed}.json").read_text(encoding="utf-8"))
    cond = adata.obs["condition"].astype(str).to_numpy()
    X = adata.X
    test_conds = list(split["test"])
    if a.max_test > 0:
        test_conds = test_conds[: a.max_test]
    run.log(f"test 条件 {len(test_conds)} 个；X {adata.shape}")

    rng = np.random.default_rng(a.seed)
    ctrl_idx = np.where(cond == "ctrl")[0]
    ctrl_sub = subsample_rows(X, ctrl_idx, 2000, rng)
    ctrl_mean = ctrl_sub.mean(axis=0)

    # 线性基线：训练条件的 one-hot Ridge（子采样加速）
    train_conds = set(split["train"])
    tr_idx = np.where(np.isin(cond, list(train_conds)))[0]
    tr_idx = rng.choice(tr_idx, min(len(tr_idx), 20000), replace=False)
    tr_rows = subsample_rows(X, tr_idx, 20000, rng)
    tr_cond = cond[tr_idx]
    uniq = sorted(train_conds)
    col = {c: i for i, c in enumerate(uniq)}
    D = np.zeros((len(tr_idx), len(uniq)), dtype=np.float32)
    for i, c in enumerate(tr_cond):
        D[i, col[c]] = 1.0
    run.log(f"Ridge 拟合（训练条件 {len(uniq)}，细胞 {len(tr_idx)}）...")
    ridge = Ridge(alpha=1.0, solver="lsqr").fit(D, tr_rows)
    # 未见条件的 one-hot 为零向量 → 预测=截距（全局均值轮廓）
    linear_mean = ridge.intercept_

    # 每条件真实统计
    def cond_idx(c):
        return np.where(cond == c)[0]

    per_cond = {b: [] for b in ["mean_passthrough", "linear", "random"]}
    perm = rng.permutation(len(test_conds))
    for k, c in enumerate(test_conds):
        ci = cond_idx(c)
        if len(ci) < 3:
            continue
        rows_c = subsample_rows(X, ci, 1000, rng)
        true_mean = rows_c.mean(axis=0)
        de_idx = M.de_top_genes(ctrl_sub, rows_c, n_top=20)
        true_stat = true_mean - ctrl_mean
        preds = {
            "mean_passthrough": (ctrl_mean, ctrl_sub),
            "linear": (linear_mean, None),
            "random": (None, None),  # 下方按置换填充
        }
        # random：置换
        donor_c = test_conds[perm[k]]
        di = cond_idx(donor_c)
        donor_mean = subsample_rows(X, di, 1000, rng).mean(axis=0) if len(di) else ctrl_mean
        donor_cells = subsample_rows(X, di, 500, rng) if len(di) else ctrl_sub
        preds["random"] = (donor_mean, donor_cells)

        for b, (pmean, psamples) in preds.items():
            if pmean is None:
                continue
            rec = {
                "condition": c,
                "pearson_deg": M.pearson_deg(pmean, true_mean, de_idx),
                "mse_deg": M.mse_top20(pmean, true_mean, de_idx),
                "r2_deg": M.r2_deg(pmean, true_mean, de_idx),
                "delta_r2_deg": delta_r2(pmean, true_mean, ctrl_mean, de_idx),
                "jaccard_top20": M.jaccard_topk(pmean - ctrl_mean, true_stat, k=20),
            }
            if psamples is not None:
                rec["e_distance"] = M.e_distance(psamples, rows_c, max_n=500, seed=a.seed)
            per_cond[b].append(rec)

    summary = {}
    for b, recs in per_cond.items():
        if not recs:
            continue
        df = pd.DataFrame(recs)
        num = df.select_dtypes(include=[float])
        summary[b] = {m: {"mean": float(num[m].mean()), "std": float(num[m].std(ddof=1))}
                      for m in num.columns}
        summary[b]["n_conditions"] = int(len(df))
        df.to_csv(run.dir / f"per_condition_{b}.csv", index=False)

    run.write_metrics({
        "meta": {"dataset": a.dataset, "seed": a.seed, "level": "T3-simulation",
                 "generated_at": datetime.now().isoformat(timespec="seconds"),
                 "de_method": "mean_diff", "n_top": 20},
        "summary": summary,
    })
    run.log("基线评测完成：" + json.dumps({b: {k: round(vv["mean"], 4) for k, vv in s.items() if isinstance(vv, dict)}
                                          for b, s in summary.items()}, ensure_ascii=False))
    print("BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()