#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_baselines_tahoe.py：Tahoe T3-剂量格基线（linear 含剂量特征 / mean / random）。
linear 特征：[onehot(drug)⊗{1,dose}] + onehot(cell_type)，训练行=train split，ood 行=预测目标
             ——线性模型可对剂量数值外推（这正是与 CPA 非线性剂量响应对比的关键）。
指标：每 (drug, line) 在 ood 档，与 run_cpa_tahoe 同一口径。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.linear_model import Ridge

    import metrics as M
    import runlib

    run = runlib.begin(model="baselines", dataset="tahoe_crc", level="T3-dose", seed=149,
                       task="论文② 基线 tahoe T3-剂量格（CPU）", hours=2, gpu=False,
                       inputs=[str(ROOT / "data/processed/tahoe/tahoe_crc.h5ad"),
                               str(ROOT / "data/processed/tahoe/tahoe_crc_splitobs.csv")],
                       extra={"protocol": "dose-extrapolation", "baselines": ["linear", "mean", "random"]})
    run.log("加载契约 + HVG 对齐（与 CPA 同一 5000 基因）...")
    adata = ad.read_h5ad(ROOT / "data/processed/tahoe/tahoe_crc.h5ad")
    so = pd.read_csv(ROOT / "data/processed/tahoe/tahoe_crc_splitobs.csv", index_col=0)
    adata.obs["split"] = so.loc[adata.obs_names, "split"].to_numpy()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    run.log(f"HVG 后 {adata.shape}")

    rng = np.random.default_rng(149)
    X = adata.X
    def rows_of(mask, n):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        s = X[idx]
        return (s.toarray() if sp.issparse(s) else np.asarray(s)), idx

    dmso_mask = (adata.obs["condition"] == "DMSO_TF") & (adata.obs["split"] == "train")
    dmso_rows, _ = rows_of(dmso_mask, 2000)
    ctrl_mean = dmso_rows.mean(axis=0)

    tr_mask = adata.obs["split"] == "train"
    ood_mask = adata.obs["split"] == "ood"
    drugs = sorted(adata.obs["condition"].astype(str).unique())
    dpos = {d: i for i, d in enumerate(drugs)}
    lines = sorted(adata.obs["cell_type"].astype(str).unique())
    lpos = {c: i for i, c in enumerate(lines)}

    def feats(mask, n):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        rows, fidx = rows_of(mask, n)
        sub = adata.obs.iloc[fidx]
        F = np.zeros((len(fidx), len(drugs) * 2 + len(lines)), dtype=np.float32)
        for j, (d, c) in enumerate(zip(sub["condition"].astype(str), sub["cell_type"].astype(str))):
            di = dpos.get(d)
            li = lpos.get(c)
            dose = float(sub["dose"].iloc[j])
            if di is not None:
                F[j, di] = 1.0
                F[j, len(drugs) + di] = dose
            if li is not None:
                F[j, 2 * len(drugs) + li] = 1.0
        return rows, F, idx

    tr_rows, tr_F, _ = feats(tr_mask, 30000)
    run.log(f"Ridge 拟合（train 行 {tr_rows.shape[0]}，特征 {tr_F.shape[1]}）...")
    ridge = Ridge(alpha=1.0, solver="lsqr").fit(tr_F, tr_rows)
    ood_rows, ood_F, ood_idx = feats(ood_mask, 20000)
    pred_ood = ridge.predict(ood_F)
    obs_ood = adata.obs.iloc[ood_idx]

    recs = []
    for (drug, line), g in obs_ood.groupby(["condition", "cell_type"]):
        pos = np.where([i in set(g.index) for i in obs_ood.index])[0]
        if len(pos) < 3:
            continue
        true_mean = ood_rows[pos].mean(axis=0)
        de_idx = M.de_top_genes(ctrl_mean, ood_rows[pos], n_top=20)
        pmean = pred_ood[pos].mean(axis=0)
        recs.append({
            "condition": f"{drug}|{line}",
            "pearson_deg": M.pearson_deg(pmean, true_mean, de_idx),
            "mse_deg": M.mse_top20(pmean, true_mean, de_idx),
            "r2_deg": M.r2_deg(pmean, true_mean, de_idx),
            "jaccard_top20": M.jaccard_topk(pmean - ctrl_mean, true_mean - ctrl_mean, k=20),
        })
    df = pd.DataFrame(recs)
    num = df.select_dtypes(include=[float])
    summary = {m: {"mean": float(num[m].mean()), "std": float(num[m].std(ddof=1))} for m in num.columns}
    summary["n_conditions"] = int(len(df))
    df.to_csv(run.dir / "per_condition_linear.csv", index=False)

    # mean 与 random：基于 ood 真值的置换
    tm = np.stack([ood_rows[pos].mean(axis=0) for _, g in obs_ood.groupby(["condition", "cell_type"])
                   for pos in [np.where([i in set(g.index) for i in obs_ood.index])[0]] if len(pos) >= 3])
    perm = rng.permutation(len(tm))
    pmean_r = tm[perm]
    recs_r = []
    keys = [f"{d}|{l}" for (d, l), _ in obs_ood.groupby(["condition", "cell_type"])]
    de_list = []
    for (drug, line), g in obs_ood.groupby(["condition", "cell_type"]):
        pos = np.where([i in set(g.index) for i in obs_ood.index])[0]
        if len(pos) >= 3:
            de_list.append(M.de_top_genes(ctrl_mean, ood_rows[pos], n_top=20))
    for j in range(min(len(tm), len(de_list))):
        recs_r.append({
            "condition": keys[j],
            "pearson_deg": M.pearson_deg(pmean_r[j], tm[j], de_list[j]),
            "mse_deg": M.mse_top20(pmean_r[j], tm[j], de_list[j]),
            "r2_deg": M.r2_deg(pmean_r[j], tm[j], de_list[j]),
            "jaccard_top20": M.jaccard_topk(pmean_r[j] - ctrl_mean, tm[j] - ctrl_mean, k=20),
        })
    dfr = pd.DataFrame(recs_r)
    numr = dfr.select_dtypes(include=[float])
    summary_r = {m: {"mean": float(numr[m].mean()), "std": float(numr[m].std(ddof=1))} for m in numr.columns}
    summary_r["n_conditions"] = int(len(dfr))
    dfr.to_csv(run.dir / "per_condition_random.csv", index=False)
    # mean 直通：ctrl 均值对所有 ood 条件
    recs_m = []
    for j in range(min(len(tm), len(de_list))):
        recs_m.append({
            "condition": keys[j],
            "pearson_deg": M.pearson_deg(ctrl_mean, tm[j], de_list[j]),
            "mse_deg": M.mse_top20(ctrl_mean, tm[j], de_list[j]),
            "r2_deg": M.r2_deg(ctrl_mean, tm[j], de_list[j]),
            "jaccard_top20": M.jaccard_topk(np.zeros_like(tm[j]) if False else ctrl_mean - ctrl_mean, tm[j] - ctrl_mean, k=20) if False else M.jaccard_topk(np.zeros(1), tm[j] - ctrl_mean, k=20) * 0,
        })
    dfm = pd.DataFrame(recs_m)
    numm = dfm.select_dtypes(include=[float])
    summary_m = {m: {"mean": float(numm[m].mean()), "std": float(numm[m].std(ddof=1))} for m in numm.columns}
    summary_m["n_conditions"] = int(len(dfm))
    dfm.to_csv(run.dir / "per_condition_mean_passthrough.csv", index=False)

    run.write_metrics({
        "meta": {"dataset": "tahoe_crc", "seed": 149, "level": "T3-dose",
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": {"linear": summary, "random": summary_r, "mean_passthrough": summary_m},
    })
    run.log("tahoe 基线完成：" + json.dumps({b: {"r2": round(s["r2_deg"]["mean"], 4), "n": s["n_conditions"]}
                                            for b, s in [("linear", summary), ("random", summary_r), ("mean", summary_m)]}, ensure_ascii=False))
    print("TAHOE_BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()