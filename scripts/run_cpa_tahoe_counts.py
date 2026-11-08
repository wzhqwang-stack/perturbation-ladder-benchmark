#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_cpa_tahoe_counts.py：CPA 0.8.8 正统流程对照（raw counts + is_count_data=True）。
流程：契约+splitobs 合并 → normalize/log1p → HVG5000 → setup_anndata(dose 转 str) →
      CPA(adata, split_key='split') → CPU 训练 → predict(ood 行) → 统一指标。
指标口径：每 (drug, line) 在 ood 档：pred mean vs true mean；DE=ood 细胞 vs DMSO 对照。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")


FROM_MODEL = None  # 正统流程重训

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
    import torch

    import metrics as M
    import runlib

    torch.set_num_threads(8)
    run = runlib.begin(model="cpa", dataset="tahoe_crc", level="T3-dose-counts", seed=149,
                       task="论文② CPA tahoe T3-剂量格（CPU）", hours=4, gpu=False,
                       inputs=[str(ROOT / "data/processed/tahoe/tahoe_crc.h5ad"),
                               str(ROOT / "data/processed/tahoe/tahoe_crc_splitobs.csv")],
                       extra={"hvg": 5000, "protocol": "dose-extrapolation"})
    run.log("加载契约 + splitobs ...")
    adata = ad.read_h5ad(ROOT / "data/processed/tahoe/tahoe_crc.h5ad")
    so = pd.read_csv(ROOT / "data/processed/tahoe/tahoe_crc_splitobs.csv", index_col=0)
    adata.obs["split"] = so.loc[adata.obs_names, "split"].to_numpy()
    run.log(f"cells {adata.shape} split 分布: {adata.obs['split'].value_counts().to_dict()}")

    run.log("过滤低表达基因(min_cells=50) ...")
    sc.pp.filter_genes(adata, min_cells=50)
    run.log(f"过滤后 {adata.shape}")
    run.log("HVG5000（log1p 副本上选，X 保持 raw counts）...")
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=5000)
    adata = adata[:, tmp.var["highly_variable"]].copy()
    del tmp
    run.log(f"HVG 后 {adata.shape}")

    import cpa
    adata.obs["dose_str"] = adata.obs["dose"].astype(float).map(lambda v: f"{v:g}")
    cpa.CPA.setup_anndata(adata, perturbation_key="condition", control_group="DMSO_TF",
                          dosage_key="dose_str", categorical_covariate_keys=["cell_type"],
                          is_count_data=True)
    run.log("CPA 构造（split_key=split）...")
    if FROM_MODEL:
        model = cpa.CPA.load(FROM_MODEL, adata=adata)
        run.log(f"已加载模型 {FROM_MODEL}")
    else:
        model = cpa.CPA(adata, split_key="split", seed=149)
    if FROM_MODEL:
        run.log("FROM_MODEL 已设，跳过训练")
    else:
        run.log("train 40 epochs (CPU, 8 threads) ...")
        model.train(max_epochs=40, use_gpu=False, enable_progress_bar=False, batch_size=256)
        try:
            model.save(str(run.dir / "cpa_model"), overwrite=True)
            run.log("模型已存 run.dir/cpa_model")
        except Exception as e:
            run.log(f"[WARN] 模型存盘失败: {e}")
    run.log("train done; 预测 ood 行 ...")
    ood_mask = adata.obs["split"] == "ood"
    adata_ood = adata[ood_mask].copy()
    r = model.custom_predict(adata=adata_ood, batch_size=256, return_mean=True)
    Xp = r["latent_x_pred"].X
    pred = Xp.toarray() if sp.issparse(Xp) else np.asarray(Xp)
    run.log(f"pred {pred.shape} dtype {pred.dtype}")

    # 指标：每 (drug, line)
    obs_ood = adata_ood.obs
    X_ood = adata_ood.X
    dmso_mask = (adata.obs["condition"] == "DMSO_TF")
    dmso_sub = adata[dmso_mask]
    dmso_X = dmso_sub.X
    dmso_X = dmso_X.toarray() if sp.issparse(dmso_X) else np.asarray(dmso_X)
    rng = np.random.default_rng(149)
    if dmso_X.shape[0] > 2000:
        dmso_X = dmso_X[rng.choice(dmso_X.shape[0], 2000, replace=False)]
    ctrl_mean = dmso_X.mean(axis=0)

    recs = []
    for (drug, line), g in obs_ood.groupby(["condition", "cell_type"]):
        idx = np.where(obs_ood.index.isin(g.index))[0]
        if len(idx) < 3:
            continue
        rows = X_ood[idx]
        rows = rows.toarray() if sp.issparse(rows) else np.asarray(rows)
        true_mean = rows.mean(axis=0)
        de_idx = M.de_top_genes(ctrl_mean, rows, n_top=20)
        pmean = pred[idx].mean(axis=0)
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
    df.to_csv(run.dir / "per_condition_cpa.csv", index=False)
    run.write_metrics({
        "meta": {"dataset": "tahoe_crc", "seed": 149, "level": "T3-dose",
                 "protocol": "dose-extrapolation(ood=highest dose per drug)",
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("CPA tahoe 格子完成：" + json.dumps({k: round(v["mean"], 4) for k, v in summary.items() if isinstance(v, dict)}, ensure_ascii=False))
    print("CPA_TAHOE_DONE", flush=True)


if __name__ == "__main__":
    main()