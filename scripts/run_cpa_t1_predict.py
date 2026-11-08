#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_cpa_t1_predict.py：CPA T1 预测阶段续跑（复用已训模型；伪细胞矩阵拷贝源训练集同条件 obsm 参考行）。
背景：run_cpa_t1.py 训练 40 epochs 后死于 custom_predict 的 obsm 缺失（'perts'/'perts_doses'）。
本脚本重建同一 src_tr（种子决定论一致）→ setup → CPA.load → 逐块预测 → 指标入档。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
SOURCE = "replogle_k562_essential"
TARGET = "replogle_rpe1_essential"
SEED = 149


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import anndata as ad
    import pandas as pd
    import torch

    import metrics as M
    import runlib

    torch.set_num_threads(8)
    model_dir = ROOT / "results/runs/cpa_t1_k562-to-rpe1_T1_s149_cpu20261031-200627-5cc0/cpa_model"
    pair = "k562-to-rpe1"
    run = runlib.begin(model="cpa", dataset=f"t1_{pair}", level="T1", seed=SEED,
                       task="论文② CPA T1 k562-to-rpe1 预测续跑（CPU，复用已训模型）", hours=1, gpu=False,
                       inputs=[str(model_dir / "model.pt"),
                               str(ROOT / f"data/processed/{SOURCE}/{SOURCE}.h5ad"),
                               str(ROOT / f"data/processed/{TARGET}/{TARGET}.h5ad")],
                       extra={"protocol": "predict-stage rerun; obsm perts/perts_doses copied from source ref rows"})
    run.log("加载两系契约 ...")
    src = ad.read_h5ad(ROOT / f"data/processed/{SOURCE}/{SOURCE}.h5ad")
    tgt = ad.read_h5ad(ROOT / f"data/processed/{TARGET}/{TARGET}.h5ad")
    common_genes = sorted(set(src.var["gene_name"].astype(str)) & set(tgt.var["gene_name"].astype(str)))
    gp_s = {g: i for i, g in enumerate(src.var["gene_name"].astype(str))}
    gp_t = {g: i for i, g in enumerate(tgt.var["gene_name"].astype(str))}
    s_cols = [gp_s[g] for g in common_genes]
    t_cols = [gp_t[g] for g in common_genes]
    src_sub = src[:, s_cols].copy()

    rng = np.random.default_rng(SEED)
    tr_idx = np.arange(src_sub.n_obs)
    if len(tr_idx) > 60000:
        tr_idx = rng.choice(tr_idx, 60000, replace=False)
    src_tr = src_sub[tr_idx].copy()
    if "dose_str" not in src_tr.obs.columns:
        src_tr.obs["dose_str"] = src_tr.obs["dose"].astype(float).map(lambda v: f"{v:g}")

    import cpa
    cpa.CPA.setup_anndata(src_tr, perturbation_key="condition", control_group="ctrl",
                          dosage_key="dose_str", categorical_covariate_keys=["cell_type"],
                          is_count_data=False)
    raw_obs_cols = set(src.obs.columns)
    derived_obs = [c for c in src_tr.obs.columns if c not in raw_obs_cols]
    obsm_keys = list(src_tr.obsm.keys())
    run.log(f"setup 后 obsm 键: {obsm_keys}；派生 obs 列: {derived_obs}")
    assert "perts" in src_tr.obsm and "perts_doses" in src_tr.obsm, "obsm 参考键缺失"
    run.log("CPA.load ...")
    model = cpa.CPA.load(str(model_dir), adata=src_tr)
    run.log("模型加载完成")

    sc_cond_all = src_tr.obs["condition"].astype(str).to_numpy()
    dose_all = src_tr.obs["dose_str"].astype(str).to_numpy()
    ref_row = {}
    for j, c in enumerate(sc_cond_all):
        if c not in ref_row and c != "ctrl" and dose_all[j] == "1":
            ref_row[c] = j  # 扰动条件取 dose=1 参考行（K562 剂量域 {0,1}）

    t_cond = tgt.obs["condition"].astype(str).to_numpy()
    tX = tgt.X
    t_ctrl_idx = np.where(t_cond == "ctrl")[0]
    t_ctrl_idx = rng.choice(t_ctrl_idx, min(len(t_ctrl_idx), 300), replace=False)
    s = tX[t_ctrl_idx]
    ctrl_rows = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    ctrl_rows = ctrl_rows[:, t_cols]
    src_celltype = str(src_sub.obs["cell_type"].astype(str).iloc[0])

    common_conds = sorted(set(t_cond) & set(sc_cond_all) - {"ctrl"})

    def t_rows(idx):
        s2 = tX[idx]
        s2 = s2.toarray() if hasattr(s2, "toarray") else np.asarray(s2)
        return s2[:, t_cols]
    true_means, de_means, conds_used = {}, {}, []
    for c in common_conds:
        ci = np.where(t_cond == c)[0]
        if len(ci) < 3 or c not in ref_row:
            continue
        rows = t_rows(ci)
        if len(rows) > 500:
            rows = rows[rng.choice(len(rows), 500, replace=False)]
        true_means[c] = rows.mean(axis=0)
        de_means[c] = M.de_top_genes(ctrl_rows, rows, n_top=20)
        conds_used.append(c)
    run.log(f"可用条件 {len(conds_used)}（ref_row 命中）")
    t_ctrl_mean = ctrl_rows.mean(axis=0)

    recs = []
    B = 60
    reps = 3
    for bi in range(0, len(conds_used), B):
        chunk = conds_used[bi: bi + B]
        rows_l, cond_l, dose_l, ct_l = [], [], [], []
        ref_l = []
        for c in chunk:
            r0 = ref_row[c]
            for rr in range(reps):
                rows_l.append(ctrl_rows[rr % len(ctrl_rows)])
                cond_l.append(c)
                dose_l.append("1")
                ct_l.append(src_celltype)
                ref_l.append(r0)
        pseudo = ad.AnnData(np.stack(rows_l).astype(np.float32))
        pseudo.obs["condition"] = cond_l
        pseudo.obs["dose_str"] = dose_l
        pseudo.obs["cell_type"] = ct_l
        # 派生 obs 列与全部 obsm 键整行拷贝自参考行（CPA 校验所需完整字段集）
        for col in derived_obs:
            pseudo.obs[col] = src_tr.obs[col].astype(object).to_numpy()[ref_l]
        for key in obsm_keys:
            pseudo.obsm[key] = np.stack([np.asarray(src_tr.obsm[key][r0]) for r0 in ref_l])
        pseudo.obs_names = [f"cp{bi}-{k}" for k in range(pseudo.n_obs)]
        r = model.custom_predict(adata=pseudo, batch_size=64, return_mean=True)
        Xp = r["latent_x_pred"].X
        pred = Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)
        for j, c in enumerate(chunk):
            if c not in true_means:
                continue
            sel = np.arange(j * reps, (j + 1) * reps)
            pmean = pred[sel].mean(axis=0)
            recs.append({"condition": c,
                         "pearson_deg": M.pearson_deg(pmean, true_means[c], de_means[c]),
                         "mse_deg": M.mse_top20(pmean, true_means[c], de_means[c]),
                         "r2_deg": M.r2_deg(pmean, true_means[c], de_means[c]),
                         "jaccard_top20": M.jaccard_topk(pmean - t_ctrl_mean, true_means[c] - t_ctrl_mean, k=20)})
        run.log(f"  chunk {bi//B + 1}/{(len(conds_used)+B-1)//B} 完成")

    df = pd.DataFrame(recs)
    num = df.select_dtypes(include=[float])
    summary = {m: {"mean": float(num[m].mean()), "std": float(num[m].std(ddof=1))} for m in num.columns}
    summary["n_conditions"] = int(len(df))
    df.to_csv(run.dir / "per_condition_cpa_t1.csv", index=False)
    run.write_metrics({
        "meta": {"source": SOURCE, "target": TARGET, "seed": SEED, "level": "T1",
                 "n_common_genes": len(common_genes), "limitation": "no target context",
                 "model_reused_from": str(model_dir.parent.name),
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("T1 CPA 预测续跑完成：" + json.dumps({k: round(v["mean"], 4) for k, v in summary.items() if isinstance(v, dict)}, ensure_ascii=False))
    print("CPA_T1_PREDICT_DONE", flush=True)


if __name__ == "__main__":
    main()