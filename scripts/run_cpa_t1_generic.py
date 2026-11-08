#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_cpa_t1_rev.py：CPA T1 反向（RPE1 训 → K562 预测）。参考行取各条件首个非 ctrl 细胞（dose 通用化）；
派生 obs 列与全部 obsm 键整行拷贝；指标在公共基因空间。"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
SOURCE = None  # via args
TARGET = None  # via args
SEED = None  # via args


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=149)
    global SOURCE, TARGET, SEED
    a = ap.parse_args()
    SOURCE, TARGET, SEED = a.source, a.target, a.seed
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
    pair = f"{SOURCE.split(chr(95))[1]}-to-{TARGET.split(chr(95))[1]}"
    run = runlib.begin(model="cpa", dataset=f"t1_{pair}", level="T1", seed=SEED,
                       task="论文② CPA T1 rpe1-to-k562 训练+预测（CPU）", hours=3, gpu=False,
                       inputs=[str(ROOT / f"data/processed/{SOURCE}/{SOURCE}.h5ad"),
                               str(ROOT / f"data/processed/{TARGET}/{TARGET}.h5ad")],
                       extra={"protocol": "source-train + target-ctrl-as-perturbed; ref-row derived-field copy"})
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
    run.log(f"源训练集 {src_tr.shape}")

    import cpa
    cpa.CPA.setup_anndata(src_tr, perturbation_key="condition", control_group="ctrl",
                          dosage_key="dose_str", categorical_covariate_keys=["cell_type"],
                          is_count_data=False)
    raw_obs_cols = set(src.obs.columns)
    derived_obs = [c for c in src_tr.obs.columns if c not in raw_obs_cols]
    obsm_keys = list(src_tr.obsm.keys())
    run.log(f"setup 后 obsm 键: {obsm_keys}；派生 obs 列: {derived_obs}")
    model = cpa.CPA(src_tr, seed=SEED)
    run.log("train 40 epochs (CPU, 8 threads) ...")
    model.train(max_epochs=40, use_gpu=False, enable_progress_bar=False, batch_size=256)
    try:
        model.save(str(run.dir / "cpa_model"), overwrite=True)
        run.log("模型已存 run.dir/cpa_model")
    except Exception as e:
        run.log(f"[WARN] 模型存盘失败: {e}")

    sc_cond_all = src_tr.obs["condition"].astype(str).to_numpy()
    dose_all = src_tr.obs["dose_str"].astype(str).to_numpy()
    ref_row = {}
    for j, c in enumerate(sc_cond_all):
        if c not in ref_row and c != "ctrl":
            ref_row[c] = j  # 首个非 ctrl 细胞（dose 通用化）

    t_cond = tgt.obs["condition"].astype(str).to_numpy()
    tX = tgt.X
    t_ctrl_idx = np.where(t_cond == "ctrl")[0]
    t_ctrl_idx = rng.choice(t_ctrl_idx, min(len(t_ctrl_idx), 300), replace=False)
    s2 = tX[t_ctrl_idx]
    ctrl_rows = s2.toarray() if hasattr(s2, "toarray") else np.asarray(s2)
    ctrl_rows = ctrl_rows[:, t_cols]
    src_celltype = str(src_sub.obs["cell_type"].astype(str).iloc[0])

    common_conds = sorted(set(t_cond) & set(sc_cond_all) - {"ctrl"})

    def t_rows(idx):
        s3 = tX[idx]
        s3 = s3.toarray() if hasattr(s3, "toarray") else np.asarray(s3)
        return s3[:, t_cols]
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
    run.log(f"可用条件 {len(conds_used)}")
    t_ctrl_mean = ctrl_rows.mean(axis=0)

    recs = []
    B = 60
    reps = 3
    for bi in range(0, len(conds_used), B):
        chunk = conds_used[bi: bi + B]
        rows_l, cond_l, dose_l, ct_l, ref_l = [], [], [], [], []
        for c in chunk:
            r0 = ref_row[c]
            for rr in range(reps):
                rows_l.append(ctrl_rows[rr % len(ctrl_rows)])
                cond_l.append(c)
                dose_l.append(dose_all[r0])
                ct_l.append(src_celltype)
                ref_l.append(r0)
        pseudo = ad.AnnData(np.stack(rows_l).astype(np.float32))
        pseudo.obs["condition"] = cond_l
        pseudo.obs["dose_str"] = dose_l
        pseudo.obs["cell_type"] = ct_l
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
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("T1 CPA 反向完成：" + json.dumps({k: round(v["mean"], 4) for k, v in summary.items() if isinstance(v, dict)}, ensure_ascii=False))
    print("CPA_T1_REV_DONE", flush=True)


if __name__ == "__main__":
    main()