#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_t1_scgen.py：T1 留出细胞系转移（scGen 主场）+ 三基线。
协议：源系全条件训练 scGen（batch_key=condition, labels_key=cell_type）；
      每公共条件 c 的源系 latent delta（mean_lat_c − mean_lat_ctrl）施加于目标系 ctrl 细胞 latent，
      解码得目标系预测轮廓；指标对目标系真实扰动细胞（公共基因空间 2832）。
基线：mean=目标系 ctrl 均值直通；random=目标真值置换；linear_transfer=目标 ctrl + 源系全局平均 delta（线性解码器等价）。
用法: python run_t1_scgen.py --source replogle_k562_essential --target replogle_rpe1_essential --seed 149 [--device cuda]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=149)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    import anndata as ad
    import scanpy as sc
    import scgen
    import scgen_compat
    import torch

    import metrics as M
    import runlib

    scgen_compat.apply()
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    pair = f"{a.source.split('_')[1]}-to-{a.target.split('_')[1]}"  # k562-to-rpe1
    run = runlib.begin(model="scgen", dataset=f"t1_{pair}", level="T1", seed=a.seed,
                       task=f"论文② scGen T1 {pair}（GPU）", hours=2, gpu=(device == "cuda"),
                       inputs=[str(ROOT / f"data/processed/{a.source}/{a.source}.h5ad"),
                               str(ROOT / f"data/processed/{a.target}/{a.target}.h5ad")],
                       extra={"device": device, "protocol": "source-delta-transfer"})
    run.log("加载两系契约 ...")
    src = ad.read_h5ad(ROOT / f"data/processed/{a.source}/{a.source}.h5ad")
    tgt = ad.read_h5ad(ROOT / f"data/processed/{a.target}/{a.target}.h5ad")
    common_genes = sorted(set(src.var["gene_name"].astype(str)) & set(tgt.var["gene_name"].astype(str)))
    gpos_s = {g: i for i, g in enumerate(src.var["gene_name"].astype(str))}
    gpos_t = {g: i for i, g in enumerate(tgt.var["gene_name"].astype(str))}
    s_cols = [gpos_s[g] for g in common_genes]
    t_cols = [gpos_t[g] for g in common_genes]
    run.log(f"公共基因 {len(common_genes)}")

    rng = np.random.default_rng(a.seed)
    sc_cond = src.obs["condition"].astype(str).to_numpy()
    tg_cond = tgt.obs["condition"].astype(str).to_numpy()
    common_conds = sorted(set(sc_cond) & set(tg_cond) - {"ctrl"})
    if "ctrl" not in set(sc_cond) or "ctrl" not in set(tg_cond):
        raise SystemExit("[ERR] 某系缺 ctrl 条件")
    run.log(f"公共扰动条件 {len(common_conds)}")

    # ---- 源系训练 ----
    tr_mask = np.ones(src.n_obs, dtype=bool)
    src_tr = src[tr_mask].copy()
    src_celltype = str(src_tr.obs["cell_type"].astype(str).iloc[0])  # 上下文细胞型=源系（转移语义）
    if src_tr.n_obs > 40000:
        sc.pp.subsample(src_tr, n_obs=40000, copy=False)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    scgen.SCGEN.setup_anndata(src_tr, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(src_tr)
    model.train(max_epochs=30, batch_size=256, early_stopping=False, enable_progress_bar=False)
    run.log("源系 scGen 训练完成")

    def lat_mean(ad_sub, tag):
        tmp = ad_sub.copy()
        tmp.obs_names = [f"{tag}{a.seed}-{j}" for j in range(tmp.n_obs)]
        L = model.get_latent_representation(tmp)
        return L.mean(axis=0)

    def decode(lat_rows, tag):
        tmp = ad.AnnData(np.asarray(lat_rows, dtype=np.float32))
        tmp.obs["condition"] = "ctrl"
        tmp.obs["cell_type"] = src_tr.obs["cell_type"].astype(str).iloc[0]
        tmp.obs_names = [f"{tag}{a.seed}-{j}" for j in range(tmp.n_obs)]
        out = model.module.generative(torch.tensor(np.asarray(lat_rows), dtype=torch.float32).to(model.device))
        px = out["px"].cpu().detach().numpy()
        return px.mean(axis=0) if px.ndim > 1 else px

    # 源系 latent 均值表（一次编码）
    lat_src = model.get_latent_representation(src_tr)
    cond_src = src_tr.obs["condition"].astype(str).to_numpy()
    lat_means = {}
    for c in set(cond_src):
        m = cond_src == c
        if m.any():
            lat_means[c] = lat_src[m].mean(axis=0)
    lat_ctrl_s = lat_means.get("ctrl")
    if lat_ctrl_s is None:
        raise SystemExit("[ERR] 源系缺 ctrl")

    # 目标系 ctrl 子采样
    t_ctrl_idx = np.where(tg_cond == "ctrl")[0]
    t_ctrl_idx = rng.choice(t_ctrl_idx, min(len(t_ctrl_idx), 2000), replace=False)
    tX = tgt.X
    def t_rows(idx):
        s = tX[idx]
        return s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    t_ctrl_rows_common = t_rows(t_ctrl_idx)[:, t_cols]
    # 目标数据映射进源基因空间（缺失基因置 0；跨库标准做法，记入 metrics.meta）
    t_ctrl_rows_src = np.zeros((t_ctrl_rows_common.shape[0], src.n_vars), dtype=np.float32)
    for j, g in enumerate(common_genes):
        t_ctrl_rows_src[:, gpos_s[g]] = t_ctrl_rows_common[:, j]
    t_ctrl_mean = t_ctrl_rows_common.mean(axis=0)
    # 目标 ctrl 编码（源基因空间 + 补 obs 列）
    tmp = ad.AnnData(t_ctrl_rows_src.astype(np.float32))
    tmp.obs["condition"] = "ctrl"
    tmp.obs["cell_type"] = src_celltype  # 源系上下文（CPA/scGen 转移语义）
    tmp.obs_names = [f"tc{a.seed}-{j}" for j in range(tmp.n_obs)]
    lat_ctrl_t = model.get_latent_representation(tmp).mean(axis=0)

    # 目标系各条件真实均值（公共基因）
    true_means, de_means, conds_used = {}, {}, []
    for c in common_conds:
        ci = np.where(tg_cond == c)[0]
        if len(ci) < 3:
            continue
        rows = t_rows(ci)[:, t_cols]
        if len(rows) > 500:
            rows = rows[rng.choice(len(rows), 500, replace=False)]
        true_means[c] = rows.mean(axis=0)
        de_means[c] = M.de_top_genes(t_ctrl_rows_common, rows, n_top=20)
        conds_used.append(c)
    run.log(f"可用条件 {len(conds_used)}")

    # 预测：每条件 = decode(lat_ctrl_t + delta_c)
    recs = []
    global_delta = np.mean([lat_means[c] - lat_ctrl_s for c in conds_used if c in lat_means], axis=0)
    pred_mean_transfer_full = decode(lat_ctrl_t[None, :] + global_delta[None, :], "gt")
    pred_mean_transfer = pred_mean_transfer_full[s_cols]
    deltas_used = []
    for c in conds_used:
        if c not in lat_means:
            continue
        deltas_used.append((c, lat_means[c] - lat_ctrl_s))
    # 分块解码（每条件一行）
    lat_in = np.stack([lat_ctrl_t + d for _, d in deltas_used])
    tmp = ad.AnnData(lat_in.astype(np.float32))
    tmp.obs["condition"] = "ctrl"
    tmp.obs["cell_type"] = src_celltype
    tmp.obs_names = [f"pd{a.seed}-{j}" for j in range(tmp.n_obs)]
    out = model.module.generative(torch.tensor(lat_in, dtype=torch.float32).to(model.device))
    pmat = out["px"].cpu().detach().numpy()
    pred_by_cond = {c: pmat[j] for j, (c, _) in enumerate(deltas_used)}

    for c in conds_used:
        true_mean = true_means[c]
        de_idx = de_means[c]
        tstat = true_mean - t_ctrl_mean
        if c in pred_by_cond:
            pmean = pred_by_cond[c][s_cols]  # 源空间裁回公共基因空间
            rec = {"condition": c,
                   "pearson_deg": M.pearson_deg(pmean, true_mean, de_idx),
                   "mse_deg": M.mse_top20(pmean, true_mean, de_idx),
                   "r2_deg": M.r2_deg(pmean, true_mean, de_idx),
                   "jaccard_top20": M.jaccard_topk(pmean - t_ctrl_mean, tstat, k=20)}
            recs.append(("scgen", rec))
        # linear_transfer
        rec = {"condition": c,
               "pearson_deg": M.pearson_deg(pred_mean_transfer, true_mean, de_idx),
               "mse_deg": M.mse_top20(pred_mean_transfer, true_mean, de_idx),
               "r2_deg": M.r2_deg(pred_mean_transfer, true_mean, de_idx),
               "jaccard_top20": M.jaccard_topk(pred_mean_transfer - t_ctrl_mean, tstat, k=20)}
        recs.append(("linear_transfer", rec))
        # mean
        rec = {"condition": c,
               "pearson_deg": M.pearson_deg(t_ctrl_mean, true_mean, de_idx),
               "mse_deg": M.mse_top20(t_ctrl_mean, true_mean, de_idx),
               "r2_deg": M.r2_deg(t_ctrl_mean, true_mean, de_idx),
               "jaccard_top20": 0.0}
        recs.append(("mean_passthrough", rec))
    # random
    keys = [c for c in conds_used]
    perm = rng.permutation(len(keys))
    for j, c in enumerate(keys):
        donor = keys[perm[j]]
        if donor not in true_means or c not in de_means:
            continue
        de_idx = de_means[c]
        pmean = true_means[donor]
        recs.append(("random", {"condition": c,
                                "pearson_deg": M.pearson_deg(pmean, true_means[c], de_idx),
                                "mse_deg": M.mse_top20(pmean, true_means[c], de_idx),
                                "r2_deg": M.r2_deg(pmean, true_means[c], de_idx),
                                "jaccard_top20": M.jaccard_topk(pmean - t_ctrl_mean, true_means[c] - t_ctrl_mean, k=20)}))

    import pandas as pd
    df = pd.DataFrame([{"method": m, **r} for m, r in recs])
    df.to_csv(run.dir / "per_condition_all.csv", index=False)
    summary = {}
    for m, g in df.groupby("method"):
        num = g.select_dtypes(include=[float])
        summary[m] = {k: {"mean": float(num[k].mean()), "std": float(num[k].std(ddof=1))} for k in num.columns}
        summary[m]["n_conditions"] = int(len(g))
    run.write_metrics({
        "meta": {"source": a.source, "target": a.target, "seed": a.seed, "level": "T1",
                 "n_common_genes": len(common_genes), "protocol": "source-delta-transfer",
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("T1 完成：" + json.dumps({m: {"r2": round(s["r2_deg"]["mean"], 4), "n": s["n_conditions"]}
                                      for m, s in summary.items()}, ensure_ascii=False))
    print("T1_SCGEN_DONE", flush=True)


if __name__ == "__main__":
    main()