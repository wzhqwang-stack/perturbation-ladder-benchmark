#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_baselines_t2.py：T2 基线三件套（mean=目标 ctrl 直通 / random=真值置换 / linear_transfer=源系全局平均 delta）。
双向各一个 run；指标与 GEARS T2 同口径（公共基因空间 1,731、DE@20）。"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
import sys as _sys
SEED = int(_sys.argv[_sys.argv.index("--seed") + 1]) if "--seed" in _sys.argv else 149


def delta_r2(pred_mean, true_mean, ctrl_mean, de_idx):
    x = (np.asarray(pred_mean, dtype=np.float64) - np.asarray(ctrl_mean, dtype=np.float64))[de_idx]
    y = (np.asarray(true_mean, dtype=np.float64) - np.asarray(ctrl_mean, dtype=np.float64))[de_idx]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((y - x) ** 2).sum()) / ss_tot


def run_dir_pair(source, target):
    import anndata as ad
    run = runlib.begin(model="baselines", dataset=f"t2_{source}-to-{target}", level="T2", seed=SEED,
                       task=f"论文② T2 基线三件套 {source}-to-{target}（CPU）", hours=1, gpu=False,
                       inputs=[str(ROOT / f"data/processed/{source}/{source}.h5ad"),
                               str(ROOT / f"data/processed/{target}/{target}.h5ad")],
                       extra={"protocol": "baselines only; common gene space"})
    run.log("加载两系契约 ...")
    src = ad.read_h5ad(ROOT / f"data/processed/{source}/{source}.h5ad")
    tgt = ad.read_h5ad(ROOT / f"data/processed/{target}/{target}.h5ad")
    common_genes = sorted(set(src.var["gene_name"].astype(str)) & set(tgt.var["gene_name"].astype(str)))
    gp_s = {g: i for i, g in enumerate(src.var["gene_name"].astype(str))}
    gp_t = {g: i for i, g in enumerate(tgt.var["gene_name"].astype(str))}
    s_cols = [gp_s[g] for g in common_genes]
    t_cols = [gp_t[g] for g in common_genes]
    rng = np.random.default_rng(SEED)

    sc_cond = src.obs["condition"].astype(str).to_numpy()
    t_cond = tgt.obs["condition"].astype(str).to_numpy()
    tX = tgt.X
    srcX = src.X

    def rows_of(X, idx, cols):
        s = X[idx]
        s = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
        return s[:, cols]

    # 源系全局平均 delta（公共基因空间）
    s_ctrl_idx = np.where(sc_cond == "ctrl")[0]
    s_ctrl_mean = rows_of(srcX, rng.choice(s_ctrl_idx, min(len(s_ctrl_idx), 2000), replace=False), s_cols).mean(axis=0)
    src_conds = sorted(set(sc_cond) - {"ctrl"})
    deltas = []
    for c in src_conds:
        ci = np.where(sc_cond == c)[0]
        if len(ci) < 3:
            continue
        if len(ci) > 300:
            ci = rng.choice(ci, 300, replace=False)
        deltas.append(rows_of(srcX, ci, s_cols).mean(axis=0) - s_ctrl_mean)
    global_delta = np.mean(deltas, axis=0)
    pred_linear = s_ctrl_mean * 0 + global_delta  # 目标 ctrl + delta（先取目标 ctrl 均值）
    # 目标 ctrl
    t_ctrl_idx = np.where(t_cond == "ctrl")[0]
    t_ctrl_rows = rows_of(tX, rng.choice(t_ctrl_idx, min(len(t_ctrl_idx), 2000), replace=False), t_cols)
    t_ctrl_mean = t_ctrl_rows.mean(axis=0)
    pred_linear = t_ctrl_mean + global_delta

    true_means, de_means, conds_used = {}, {}, []
    for c in sorted(set(t_cond) - {"ctrl"}):
        ci = np.where(t_cond == c)[0]
        if len(ci) < 3:
            continue
        rows = rows_of(tX, ci, t_cols)
        if len(rows) > 500:
            rows = rows[rng.choice(len(rows), 500, replace=False)]
        true_means[c] = rows.mean(axis=0)
        de_means[c] = M.de_top_genes(t_ctrl_rows, rows, n_top=20)
        conds_used.append(c)
    run.log(f"可用条件 {len(conds_used)}")

    recs = []
    for c in conds_used:
        tm = true_means[c]
        de = de_means[c]
        for method, pmean, jac_ref in (
            ("linear_transfer", pred_linear, pred_linear - t_ctrl_mean),
            ("mean_passthrough", t_ctrl_mean, None),
        ):
            recs.append({"method": method, "condition": c,
                         "pearson_deg": M.pearson_deg(pmean, tm, de),
                         "mse_deg": M.mse_top20(pmean, tm, de),
                         "r2_deg": M.r2_deg(pmean, tm, de),
                         "delta_r2_deg": delta_r2(pmean, tm, t_ctrl_mean, de),
                         "jaccard_top20": 0.0 if jac_ref is None else M.jaccard_topk(jac_ref, tm - t_ctrl_mean, k=20)})
    keys = list(conds_used)
    perm = rng.permutation(len(keys))
    for j, c in enumerate(keys):
        donor = keys[perm[j]]
        tm, de = true_means[c], de_means[c]
        recs.append({"method": "random", "condition": c,
                     "pearson_deg": M.pearson_deg(true_means[donor], tm, de),
                     "mse_deg": M.mse_top20(true_means[donor], tm, de),
                     "r2_deg": M.r2_deg(true_means[donor], tm, de),
                     "delta_r2_deg": delta_r2(true_means[donor], tm, t_ctrl_mean, de),
                     "jaccard_top20": M.jaccard_topk(true_means[donor] - t_ctrl_mean, tm - t_ctrl_mean, k=20)})
    import pandas as pd
    df = pd.DataFrame(recs)
    summary = {}
    for m, g in df.groupby("method"):
        num = g.select_dtypes(include=[float])
        summary[m] = {k: {"mean": float(num[k].mean()), "std": float(num[k].std(ddof=1))} for k in num.columns}
        summary[m]["n_conditions"] = int(len(g))
    df.to_csv(run.dir / "per_condition_baselines_t2.csv", index=False)
    run.write_metrics({
        "meta": {"source": source, "target": target, "seed": SEED, "level": "T2",
                 "n_common_genes": len(common_genes),
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("T2 基线完成：" + json.dumps({m: {"r2": round(s["r2_deg"]["mean"], 4)} for m, s in summary.items()}, ensure_ascii=False))
    print("BASELINES_T2_DONE", source, target, flush=True)


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import metrics as M  # noqa: F401
    global runlib
    import runlib
    global M
    run_dir_pair("norman", "adamson")
    run_dir_pair("adamson", "norman")


if __name__ == "__main__":
    main()