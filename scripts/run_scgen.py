#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_scgen.py：scGen 网格格子（T3 全局平均 delta 协议）。
协议（记入 metrics.meta，与协议文档 §2 的偏差显式声明）：
  每数据集×seed 训练一次 SCGEN（train 条件全集，batch_key=condition, labels_key=cell_type）；
  对每个 test 扰动 p：pred = ctrl 解码(平均 latent delta 加到 ctrl latent)。
  ——scGen 无法零样本外推全新扰动（其 OOD 能力在 unseen 细胞型/物种），此协议是其 T3 可行上界近似。
用法: python run_scgen.py --dataset norman --seed 149 [--device cuda] [--max-test 0]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=149)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-test", type=int, default=0)
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
    split_json = ROOT / "data" / "processed" / a.dataset / "splits" / f"split_T3-simulation_s{a.seed}.json"
    contract_h5ad = ROOT / "data" / "processed" / a.dataset / f"{a.dataset}.h5ad"
    run = runlib.begin(model="scgen", dataset=a.dataset, level="T3-simulation", seed=a.seed,
                       task=f"论文② scGen {a.dataset} T3 s{a.seed} 网格格子",
                       hours=2, gpu=(device == "cuda"),
                       inputs=[str(contract_h5ad), str(split_json)],
                       extra={"device": device, "protocol": "global_mean_delta"})
    run.log("加载契约数据 ...")
    adata = ad.read_h5ad(contract_h5ad)
    split = json.loads(split_json.read_text(encoding="utf-8"))
    cond = adata.obs["condition"].astype(str).to_numpy()
    X = adata.X
    test_conds = list(split["test"])
    if a.max_test > 0:
        test_conds = test_conds[: a.max_test]

    train_conds = set(split["train"]) | {"ctrl"}
    mask = np.isin(cond, list(train_conds))
    adata_tr = adata[mask].copy()
    run.log(f"训练子集 {adata_tr.shape}（条件 {adata_tr.obs['condition'].nunique()}）")
    if adata_tr.n_obs > 40000:
        sc.pp.subsample(adata_tr, n_obs=40000, copy=False)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    scgen.SCGEN.setup_anndata(adata_tr, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(adata_tr)
    model.train(max_epochs=30, batch_size=256, early_stopping=False, enable_progress_bar=False)
    run.log("train done")

    rng = np.random.default_rng(a.seed)
    ctrl_idx = np.where(cond == "ctrl")[0]
    def sub(idx, n):
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        s = X[idx]
        return s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    ctrl_sub = sub(ctrl_idx, 2000)
    ctrl_mean = ctrl_sub.mean(axis=0)

    def lat_of_rows(rows, obs_rows, tag):
        tmp = ad.AnnData(rows.astype(np.float32))
        tmp.obs["condition"] = obs_rows["condition"].to_numpy()
        tmp.obs["cell_type"] = obs_rows["cell_type"].to_numpy()
        tmp.obs_names = [f"{tag}{a.seed}-{j}" for j in range(len(rows))]
        return model.get_latent_representation(tmp)

    train_pert_conds = sorted(split["train"])
    deltas = []
    lat_ctrl = None
    for c in train_pert_conds:
        ci = np.where(cond == c)[0]
        if len(ci) < 3:
            continue
        ci = rng.choice(ci, min(len(ci), 200), replace=False)
        rows = sub(ci, 200)
        deltas.append(lat_of_rows(rows, adata.obs.iloc[ci], "t").mean(axis=0))
        if lat_ctrl is None:
            cidx = np.where(cond == "ctrl")[0]
            cidx = rng.choice(cidx, min(len(cidx), 200), replace=False)
            crows = sub(cidx, 200)
            lat_ctrl = lat_of_rows(crows, adata.obs.iloc[cidx], "c").mean(axis=0)
    mean_delta = np.mean(deltas, axis=0) - lat_ctrl
    cidx_full = rng.choice(ctrl_idx, min(len(ctrl_idx), 500), replace=False)
    crows_full = sub(cidx_full, 500)
    lat_pred = lat_of_rows(crows_full, adata.obs.iloc[cidx_full], "cf") + mean_delta
    run.log("解码预测轮廓 ...")
    import torch as th
    out = model.module.generative(th.tensor(np.asarray(lat_pred), dtype=th.float32).to(model.device))
    pmean_all = out["px"].cpu().detach().numpy()
    ctrl_mean_pred_basis = crows_full.mean(axis=0)

    recs = []
    for k, c in enumerate(test_conds):
        ci = np.where(cond == c)[0]
        if len(ci) < 3:
            continue
        rows_c = sub(ci, 1000)
        true_mean = rows_c.mean(axis=0)
        de_idx = M.de_top_genes(ctrl_sub, rows_c, n_top=20)
        pmean = pmean_all[k] if pmean_all.ndim > 1 and k < len(pmean_all) else pmean_all.mean(axis=0)
        recs.append({
            "condition": c,
            "pearson_deg": M.pearson_deg(pmean, true_mean, de_idx),
            "mse_deg": M.mse_top20(pmean, true_mean, de_idx),
            "r2_deg": M.r2_deg(pmean, true_mean, de_idx),
            "jaccard_top20": M.jaccard_topk(pmean - ctrl_mean, true_mean - ctrl_mean, k=20),
        })
    import pandas as pd
    df = pd.DataFrame(recs)
    num = df.select_dtypes(include=[float])
    summary = {m: {"mean": float(num[m].mean()), "std": float(num[m].std(ddof=1))} for m in num.columns}
    summary["n_conditions"] = int(len(df))
    df.to_csv(run.dir / "per_condition_scgen.csv", index=False)
    run.write_metrics({
        "meta": {"dataset": a.dataset, "seed": a.seed, "level": "T3-simulation",
                 "protocol": "global_mean_delta",
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("scGen 格子完成：" + json.dumps({k: round(v["mean"], 4) for k, v in summary.items() if isinstance(v, dict)}, ensure_ascii=False))
    print("SCGEN_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()