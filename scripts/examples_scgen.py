#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方示例冒烟：scGen API 全链路（SCGEN 训练 + 条件预测 + batch_removal）。
数据：合成两条件小数据集（ctrl/stim），目的=API 跑通与耗时标定；正式跑分由统一接口供数据。
用法: python examples_scgen.py [--seed 149]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=149)
    a = ap.parse_args()

    import numpy as np
    import scanpy as sc
    import anndata as ad
    import scgen
    import scgen_compat
    scgen_compat.apply()
    import torch
    from importlib.metadata import version

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = runlib.begin(model="scgen", dataset="synthetic", level="smoke", seed=a.seed,
                       task="论文② scGen API 官方示例冒烟", hours=1,
                       extra={"scgen": version("scgen"), "example": "api-smoke-synthetic"})
    rng = np.random.default_rng(a.seed)
    n_genes, n_cells = 100, 600
    base = rng.normal(0, 1, (n_cells, n_genes))
    cond = np.array(["ctrl"] * (n_cells // 2) + ["stim"] * (n_cells // 2))
    eff = np.zeros(n_genes); eff[:20] = 2.0
    X = base + np.where(cond[:, None] == "stim", eff, 0.0)
    adata = ad.AnnData(X.astype(np.float32))
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]  # obs_names 必须唯一，否则 scvi 转移注册返回 None
    adata.obs["condition"] = cond
    adata.obs["cell_type"] = "CT"
    adata.obs["batch"] = "b0"
    run.log(f"synthetic adata {adata.shape}")
    scgen.SCGEN.setup_anndata(adata, batch_key="condition", labels_key="cell_type")  # scgen>=2.1.1: batch=condition, labels=cell_type
    model = scgen.SCGEN(adata)
    model.train(max_epochs=20, batch_size=64, early_stopping=False, enable_progress_bar=False)
    run.log("train done; predict ctrl->stim ...")
    pred_adata, delta = model.predict(ctrl_key="ctrl", stim_key="stim", celltype_to_predict="CT")  # 2.1.1 返回 (pred, delta)
    run.log(f"pred shape={pred_adata.shape}")
    latent = model.get_latent_representation(adata)
    run.log(f"latent shape={latent.shape}")
    run.write_metrics({"smoke": "ok", "pred_shape": list(pred_adata.shape), "latent_shape": list(latent.shape)})

if __name__ == "__main__":
    main()