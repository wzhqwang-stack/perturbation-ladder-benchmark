#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方示例冒烟：CPA 0.8.8 API（ComPertAPI + CPA）全链路，CPU。
数据：合成数据（ctrl/DrugA/DrugB × 剂量 × 两细胞型），目的=API 跑通；正式跑分由统一接口供数据。
用法: python examples_cpa.py [--seed 149]
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
    import pandas as pd
    import anndata as ad
    import cpa
    import cpa  # cpa-tools 0.8.8：CPA(adata, **keys) 官方流程
    import torch
    from importlib.metadata import version

    run = runlib.begin(model="cpa", dataset="synthetic", level="smoke", seed=a.seed,
                       task="论文② CPA 0.8.8 官方示例冒烟(CPU)", hours=1,
                       extra={"cpa": version("cpa-tools"), "example": "compertapi-smoke-synthetic"})
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    n_genes, n_per = 200, 50
    rows = []
    # CPA 约定：ctrl 的 dose='0'；扰动 '1'/'10' 两档
    for d, z in [("ctrl", "0"), ("DrugA", "1"), ("DrugA", "10"), ("DrugB", "1"), ("DrugB", "10")]:
        for i in range(n_per):
            rows.append((d, z, "CT-A" if (i % 2 == 0) else "CT-B"))
    n = len(rows)
    base = rng.poisson(5.0, (n, n_genes)).astype(np.float32)  # 非负计数底座（CPA 内部 log1p 不可见负值）
    de = {"DrugA": 1, "DrugB": 2}
    X = base.copy()
    for i, (d, z, ct) in enumerate(rows):
        if d in de:
            X[i] += float(z) * 3.0 * de[d]
    adata = ad.AnnData(X.astype(np.float32))
    adata.obs["condition"] = [r[0] for r in rows]
    adata.obs["dose"] = [r[1] for r in rows]
    adata.obs["cell_type"] = [r[2] for r in rows]
    run.log(f"synthetic adata {adata.shape}")

    # cpa-tools 0.8.8 官方流程：setup_anndata -> CPA(adata) -> train
    cpa.CPA.setup_anndata(adata, perturbation_key="condition", control_group="ctrl",
                          dosage_key="dose", categorical_covariate_keys=["cell_type"])
    run.log("setup_anndata ok; CPA construct ...")
    model = cpa.CPA(adata, seed=a.seed)
    run.log("CPA constructed; train 10 epochs (CPU) ...")
    model.train(max_epochs=10, use_gpu=False, enable_progress_bar=False)
    run.log("train done; latent ...")
    latent = model.get_latent_representation()
    if isinstance(latent, dict):  # cpa 0.8.8 返回 dict（键名为扰动组合）
        shapes = {k: list(np.asarray(v).shape) for k, v in latent.items()}
        run.log(f"latent dict keys={list(latent.keys())[:3]} shapes={list(shapes.values())[:3]}")
        run.write_metrics({"smoke": "ok", "device": "cpu", "epochs": 10, "latent_shapes": shapes})
    else:
        run.log(f"latent shape={latent.shape}")
        run.write_metrics({"smoke": "ok", "device": "cpu", "epochs": 10, "latent_shape": list(latent.shape)})

if __name__ == "__main__":
    main()