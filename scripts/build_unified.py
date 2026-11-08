#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_unified.py：GEARS 预处理数据 → 论文②统一契约 h5ad（docs/统一评测接口设计.md §1）。

契约:
  X = log-normalized（沿用 GEARS 预处理值；转换器抽查非零值域并写入 uns['processing_log']）
  obs 强制列: condition('ctrl' 或扰动，多基因 '+'), cell_type, dose(float: ctrl=0.0, 扰动=1.0 占位),
              dataset, batch(无则 'b0')
  var 强制列: gene_name；可选 ensembl_id
  splits/split_<T级>_<名>.json: 按 (condition, cell_type, dose) 三元组的 train/val/test 划分

用法:
  python build_unified.py --name norman --dataset norman --split-t3 simulation --seed 149
  （GEARS 数据根固定为 论文②/data/raw/gears/<name>/；split pkl 为 <name>_<split>_s<seed>_0.75.pkl）
"""
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import anndata as ad
import numpy as np
import scipy.sparse as sp

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
RAW_GEARS = ROOT / "data" / "raw" / "gears"
OUT_ROOT = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="GEARS 数据名，如 norman")
    ap.add_argument("--dataset", default=None, help="契约中的 dataset 标签，默认同 --name")
    ap.add_argument("--split-t3", default=None, help="GEARS split 名（如 simulation）；提供则同时产出 T3 split JSON")
    ap.add_argument("--seed", type=int, default=149)
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    dataset = a.dataset or a.name
    src = RAW_GEARS / a.name / "perturb_processed.h5ad"
    out_dir = OUT_ROOT / a.name
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}.h5ad"
    log = [f"[{datetime.now().isoformat(timespec='seconds')}] src={src}"]

    print(f"读取 {src} ...", flush=True)
    adata = ad.read_h5ad(src)
    log.append(f"src shape={adata.shape}")

    # --- obs 契约 ---
    cond = adata.obs["condition"].astype(str)
    cell_type = adata.obs["cell_type"].astype(str)
    if "control" in adata.obs.columns:
        is_ctrl = adata.obs["control"].astype(int).to_numpy() == 1
    else:
        is_ctrl = (cond == "ctrl").to_numpy()
    if "dose_val" in adata.obs.columns:
        # CRISPR 无量纲：ctrl=0.0，扰动=1.0（dose_val '1+1'/'1' 等原值另存）
        dose = np.where(is_ctrl, 0.0, 1.0)
    else:
        dose = np.where(is_ctrl, 0.0, 1.0)
    obs_new = pd_df = None
    import pandas as pd
    obs_new = pd.DataFrame({
        "condition": cond.to_numpy(),
        "cell_type": cell_type.to_numpy(),
        "dose": dose,
        "dataset": dataset,
        "batch": "b0",
    }, index=adata.obs_names)
    if "dose_val" in adata.obs.columns:
        obs_new["dose_val_raw"] = adata.obs["dose_val"].astype(str).to_numpy()
    adata.obs = obs_new

    # --- var 契约 ---
    if "gene_name" not in adata.var.columns:
        raise SystemExit("[ERR] var 无 gene_name 列，需人工映射")
    var_new = adata.var[["gene_name"]].copy()
    var_new["ensembl_id"] = var_new.index.astype(str)
    adata.var = var_new

    # --- X 值域抽查（log-normalized 验证，不改值）---
    X = adata.X
    idx = np.linspace(0, X.shape[0] - 1, 64).astype(int)
    sub = X[idx]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    nz = sub[sub > 0]
    log.append(f"X type={type(X).__name__} nonzero_sample min={nz.min():.3f} max={nz.max():.3f} "
               f"median={np.median(nz):.3f} (log1p 域应 ~0-10)")
    if nz.max() > 20:
        print("[WARN] X 非零值域超 log1p 预期（>20），请人工核对是否为原始计数", flush=True)

    adata.uns["processing_log"] = " | ".join(log)
    adata.uns["contract_version"] = "paper2-unified-v1"
    adata.write_h5ad(out, compression="gzip")
    print(f"写出 {out} ({out.stat().st_size/1e9:.2f} GB)", flush=True)

    # --- T3 split JSON ---
    if a.split_t3:
        pkl = RAW_GEARS / f"{a.name}_{a.split_t3}_{a.seed}_0.75.pkl"  # split pkl 在 gears 根目录（2026-10-29 实证）  # GEARS 实际命名无 s 前缀（2026-10-29 实证）
        if pkl.exists():
            with open(pkl, "rb") as f:
                sp_ = pickle.load(f)
            js = {k: (list(v) if not isinstance(v, dict) else {kk: list(vv) for kk, vv in v.items()})
                  for k, v in sp_.items() if isinstance(v, (list, dict))}
            dst = out_dir / "splits" / f"split_T3-{a.split_t3}_s{a.seed}.json"
            dst.write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
            n = {k: (len(v) if not isinstance(v, dict) else sum(len(x) for x in v.values()))
                 for k, v in js.items()}
            print(f"split JSON -> {dst} ({n})", flush=True)
        else:
            print(f"[WARN] split pkl 不存在: {pkl}", flush=True)

    # --- 指纹入库 ---
    import subprocess
    repro = str(ROOT / "scripts" / "repro.py")
    subprocess.run([sys.executable, repro, "sha256", str(out),
                    "--csv", str(ROOT / "checksums.csv"),
                    "--note", f"unified contract copy of GEARS {a.name} (build_unified.py)"], check=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()