#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_split_t3dose.py：Tahoe T3-剂量 split（协议 v1）。
规则：每药物剂量升序 d1..dk（k>=3）：train=d1..d(k-2)，val=d(k-1)，ood(test)=dk；
      k=2：train=d1，ood=d2；k=1：整药计入 train（不进剂量网格）。
      DMSO_TF（对照，dose=0）恒为 train。
输出: data/processed/tahoe/tahoe_crc_splitobs.csv（obs_names->split）+ docs/tahoe_t3dose_split.json（协议落盘）。
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    a = ad.read_h5ad(ROOT / "data/processed/tahoe/tahoe_crc.h5ad", backed="r")
    obs = a.obs[["condition", "dose", "cell_type"]].copy()
    obs["dose"] = obs["dose"].astype(float)
    assign = pd.Series(index=obs.index, dtype=object, data="train")
    protocol = {}
    ctrl_mask = obs["condition"] == "DMSO_TF"
    assign[ctrl_mask] = "train"
    for drug, g in obs[~ctrl_mask].groupby("condition"):
        doses = sorted(g["dose"].unique())
        k = len(doses)
        if k >= 3:
            tr, val, ood = doses[:-2], [doses[-2]], [doses[-1]]
        elif k == 2:
            tr, val, ood = [doses[0]], [], [doses[1]]
        else:
            continue
        protocol[drug] = {"doses": doses, "train": tr, "val": val, "ood": ood, "n_cells": int(len(g))}
        m = g["dose"].isin(tr + val)
        assign.loc[g.index[m]] = "train"
        m_ood = g["dose"].isin(ood)
        assign.loc[g.index[m_ood]] = "ood"
        if val:
            m_val = g["dose"].isin(val)
            assign.loc[g.index[m_val]] = "val"
    out = pd.DataFrame({"split": assign})
    out.to_csv(ROOT / "data/processed/tahoe/tahoe_crc_splitobs.csv", encoding="utf-8")
    summary = {
        "protocol": "per-drug dose extrapolation: train=low, val=second-highest(k>=3), ood=highest",
        "n_drugs_in_grid": len(protocol),
        "n_cells": {"train": int((assign == "train").sum()), "val": int((assign == "val").sum()),
                    "ood": int((assign == "ood").sum())},
        "per_drug": protocol,
    }
    (ROOT / "docs/tahoe_t3dose_split.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("protocol", "n_drugs_in_grid", "n_cells")}, ensure_ascii=False))
    print("SPLIT_DOSE_DONE", flush=True)


if __name__ == "__main__":
    main()