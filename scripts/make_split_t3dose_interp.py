#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_split_t3dose_interp.py：Tahoe T3-剂量【插值】split（协议 v1i）。
规则（插值）：每药物剂量升序 d1..dk（k>=3）：ood(test)=中位档，train=其余全部（含最高档）；
      k<=2：整药计入 train（不进插值网格）。
      DMSO_TF（对照，dose=0）恒为 train。
输出: data/processed/tahoe/tahoe_crc_splitobs_interp.csv（obs_names->split）+ docs/tahoe_t3dose_split_interp.json（协议落盘）。
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
            mid = k // 2
            ood = [doses[mid]]
            tr = [d for i, d in enumerate(doses) if i != mid]
            val = []
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
    out.to_csv(ROOT / "data/processed/tahoe/tahoe_crc_splitobs_interp.csv", encoding="utf-8")
    summary = {
        "protocol": "per-drug dose INTERPOLATION: ood=median dose, train=rest incl. highest",
        "n_drugs_in_grid": len(protocol),
        "n_cells": {"train": int((assign == "train").sum()), "val": int((assign == "val").sum()),
                    "ood": int((assign == "ood").sum())},
        "per_drug": protocol,
    }
    (ROOT / "docs/tahoe_t3dose_split_interp.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("protocol", "n_drugs_in_grid", "n_cells")}, ensure_ascii=False))
    print("SPLIT_DOSE_DONE", flush=True)


if __name__ == "__main__":
    main()