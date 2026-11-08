#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate.py：扫描 results/runs 的 T3 格子 metrics，产出全格矩阵 + GEARS/linear 配对 Wilcoxon。
用法: python aggregate.py [--out docs/网格汇总_stage02.md]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
RUNS = ROOT / "results" / "runs"
DATASETS = ["norman", "adamson", "replogle_k562_essential"]
SEEDS = [149, 150, 151]


def load_cell(model, ds, seed, fname):
    """取该格最新 run 的 per-condition CSV（兼容 T3-simulation / T3-smoke 目录名）。"""
    for level in ("T3-simulation", "T3-smoke"):
        cands = sorted(RUNS.glob(f"{model}_{ds}_{level}_s{seed}_*/{fname}"))
        if cands:
            return pd.read_csv(cands[-1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "网格汇总_stage02.md"))
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            for model, fname in [("gears", "per_condition_gears.csv"),
                                 ("scgen", "per_condition_scgen.csv")]:
                csv = load_cell(model, ds, seed, fname)
                if csv is None:
                    rows.append({"dataset": ds, "seed": seed, "method": model, "missing": True})
                    continue
                num = csv.select_dtypes(include=[float])
                r = {"dataset": ds, "seed": seed, "method": model, "n": len(csv)}
                for m in ["pearson_deg", "mse_deg", "r2_deg", "jaccard_top20"]:
                    if m in num.columns:
                        r[m] = num[m].mean()
                rows.append(r)
            bl = load_cell("baselines", ds, seed, "per_condition_linear.csv")
            if bl is not None:
                num = bl.select_dtypes(include=[float])
                r = {"dataset": ds, "seed": seed, "method": "linear", "n": len(bl)}
                for m in ["pearson_deg", "mse_deg", "r2_deg", "jaccard_top20"]:
                    if m in num.columns:
                        r[m] = num[m].mean()
                rows.append(r)
            for b in ["mean_passthrough", "random"]:
                csv = load_cell("baselines", ds, seed, f"per_condition_{b}.csv")
                if csv is None:
                    continue
                num = csv.select_dtypes(include=[float])
                r = {"dataset": ds, "seed": seed, "method": b, "n": len(csv)}
                for m in ["pearson_deg", "mse_deg", "r2_deg", "jaccard_top20"]:
                    if m in num.columns:
                        r[m] = num[m].mean()
                rows.append(r)
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["dataset", "method"], columns="seed",
                           values=["r2_deg", "pearson_deg", "mse_deg", "jaccard_top20"], aggfunc="mean")

    # 配对 Wilcoxon：GEARS vs linear（逐条件，按 seed 配对取均值轮廓）
    sig_rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            g = load_cell("gears", ds, seed, "per_condition_gears.csv")
            l = load_cell("baselines", ds, seed, "per_condition_linear.csv")
            if g is None or l is None:
                continue
            m = g.merge(l, on="condition", suffixes=("_gears", "_linear"))
            for metric in ["r2_deg", "pearson_deg"]:
                if f"{metric}_gears" not in m.columns:
                    continue
                d = m[f"{metric}_gears"] - m[f"{metric}_linear"]
                if d.abs().sum() == 0:
                    continue
                try:
                    w = stats.wilcoxon(m[f"{metric}_gears"], m[f"{metric}_linear"])
                    sig_rows.append({"dataset": ds, "seed": seed, "metric": metric,
                                     "n": int(len(m)), "median_diff": float(d.median()),
                                     "win_rate": float((d > 0).mean()), "p": float(w.pvalue)})
                except ValueError as e:
                    sig_rows.append({"dataset": ds, "seed": seed, "metric": metric, "error": str(e)[:60]})
    sig = pd.DataFrame(sig_rows)
    # BH-FDR 校正（家族=表内全部有效 Wilcoxon 检验；2026-11-02 审查修正）
    if len(sig) and "p" in sig.columns:
        pv = sig["p"].to_numpy(dtype=float)
        ok = ~np.isnan(pv)
        pbh = np.full(len(sig), np.nan)
        if ok.sum():
            m_ = int(ok.sum())
            o = np.argsort(pv[ok])
            r_ = pv[ok][o]
            q = np.minimum.accumulate((r_ * m_ / (np.arange(m_) + 1))[::-1])[::-1]
            q = np.clip(q, 0, 1)
            tmp = np.empty(m_)
            tmp[o] = q
            pbh[ok] = tmp
        sig["p_BH"] = pbh
        sig["sig_BH<0.05"] = np.where(np.isnan(pbh), "", np.where(pbh < 0.05, "yes", "no"))

    lines = ["# T3 网格汇总（stage02，自动生成）", "",
             f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}", "",
             "## 全格矩阵（R²）", "",
             pivot["r2_deg"].round(4).to_markdown(), "",
             "## 全格矩阵（Pearson）", "",
             pivot["pearson_deg"].round(4).to_markdown(), "",
             "## GEARS vs linear 配对 Wilcoxon（逐条件差值；p_BH=全表 BH 校正）", "",
             (sig.round(4).to_markdown(index=False) if len(sig) else "（无可用配对）"), ""]
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("AGGREGATE_DONE", flush=True)


if __name__ == "__main__":
    main()