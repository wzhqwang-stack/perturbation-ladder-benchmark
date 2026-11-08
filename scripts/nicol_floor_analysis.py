#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nicol_floor_analysis.py：Nicol et al. (bioRxiv 2026.05.07.723486) 控制复用偏置的正面处理。
输入：docs/appendix_delta_r2.csv（共享控制列 + 控制拆分列）。
输出：docs/nicol_floor_analysis.md —— (a) 偏置量级（逐条件配对差）；(b) 噪声地板（拆分口径 mean 直通的真实 delta vs 控制噪声差）；
      (c) 超地板信号（各方法 split delta-Pearson 减去地板，含逐条件胜率）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
CSV = ROOT / "docs" / "appendix_delta_r2.csv"
OUT = ROOT / "docs" / "nicol_floor_analysis.md"


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    df = pd.read_csv(CSV)
    rows = []
    for (lv, dirn), g in df.groupby(["level", "direction"]):
        r = {"level": lv, "direction": dirn, "n": len(g)}
        for meth, sh, sp in [("GEARS", "delta_pearson_GEARS", "delta_pearson_GEARS_split"),
                             ("random", "delta_pearson_random", "delta_pearson_random_split")]:
            d = (g[sh] - g[sp]).dropna()
            r[f"{meth}_bias_median"] = float(d.median())
            r[f"{meth}_bias_p5"] = float(d.quantile(0.05))
            r[f"{meth}_bias_p95"] = float(d.quantile(0.95))
        fl = g["delta_pearson_mean_split"].dropna()
        r["floor_mean"] = float(fl.mean())
        r["floor_sd"] = float(fl.std(ddof=1))
        r["floor_median"] = float(fl.median())
        for meth, sp in [("GEARS", "delta_pearson_GEARS_split"), ("random", "delta_pearson_random_split")]:
            v = g[sp].dropna()
            r[f"{meth}_split_mean"] = float(v.mean())
            r[f"{meth}_above_floor"] = float(v.mean() - fl.mean())
            r[f"{meth}_above_win"] = float((v > fl.mean()).mean())
        rows.append(r)
    r = pd.DataFrame(rows)
    fmt = lambda x, k=3: f"{x:.{k}f}"
    lines = ["# Nicol 偏置的正面处理：偏置量级 · 噪声地板 · 超地板信号", "",
             "来源文献：Nicol PB, Shivakumar S, Irizarry RA. Spurious correlation inflates performance in single-cell "
             "perturbation prediction. bioRxiv 2026.05.07.723486（doi:10.64898/2026.05.07.723486）。",
             "方法：控制拆分（ctrl_A=真实 delta 中心，ctrl_B=预测 delta 中心；各 1,000 细胞，源 2,000 控制细胞等分）；"
             "地板=拆分口径下「控制组直通」的真实 delta 与控制噪声差之间的 delta-Pearson（无信息预测的期望水平）。",
             "",
             "## 1. 偏置量级（共享控制 − 控制拆分，逐条件配对）",
             "",
             "| 层 | 方向 | n | GEARS 中位偏置 [5%,95%] | random 中位偏置 [5%,95%] |",
             "|---|---|---|---|---|"]
    for _, x in r.iterrows():
        lines.append(f"| {x['level']} | {x['direction']} | {int(x['n'])} | "
                     f"{fmt(x['GEARS_bias_median'],4)} [{fmt(x['GEARS_bias_p5'],3)},{fmt(x['GEARS_bias_p95'],3)}] | "
                     f"{fmt(x['random_bias_median'],4)} [{fmt(x['random_bias_p5'],3)},{fmt(x['random_bias_p95'],3)}] |")
    lines += ["", "结论：在本基准（每层 2,000 控制细胞、每条件 ≤500 扰动细胞）下，控制复用偏置的中位量级为 "
              "0.0001–0.0036（90% 区间约 ±0.02–0.06），实际可忽略——与 Nicol 等报告的「控制细胞有限时偏置严重」一致，"
              "本数据不在其高危区间。",
              "",
              "## 2. 噪声地板与超地板信号（拆分口径）",
              "",
              "| 层 | 方向 | 地板 mean±sd | GEARS split | 超地板 | 逐条件胜率 | random split | 超地板 | 逐条件胜率 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, x in r.iterrows():
        lines.append(f"| {x['level']} | {x['direction']} | {fmt(x['floor_mean'],3)}±{fmt(x['floor_sd'],3)} | "
                     f"{fmt(x['GEARS_split_mean'],3)} | {fmt(x['GEARS_above_floor'],3)} | {x['GEARS_above_win']:.0%} | "
                     f"{fmt(x['random_split_mean'],3)} | {fmt(x['random_above_floor'],3)} | {x['random_above_win']:.0%} |")
    lines += ["", "## 3. 判读", "",
              "1. **T1 K→R：GEARS 的 delta 信号与噪声地板不可区分**（0.149 vs 地板 0.145，超地板 +0.004，逐条件胜率 52%）"
              "——跨系前向方向上 GEARS 无任何可迁移的方向信息；random（0.660，超地板 +0.515，胜率 91%）的强表现是真实的借用效应。",
              "2. **T1 R→K / T2 双向：两法皆超地板**（GEARS +0.22~+0.32，random +0.28~+0.67）——借用效应强于模型迁移，"
              "但 GEARS 亦非零信号（胜率 77–89%），负结果是「不够强」而非「完全没有」。",
              "3. **T3：GEARS 超地板 +0.59~+0.73、胜率 92–95%**——分布内条件特异信号真实且稳健（adamson 的 random 亦高，"
              "与该面板条件同质性一致，见正文 2.2）。",
              "4. 地板本身分层差异大（−0.07~+0.145）：T1 K→R 的 RPE1 控制结构贡献了高地板——报告「超地板量」比报告原始"
              "delta-Pearson 更严格，本文采用后者为主口径。", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()