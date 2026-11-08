#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate_transfer.py v2：T1/T2/T4 转移层聚合（多种子版）。
按 (模型, 层级, 方向, seed) 收集正典 run（同键取 metrics 最新），再跨 seed 求均值±std。
summary 形状自适应：gears/cpa=指标名键；scgen/baselines=方法名键。
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "docs" / "转移层汇总_T1T2T4.md"
METRICS = ["r2_deg", "pearson_deg", "jaccard_top20", "mse_deg"]
BASELINES = ["mean_passthrough", "linear_transfer", "random"]


def extract_methods(model, summary):
    """返回 {method: {metric: mean, n: int}}。"""
    out = {}
    if model in ("gears", "cpa"):
        out[model] = {m: (summary.get(m, {}).get("mean") if isinstance(summary.get(m), dict) else None)
                      for m in METRICS}
        out[model]["n"] = summary.get("n_conditions")
    else:  # scgen / baselines：方法名键
        keys = (["scgen"] + BASELINES) if model == "scgen" else [k for k in summary if isinstance(summary[k], dict)]
        for meth in keys:
            s = summary.get(meth, {})
            if not isinstance(s, dict):
                continue
            out[meth] = {m: (s.get(m, {}).get("mean") if isinstance(s.get(m), dict) else None) for m in METRICS}
            out[meth]["n"] = s.get("n_conditions")
    return out


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    # (model, lv, src, tgt, seed) -> latest (mtime, summary)
    best = {}
    for d in RUNS.iterdir():
        mj = d / "metrics.json"
        if not d.is_dir() or not mj.exists():
            continue
        try:
            j = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = j.get("meta", {})
        lv = meta.get("level")
        if lv not in ("T1", "T2", "T4"):
            continue
        model = d.name.split("_")[0]
        m = re.search(r"_s(\d+)_", d.name)
        seed = int(m.group(1)) if m else 0
        if lv == "T4":
            src, tgt = "K562-CRISPR", str(meta.get("gse", "?"))
        else:
            src, tgt = str(meta.get("source", "?")), str(meta.get("target", "?"))
        key = (model, lv, src, tgt, seed)
        mt = mj.stat().st_mtime
        if key not in best or mt > best[key][0]:
            best[key] = (mt, j["summary"])
    # 聚合： (model, lv, src, tgt) -> {method: {metric: [per-seed values]}}
    acc = {}
    for (model, lv, src, tgt, seed), (_, summary) in sorted(best.items()):
        key = (model, lv, src, tgt)
        meths = extract_methods(model, summary)
        a = acc.setdefault(key, {"seeds": set(), "vals": {}})
        a["seeds"].add(seed)
        for meth, mm in meths.items():
            for m in METRICS:
                if mm.get(m) is not None:
                    a["vals"].setdefault((meth, m), []).append(float(mm[m]))
            if mm.get("n") is not None:
                a["vals"].setdefault((meth, "n"), []).append(float(mm["n"]))
    rows = []
    for (model, lv, src, tgt), a in sorted(acc.items()):
        dirn = f"{src} → {tgt}"
        for (meth, m), vals in sorted(a["vals"].items()):
            if m == "n":
                continue
            rows.append((lv, dirn, meth, m, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                         len(vals), int(round(np.mean(a["vals"].get((meth, "n"), [0]))))))
    lines = ["# 转移层结果汇总（T1 跨细胞系 / T2 跨研究 / T4 跨体系 CKI）", "",
             f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
             "- 口径：每条件先算再平均（mean-of-conds），再跨 seed 平均±std；n=每 seed 条件数（跨 seed 均值，四舍五入）。",
             "- 正典 run：同 (模型,层级,方向,seed) 取 metrics 最新；转移层 run 按 meta.level 与 T3 网格隔离。",
             "- T1 协议：源系全条件训练 → 目标系 ctrl 施加源机制；上下文细胞型=源系（无目标上下文，诚实标注）。",
             "- T2 协议：零公共条件全未见扰动跨研究转移（norman∩adamson 扰动基因宇宙=∅）；scGen/CPA 机制性 N/A。",
             "- T4 协议：跨体系 sc→bulk 演示；GEARS/scGen/CPA 机制性 N/A；签名检索见 run 目录 t4_signature_retrieval.csv。",
             "- random=目标真值置换基线；mean=目标 ctrl 直通；linear_transfer=源系全局平均 delta 线性迁移。", ""]
    for lv in ("T1", "T2", "T4"):
        sub = [r for r in rows if r[0] == lv]
        if not sub:
            continue
        lines += [f"## {lv}", "",
                  "| 方向 | 方法 | R²_deg (mean±std) | pearson_deg | Jaccard@20 | MSE_deg | seeds | n |",
                  "|---|---|---|---|---|---|---|---|"]
        seen = set()
        for _, dirn, meth, m, mu, sd, k, n in sub:
            if (dirn, meth) in seen:
                continue
            if m != "r2_deg":
                continue
            seen.add((dirn, meth))
        # 按方法展开一行多指标
        for dirn in sorted({r[1] for r in sub}):
            for meth in sorted({r[2] for r in sub if r[1] == dirn}):
                cells = {}
                seeds_k = 0
                n_mean = 0
                for _, d2, meth2, m2, mu, sd, k, n in sub:
                    if d2 == dirn and meth2 == meth:
                        cells[m2] = (mu, sd)
                        seeds_k = max(seeds_k, k)
                        n_mean = max(n_mean, n)
                if "r2_deg" not in cells:
                    continue

                def f(m2):
                    if m2 not in cells:
                        return "—"
                    mu, sd = cells[m2]
                    return f"{mu:.4f}±{sd:.3f}" if sd > 0 else f"{mu:.4f}"
                lines.append(f"| {dirn} | {meth} | {f('r2_deg')} | {f('pearson_deg')} | {f('jaccard_top20')} | {f('mse_deg')} | {seeds_k} | {n_mean} |")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {OUT} rows={len(rows)}")


if __name__ == "__main__":
    main()