#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figures.py：P1-P6 图表起草（英文图签，300dpi，PNG+PDF）。
数据源：results/runs/*/metrics.json（同键取最新为正典）+ per_condition CSV。
输出：results/figures/P*.png/.pdf + figures_index.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
RUNS = ROOT / "results" / "runs"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

C = {"GEARS": "#4C72B0", "scGen": "#DD8452", "CPA": "#55A868",
     "Linear": "#C44E52", "Mean": "#8172B3", "Random": "#937860"}
ORDER = ["GEARS", "scGen", "CPA", "Linear", "Mean", "Random"]
T3_DS = ["norman", "adamson", "replogle_k562_essential", "replogle_rpe1_essential"]


def canon(pred):
    """返回满足 pred(dir_name, meta) 的 metrics.json 最新 run。"""
    best = None
    for d in RUNS.iterdir():
        mj = d / "metrics.json"
        if not (d.is_dir() and mj.exists()):
            continue
        try:
            j = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = j.get("meta", {})
        if pred(d.name, meta):
            if best is None or mj.stat().st_mtime > best[0]:
                best = (mj.stat().st_mtime, j, d)
    return best


def metric_keyed(summary):
    return "r2_deg" in summary


def get_val(summary, metric):
    v = summary.get(metric)
    return v.get("mean") if isinstance(v, dict) else None


def load_t3():
    """(dataset, model, seed) -> summary。model in gears/scgen/baselines。"""
    out = {}
    for d in RUNS.iterdir():
        mj = d / "metrics.json"
        if not (d.is_dir() and mj.exists()):
            continue
        name = d.name
        if "_T3-simulation_" not in name and "_T3-smoke_" not in name:
            continue  # aggregate.py 双前缀惯例：norman/adamson 基线 run 用 T3-smoke 命名
        model = name.split("_")[0]
        if model not in ("gears", "scgen", "baselines"):
            continue
        import re as _re
        _m = _re.search(r"_s(\d+)_", name)
        if not _m:
            continue
        seed = int(_m.group(1))
        ds = next((x for x in T3_DS if x in name), None)
        if ds is None:
            continue
        try:
            j = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j.get("meta", {}).get("level") in ("T1", "T2"):
            continue  # 转移层 run 沿用 T3 式目录名，按 meta.level 排除
        key = (ds, model, seed)
        mt = mj.stat().st_mtime
        if key not in out or mt > out[key][0]:
            out[key] = (mt, j["summary"])
    return {k: v[1] for k, v in out.items()}


def t3_cell(t3, ds, method):
    """T3 单格：跨 seed 均值。method 映射到 run 模型与 summary 键。"""
    seeds = sorted(s for (d2, m2, s) in [(a, b, c) for a, b, c in t3] if False) if False else None
    vals = []
    for (d2, m2, s), summ in t3.items():
        if d2 != ds:
            continue
        if method in ("GEARS",) and m2 == "gears":
            v = get_val(summ, "r2_deg")
        elif method == "scGen" and m2 == "scgen":
            v = get_val(summ, "r2_deg")
        elif method == "Linear" and m2 == "baselines":
            v = get_val(summ.get("linear", {}), "r2_deg") if not metric_keyed(summ) else get_val(summ, "r2_deg")
        elif method == "Mean" and m2 == "baselines":
            v = get_val(summ.get("mean_passthrough", {}), "r2_deg") if not metric_keyed(summ) else None
        elif method == "Random" and m2 == "baselines":
            v = get_val(summ.get("random", {}), "r2_deg") if not metric_keyed(summ) else None
        else:
            continue
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan, len(vals)


def load_transfer(level):
    """多种子版：key=(model,src,tgt) -> [(seed, summary)...]（同 seed 取最新 run）。"""
    import re as _re
    raw = {}
    for d in RUNS.iterdir():
        mj = d / "metrics.json"
        if not (d.is_dir() and mj.exists()):
            continue
        try:
            j = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = j.get("meta", {})
        if meta.get("level") != level:
            continue
        model = d.name.split("_")[0]
        _m = _re.search(r"_s(\d+)_", d.name)
        seed = int(_m.group(1)) if _m else 0
        key = (model, str(meta.get("source")), str(meta.get("target")))
        mt = mj.stat().st_mtime
        pk = (key, seed)
        if pk not in raw or mt > raw[pk][0]:
            raw[pk] = (mt, seed, j["summary"])
    out = {}
    for (key, seed), (_, sd, summ) in raw.items():
        out.setdefault(key, []).append((seed, summ))
    return {k: sorted(v) for k, v in out.items()}


def xfer_val(recs, model, src, tgt, method, metric):
    """跨 seed 均值（recs=load_transfer 输出）。gears/cpa=指标键；scgen/baselines=方法键。"""
    lst = recs.get((model, src, tgt)) if recs else None
    if not lst:
        return None
    vals = []
    for _seed, summ in lst:
        if model in ("gears", "cpa"):
            v = summ.get(metric, {}).get("mean") if isinstance(summ.get(metric), dict) else None
        else:
            v = summ.get(method, {}).get(metric, {}).get("mean") if isinstance(summ.get(method), dict) else None
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def norm_name(x, keep=True):
    return x if keep else x


def build_p1(t3, t1, t2, tahoe):
    rows = []
    for ds, label in [("norman", "T3 norman"), ("adamson", "T3 adamson"),
                      ("replogle_k562_essential", "T3 K562"), ("replogle_rpe1_essential", "T3 RPE1")]:
        vals = {}
        for m in ORDER:
            vals[m] = t3_cell(t3, ds, m)[0] if m != "CPA" else np.nan
        rows.append((label, vals, "T3"))
    # T3-dose (Tahoe)
    b = tahoe["base_sum"]
    rows.append(("T3-dose ext (Tahoe)", {
        "GEARS": np.nan, "scGen": np.nan, "CPA": tahoe["cpa_ext"],
        "Linear": get_val(b.get("linear", {}), "r2_deg"),
        "Mean": get_val(b.get("mean_passthrough", {}), "r2_deg"),
        "Random": get_val(b.get("random", {}), "r2_deg")}, "T3"))
    rows.append(("T3-dose ext (CPA counts)", {
        "GEARS": np.nan, "scGen": np.nan, "CPA": tahoe["cpa_counts"],
        "Linear": np.nan, "Mean": np.nan, "Random": np.nan}, "T3"))
    rows.append(("T3-dose interp (Tahoe)", {
        "GEARS": np.nan, "scGen": np.nan, "CPA": tahoe["cpa_int"],
        "Linear": np.nan, "Mean": np.nan, "Random": np.nan}, "T3"))
    # T1（多种子均值）
    for src, tgt, lab in [("replogle_k562_essential", "replogle_rpe1_essential", "T1 K→R"),
                          ("replogle_rpe1_essential", "replogle_k562_essential", "T1 R→K")]:
        vals = {
            "GEARS": xfer_val(t1, "gears", src, tgt, "gears", "r2_deg"),
            "scGen": xfer_val(t1, "scgen", src, tgt, "scgen", "r2_deg"),
            "CPA": xfer_val(t1, "cpa", src, tgt, "cpa", "r2_deg"),
            "Linear": xfer_val(t1, "scgen", src, tgt, "linear_transfer", "r2_deg"),
            "Mean": xfer_val(t1, "scgen", src, tgt, "mean_passthrough", "r2_deg"),
            "Random": xfer_val(t1, "scgen", src, tgt, "random", "r2_deg"),
        }
        rows.append((lab, vals, "T1"))
    # T2（多种子均值）
    for src, tgt, lab in [("norman", "adamson", "T2 N→A"),
                          ("adamson", "norman", "T2 A→N")]:
        vals = {"GEARS": xfer_val(t2, "gears", src, tgt, "gears", "r2_deg"),
                "scGen": np.nan, "CPA": np.nan,
                "Linear": xfer_val(t2, "baselines", src, tgt, "linear_transfer", "r2_deg"),
                "Mean": xfer_val(t2, "baselines", src, tgt, "mean_passthrough", "r2_deg"),
                "Random": xfer_val(t2, "baselines", src, tgt, "random", "r2_deg")}
        rows.append((lab, vals, "T2"))
    return rows


def p1(rows):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    mat = np.full((len(rows), len(ORDER)), np.nan)
    for i, (_, vals, _) in enumerate(rows):
        for j, m in enumerate(ORDER):
            v = vals.get(m)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                mat[i, j] = v
    cmap = LinearSegmentedColormap.from_list("rb", ["#B2182B", "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"])
    cmap.set_bad("#E8E8E8")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7.5, color="#888888")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.55 else "black")
    ax.set_xticks(range(len(ORDER)), ORDER, fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    for i, (_, vals, lv) in enumerate(rows):
        if lv in ("T1", "T2"):
            ax.add_patch(Rectangle((-0.5, i - 0.5), len(ORDER), 1, fill=False, edgecolor="#444444", lw=1.0))
    ax.set_title("P1  R$^2_{DE}$ across generalization levels (mean over seeds)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("R$^2_{DE}$", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"P1_campaign_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def p2(t3, t1, t2):
    methods = ["GEARS", "scGen", "CPA", "Linear", "Mean", "Random"]
    levels = ["T3\n(same-study,\nunseen combos)", "T1\n(cross\n-cell-line)", "T2\n(cross\n-study)"]
    def t3_mean(m):
        vs = [t3_cell(t3, ds, m)[0] for ds in T3_DS[:3]]
        vs = [v for v in vs if not np.isnan(v)]
        return (np.mean(vs), (np.min(vs), np.max(vs))) if vs else (np.nan, None)
    def t1_mean(m):
        vals = []
        for src, tgt in [("replogle_k562_essential", "replogle_rpe1_essential"),
                         ("replogle_rpe1_essential", "replogle_k562_essential")]:
            if m == "GEARS":
                v = xfer_val(t1, "gears", src, tgt, "gears", "r2_deg")
            elif m == "CPA":
                v = xfer_val(t1, "cpa", src, tgt, "cpa", "r2_deg")
            elif m == "scGen":
                v = xfer_val(t1, "scgen", src, tgt, "scgen", "r2_deg")
            else:
                b = {"Linear": "linear_transfer", "Mean": "mean_passthrough", "Random": "random"}[m]
                v = xfer_val(t1, "scgen", src, tgt, b, "r2_deg")
            if v is not None:
                vals.append(v)
        return (np.mean(vals), (np.min(vals), np.max(vals))) if vals else (np.nan, None)
    def t2_mean(m):
        vals = []
        for src, tgt in [("norman", "adamson"), ("adamson", "norman")]:
            if m == "GEARS":
                v = xfer_val(t2, "gears", src, tgt, "gears", "r2_deg")
            else:
                b = {"Linear": "linear_transfer", "Mean": "mean_passthrough", "Random": "random"}.get(m)
                v = xfer_val(t2, "baselines", src, tgt, b, "r2_deg") if b else None
            if v is not None:
                vals.append(v)
        return (np.mean(vals), (np.min(vals), np.max(vals))) if vals else (np.nan, None)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(3)
    for m in methods:
        pts, bands = [], []
        for fn in (t3_mean, t1_mean, t2_mean):
            mu, band = fn(m)
            pts.append(mu)
            bands.append(band)
        xs = [i for i, v in enumerate(pts) if not np.isnan(v)]
        ys = [pts[i] for i in xs]
        ax.plot(xs, ys, "-o", color=C[m], label=m, ms=4.5, lw=1.6)
        for i, b in zip(xs, bands):
            if b:
                ax.plot([i, i], [b[0], b[1]], color=C[m], lw=0.8, alpha=0.5)
    ax.axhline(0, color="#666666", lw=0.8, ls="--")
    ax.set_xticks(x, levels, fontsize=8)
    ax.set_ylabel("mean R$^2_{DE}$", fontsize=9)
    ax.set_title("P2  Method advantage decays with generalization distance", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"P2_level_decay.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def p3(t3):
    from scipy.stats import wilcoxon
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=False)
    for ax, ds, lab in zip(axes, ["norman", "adamson", "replogle_k562_essential"],
                           ["norman", "adamson", "replogle-K562"]):
        def _t3run(n, m, d=ds):
            return (n.startswith("gears_") or n.startswith("baselines_")) and d in n and "s149" in n and m.get("level") not in ("T1", "T2")
        g = canon(lambda n, m, d=ds: n.startswith("gears_") and d in n and "_T3-simulation_s149_" in n and m.get("level") not in ("T1", "T2"))
        b = canon(lambda n, m, d=ds: n.startswith("baselines_") and d in n and "s149" in n and m.get("level") not in ("T1", "T2"))
        if g is None or b is None:
            ax.set_title(lab + " (missing)", fontsize=9)
            continue
        gd = pd.read_csv(g[2] / "per_condition_gears.csv")
        bd = pd.read_csv(b[2] / "per_condition_linear.csv")
        mrg = gd[["condition", "r2_deg"]].merge(bd[["condition", "r2_deg"]], on="condition", suffixes=("_g", "_l"))
        diff = mrg["r2_deg_g"] - mrg["r2_deg_l"]
        ax.hist(diff, bins=24, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="#333333", lw=1.0)
        med = float(diff.median())
        ax.axvline(med, color="#C44E52", lw=1.4)
        try:
            p = wilcoxon(mrg["r2_deg_g"], mrg["r2_deg_l"]).pvalue
        except Exception:
            p = np.nan
        wr = float((diff > 0).mean())
        ax.set_title(f"{lab}  n={len(mrg)}\nmedian Δ={med:+.3f}  win={wr:.0%}  p={p:.1e}", fontsize=8.5)
        ax.set_xlabel("ΔR$^2_{DE}$ (GEARS − Linear)", fontsize=8.5)
    axes[0].set_ylabel("conditions", fontsize=8.5)
    fig.suptitle("P3  GEARS vs Linear, paired per condition (seed 149)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"P3_gears_vs_linear.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def p4(tahoe):
    items = [("CPA ext\n(log)", tahoe["cpa_ext"], tahoe["jac_ext"], C["CPA"]),
             ("CPA ext\n(counts)", tahoe["cpa_counts"], tahoe["jac_counts"], "#7FBF7F"),
             ("CPA interp\n(log)", tahoe["cpa_int"], tahoe["jac_int"], "#A8D5A8"),
             ("Linear", tahoe["lin"], get_val(tahoe["base_sum"].get("linear", {}), "jaccard_top20"), C["Linear"]),
             ("Mean", tahoe["mean"], get_val(tahoe["base_sum"].get("mean_passthrough", {}), "jaccard_top20"), C["Mean"]),
             ("Random", tahoe["rand"], get_val(tahoe["base_sum"].get("random", {}), "jaccard_top20"), C["Random"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, idx, title in zip(axes, (1, 2), ("R$^2_{DE}$", "Jaccard@20")):
        names = [i[0] for i in items]
        vals = [np.nan if i[idx] is None else i[idx] for i in items]
        cols = [i[3] for i in items]
        ax.bar(range(len(items)), vals, color=cols, edgecolor="white")
        ax.set_xticks(range(len(items)), names, fontsize=7.5)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(title, fontsize=9.5)
        for k, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(k, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    axes[0].set_ylabel("Tahoe T3-dose", fontsize=9)
    fig.suptitle("P4  Dose dimension (Tahoe CRC, 1,135 drug×cell-line pairs; seed 149)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"P4_dose_dimension.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def p5(t1):
    dirs = [("replogle_k562_essential", "replogle_rpe1_essential", "K→R"),
            ("replogle_rpe1_essential", "replogle_k562_essential", "R→K")]
    methods = ["GEARS", "scGen", "CPA", "Linear", "Mean", "Random"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, metric, title in zip(axes, ("r2_deg", "jaccard_top20"), ("R$^2_{DE}$", "Jaccard@20")):
        w = 0.13
        for k, m in enumerate(methods):
            vals = []
            for src, tgt, _ in dirs:
                if m == "GEARS":
                    v = xfer_val(t1, "gears", src, tgt, "gears", metric)
                elif m == "CPA":
                    v = xfer_val(t1, "cpa", src, tgt, "cpa", metric)
                elif m == "scGen":
                    v = xfer_val(t1, "scgen", src, tgt, "scgen", metric)
                else:
                    b = {"Linear": "linear_transfer", "Mean": "mean_passthrough", "Random": "random"}[m]
                    v = xfer_val(t1, "scgen", src, tgt, b, metric)
                vals.append(np.nan if v is None else v)
            xs = np.arange(len(dirs)) + (k - 2.5) * w
            ax.bar(xs, vals, width=w, color=C[m], label=m, edgecolor="white")
        ax.set_xticks(range(len(dirs)), [d[2] for d in dirs], fontsize=9)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("T1 cross-cell-line", fontsize=9)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("P5  T1 transfer detail (848 common conditions, seed 149)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"P5_t1_detail.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def p6(t2):
    from pathlib import Path as _P
    import re as _re

    def canon_t2_all(model, src, tgt):
        """[(seed, dir)...] 每 seed 最新。"""
        raw = {}
        for d in RUNS.iterdir():
            mj = d / "metrics.json"
            if not (d.is_dir() and mj.exists()):
                continue
            try:
                j = json.loads(mj.read_text(encoding="utf-8"))
            except Exception:
                continue
            m = j.get("meta", {})
            if m.get("level") == "T2" and d.name.split("_")[0] == model and str(m.get("source")) == src and str(m.get("target")) == tgt:
                _m = _re.search(r"_s(\d+)_", d.name)
                seed = int(_m.group(1)) if _m else 0
                if (model, seed) not in raw or mj.stat().st_mtime > raw[(model, seed)][0]:
                    raw[(model, seed)] = (mj.stat().st_mtime, seed, d)
        return [(sd, dd) for _, sd, dd in sorted(raw.values())]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    methods = ["GEARS", "Linear", "Mean", "Random"]
    for ax, metric, title in zip(axes, ("r2_deg", "jaccard_top20"), ("R$^2_{DE}$", "Jaccard@20")):
        for di, (src, tgt, lab) in enumerate([("norman", "adamson", "N→A"),
                                              ("adamson", "norman", "A→N")]):
            gs = canon_t2_all("gears", src, tgt)
            bs = canon_t2_all("baselines", src, tgt)

            def _seed_matched(metric=metric):
                per_seed = []
                for sd, gd in gs:
                    bd = [dd for s2, dd in bs if s2 == sd]
                    if not bd:
                        continue
                    gdf = pd.read_csv(gd / "per_condition_gears_t1.csv")
                    bdf = pd.read_csv(bd[0] / "per_condition_baselines_t2.csv")
                    conds = set(gdf["condition"])
                    bsub = bdf[bdf["condition"].isin(conds)]
                    per_seed.append({
                        "GEARS": gdf[metric].mean(),
                        "Linear": bsub[bsub["method"] == "linear_transfer"][metric].mean(),
                        "Mean": bsub[bsub["method"] == "mean_passthrough"][metric].mean(),
                        "Random": bsub[bsub["method"] == "random"][metric].mean()})
                return pd.DataFrame(per_seed).mean() if per_seed else {}
            ms = _seed_matched()
            w = 0.19
            items = [("GEARS", ms.get("GEARS"), C["GEARS"]),
                     ("Linear", ms.get("Linear"), C["Linear"]),
                     ("Mean", ms.get("Mean"), C["Mean"]),
                     ("Random", ms.get("Random"), C["Random"])]
            for k, (m, v, col) in enumerate(items):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                x = di + (k - 1.5) * w
                ax.bar(x, v, width=w, color=col, label=m if di == 0 else None, edgecolor="white")
                ax.text(x, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        ax.set_xticks(range(2), ["norman→adamson", "adamson→norman"], fontsize=8.5)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("T2 cross-study", fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("P6  T2 cross-study, matched condition sets (GEARS-feasible subset)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"P6_t2_detail.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t3 = load_t3()
    t1 = load_transfer("T1")
    t2 = load_transfer("T2")

    def tahoe_latest(kind):
        best = None
        for d in RUNS.iterdir():
            mj = d / "metrics.json"
            ok = (("T3-dose" in d.name and kind == "ext" and d.name.startswith("cpa_tahoe") and "interp" not in d.name and "counts" not in d.name)
                  or ("T3-dose-interp" in d.name and kind == "int")
                  or ("T3-dose-counts" in d.name and kind == "counts")
                  or (kind == "baselines" and d.name.startswith("baselines_tahoe")))
            if not (d.is_dir() and mj.exists() and ok):
                continue
            if best is None or mj.stat().st_mtime > best[0]:
                best = (mj.stat().st_mtime, j := json.loads(mj.read_text(encoding="utf-8")))
        return best[1]["summary"]
    s_ext = tahoe_latest("ext")
    s_int = tahoe_latest("int")
    s_cnt = tahoe_latest("counts")
    s_base = tahoe_latest("baselines")
    tahoe = {"cpa_ext": get_val(s_ext, "r2_deg"), "cpa_int": get_val(s_int, "r2_deg"),
             "cpa_counts": get_val(s_cnt, "r2_deg"),
             "jac_ext": get_val(s_ext, "jaccard_top20"), "jac_int": get_val(s_int, "jaccard_top20"),
             "jac_counts": get_val(s_cnt, "jaccard_top20"),
             "lin": get_val(s_base.get("linear", {}), "r2_deg"),
             "mean": get_val(s_base.get("mean_passthrough", {}), "r2_deg"),
             "rand": get_val(s_base.get("random", {}), "r2_deg"),
             "jac_lin": get_val(s_base, "jaccard_top20") if not metric_keyed(s_base) else None,
             "jac_rand": get_val(s_base.get("random", {}), "jaccard_top20"),
             "base_sum": s_base}

    rows = build_p1(t3, t1, t2, tahoe)
    p1(rows)
    p2(t3, t1, t2)
    p3(t3)
    p4(tahoe)
    p5(t1)
    p6(t2)
    print("FIGURES DONE:", sorted(f.name for f in FIGS.glob("P*.png")))


if __name__ == "__main__":
    main()