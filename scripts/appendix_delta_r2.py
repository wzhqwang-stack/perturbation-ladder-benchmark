#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""appendix_delta_r2.py：delta 中心化 R² 附录（2026-11-02 审查行动 #2）。
问题：绝对轮廓 R² 在同细胞系条件间共享主变异轴 → random 置换基线高企（T1 0.48/0.64）。
检验：delta-R²（预测 delta vs 真实 delta，同 DE 基因集）下 random 是否塌缩、GEARS 是否保留。
覆盖：GEARS（preds.pkl 重放，T3×9 / T1×6 / T2×6）+ mean（零 delta）/ random（donor 真实 delta）解析；
linear 取自重跑基线 CSV（baselines.py 已加 delta_r2_deg）。
输出：docs/appendix_delta_r2.md + results/runs/…（附录 CSV 直写 docs 旁路）。
"""
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import metrics as M

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
RUNS = ROOT / "results" / "runs"
OUT_MD = ROOT / "docs" / "appendix_delta_r2.md"
OUT_CSV = ROOT / "docs" / "appendix_delta_r2.csv"
SEEDS = [149, 150, 151]
T3_DS = ["norman", "adamson", "replogle_k562_essential"]


def delta_pearson(pred_mean, true_mean, ctrl_mean, de_idx):
    """delta Pearson（偏移/尺度不变；GEARS/CPA 论文标准口径）。"""
    from scipy import stats as _st
    x = (np.asarray(pred_mean, dtype=np.float64) - np.asarray(ctrl_mean, dtype=np.float64))[de_idx]
    y = (np.asarray(true_mean, dtype=np.float64) - np.asarray(ctrl_mean, dtype=np.float64))[de_idx]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(_st.pearsonr(x, y)[0])


def delta_pearson2(pred_mean, true_mean, ctrl_pred, ctrl_true, de_idx):
    """控制拆分版 delta-Pearson：预测 delta 与真实 delta 分别对不同的控制子集取中心
    （消除 Nicol et al. 2026 指出的「复用同一控制群体」统计偏置）。"""
    from scipy import stats as _st
    x = (np.asarray(pred_mean, dtype=np.float64) - np.asarray(ctrl_pred, dtype=np.float64))[de_idx]
    y = (np.asarray(true_mean, dtype=np.float64) - np.asarray(ctrl_true, dtype=np.float64))[de_idx]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(_st.pearsonr(x, y)[0])


def delta_r2(pred_mean, true_mean, ctrl_mean, de_idx):
    x = (np.asarray(pred_mean, dtype=np.float64) - np.asarray(ctrl_mean, dtype=np.float64))[de_idx]
    y = (np.asarray(true_mean, dtype=np.float64) - np.asarray(ctrl_mean, dtype=np.float64))[de_idx]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((y - x) ** 2).sum()) / ss_tot


def load_contract(name, genes=None):
    import anndata as ad
    a = ad.read_h5ad(ROOT / f"data/processed/{name}/{name}.h5ad")
    if genes is not None:
        gp = {g: i for i, g in enumerate(a.var["gene_name"].astype(str))}
        cols = [gp[g] for g in genes if g in gp]
        keep = [g for g in genes if g in gp]
        X = a.X[:, cols]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return a, X, keep
    return a, a.X, list(a.var["gene_name"].astype(str))


def rows_of(X, idx):
    s = X[idx]
    return s.toarray() if hasattr(s, "toarray") else np.asarray(s)


def canon_run(prefix, level=None, seed=None, src=None, tgt=None):
    best = None
    for d in RUNS.iterdir():
        mj = d / "metrics.json"
        if not (d.is_dir() and mj.exists() and d.name.startswith(prefix)):
            continue
        try:
            j = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = j.get("meta", {})
        lv = meta.get("level")
        if level is not None and lv != level:
            continue
        if level is None and lv in ("T1", "T2", "T4"):
            continue  # T3 查找须排除转移层 run（目录名沿用 T3 式，meta.level 才是真元数据）
        if src is not None and str(meta.get("source")) != src:
            continue
        if tgt is not None and str(meta.get("target")) != tgt:
            continue
        if seed is not None:
            m = re.search(r"_s(\d+)_", d.name)
            if not m or int(m.group(1)) != seed:
                continue
        if best is None or mj.stat().st_mtime > best[0]:
            best = (mj.stat().st_mtime, d)
    return best[1] if best else None


def preds_lookup(pp):
    with open(pp, "rb") as f:
        preds = pickle.load(f)
    lut = {}
    for k, v in preds.items():
        if isinstance(k, str):
            parts = tuple(g for g in k.replace("_", "+").split("+") if g != "ctrl")
        elif isinstance(k, (list, tuple)):
            parts = tuple(g for g in k if g != "ctrl")
        else:
            parts = (str(k),)
        df = v
        mean_vec = None
        if hasattr(df, "columns") and "mean" in getattr(df, "columns", []):
            mean_vec = df["mean"]
            idx_names = [str(i) for i in df.index]
        else:
            arr = df.values if hasattr(df, "values") else np.asarray(df)
            mean_vec = arr.ravel()
            idx_names = None
        lut.setdefault("+".join(parts), (mean_vec, idx_names))
        lut.setdefault(parts, (mean_vec, idx_names))
        lut.setdefault(k, (mean_vec, idx_names))
    return lut


def get_pred(lut, genes):
    tg = tuple(genes)
    sg = tuple(sorted(genes))
    for probe in (tg, "+".join(tg), "_".join(tg), sg, "+".join(sg), "_".join(sg)):
        if probe in lut:
            return lut[probe]
    return None


def rows_from_pred(vec, idx_names, gene_index):
    """GEARS preds：DataFrame(index=基因名, col=mean) → 全基因空间向量。"""
    if idx_names is None:
        return np.asarray(vec, dtype=np.float64)
    out = np.zeros(len(gene_index), dtype=np.float64)
    gp = {g: i for i, g in enumerate(gene_index)}
    for n, v in zip(idx_names, np.asarray(vec, dtype=np.float64)):
        j = gp.get(str(n))
        if j is not None:
            out[j] = v
    return out


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rng_global = np.random.default_rng(SEEDS[0])
    rows = []

    # ---------- T3：GEARS + mean + random（统一协议） ----------
    for ds in T3_DS:
        a, X, gene_index = load_contract(ds)
        cond = a.obs["condition"].astype(str).to_numpy()
        for seed in SEEDS:
            run = canon_run(f"gears_{ds}_", seed=seed)
            if run is None or not (run / "preds.pkl").exists():
                continue
            with open(ROOT / f"data/processed/{ds}/splits/split_T3-simulation_s{seed}.json", encoding="utf-8") as f:
                split = json.load(f)
            test_conds = list(split["test"]) if isinstance(split.get("test"), list) else \
                [c for v in split.get("test", {}).values() for c in v]
            rng = np.random.default_rng(seed)
            ctrl_idx = np.where(cond == "ctrl")[0]
            ctrl_sub = rows_of(X, rng.choice(ctrl_idx, min(len(ctrl_idx), 2000), replace=False))
            ctrl_mean = ctrl_sub.mean(axis=0)
            _h = len(ctrl_sub) // 2
            ctrl_mean_a = ctrl_sub[:_h].mean(axis=0)
            ctrl_mean_b = ctrl_sub[_h:].mean(axis=0)
            lut = preds_lookup(run / "preds.pkl")
            donors = [c for c in test_conds if c in set(cond)]
            for c in test_conds:
                ci = np.where(cond == c)[0]
                if len(ci) < 3:
                    continue
                if len(ci) > 500:
                    ci = rng.choice(ci, 500, replace=False)
                rc = rows_of(X, ci)
                true_mean = rc.mean(axis=0)
                de = M.de_top_genes(ctrl_sub, rc, n_top=20)
                genes = [g for g in c.split("+") if g != "ctrl"]
                hit = get_pred(lut, genes)
                entry = {"level": "T3", "dataset": ds, "direction": ds, "seed": seed, "condition": c, "n_de": 20}
                entry["delta_pearson_GEARS"] = delta_pearson(
                    rows_from_pred(hit[0], hit[1], gene_index) if hit else np.full(len(gene_index), np.nan),
                    true_mean, ctrl_mean, de) if hit else np.nan
                entry["delta_r2_GEARS"] = delta_r2(
                    rows_from_pred(hit[0], hit[1], gene_index) if hit else np.full(len(gene_index), np.nan),
                    true_mean, ctrl_mean, de) if hit else np.nan
                entry["delta_r2_mean"] = delta_r2(np.zeros(len(gene_index)), true_mean, ctrl_mean, de)
                donor = donors[rng.permutation(len(donors))[0]] if donors else c
                di = np.where(cond == donor)[0]
                if len(di) > 500:
                    di = rng.choice(di, 500, replace=False)
                dmean = rows_of(X, di).mean(axis=0)
                entry["delta_r2_random"] = delta_r2(dmean, true_mean, ctrl_mean, de)
                entry["delta_pearson_random"] = delta_pearson(dmean, true_mean, ctrl_mean, de)
                if hit:
                    entry["delta_pearson_GEARS_split"] = delta_pearson2(
                        rows_from_pred(hit[0], hit[1], gene_index), true_mean, ctrl_mean_b, ctrl_mean_a, de)
                entry["delta_pearson_mean_split"] = delta_pearson2(ctrl_mean_b, true_mean, ctrl_mean_a, ctrl_mean_a, de)
                entry["delta_pearson_random_split"] = delta_pearson2(dmean, true_mean, ctrl_mean_b, ctrl_mean_a, de)
                rows.append(entry)
        print(f"T3 {ds} done", flush=True)

    # ---------- T1：GEARS + mean + random（公共基因空间） ----------
    for src, tgt in [("replogle_k562_essential", "replogle_rpe1_essential"),
                     ("replogle_rpe1_essential", "replogle_k562_essential")]:
        tgt_a, tX_full, tgt_genes = load_contract(tgt)
        t_cond = tgt_a.obs["condition"].astype(str).to_numpy()
        for seed in SEEDS:
            run = canon_run("gears_", level="T1", seed=seed, src=src, tgt=tgt)
            if run is None or not (run / "preds.pkl").exists():
                continue
            src_a, _, src_genes = load_contract(src, genes=tgt_genes)
            s_cond = src_a.obs["condition"].astype(str).to_numpy()
            common = sorted(set(src_genes) & set(tgt_genes))
            gp_t = {g: i for i, g in enumerate(tgt_genes)}
            gp_s = {g: i for i, g in enumerate(src_genes)}
            t_cols = [gp_t[g] for g in common]
            s_cols = [gp_s[g] for g in common]
            rng = np.random.default_rng(seed)
            t_ctrl_idx = np.where(t_cond == "ctrl")[0]
            t_ctrl_rows = rows_of(tX_full, rng.choice(t_ctrl_idx, min(len(t_ctrl_idx), 2000), replace=False))[:, t_cols]
            ctrl_mean_t = t_ctrl_rows.mean(axis=0)
            _h = len(t_ctrl_rows) // 2
            ctrl_mean_a = t_ctrl_rows[:_h].mean(axis=0)
            ctrl_mean_b = t_ctrl_rows[_h:].mean(axis=0)
            lut = preds_lookup(run / "preds.pkl")
            srcXc = src_a.X[:, s_cols]
            srcXc = srcXc.toarray() if hasattr(srcXc, "toarray") else np.asarray(srcXc)
            t_common_conds = sorted(set(t_cond) & set(s_cond) - {"ctrl"})
            donors = list(t_common_conds)
            for c in t_common_conds:
                ci = np.where(t_cond == c)[0]
                if len(ci) < 3:
                    continue
                if len(ci) > 500:
                    ci = rng.choice(ci, 500, replace=False)
                rc = rows_of(tX_full, ci)[:, t_cols]
                true_mean = rc.mean(axis=0)
                de = M.de_top_genes(t_ctrl_rows, rc, n_top=20)
                genes = [g for g in c.split("+") if g != "ctrl"]
                hit = get_pred(lut, genes)
                if hit is None:
                    continue
                pfull = rows_from_pred(hit[0], hit[1], src_genes)
                pmean = pfull[s_cols]
                entry = {"level": "T1", "dataset": f"{src}→{tgt}", "direction": f"{src.split('_')[1]}→{tgt.split('_')[1]}",
                         "seed": seed, "condition": c, "n_de": 20}
                entry["delta_r2_GEARS"] = delta_r2(pmean, true_mean, ctrl_mean_t, de)
                entry["delta_pearson_GEARS"] = delta_pearson(pmean, true_mean, ctrl_mean_t, de)
                entry["delta_r2_mean"] = delta_r2(np.zeros(len(common)), true_mean, ctrl_mean_t, de)
                donor = donors[rng.permutation(len(donors))[0]]
                di = np.where(t_cond == donor)[0]
                if len(di) > 500:
                    di = rng.choice(di, 500, replace=False)
                dmean = rows_of(tX_full, di)[:, t_cols].mean(axis=0)
                entry["delta_r2_random"] = delta_r2(dmean, true_mean, ctrl_mean_t, de)
                entry["delta_pearson_random"] = delta_pearson(dmean, true_mean, ctrl_mean_t, de)
                entry["delta_pearson_GEARS_split"] = delta_pearson2(pmean, true_mean, ctrl_mean_b, ctrl_mean_a, de)
                entry["delta_pearson_mean_split"] = delta_pearson2(ctrl_mean_b, true_mean, ctrl_mean_a, ctrl_mean_a, de)
                entry["delta_pearson_random_split"] = delta_pearson2(dmean, true_mean, ctrl_mean_b, ctrl_mean_a, de)
                rows.append(entry)
        print(f"T1 {src}→{tgt} done", flush=True)

    # ---------- T2：GEARS（可行子集） + mean + random ----------
    for src, tgt in [("norman", "adamson"), ("adamson", "norman")]:
        tgt_a, tX_full, tgt_genes = load_contract(tgt)
        t_cond = tgt_a.obs["condition"].astype(str).to_numpy()
        for seed in SEEDS:
            run = canon_run("gears_", level="T2", seed=seed, src=src, tgt=tgt)
            if run is None or not (run / "preds.pkl").exists():
                continue
            src_a, _, src_genes = load_contract(src, genes=tgt_genes)
            common = sorted(set(src_genes) & set(tgt_genes))
            gp_t = {g: i for i, g in enumerate(tgt_genes)}
            gp_s = {g: i for i, g in enumerate(src_genes)}
            t_cols = [gp_t[g] for g in common]
            s_cols = [gp_s[g] for g in common]
            rng = np.random.default_rng(seed)
            t_ctrl_idx = np.where(t_cond == "ctrl")[0]
            t_ctrl_rows = rows_of(tX_full, rng.choice(t_ctrl_idx, min(len(t_ctrl_idx), 2000), replace=False))[:, t_cols]
            ctrl_mean_t = t_ctrl_rows.mean(axis=0)
            _h = len(t_ctrl_rows) // 2
            ctrl_mean_a = t_ctrl_rows[:_h].mean(axis=0)
            ctrl_mean_b = t_ctrl_rows[_h:].mean(axis=0)
            lut = preds_lookup(run / "preds.pkl")
            test_conds = sorted(set(t_cond) - {"ctrl"})
            donors = list(test_conds)
            done = 0
            for c in test_conds:
                genes = [g for g in c.split("+") if g != "ctrl"]
                hit = get_pred(lut, genes)
                if hit is None:
                    continue
                ci = np.where(t_cond == c)[0]
                if len(ci) < 3:
                    continue
                if len(ci) > 500:
                    ci = rng.choice(ci, 500, replace=False)
                rc = rows_of(tX_full, ci)[:, t_cols]
                true_mean = rc.mean(axis=0)
                de = M.de_top_genes(t_ctrl_rows, rc, n_top=20)
                pfull = rows_from_pred(hit[0], hit[1], src_genes)
                pmean = pfull[s_cols]
                entry = {"level": "T2", "dataset": f"{src}→{tgt}", "direction": f"{src}→{tgt}",
                         "seed": seed, "condition": c, "n_de": 20}
                entry["delta_r2_GEARS"] = delta_r2(pmean, true_mean, ctrl_mean_t, de)
                entry["delta_pearson_GEARS"] = delta_pearson(pmean, true_mean, ctrl_mean_t, de)
                entry["delta_r2_mean"] = delta_r2(np.zeros(len(common)), true_mean, ctrl_mean_t, de)
                donor = donors[rng.permutation(len(donors))[0]]
                di = np.where(t_cond == donor)[0]
                if len(di) > 500:
                    di = rng.choice(di, 500, replace=False)
                dmean = rows_of(tX_full, di)[:, t_cols].mean(axis=0)
                entry["delta_r2_random"] = delta_r2(dmean, true_mean, ctrl_mean_t, de)
                entry["delta_pearson_random"] = delta_pearson(dmean, true_mean, ctrl_mean_t, de)
                entry["delta_pearson_GEARS_split"] = delta_pearson2(pmean, true_mean, ctrl_mean_b, ctrl_mean_a, de)
                entry["delta_pearson_mean_split"] = delta_pearson2(ctrl_mean_b, true_mean, ctrl_mean_a, ctrl_mean_a, de)
                entry["delta_pearson_random_split"] = delta_pearson2(dmean, true_mean, ctrl_mean_b, ctrl_mean_a, de)
                rows.append(entry)
                done += 1
            print(f"T2 {src}→{tgt} s{seed}: {done} conds", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    agg = df.groupby(["level", "direction"]).agg(
        GEARS_dp=("delta_pearson_GEARS", "mean"), GEARS_dp_std=("delta_pearson_GEARS", "std"),
        random_dp=("delta_pearson_random", "mean"), random_dp_std=("delta_pearson_random", "std"),
        GEARS_r2=("delta_r2_GEARS", "mean"),
        GEARS_dp_split=("delta_pearson_GEARS_split", "mean"),
        random_dp_split=("delta_pearson_random_split", "mean"),
        mean_dp_split=("delta_pearson_mean_split", "mean"),
        n=("delta_pearson_GEARS", "count")).reset_index()
    lines = ["# 附录：delta 中心化 R²（审查行动 #2）", "",
             f"- 生成：{datetime.now().isoformat(timespec='seconds')}；口径：pred delta vs true delta（同 top-20 DE 基因集，附录内统一协议重算，与正文 r2_deg 数值不可直接混排）。",
             "- 控制拆分列（*_split）：预测 delta 用控制子集 B、真实 delta 用子集 A 取中心，消除复用同一控制群体的统计偏置（Nicol, Shivakumar & Irizarry, bioRxiv 2026.05.07.723486）。",
             "- 检验逻辑（主口径 delta-Pearson，偏移/尺度不变）：若 random 在 delta 空间塌缩而 GEARS 保留 → 绝对 R² 的 random 高企确系天花板效应且模型有条件特异信号；若两者皆塌 → 天花板解释主导。delta-R²（无截距）受均值偏移支配，仅作参考。",
             "", "## 汇总（跨 seed/条件均值）", "",
             agg.round(4).to_markdown(index=False), "",
             "## 明细", "", df.round(4).to_markdown(index=False), ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("APPENDIX_DELTA_DONE", flush=True)


if __name__ == "__main__":
    main()