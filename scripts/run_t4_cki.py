#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_t4_cki.py：T4 CKI 锚点迁移（跨体系演示，go/no-go 信号）。
数据：GSE124715（HepG2+MDA-MB-231，CKI/5FU/VC/UT ×24h/48h，Entrez 行 ID→Symbol）
      GSE125933（H1975，CKI 1/2mg ±gefitinib，UT24 对照，符号直读）
协议（诚实性优先）：
  - GEARS/scGen/CPA 对全新药物扰动机制性 N/A（扰动词表/嵌入/delta 均无 CKI 概念），metrics.meta 记录；
  - 可评方法=mean_passthrough（ctrl 直通）与 linear_transfer（源系 CRISPR 全局平均 delta 施加于 bulk ctrl，跨模态 caveat 记录）；
  - 签名检索：CKI/5FU delta 与 K562 essential 全部 CRISPR 条件 delta 的余弦 top-K（论文③选型证据链）；
  - 5FU 臂作阳性对照（已知抗代谢药）。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
SEED = 149


def perm_null(Dk, d, rng, B=1000):
    """gene-label 置换零假设：打乱 query delta 的基因标签 B 次，取每次与库内最大余弦，
    返回 (观测 max cos, 零假设 max 均值, 95 分位, 经验 p=P(perm_max>=obs))。"""
    rn = np.linalg.norm(Dk, axis=1) + 1e-12
    Dkn = Dk / rn[:, None]
    obs = float(np.max((Dk @ d) / (rn * (np.linalg.norm(d) + 1e-12))))
    G = d.shape[0]
    mx = np.empty(B)
    for b in range(B):
        dp = d[rng.permutation(G)]
        mx[b] = np.max((Dkn @ dp) / (np.linalg.norm(dp) + 1e-12))
    p_emp = float((mx >= obs).mean())
    return obs, float(mx.mean()), float(np.percentile(mx, 95)), max(p_emp, 1.0 / (B + 1))


def cpm_log1p(counts):
    lib = counts.sum(axis=0)
    scale = 1e4 / np.where(lib <= 0, 1.0, lib)
    return np.log1p(counts * scale)


def k562_global_delta(common_genes):
    """K562 essential 各条件均值 delta 的全局平均（公共基因空间），稀疏逐条件。"""
    import anndata as ad
    a = ad.read_h5ad(ROOT / "data/processed/replogle_k562_essential/replogle_k562_essential.h5ad", backed="r")
    gpos = {g: i for i, g in enumerate(a.var["gene_name"].astype(str))}
    cols = [gpos[g] for g in common_genes if g in gpos]
    keep = [g for g in common_genes if g in gpos]
    cond = a.obs["condition"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    deltas = []
    for c in sorted(set(cond) - {"ctrl"}):
        idx = np.where(cond == c)[0]
        if len(idx) < 10:
            continue
        if len(idx) > 200:
            idx = rng.choice(idx, 200, replace=False)
        s = a.X[idx][:, cols]
        m = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
        deltas.append(m.mean(axis=0))
    a.file.close()
    ctrl_idx = np.where(cond == "ctrl")[0]
    if len(ctrl_idx) > 2000:
        ctrl_idx = rng.choice(ctrl_idx, 2000, replace=False)
    s = a.X[ctrl_idx][:, cols]
    m = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    deltas.append(m.mean(axis=0) - m.mean(axis=0))  # ctrl 的 delta=0 占位对齐
    D = np.stack(deltas[:-1]) - 0.0
    ctrl_mean = None
    # 重新精确计算：delta_c = mean_c - mean_ctrl
    a = ad.read_h5ad(ROOT / "data/processed/replogle_k562_essential/replogle_k562_essential.h5ad", backed="r")
    s = a.X[ctrl_idx][:, cols]
    m = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    ctrl_mean = m.mean(axis=0)
    a.file.close()
    deltas = []
    for c in sorted(set(cond) - {"ctrl"}):
        idx = np.where(cond == c)[0]
        if len(idx) < 10:
            continue
        if len(idx) > 200:
            idx = rng.choice(idx, 200, replace=False)
        s = a.file and None
        break
    # 第二遍被简化：直接用第一遍的均值缓存不可行（a 已关）——改为收集均值列表
    return None, keep


def k562_deltas(common_genes):
    """一次遍历：返回 (条件 delta 矩阵 [n_cond × G], 条件名列表, ctrl_mean)。"""
    import anndata as ad
    a = ad.read_h5ad(ROOT / "data/processed/replogle_k562_essential/replogle_k562_essential.h5ad", backed="r")
    gpos = {g: i for i, g in enumerate(a.var["gene_name"].astype(str))}
    cols = np.array([gpos[g] for g in common_genes if g in gpos], dtype=int)
    keep = [g for g in common_genes if g in gpos]
    cond = a.obs["condition"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    conds = sorted(set(cond) - {"ctrl"})
    means = np.zeros((len(conds) + 1, len(cols)), dtype=np.float64)
    for j, c in enumerate(conds):
        idx = np.where(cond == c)[0]
        if len(idx) > 200:
            idx = rng.choice(idx, 200, replace=False)
        s = a.X[idx][:, cols]
        m = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
        means[j] = m.mean(axis=0)
    ctrl_idx = np.where(cond == "ctrl")[0]
    if len(ctrl_idx) > 2000:
        ctrl_idx = rng.choice(ctrl_idx, 2000, replace=False)
    s = a.X[ctrl_idx][:, cols]
    m = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    means[-1] = m.mean(axis=0)
    a.file.close()
    D = means[:-1] - means[-1][None, :]
    return D, conds, means[-1], keep


def run_124715():
    import metrics as M
    import runlib
    run = runlib.begin(model="baselines", dataset="t4_gse124715", level="T4", seed=SEED,
                       task="论文② T4 CKI 锚点迁移 GSE124715（CPU）", hours=1, gpu=False,
                       inputs=[str(ROOT / "data/external/GSE124715/GSE124715_Hep_G2_refGenes_raw_counts.csv.gz"),
                               str(ROOT / "data/external/GSE124715/GSE124715_MDA-MB-231_refGenes_raw_counts.csv.gz"),
                               str(ROOT / "data/external/ncbi/Homo_sapiens.gene_info.gz")],
                       extra={"protocol": "cross-system transfer demo; models mechanism-N/A for unseen drug",
                              "source_panel": "replogle_k562_essential"})
    run.log("Entrez→Symbol 映射 ...")
    gi = pd.read_csv(ROOT / "data/external/ncbi/Homo_sapiens.gene_info.gz", sep="\t", compression="gzip",
                     usecols=["GeneID", "Symbol"]).drop_duplicates("GeneID")
    e2s = dict(zip(gi["GeneID"].astype(str), gi["Symbol"].astype(str)))
    mats = {}
    for f, tag in [("GSE124715_Hep_G2_refGenes_raw_counts.csv.gz", "HEPG2"),
                   ("GSE124715_MDA-MB-231_refGenes_raw_counts.csv.gz", "MDA")]:
        df = pd.read_csv(ROOT / f"data/external/GSE124715/{f}", index_col=0)
        syms = [e2s.get(str(i)) for i in df.index]
        ok = [s is not None for s in syms]
        df = df.loc[ok]
        df.index = [s for s in syms if s is not None]
        df = df.groupby(level=0).mean()
        mats[tag] = df
        run.log(f"{tag}: {df.shape[0]} 基因 × {df.shape[1]} 样本")
    run.log("CPM+log1p 归一化 ...")
    for tag in mats:
        mats[tag] = pd.DataFrame(cpm_log1p(mats[tag].to_numpy(dtype=np.float64)),
                                 index=mats[tag].index, columns=mats[tag].columns)
    # K562 CRISPR delta 库
    all_syms = set(mats["HEPG2"].index) | set(mats["MDA"].index)
    D, conds_k, ctrl_k, keep = k562_deltas(sorted(all_syms))
    sym_pos = {g: i for i, g in enumerate(keep)}
    global_delta = D.mean(axis=0)
    run.log(f"K562 delta 库：{len(conds_k)} 条件 × {len(keep)} 公共基因")

    recs, retr_rows, null_rows = [], [], []
    for tag in ["HEPG2", "MDA"]:
        df = mats[tag]
        common = [g for g in keep if g in df.index]
        pos = {g: i for i, g in enumerate(df.index)}
        cols = np.array([pos[g] for g in common], dtype=int)
        kcol = np.array([sym_pos[g] for g in common], dtype=int)
        X = df.to_numpy()[cols].T  # samples × common genes（log-CPM）
        samples = list(df.columns)
        def mean_of(pat):
            cs = [i for i, s in enumerate(samples) if s.startswith(pat)]
            return X[cs].mean(axis=0), len(cs)
        for time in ["24h", "48h"]:
            ctrl, n_c = mean_of(f"{time}-VC")
            for treat, label in [("CKI", "CKI"), ("5FU", "5FU")]:
                true, n_t = mean_of(f"{time}-{treat}")
                de = M.de_top_genes(X[[i for i, s in enumerate(samples) if s.startswith(f"{time}-VC")]],
                                    X[[i for i, s in enumerate(samples) if s.startswith(f"{time}-{treat}")]], n_top=20)
                preds = {"mean_passthrough": ctrl,
                         "linear_transfer": ctrl + global_delta[kcol]}
                for method, p in preds.items():
                    recs.append({"line": tag, "condition": f"{label}@{time}", "method": method,
                                 "pearson_deg": M.pearson_deg(p, true, de),
                                 "mse_deg": M.mse_top20(p, true, de),
                                 "r2_deg": M.r2_deg(p, true, de),
                                 "jaccard_top20": M.jaccard_topk(p - ctrl, true - ctrl, k=20)})
                # 检索行
                d = true - ctrl
                Dk = D[:, kcol]
                cos = (Dk @ d) / (np.linalg.norm(Dk, axis=1) * np.linalg.norm(d) + 1e-12)
                top = np.argsort(-np.abs(cos))[:10]
                for r, ti in enumerate(top):
                    retr_rows.append({"line": tag, "query": f"{label}@{time}", "rank": r + 1,
                                      "k562_condition": conds_k[ti], "cosine": float(cos[ti])})
                obs_m, nm, p95, pe = perm_null(Dk, d, np.random.default_rng(SEED * 7919 + len(null_rows)))
                null_rows.append({"line": tag, "query": f"{label}@{time}", "obs_max_cos": obs_m,
                                  "null_max_mean": nm, "null_max_p95": p95, "p_emp": pe, "B": 1000})
                if treat == "CKI":
                    fu, _ = mean_of(f"{time}-5FU")
                    cp = float(np.corrcoef(d, fu - ctrl)[0, 1])
                    run.log(f"{tag} {time}: CKI vs 5FU delta pearson={cp:.3f}")
    df_r = pd.DataFrame(retr_rows)
    df_r.to_csv(run.dir / "t4_signature_retrieval.csv", index=False)
    pd.DataFrame(null_rows).to_csv(run.dir / "t4_retrieval_null.csv", index=False)
    df_c = pd.DataFrame(recs)
    df_c.to_csv(run.dir / "per_condition_t4.csv", index=False)
    summary = {}
    for m, g in df_c.groupby("method"):
        num = g.select_dtypes(include=[float])
        summary[m] = {k: {"mean": float(num[k].mean()), "std": float(num[k].std(ddof=1))} for k in num.columns}
        summary[m]["n_conditions"] = int(len(g))
    top_cki = df_r[df_r["query"].str.startswith("CKI")].nsmallest(10, "rank")
    null_cki = [r for r in null_rows if str(r["query"]).startswith("CKI")]
    run.write_metrics({
        "meta": {"level": "T4", "gse": "GSE124715", "seed": SEED,
                 "na_note": "GEARS/scGen/CPA mechanism-N/A for unseen drug perturbation (no gene/embedding/delta for CKI)",
                 "caveat": "cross-modality sc->bulk, log-CPM both sides",
                 "retrieval_top5_CKI": top_cki[["line", "query", "k562_condition", "cosine"]].to_dict("records"),
                 "retrieval_null_CKI": null_cki,
                 "paper_anchor": "Cui & Adelson PLoS One 2020, PMID 32730293: CKI induces cell-cycle arrest & apoptosis",
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("T4 GSE124715 完成：" + json.dumps({m: {"r2": round(s["r2_deg"]["mean"], 4)} for m, s in summary.items()}, ensure_ascii=False))
    print("T4_124715_DONE", flush=True)


def run_125933():
    import metrics as M
    import runlib
    run = runlib.begin(model="baselines", dataset="t4_gse125933", level="T4", seed=SEED,
                       task="论文② T4 CKI 锚点迁移 GSE125933（CPU）", hours=1, gpu=False,
                       inputs=[str(ROOT / "data/external/GSE125933/GSE125933_refGenes_raw_count.txt.gz"),
                               str(ROOT / "data/external/ncbi/Homo_sapiens.gene_info.gz")],
                       extra={"protocol": "cross-system transfer demo; YFSJ arm excluded per registry",
                              "source_panel": "replogle_k562_essential"})
    run.log("读取计数矩阵（符号直读）...")
    df = pd.read_csv(ROOT / "data/external/GSE125933/GSE125933_refGenes_raw_count.txt.gz", sep="\t", index_col=0)
    df = df.groupby(level=0).mean()
    run.log(f"矩阵 {df.shape}")
    df = pd.DataFrame(cpm_log1p(df.to_numpy(dtype=np.float64)), index=df.index, columns=df.columns)
    all_syms = set(df.index)
    D, conds_k, ctrl_k, keep = k562_deltas(sorted(all_syms))
    sym_pos = {g: i for i, g in enumerate(keep)}
    global_delta = D.mean(axis=0)
    run.log(f"K562 delta 库：{len(conds_k)} 条件 × {len(keep)} 公共基因")
    common = [g for g in keep if g in df.index]
    pos = {g: i for i, g in enumerate(df.index)}
    cols = np.array([pos[g] for g in common], dtype=int)
    kcol = np.array([sym_pos[g] for g in common], dtype=int)
    X = df.to_numpy()[cols].T
    samples = list(df.columns)
    def mean_of(pat):
        cs = [i for i, s in enumerate(samples) if s.startswith(pat + "_") or s == pat]
        return X[cs].mean(axis=0), len(cs)
    recs, retr_rows, null_rows = [], [], []
    ctrl, n_c = mean_of("UT24")
    ctrl_rows = X[[i for i, s in enumerate(samples) if s.startswith("UT24")]]
    for label, pat, ctrl_pat in [("CKI_1mg", "C1", None), ("CKI_2mg", "C2", None),
                                 ("CKI_1mg+G50", "C1_G50", "G50"), ("CKI_1mg+G75", "C1_G75", "G75"),
                                 ("CKI_2mg+G50", "C2_G50", "G50"), ("CKI_2mg+G75", "C2_G75", "G75")]:
        true, n_t = mean_of(pat)
        rows_true = X[[i for i, s in enumerate(samples) if s.startswith(pat + "_")]]
        base_ctrl = ctrl if ctrl_pat is None else mean_of(ctrl_pat)[0]
        base_rows = ctrl_rows if ctrl_pat is None else X[[i for i, s in enumerate(samples) if s.startswith(ctrl_pat + "_")]]
        de = M.de_top_genes(base_rows, rows_true, n_top=20)
        preds = {"mean_passthrough": base_ctrl, "linear_transfer": base_ctrl + global_delta[kcol]}
        for method, p in preds.items():
            recs.append({"line": "H1975", "condition": label, "method": method,
                         "pearson_deg": M.pearson_deg(p, true, de),
                         "mse_deg": M.mse_top20(p, true, de),
                         "r2_deg": M.r2_deg(p, true, de),
                         "jaccard_top20": M.jaccard_topk(p - base_ctrl, true - base_ctrl, k=20)})
        d = true - base_ctrl
        Dk = D[:, kcol]
        cos = (Dk @ d) / (np.linalg.norm(Dk, axis=1) * np.linalg.norm(d) + 1e-12)
        obs_m, nm, p95, pe = perm_null(Dk, d, np.random.default_rng(SEED * 7919 + len(null_rows)))
        null_rows.append({"line": "H1975", "query": label, "obs_max_cos": obs_m,
                          "null_max_mean": nm, "null_max_p95": p95, "p_emp": pe, "B": 1000})
        top = np.argsort(-np.abs(cos))[:10]
        for r, ti in enumerate(top):
            retr_rows.append({"line": "H1975", "query": label, "rank": r + 1,
                              "k562_condition": conds_k[ti], "cosine": float(cos[ti])})
    df_r = pd.DataFrame(retr_rows)
    df_r.to_csv(run.dir / "t4_signature_retrieval.csv", index=False)
    pd.DataFrame(null_rows).to_csv(run.dir / "t4_retrieval_null.csv", index=False)
    df_c = pd.DataFrame(recs)
    df_c.to_csv(run.dir / "per_condition_t4.csv", index=False)
    summary = {}
    for m, g in df_c.groupby("method"):
        num = g.select_dtypes(include=[float])
        summary[m] = {k: {"mean": float(num[k].mean()), "std": float(num[k].std(ddof=1))} for k in num.columns}
        summary[m]["n_conditions"] = int(len(g))
    top_cki = df_r[df_r["query"].str.startswith("CKI")].nsmallest(10, "rank")
    null_cki = [r for r in null_rows if str(r["query"]).startswith("CKI")]
    run.write_metrics({
        "meta": {"level": "T4", "gse": "GSE125933", "seed": SEED,
                 "na_note": "GEARS/scGen/CPA mechanism-N/A for unseen drug perturbation",
                 "caveat": "cross-modality sc->bulk; no vehicle arm, UT24 as control",
                 "retrieval_top5_CKI": top_cki[["line", "query", "k562_condition", "cosine"]].to_dict("records"),
                 "retrieval_null_CKI": null_cki,
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("T4 GSE125933 完成：" + json.dumps({m: {"r2": round(s["r2_deg"]["mean"], 4)} for m, s in summary.items()}, ensure_ascii=False))
    print("T4_125933_DONE", flush=True)


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run_124715()
    run_125933()


if __name__ == "__main__":
    main()