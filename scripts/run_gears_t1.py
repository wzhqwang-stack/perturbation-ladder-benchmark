#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_gears_t1.py：GEARS T1 跨细胞系尝试（复用源系已训模型 → 预测目标系条件 → 公共基因空间指标）。
用法: python run_gears.py --dataset norman --seed 149 [--epochs 3] [--device cuda] [--max-test 0]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=149)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--from-preds", default=None, help="跳过训练，直接用已存 preds.pkl")
    ap.add_argument("--from-model", default=None, help="跳过训练，加载已存 gears_model 目录")
    ap.add_argument("--target", default=None, help="T1：目标系契约名（预测其公共条件）")
    ap.add_argument("--level", default="T1", help="转移层级标注（T1/T2）")
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    import anndata as ad
    import torch
    from gears import PertData, GEARS

    import metrics as M
    import runlib

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    split_json = ROOT / "data" / "processed" / a.dataset / "splits" / f"split_T3-simulation_s{a.seed}.json"
    contract_h5ad = ROOT / "data" / "processed" / a.dataset / f"{a.dataset}.h5ad"
    run = runlib.begin(model="gears", dataset=a.dataset, level="T3-simulation", seed=a.seed,
                       task=f"论文② GEARS {a.dataset} T3 s{a.seed} 网格格子",
                       hours=2, gpu=(device == "cuda"),
                       inputs=[str(contract_h5ad), str(split_json)],
                       extra={"device": device, "epochs": a.epochs})
    run.log(f"PertData 加载 {a.dataset}（缓存）...")
    pd_ = PertData(str(ROOT / "data" / "raw" / "gears"))
    pd_.load(data_name=a.dataset)
    pd_.prepare_split(split="simulation", seed=a.seed)
    pd_.get_dataloader(batch_size=64, test_batch_size=128)

    model = GEARS(pd_, device=device)
    if a.from_model:
        model.model_initialize(hidden_size=64)
        model.load_pretrained(a.from_model)
        run.log(f"已加载模型 {a.from_model}")
    elif a.from_preds:
        import pickle as _pkl
        run.log(f"跳过训练，加载 preds: {a.from_preds}")
        preds = _pkl.load(open(a.from_preds, "rb"))
        model.model_initialize(hidden_size=64)
    else:
        run.log(f"GEARS model_initialize(hidden_size=64) → train {a.epochs} epochs ({device}) ...")
        model.model_initialize(hidden_size=64)
        model.train(epochs=a.epochs, lr=1e-3)
    try:
        model.save_model(str(run.dir / "gears_model"))
        run.log("模型已存 run.dir/gears_model")
    except Exception as e:
        run.log(f"[WARN] 模型存盘失败: {e}")

    adata = ad.read_h5ad(contract_h5ad)
    gene_names = np.array(adata.var["gene_name"].astype(str))
    if a.target:
        # T1 模式：条件取目标系契约，预测在源系基因空间，指标裁到公共基因空间
        tgt = ad.read_h5ad(str(ROOT / "data" / "processed" / a.target / f"{a.target}.h5ad"))
        common_genes = sorted(set(adata.var["gene_name"].astype(str)) & set(tgt.var["gene_name"].astype(str)))
        gpos_t = {g: i for i, g in enumerate(tgt.var["gene_name"].astype(str))}
        t_cols = [gpos_t[g] for g in common_genes]
        t_cond = tgt.obs["condition"].astype(str).to_numpy()
        tX = tgt.X
        src_gene_set = set(np.asarray(model.pert_list).ravel())  # 模型真实词表（契约词表≠GEARS 内部面板）
        if a.level == "T2":
            # T2 协议（零公共条件跨研究转移）：预测目标系全部条件（GEARS 经 GO 词表处理未见基因）
            test_conds = sorted(set(t_cond) - {"ctrl"})
        else:
            # T1 协议（同板块跨系）：条件=两契约字符串交集
            test_conds = sorted(set(t_cond) & set(adata.obs["condition"].astype(str)) - {"ctrl"})
        if a.max_test > 0:
            test_conds = test_conds[: a.max_test]
        pert_list = []
        keep_conds = []
        for c in test_conds:
            genes = [g for g in c.split("+") if g != "ctrl"]
            if not genes:
                continue
            if not all(g in src_gene_set for g in genes):
                continue  # 扰动基因须在源系词表（GEARS 嵌入空间边界；T2 可行子集如实计数）
            pert_list.append(genes)
            keep_conds.append(c)
        run.log(f"{a.level} 模式：可用条件 {len(keep_conds)}/{len(test_conds)}，公共基因 {len(common_genes)}")
    else:
        split = json.loads(split_json.read_text(encoding="utf-8"))
        test_conds = list(split["test"])
        if a.max_test > 0:
            test_conds = test_conds[: a.max_test]
        pert_list = []
        keep_conds = []
        for c in test_conds:
            genes = [g for g in c.split("+") if g != "ctrl"]
            if genes:
                pert_list.append(genes)
                keep_conds.append(c)
    run.log(f"预测 {len(keep_conds)} 个 test 条件 ...")
    run.log(f"预测 {len(keep_conds)} 个条件 ...")
    if a.from_preds:
        import pickle as _pkl
        preds = _pkl.load(open(a.from_preds, "rb"))
        run.log(f"使用已存 preds（{len(preds)} 条）")
    elif a.from_model:
        preds = model.predict(pert_list)
        import pickle
        with open(run.dir / "preds.pkl", "wb") as f:
            pickle.dump(preds, f)
        run.log(f"preds.pkl 已存（{len(preds)} 条）")
    else:
        raise SystemExit("[ERR] T1 需要 --from-model 或 --from-preds")

    preds_norm = {}
    for k in preds.keys():
        if isinstance(k, str):
            parts = tuple(g for g in k.replace("_", "+").split("+") if g != "ctrl")
        elif isinstance(k, (list, tuple)):
            parts = tuple(g for g in k if g != "ctrl")
        else:
            parts = (str(k),)
        preds_norm.setdefault(parts, preds[k])
        preds_norm.setdefault("+".join(parts), preds[k])

    # ---- 目标侧真实统计（公共基因空间）----
    rng = np.random.default_rng(a.seed)
    gpos_s = {g: i for i, g in enumerate(gene_names)}
    s_cols = [gpos_s[g] for g in common_genes]
    t_ctrl_idx = np.where(t_cond == "ctrl")[0]
    t_ctrl_idx = rng.choice(t_ctrl_idx, min(len(t_ctrl_idx), 2000), replace=False)
    def t_rows(idx):
        s = tX[idx]
        s = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
        return s[:, t_cols]
    t_ctrl_rows = t_rows(t_ctrl_idx)
    t_ctrl_mean = t_ctrl_rows.mean(axis=0)
    true_means, de_means = {}, {}
    for c in keep_conds:
        ci = np.where(t_cond == c)[0]
        if len(ci) < 3:
            continue
        rows = t_rows(ci)
        if len(rows) > 500:
            rows = rows[rng.choice(len(rows), 500, replace=False)]
        true_means[c] = rows.mean(axis=0)
        de_means[c] = M.de_top_genes(t_ctrl_rows, rows, n_top=20)

    recs = []
    for c, genes in zip(keep_conds, pert_list):
        pdf = None
        for probe in (tuple(genes), "+".join(genes), "_".join(genes),
                      tuple(sorted(genes)), "+".join(sorted(genes)), "_".join(sorted(genes)), c):
            if probe in preds_norm:
                pdf = preds_norm[probe]
                break
        if pdf is None or c not in true_means:
            continue
        if hasattr(pdf, "columns") and "mean" in pdf.columns:
            idx_names = np.array([str(i) for i in pdf.index])
            out = np.zeros(len(gene_names), dtype=np.float64)
            okm = np.isin(idx_names, gene_names)
            pos = np.array([gpos_s[g] for g in idx_names[okm] if g in gpos_s])
            out[pos] = pdf.loc[idx_names[okm], "mean"].to_numpy()
            pmean_full = out
        else:
            pmean_full = np.asarray(pdf).ravel()
        pmean = pmean_full[s_cols]  # 源空间裁到公共基因空间
        true_mean = true_means[c]
        de_idx = de_means[c]
        recs.append({"condition": c,
                     "pearson_deg": M.pearson_deg(pmean, true_mean, de_idx),
                     "mse_deg": M.mse_top20(pmean, true_mean, de_idx),
                     "r2_deg": M.r2_deg(pmean, true_mean, de_idx),
                     "jaccard_top20": M.jaccard_topk(pmean - t_ctrl_mean, true_mean - t_ctrl_mean, k=20)})
    import pandas as pd
    df = pd.DataFrame(recs)
    num = df.select_dtypes(include=[float])
    summary = {m: {"mean": float(num[m].mean()), "std": float(num[m].std(ddof=1))} for m in num.columns}
    summary["n_conditions"] = int(len(df))
    df.to_csv(run.dir / "per_condition_gears_t1.csv", index=False)
    run.write_metrics({
        "meta": {"source": a.dataset, "target": a.target, "seed": a.seed, "level": a.level,
                 "n_common_genes": len(common_genes), "epochs": a.epochs,
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("T1 GEARS 完成：" + json.dumps({k: round(v["mean"], 4) for k, v in summary.items() if isinstance(v, dict)}, ensure_ascii=False))
    print("GEARS_T1_DONE", flush=True)


if __name__ == "__main__":
    main()