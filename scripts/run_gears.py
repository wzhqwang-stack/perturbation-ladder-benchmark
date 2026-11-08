#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_gears.py：GEARS 网格格子（数据集×T3 split×seed）训练+预测+统一指标。
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

    split = json.loads(split_json.read_text(encoding="utf-8"))
    adata = ad.read_h5ad(contract_h5ad)
    cond = adata.obs["condition"].astype(str).to_numpy()
    X = adata.X
    gene_names = np.array(adata.var["gene_name"].astype(str))

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
    if a.from_preds:
        import pickle as _pkl2
        preds = _pkl2.load(open(a.from_preds, "rb"))
        run.log(f"使用已存 preds（{len(preds)} 条）")
    else:
        try:
            preds = model.predict(pert_list)
        except Exception as e:
            run.log(f"[WARN] cuda 预测失败（{type(e).__name__}: {str(e)[:120]}），回退 CPU 重试 ...")
            model = model.to("cpu")
            run.log("模型已移至 CPU")
            preds = model.predict(pert_list)
        import pickle
        with open(run.dir / "preds.pkl", "wb") as f:
            pickle.dump(preds, f)
        run.log(f"preds.pkl 已存（{len(preds)} 条）")

    rng = np.random.default_rng(a.seed)
    ctrl_idx = np.where(cond == "ctrl")[0]
    def sub(idx, n):
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        s = X[idx]
        return s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    ctrl_sub = sub(ctrl_idx, 2000)
    ctrl_mean = ctrl_sub.mean(axis=0)
    gpos = {g: i for i, g in enumerate(gene_names)}

    recs = []
    preds_norm = {}
    for k in preds.keys():
        if isinstance(k, str):
            parts = tuple(g for g in k.split("+") if g != "ctrl")
        elif isinstance(k, (list, tuple)):
            parts = tuple(g for g in k if g != "ctrl")
        else:
            parts = (str(k),)
        preds_norm.setdefault(parts, preds[k])
        preds_norm.setdefault("+".join(parts), preds[k])
    run.log("preds 键样例: " + str(list(preds.keys())[:2]))
    for c, genes in zip(keep_conds, pert_list):
        pdf = None
        for probe in (tuple(genes), "+".join(genes), "_".join(genes),
                      tuple(sorted(genes)), "+".join(sorted(genes)), "_".join(sorted(genes)), c):
            if probe in preds_norm:
                pdf = preds_norm[probe]
                break
        if pdf is None:
            continue
        if hasattr(pdf, "columns") and "mean" in pdf.columns:
            idx_names = np.array([str(i) for i in pdf.index])
            out = np.zeros(len(gene_names), dtype=np.float64)
            ok = np.isin(idx_names, gene_names)
            pos = np.array([gpos[g] for g in idx_names[ok] if g in gpos])
            out[pos] = pdf.loc[idx_names[ok], "mean"].to_numpy()
            pmean = out
        else:
            pmean = np.asarray(pdf).ravel()
        ci = np.where(cond == c)[0]
        if len(ci) < 3:
            continue
        rows_c = sub(ci, 1000)
        true_mean = rows_c.mean(axis=0)
        de_idx = M.de_top_genes(ctrl_sub, rows_c, n_top=20)
        recs.append({
            "condition": c,
            "pearson_deg": M.pearson_deg(pmean, true_mean, de_idx),
            "mse_deg": M.mse_top20(pmean, true_mean, de_idx),
            "r2_deg": M.r2_deg(pmean, true_mean, de_idx),
            "jaccard_top20": M.jaccard_topk(pmean - ctrl_mean, true_mean - ctrl_mean, k=20),
        })
    import pandas as pd
    df = pd.DataFrame(recs)
    num = df.select_dtypes(include=[float])
    summary = {m: {"mean": float(num[m].mean()), "std": float(num[m].std(ddof=1))} for m in num.columns}
    summary["n_conditions"] = int(len(df))
    df.to_csv(run.dir / "per_condition_gears.csv", index=False)
    run.write_metrics({
        "meta": {"dataset": a.dataset, "seed": a.seed, "level": "T3-simulation", "epochs": a.epochs,
                 "device": device, "generated_at": datetime.now().isoformat(timespec="seconds")},
        "summary": summary,
    })
    run.log("GEARS 格子完成：" + json.dumps({k: round(v["mean"], 4) for k, v in summary.items() if isinstance(v, dict)}, ensure_ascii=False))
    print("GEARS_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()