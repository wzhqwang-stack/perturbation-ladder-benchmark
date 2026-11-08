#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方示例冒烟：GEARS 在 Norman 2019 上的最小训练+预测（跑通即过，不追求指标）。
数据：PertData 自动下载（figshare 通道）到 论文②/data/raw/gears/。
用法: python examples_gears.py [--epochs 3] [--device cuda] [--seed 149]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=149)
    a = ap.parse_args()
    sys.path.insert(0, r"D:\资料\课题\虚拟细胞\论文②\venv\Lib\site-packages")

    import torch
    from gears import PertData, GEARS

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    run = runlib.begin(model="gears", dataset="norman", level="T3-smoke", seed=a.seed,
                       task=f"论文② GEARS norman 官方示例冒烟", hours=2,
                       extra={"device": device, "epochs": a.epochs, "example": "official-quickstart"})
    run.log("PertData 下载/加载 norman ...")
    data_dir = Path(r"D:\资料\课题\虚拟细胞\论文②\data\raw\gears")
    data_dir.mkdir(parents=True, exist_ok=True)
    pert_data = PertData(str(data_dir))
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split="simulation", seed=a.seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    run.log("model_initialize hidden_size=64 ...")
    model = GEARS(pert_data, device=device)
    model.model_initialize(hidden_size=64)
    run.log(f"train epochs={a.epochs} ...")
    model.train(epochs=a.epochs, lr=1e-3)
    run.log("predict FEV 单基因扰动 ...")
    pred = model.predict([["FEV"]])
    run.log(f"pred shape/type ok: {type(pred)}")
    out = run.dir / "gears_model_smoke"
    model.save_model(str(out))
    run.log(f"模型已存 {out}")
    run.write_metrics({"smoke": "ok", "device": device, "epochs": a.epochs})

if __name__ == "__main__":
    main()