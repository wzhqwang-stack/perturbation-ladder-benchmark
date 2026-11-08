#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_splits.py：为已缓存数据集生成 GEARS simulation split（多 seed），供网格 T3 维度使用。
用法: python make_splits.py --datasets norman adamson replogle_k562_essential --seeds 150 151
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
RAW = ROOT / "data" / "raw" / "gears"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--split", default="simulation")
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from gears import PertData
    pd_ = None
    for ds in a.datasets:
        for seed in a.seeds:
            pkl = RAW / f"{ds}_{a.split}_{seed}_0.75.pkl"
            if pkl.exists():
                print(f"[skip] {pkl.name} 已存在", flush=True)
                continue
            if pd_ is None:
                pd_ = PertData(str(RAW))
            pd_.load(data_name=ds)
            pd_.prepare_split(split=a.split, seed=seed)
            print(f"[OK] {ds} {a.split} s{seed}", flush=True)
    print("ALL_SPLITS_DONE", flush=True)


if __name__ == "__main__":
    main()