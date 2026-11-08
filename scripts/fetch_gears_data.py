#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_gears_data.py：经 GEARS PertData 官方通道下载指定数据集 + 生成 seed149 simulation split + zip 指纹入库。

用法: python fetch_gears_data.py --name adamson
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
RAW = ROOT / "data" / "raw" / "gears"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=149)
    ap.add_argument("--split", default="simulation")
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    RAW.mkdir(parents=True, exist_ok=True)
    from gears import PertData
    pd_ = PertData(str(RAW))
    pd_.load(data_name=a.name)
    pd_.prepare_split(split=a.split, seed=a.seed)
    print(f"[OK] {a.name} loaded, split {a.split} s{a.seed} ready")
    z = RAW / f"{a.name}.zip"
    if z.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "repro.py"), "sha256", str(z),
                        "--csv", str(ROOT / "checksums.csv"),
                        "--note", f"GEARS PertData figshare channel {a.name}.zip"], check=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()