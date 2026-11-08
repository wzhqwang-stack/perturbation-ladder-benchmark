#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""campaign_stage02.py：T3 正式网格战役（顺序执行，逐 run 过 GPU 守卫，claim 被占则退避重试）。
矩阵：GEARS 20ep × {norman, adamson, replogle_k562_essential} × {149,150,151}
      → scGen 30ep × 同 9 格 → baselines × 同 9 格（CPU）
每个 run 由各自脚本自行 claim/release 与落账；本脚本只编排与断点续跑（metrics 已存在的格跳过）。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
VENV = ROOT / "venv" / "Scripts" / "python.exe"
RUNS = ROOT / "results" / "runs"
DATASETS = ["norman", "adamson", "replogle_k562_essential"]
SEEDS = [149, 150, 151]


def runs_with_metrics(prefix):
    out = []
    if not RUNS.exists():
        return out
    for d in RUNS.glob(prefix + "*"):
        m = d / "metrics.json"
        if m.exists() and m.stat().st_size > 10:
            out.append(d.name)
    return out


def launch(script, args, prefix, expect_epochs=None, retries=8):
    # 断点续跑：若该格已有含 meta 的 metrics.json 则跳过
    for d in RUNS.glob(prefix + "*"):
        m = d / "metrics.json"
        if m.exists() and m.stat().st_size > 10:
            try:
                import json
                meta = json.loads(m.read_text(encoding="utf-8")).get("meta", {})
                if expect_epochs is None or meta.get("epochs") == expect_epochs:
                    print(f"[skip] {prefix} 已有 metrics（{d.name}）", flush=True)
                    return 0
            except Exception:
                pass
    for i in range(retries):
        p = subprocess.run([str(VENV), str(ROOT / "scripts" / script), *args],
                           capture_output=True, text=True)
        tail = (p.stdout or "").strip().splitlines()[-3:] if p.stdout else []
        etail = (p.stderr or "").strip().splitlines()[-3:] if p.stderr else []
        print(f"[{script} {' '.join(args)}] rc={p.returncode} (try {i+1}) tail={tail} err={etail}", flush=True)
        if p.returncode == 0:
            return 0
        if "GPU 被占用" in (p.stdout or "") or "BLOCKED" in (p.stdout or ""):
            print("  GPU 被占，退避 10 分钟后重试", flush=True)
            time.sleep(600)
            continue
        return p.returncode
    return 1


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    for ds in DATASETS:
        for seed in SEEDS:
            rc = launch("run_gears.py",
                        ["--dataset", ds, "--seed", str(seed), "--epochs", "20", "--device", "cuda"],
                        prefix=f"gears_{ds}_T3-simulation_s{seed}_", expect_epochs=20)
            if rc != 0:
                print(f"[WARN] GEARS {ds} s{seed} rc={rc}，继续下一格", flush=True)
        for seed in SEEDS:
            rc = launch("run_scgen.py",
                        ["--dataset", ds, "--seed", str(seed), "--device", "cuda"],
                        prefix=f"scgen_{ds}_T3-simulation_s{seed}_", expect_epochs=30)
            if rc != 0:
                print(f"[WARN] scGen {ds} s{seed} rc={rc}", flush=True)
        for seed in SEEDS:
            rc = launch("baselines.py",
                        ["--dataset", ds, "--seed", str(seed)],
                        prefix=f"baselines_{ds}_T3-smoke_s{seed}_")
            if rc != 0:
                print(f"[WARN] baselines {ds} s{seed} rc={rc}", flush=True)
    print("CAMPAIGN_DONE", flush=True)


if __name__ == "__main__":
    main()