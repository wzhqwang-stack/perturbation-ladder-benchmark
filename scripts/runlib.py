#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""runlib（论文②）：GPU 守卫 + run 目录 + run_config 的一站式辅助。

用法（在训练脚本内）:
    import runlib
    run = runlib.begin(model="gears", dataset="norman", level="T3-simulation", seed=149,
                       task="论文② GEARS norman T3 网格", hours=4, inputs=[...])
    ... 训练/预测，产物写进 run.dir ...
    run.finish()
begin() 会：claim GPU（被占则 SystemExit(1)）-> 建 results/runs/<run_id>/ -> 写 run_config.json。
finish() 会：回填 ended_at + 汇总 metrics.json（若脚本已写）-> release GPU。
"""
import atexit
import json
import subprocess
import uuid
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
GUARD = ROOT / "scripts" / "gpu_guard.py"
RUNS = ROOT / "results" / "runs"
REPRO = ROOT / "scripts" / "repro.py"
VENV_PY = sys.executable


def _guard(*args):
    return subprocess.run([VENV_PY, str(GUARD), *args], capture_output=True, text=True)


class Run:
    def __init__(self, run_id, dir_path, cfg):
        self.run_id = run_id
        self.dir = Path(dir_path)
        self.cfg = cfg

    def log(self, msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with (self.dir / "log.txt").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_metrics(self, metrics: dict):
        (self.dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def finish(self):
        cfgp = self.dir / "run_config.json"
        try:
            cfg = json.loads(cfgp.read_text(encoding="utf-8"))
            cfg["ended_at"] = datetime.now().isoformat(timespec="seconds")
            cfgp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] run_config 回填失败: {e}")
        _guard("release", self.run_id)


def begin(model, dataset, level, seed=149, task=None, hours=4.0, inputs=None, extra=None, gpu=True):
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    task = task or f"论文② {model} {dataset} {level}"
    if gpu:
        p = _guard("claim", task, "--hours", str(hours))
        if p.returncode != 0:
            print(p.stdout)
            raise SystemExit("[runlib] GPU 被占用，任务拒绝启动（错峰保护）")
        run_id = p.stdout.strip().splitlines()[-1]
    else:
        import uuid as _uuid
        run_id = datetime.now().strftime("cpu%Y%m%d-%H%M%S") + "-" + _uuid.uuid4().hex[:4]
    dir_name = f"{model}_{dataset}_{level}_s{seed}_{run_id}"
    run_dir = RUNS / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [VENV_PY, str(REPRO), "run-config", str(run_dir),
           "--script", str(Path(sys.argv[0]).resolve()),
           "--args", " ".join(sys.argv[1:]), "--seed", str(seed)]
    for i in (inputs or []):
        cmd += ["--input", str(i)]
    subprocess.run(cmd, check=True, capture_output=True)
    if extra:
        cfgp = run_dir / "run_config.json"
        cfg = json.loads(cfgp.read_text(encoding="utf-8"))
        cfg["extra"] = extra
        cfgp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    run = Run(run_id, run_dir, None)
    atexit.register(run.finish)
    return run