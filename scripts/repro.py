#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现性工具（论文②）：SHA256 指纹 + run_config.json（《复现性加固清单》第 2/4 项的落地件）。

用法:
  python repro.py sha256 <文件> [--csv checksums.csv] [--note 来源URL]
  python repro.py sha256dir <目录> [--csv checksums.csv] [--note ...]   # 目录内新文件批量入库（跳过已登记路径）
  python repro.py run-config <run目录> --script <脚本> [--args "..."] [--seed 149] [--input 路径 ...]
      # 生成 run_config.json（seed+库版本+输入指纹+起始时间）；训练脚本结束时应回填 ended_at
"""
import argparse
import csv
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def append_csv(csv_path, row):
    p = Path(csv_path)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "sha256", "bytes", "mtime", "note"])
        w.writerow(row)


def read_csv_paths(csv_path):
    done = set()
    p = Path(csv_path)
    if p.exists():
        with p.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add(r["path"])
    return done


def lib_versions():
    from importlib.metadata import version
    vers = {}
    for name in ["torch", "scanpy", "anndata", "scvi-tools", "scgen", "cell-gears", "cpa-tools",
                 "torch-geometric", "numpy", "scipy", "pandas", "h5py", "huggingface_hub"]:
        try:
            vers[name] = version(name)
        except Exception:
            pass
    return vers


def stat_row(p, digest, note):
    return [str(p.resolve()), digest, p.stat().st_size,
            datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"), note or ""]


def cmd_sha256(a):
    p = Path(a.file)
    if not p.exists():
        print(f"[ERR] 文件不存在: {p}")
        return 1
    digest = sha256(p)
    print(digest)
    if a.csv:
        append_csv(a.csv, stat_row(p, digest, a.note))
        print(f"[OK] 已记入 {a.csv}")
    return 0


def cmd_sha256dir(a):
    root = Path(a.dir)
    if not root.exists():
        print(f"[ERR] 目录不存在: {root}")
        return 1
    done = read_csv_paths(a.csv)
    n = 0
    for p in sorted(root.rglob("*")):
        if (p.is_file() and p.suffix.lower() not in (".tmp", ".part", ".crdownload")
                and str(p.resolve()) not in done):
            append_csv(a.csv, stat_row(p, sha256(p), a.note))
            n += 1
    print(f"[OK] 新入库 {n} 个文件 -> {a.csv}")
    return 0


def cmd_run_config(a):
    outdir = Path(a.run_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "run_id": outdir.name,
        "script": a.script,
        "args": a.args or "",
        "seed": a.seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lib_versions": lib_versions(),
        "inputs_sha256": {},
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    for p in (a.input or []):
        pp = Path(p)
        if pp.exists():
            cfg["inputs_sha256"][str(pp.resolve())] = sha256(pp)
        else:
            cfg["inputs_sha256"][str(pp)] = "MISSING"
    (outdir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(outdir / "run_config.json"))
    return 0


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sha256")
    s.add_argument("file")
    s.add_argument("--csv")
    s.add_argument("--note")
    d = sub.add_parser("sha256dir")
    d.add_argument("dir")
    d.add_argument("--csv", default="checksums.csv")
    d.add_argument("--note")
    r = sub.add_parser("run-config")
    r.add_argument("run_dir")
    r.add_argument("--script", required=True)
    r.add_argument("--args", default="")
    r.add_argument("--seed", type=int, default=149)
    r.add_argument("--input", action="append")
    a = ap.parse_args()
    if a.cmd == "sha256":
        sys.exit(cmd_sha256(a))
    if a.cmd == "sha256dir":
        sys.exit(cmd_sha256dir(a))
    if a.cmd == "run-config":
        sys.exit(cmd_run_config(a))


if __name__ == "__main__":
    main()