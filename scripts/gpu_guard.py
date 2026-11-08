#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 排队守卫（论文①/②共享 RTX 5060 Ti）。

判忙规则（WDDM 下桌面进程也会出现在 compute-apps，故以三信号判忙）:
  1) compute-apps 中出现 python 进程；2) GPU 利用率 >= 15%；3) 登记表存在未到期 RUNNING 行。

用法:
  python gpu_guard.py claim "<任务名>" [--hours 4]   # 冲突检查+登记，通过打印 run_id（rc=0），占用拒绝（rc=1）
  python gpu_guard.py release <run_id>              # 标记完成
  python gpu_guard.py check                         # 只查不登记，JSON 输出
登记表: D:/资料/课题/虚拟细胞/GPU排队登记.md
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

LEDGER = Path(r"D:/资料/课题/虚拟细胞/GPU排队登记.md")


def nvidia_compute_procs():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception as e:
        print(f"[WARN] nvidia-smi 查询失败: {e}")
        return ["unknown"]


def gpu_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1.0


def os_busy():
    procs = nvidia_compute_procs()
    py = [p for p in procs if re.search(r"python", p, re.I)]
    util = gpu_util()
    busy = bool(py) or util >= 15.0
    return procs, py, util, busy


def parse_claims():
    claims = []
    if not LEDGER.exists():
        return claims
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        m = re.match(
            r"^\|\s*([0-9A-Za-z\-]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|", line
        )
        if m:
            claims.append({
                "run_id": m.group(1), "start": m.group(2).strip(), "paper": m.group(3).strip(),
                "task": m.group(4).strip(), "expires": m.group(5).strip(), "status": m.group(6).strip(),
            })
    return claims


def active_claims(now=None):
    now = now or datetime.datetime.now()
    act = []
    for c in parse_claims():
        if c["status"] != "RUNNING":
            continue
        try:
            exp = datetime.datetime.strptime(c["expires"], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if exp > now:
            act.append(c)
    return act


def cmd_check():
    procs, py, util, busy = os_busy()
    act = active_claims()
    free = (not busy) and (not act)
    print(json.dumps({"python_compute_procs": py, "gpu_util_pct": util,
                      "active_claims": act, "free": free},
                     ensure_ascii=False, indent=2))
    return 0 if free else 1


def cmd_claim(task, hours):
    now = datetime.datetime.now()
    procs, py, util, busy = os_busy()
    act = active_claims(now)
    if busy or act:
        print("[BLOCKED] GPU 被占用或存在未到期登记，任务未启动。")
        print(json.dumps({"python_compute_procs": py, "gpu_util_pct": util,
                          "active_claims": act}, ensure_ascii=False, indent=2))
        return 1
    run_id = now.strftime("r%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    expires = (now + datetime.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M")
    if ("论文②" in task) or ("论文2" in task):
        paper = "论文②"
    elif "论文①" in task:
        paper = "论文①"
    else:
        paper = "未注明"
    row = f"| {run_id} | {now.strftime('%Y-%m-%d %H:%M')} | {paper} | {task} | {expires} | RUNNING |"
    with LEDGER.open("a", encoding="utf-8") as f:
        cur = LEDGER.read_text(encoding="utf-8") if LEDGER.exists() else ""
        if cur and not cur.endswith("\n"):
            f.write("\n")
        f.write(row + "\n")
    print(run_id)
    return 0


def cmd_release(run_id):
    if not LEDGER.exists():
        print(f"[ERR] 登记表不存在: {LEDGER}")
        return 1
    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    hit = False
    for i, line in enumerate(lines):
        if re.match(r"^\|\s*" + re.escape(run_id) + r"\s*\|", line) and "RUNNING" in line:
            lines[i] = line.replace("RUNNING", "DONE")
            hit = True
    LEDGER.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("released" if hit else f"[WARN] 未找到 RUNNING 状态的 {run_id}")
    return 0 if hit else 1


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if not LEDGER.exists():
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(
            "# GPU 排队登记表（论文①/② 共享 RTX 5060 Ti 16GB）\n\n"
            "规则：先登记先用；开跑前 claim 自动查卡+写入本表；跑完 release <run_id>。\n\n"
            "| run_id | 起始时间 | 论文 | 任务 | 预计释放 | 状态 |\n|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("claim")
    c.add_argument("task")
    c.add_argument("--hours", type=float, default=4.0)
    r = sub.add_parser("release")
    r.add_argument("run_id")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(cmd_check())
    if a.cmd == "claim":
        sys.exit(cmd_claim(a.task, a.hours))
    if a.cmd == "release":
        sys.exit(cmd_release(a.run_id))


if __name__ == "__main__":
    main()