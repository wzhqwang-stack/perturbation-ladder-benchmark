#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEARS × scipy 1.18 兼容补丁（论文②，可复现性记录件）。

根因（2026-10-29 实测，scipy 1.18.1）：gears/gears.py:87
    np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'], axis=0)
把 pandas Series 布尔掩码直接传给 scipy 稀疏矩阵索引；scipy 1.18 的
_validate_indices 调 idx.nonzero()，pandas Series 无该方法 →
AttributeError: 'Series' object has no attribute 'nonzero'。
（与论文① scanpy×scipy1.18 兼容坑同族；GEARS 全包仅此一处 Series 掩码直进稀疏索引。）

补丁：掩码外层加 .to_numpy()（最小修改、语义等价）。
用法:
  python patch_gears_scipy118.py apply   # 幂等；前后 SHA256 记入 checksums.csv
  python patch_gears_scipy118.py check   # 只检查状态，不改文件
"""
import csv
import datetime
import hashlib
import sys
from pathlib import Path

GEARS_PY = Path(r"D:/资料/课题/虚拟细胞/论文②/venv/Lib/site-packages/gears/gears.py")
CHECKSUMS = Path(r"D:/资料/课题/虚拟细胞/论文②/checksums.csv")
OLD = "self.adata.X[self.adata.obs.condition == 'ctrl']"
NEW = "self.adata.X[(self.adata.obs.condition == 'ctrl').to_numpy()]"


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if not GEARS_PY.exists():
        print(f"[ERR] 目标不存在: {GEARS_PY}")
        return 1
    src = GEARS_PY.read_text(encoding="utf-8")
    if NEW in src:
        print("patched-already")
        return 0
    if OLD not in src:
        print("PATTERN-NOT-FOUND（GEARS 版本变化，需人工复核）")
        return 1
    if cmd == "check":
        print("needs-patch")
        return 0
    before = sha256(GEARS_PY)
    GEARS_PY.write_text(src.replace(OLD, NEW), encoding="utf-8")
    after = sha256(GEARS_PY)
    new = not CHECKSUMS.exists()
    with CHECKSUMS.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "sha256", "bytes", "mtime", "note"])
        for digest, note in (
            (before, "gears.py BEFORE patch_gears_scipy118 (upstream cell-gears 0.1.2)"),
            (after, "gears.py AFTER patch_gears_scipy118 (Series mask -> .to_numpy(), scipy 1.18 compat)"),
        ):
            w.writerow([str(GEARS_PY), digest, GEARS_PY.stat().st_size,
                        datetime.datetime.now().isoformat(timespec="seconds"), note])
    print(f"patched before={before[:12]} after={after[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())