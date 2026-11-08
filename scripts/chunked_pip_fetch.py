#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chunked_pip_fetch.py：PyPI JSON API 解析 wheel 直链 + chunked 分块下载（pip 长会话挂死时的补给线）。

用法: python chunked_pip_fetch.py --list pkg1==1.0 pkg2 pkg3 ... --wheels-dir <dir> [--py cp310]
说明:
  - pinned 包查 /pypi/<name>/<ver>/json；未钉版本查 /pypi/<name>/json 并自动挑「有 cp310 win_amd64 wheel 的最新版」
  - 每个 wheel 用 8MB Range 分块下载（复用 chunked_dl 逻辑），已存在且大小一致则跳过
  - 下载后用 pip install --no-index --find-links <dir> 离线安装，缺依赖再补进 --list 迭代
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
CHUNK = 8 << 20
PYTAG = "cp310"  # 默认；可用 --py 覆盖


def http_get(url, timeout=20, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "chunked-pip-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            print(f"  [retry {i+1}] {type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(5)
    raise last


def pick_wheel(rels, pytag=PYTAG):  # pytag 参数化
    cands = []
    for f in rels:
        fn = f["filename"]
        if fn.endswith(".whl") and "win_amd64" in fn and pytag in fn:
            cands.append(f)
        elif fn.endswith(".whl") and "py3-none-any" in fn:
            cands.append(f)
    if not cands:
        return None
    cands.sort(key=lambda f: f["upload_time"], reverse=True)
    return cands[0]


def chunked_download(url, out: Path, expect=None):
    if out.exists():
        if expect is not None and out.stat().st_size == expect:
            print(f"  [skip] {out.name} 已在库", flush=True)
            return
        if expect is None:
            print(f"  [skip?] {out.name} 已存在（大小 {out.stat().st_size}），重新校验", flush=True)
            out.unlink()
    t0 = time.time()
    have = 0
    if expect is None:
        meta = http_get(url, timeout=20)  # 太大的 wheel 不整取——这里只用于无 Content-Length 的兜底
        raise SystemExit("[ERR] expect 为空且无法走 HEAD，请传入 expected size")
    with open(out, "wb") as f:
        f.truncate(expect)
    while have < expect:
        s = have
        e = min(s + CHUNK, expect) - 1
        blob = None
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e}"})
                with urllib.request.urlopen(req, timeout=45) as r:
                    blob = r.read()
                break
            except Exception as ex:
                print(f"  [retry {attempt+1}] range {s}-{e}: {type(ex).__name__}: {str(ex)[:80]}", flush=True)
                time.sleep(8)
        if not blob:
            raise SystemExit(f"[ERR] range {s}-{e} 多次失败")
        with open(out, "r+b") as f:
            f.seek(s)
            f.write(blob)
        have = s + len(blob)
        print(f"  [{time.time()-t0:6.1f}s] {have*100//expect}% ({have}/{expect})", flush=True)
    print(f"  [done] {out.name} ({round(expect/1e6,1)} MB, {round(time.time()-t0,1)}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", nargs="+", required=True)
    ap.add_argument("--wheels-dir", required=True)
    ap.add_argument("--py", default=PYTAG)
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    wd = Path(a.wheels_dir)
    wd.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []
    for spec in a.list:
        if "==" in spec:
            name, ver = spec.split("==", 1)
        else:
            name, ver = spec, None
        print(f"== {spec}", flush=True)
        try:
            api = f"https://pypi.org/pypi/{name}/{ver}/json" if ver else f"https://pypi.org/pypi/{name}/json"
            data = json.loads(http_get(api, timeout=20))
            whl = None
            if ver is not None:
                whl = pick_wheel(data["urls"], a.py)
            else:
                # 未钉版本：按上传时间从新到旧回退，找第一个有 cp310 win wheel 的版本
                rels = data["releases"]
                order = sorted(rels.items(), key=lambda kv: max((f["upload_time"] for f in kv[1]), default=""), reverse=True)
                for v, files in order:
                    w = pick_wheel(files, a.py)
                    if w is not None:
                        ver = v
                        whl = w
                        break
                if whl is None:
                    ver = data["info"]["version"]
                    data2 = json.loads(http_get(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=20))
                    whl = pick_wheel(data2["urls"])
            if whl is None:
                print(f"  [WARN] 无 cp310 win wheel（可能仅 sdist），跳过: {spec}", flush=True)
                fail.append(spec + " (no wheel)")
                continue
            out = wd / whl["filename"]
            chunked_download(whl["url"], out, whl["size"])
            ok.append(whl["filename"])
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [FAIL] {spec}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            fail.append(spec)
    print(f"\nOK={len(ok)} FAIL={len(fail)}")
    if fail:
        print("FAIL list:", fail, flush=True)
    print("FETCH_DONE", flush=True)


if __name__ == "__main__":
    main()