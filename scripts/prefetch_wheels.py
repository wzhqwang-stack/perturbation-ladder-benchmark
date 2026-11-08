# -*- coding: utf-8 -*-
"""预取大 wheel（论文①环境抢救第二步）：
读 pip dry-run 报告解析出的 URL 清单 -> HEAD 大小 -> >15MB 的用 chunked_dl.py 分块落盘。
pip 安装时用 --find-links 指向该目录，避免 pip 单长连接撞本机网络挂起病。
"""
import os, sys, time, subprocess, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
PY = r'C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe'
CDL = r'D:\资料\课题\虚拟细胞\论文①\scripts\chunked_dl.py'
WDIR = r'D:\资料\课题\虚拟细胞\论文①\data\wheels'
URLS = os.path.join(os.environ['TEMP'], 'wheel_urls.txt')
THRESH = 15 << 20
os.makedirs(WDIR, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

rows = []
for ln in open(URLS, encoding='utf-8-sig'):
    ln = ln.strip()
    if not ln or '  ' not in ln:
        continue
    nv, url = ln.split('  ', 1)
    rows.append((nv, url))
log(f'{len(rows)} urls')

def head_size(url):
    for a in range(3):
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=25) as r:
                return int(r.headers['Content-Length'])
        except Exception as e:
            if a == 2:
                log(f'HEAD fail {url[-60:]}: {type(e).__name__}')
                return -1
            time.sleep(5)

big, small, failed = [], [], 0
for i, (nv, url) in enumerate(rows):
    if not url.endswith('.whl'):
        continue
    fn = url.rsplit('/', 1)[-1]
    out = os.path.join(WDIR, fn)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        log(f'[{i+1}/{len(rows)}] have {fn}')
        continue
    sz = head_size(url)
    if sz < 0:
        failed += 1
        continue
    if sz <= THRESH:
        small.append((fn, sz))
        continue
    log(f'[{i+1}/{len(rows)}] BIG {fn} {sz/1e6:.0f}MB -> chunked download')
    r = subprocess.run([PY, CDL, url, out, str(sz)], capture_output=True, text=True)
    ok = os.path.exists(out) and os.path.getsize(out) == sz
    log(f'  {"OK" if ok else "FAIL rc=" + str(r.returncode)}')
    if not ok:
        failed += 1

log(f'big wheels pre-fetched; small wheels {len(small)} left to pip; failed {failed}')
log('PREFETCH_DONE')
