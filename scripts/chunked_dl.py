# -*- coding: utf-8 -*-
"""通用 HTTPS 分块续传下载器（论文①环境抢救用）。
用法: python chunked_dl.py <url> <out> [expected_size]
针对本机网络病：单条长连接会挂起 → 用 Range 分块（8MB）+ 文件续传 + IncompleteRead 抢救。
"""
import os, sys, time, urllib.request, http.client
sys.stdout.reconfigure(encoding='utf-8')
URL = sys.argv[1]
OUT = sys.argv[2]
EXPECT = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
CHUNK = 8 << 20
t0 = time.time()
log = lambda *a: print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

def fetch_range(s, e, timeout=90):
    req = urllib.request.Request(URL, headers={'Range': f'bytes={s}-{e}'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        chunks = []
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            chunks.append(b)
        return b''.join(chunks)

attempt = 0
while True:
    attempt += 1
    have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if EXPECT and have > EXPECT:
        log('file larger than expected; restarting clean')
        os.remove(OUT); continue
    if EXPECT and have == EXPECT:
        log(f'complete: {have} bytes')
        break
    if not EXPECT and attempt == 1:
        req = urllib.request.Request(URL, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as r:
            EXPECT = int(r.headers['Content-Length'])
        log(f'size: {EXPECT}')
    s = have
    e = min(s + CHUNK, EXPECT) - 1
    try:
        blob = fetch_range(s, e)
    except http.client.IncompleteRead as ie:
        blob = ie.partial or b''
        log(f'attempt {attempt}: IncompleteRead, salvaged {len(blob)}B partial')
    except Exception as ex:
        log(f'attempt {attempt}: {type(ex).__name__}: {str(ex)[:120]}; sleep 20s')
        time.sleep(20)
        continue
    if not blob:
        log(f'attempt {attempt}: empty; sleep 15s')
        time.sleep(15)
        continue
    with open(OUT, 'r+b' if os.path.exists(OUT) else 'wb') as fo:
        fo.seek(s)
        fo.write(blob)
    log(f'attempt {attempt}: +{len(blob)}B -> {s+len(blob)}/{EXPECT} ({(s+len(blob))*100//EXPECT}%)')
log('DL_COMPLETE')
