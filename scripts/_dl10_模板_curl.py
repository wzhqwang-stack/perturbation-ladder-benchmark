# -*- coding: utf-8 -*-
"""GSE144735 下载器 v10（终版方案）：curl --ssl-no-revoke 原生断点续传 + gzip 前缀校验。
背景：本机 schannel 吊销检查离线导致 curl HTTPS 全挂；--ssl-no-revoke 修复后畅通（实测）。
NCBI 对 python-urllib 指纹限流（v8/v9 全垃圾），curl 指纹实测可用。
每轮循环：curl -C - 续传 -> zlib 校验合法 gzip 前缀 -> 截掉损坏尾部（防「长传输被篡改」）。
"""
import os, sys, time, zlib, subprocess
sys.stdout.reconfigure(encoding='utf-8')
URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144735/suppl'
MNAME = 'GSE144735_processed_KUL3_CRC_10X_raw_UMI_count_matrix.txt.gz'
ANAME = 'GSE144735_processed_KUL3_CRC_10X_annotation.txt.gz'
SIZE = 56598691
ASIZE = 194418
OUT = r'D:\资料\课题\虚拟细胞\预实验\数据\GSE144735'
FINAL = os.path.join(OUT, MNAME)
ANNO = os.path.join(OUT, ANAME)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

def good_prefix_len(path):
    z = zlib.decompressobj(31)
    good = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            try:
                z.decompress(b)
                good += len(b)
            except zlib.error:
                return good
    return good

for a in range(1, 60):
    good = good_prefix_len(FINAL) if os.path.exists(FINAL) else 0
    if os.path.exists(FINAL) and os.path.getsize(FINAL) > good:
        log(f'truncate corrupt tail {os.path.getsize(FINAL)} -> {good}')
        with open(FINAL, 'r+b') as fh:
            fh.truncate(good)
    if good >= SIZE:
        break
    r = subprocess.run(['curl.exe', '-4', '-s', '-S', '--ssl-no-revoke', '--connect-timeout', '20',
                        '--max-time', '600', '--speed-time', '30', '--speed-limit', '20480',
                        '-C', '-', '-o', FINAL, f'{URL}/{MNAME}'], capture_output=True)
    sz = os.path.getsize(FINAL)
    log(f'loop {a}: curl rc={r.returncode} size={sz}/{SIZE} ({sz*100//SIZE}%)')
    if r.returncode == 33:
        log('range not satisfiable -> treat as complete')
        break
    time.sleep(8)

good = good_prefix_len(FINAL)
if good < SIZE:
    log(f'FAIL: only {good}/{SIZE} after retries')
    sys.exit(3)
z = zlib.decompressobj(31)
n = 0
with open(FINAL, 'rb') as f:
    while True:
        b = f.read(1 << 20)
        if not b:
            break
        n += len(z.decompress(b))
assert z.eof and not z.unused_data, 'gzip incomplete'
log(f'FULL verified: {SIZE} compressed -> {n} decompressed bytes')

if not os.path.exists(ANNO) or os.path.getsize(ANNO) < ASIZE:
    subprocess.run(['curl.exe', '-4', '-s', '-S', '--ssl-no-revoke', '-C', '-', '-o', ANNO, f'{URL}/{ANAME}'], capture_output=True)
log(f'annotation: {os.path.getsize(ANNO) if os.path.exists(ANNO) else 0}/{ASIZE}')
# 基因行数核验（首行=细胞名表头；基因数=总行数-1）
import gzip
g = 0
with gzip.open(FINAL, 'rb') as f:
    hdr = f.readline()
    for _ in f:
        g += 1
log(f'header cells~={hdr.decode().count(chr(9))}, gene rows={g}')
log('DONE_MARKER_DL10')
