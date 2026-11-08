import sys, json
sys.stdout.reconfigure(encoding="utf-8")
import pyarrow.parquet as pq
import numpy as np

f = r"D:\资料\课题\虚拟细胞\论文②\data\raw\tahoe\obs_metadata.parquet"
pf = pq.ParquetFile(f)
targets = {"HCT116", "HCT 116", "SW480", "COLO 205", "COLO205", "LoVo", "SW620", "HT-29", "T84"}
hits = {c: [] for c in targets}
total = 0
for rg in range(pf.metadata.num_row_groups):
    tbl = pf.read_row_group(rg, columns=["cell_name", "sample", "pass_filter"])
    cn2 = tbl["cell_name"].to_pylist()
    sm = tbl["sample"].to_pylist()
    pf2 = [v.as_py() if hasattr(v, "as_py") else v for v in tbl["pass_filter"].to_pylist()]
    for i, c in enumerate(cn2):
        if c in targets:
            hits[c].append((total + i, sm[i], pf2[i]))
    total += tbl.num_rows
    if rg % 100 == 0:
        print(f"rg {rg}/{pf.metadata.num_row_groups} rows={total}", flush=True)

print("总行数:", total)
out = {}
for c, lst in hits.items():
    if not lst:
        out[c] = {"n": 0}
        continue
    idx = np.array([x[0] for x in lst], dtype=np.int64)
    samples = sorted(set(x[1] for x in lst))
    npass = sum(1 for x in lst if str(x[2]) in ("True", "true"))
    q = np.percentile(idx, [0, 25, 50, 75, 100]).astype(np.int64)
    out[c] = {"n": int(len(idx)), "pass_true": int(npass), "row_min": int(q[0]), "q50": int(q[2]),
              "row_max": int(q[4]), "n_samples": len(samples), "samples_first20": samples[:20]}
    print(c, {k: v for k, v in out[c].items() if k != "samples_first20"}, flush=True)
json.dump({"total_rows": total, "targets": out}, open(r"D:\资料\课题\虚拟细胞\论文②\data\raw\tahoe\crc_scan.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("SCAN_DONE", flush=True)