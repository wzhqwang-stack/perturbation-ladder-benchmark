import sys
sys.stdout.reconfigure(encoding="utf-8")
import pyarrow.parquet as pq
pf = pq.ParquetFile(r"D:\资料\课题\虚拟细胞\论文②\data\raw\tahoe\obs_metadata.parquet")
names = set()
for rg in range(pf.metadata.num_row_groups):
    tbl = pf.read_row_group(rg, columns=["cell_name"])
    names.update(tbl["cell_name"].to_pylist())
    if rg % 20 == 0:
        print(f"rg {rg}", len(names), flush=True)
import json
json.dump(sorted(map(str, names)), open(r"D:\资料\课题\虚拟细胞\论文②\data\raw\tahoe\cell_names_all.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
crc_like = sorted(n for n in names if any(k in str(n).upper().replace(" ", "").replace("-", "") for k in ("HCT116", "COLO205", "SW480", "LOVO", "SW620", "HT29", "T84", "HCT15")))
print("CRC-like 命名:", crc_like, flush=True)
print("NAMES_DONE", len(names), flush=True)