#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_tahoe.py：Tahoe-100M CRC 抽样契约构建（随机分片种子化 + CVCL 过滤 + 配额停止）。
协议（docs/数据与引用登记.md Tahoe 抽样协议 v1）：
  CRC 系=SW480/COLO 205/LoVo/HT-29/HCT15（HCT116 不在该数据集，如实记录）；
  分片顺序 seed=149 随机；逐片下载→过滤 CRC CVCL→按系配额 25k 累计；剂量经 sample_metadata 连接。
输出: data/processed/tahoe/tahoe_crc.h5ad + checksums。
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(r"D:/资料/课题/虚拟细胞/论文②")
TD = ROOT / "data" / "raw" / "tahoe"
OUT = ROOT / "data" / "processed" / "tahoe"
MIRROR = "https://hf-mirror.com/datasets/tahoebio/Tahoe-100M/resolve/main"
SEED = 149
QUOTA = 25000
MAX_SHARDS = 25


def log(msg):
    print(msg, flush=True)


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import scipy.sparse as sp  # noqa
    import anndata as ad
    import runlib

    run = runlib.begin(model="tahoe", dataset="tahoe_crc", level="build", seed=SEED,
                       task="论文② Tahoe CRC 抽样契约构建", hours=4, gpu=False,
                       inputs=[str(TD / "cell_line_metadata.parquet"), str(TD / "sample_metadata.parquet"),
                               str(TD / "drug_metadata.parquet"), str(TD / "gene_metadata.parquet")],
                       extra={"seed": SEED, "quota_per_line": QUOTA, "max_shards": MAX_SHARDS})

    cl = pd.read_parquet(TD / "cell_line_metadata.parquet")
    name2cvcl = dict(zip(cl["cell_name"], cl["Cell_ID_Cellosaur"]))
    crc_names = ["SW480", "COLO 205", "LoVo", "HT-29", "HCT15"]
    crc_ids = {name2cvcl[n]: n for n in crc_names if n in name2cvcl}
    log(f"CRC CVCL 映射: {crc_ids}")

    smeta = pd.read_parquet(TD / "sample_metadata.parquet")
    smeta["dose_uM"] = smeta["drugname_drugconc"].apply(
        lambda s: (ast.literal_eval(s)[0][1] if isinstance(s, str) and s.startswith("[") else np.nan))
    smeta["drugname"] = smeta["drugname_drugconc"].apply(
        lambda s: (ast.literal_eval(s)[0][0].strip() if isinstance(s, str) and s.startswith("[") else s))
    dose_of = dict(zip(smeta["sample"], smeta["dose_uM"]))
    drugname_of = dict(zip(smeta["sample"], smeta["drugname"]))
    log(f"sample 表 {len(smeta)} 行（剂量解析完成）")

    gene_meta = pd.read_parquet(TD / "gene_metadata.parquet")
    # genes 列索引 = token_id（非行序）；按 token_id 建基因轴
    n_genes = int(gene_meta["token_id"].max()) + 1
    gene_names = np.empty(n_genes, dtype=object)
    gene_names[gene_meta["token_id"].to_numpy()] = gene_meta["gene_symbol"].astype(str).to_numpy()
    log(f"基因轴 {n_genes}（token_id 最大 {int(gene_meta['token_id'].max())}）")

    import pyarrow  # noqa
    shards = [f"data/train-{i:05d}-of-03388.parquet" for i in range(3388)]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(shards))

    counts = {n: 0 for n in crc_names}
    blocks, obs_rows = [], []
    n_downloaded = 0
    for si in order:
        if n_downloaded >= MAX_SHARDS:
            log(f"到达分片上限 {MAX_SHARDS}")
            break
        if all(v >= QUOTA for v in counts.values()):
            log("全部系配额达成，提前停止")
            break
        shard_rel = shards[si]
        shard_name = shard_rel.split("/")[-1]
        local = TD / shard_name
        if not local.exists():
            log(f"下载 {shard_name} ...")
            import subprocess
            p = subprocess.run([sys.executable, str(ROOT / "scripts" / "chunked_dl.py"),
                                f"{MIRROR}/{shard_rel}", str(local)], capture_output=True, text=True)
            if p.returncode != 0:
                log(f"[WARN] 下载失败 {shard_name}: {p.stdout[-200:]}")
                continue
            n_downloaded += 1
        else:
            n_downloaded += 1
        tbl = pq.read_table(local)
        cl_col = tbl["cell_line_id"].to_pylist()
        keep_local = [i for i, cid in enumerate(cl_col) if cid in crc_ids]
        if not keep_local:
            local.unlink(missing_ok=True)
            continue
        sub = tbl.take(keep_local)
        genes = sub["genes"].to_pylist()
        exprs = sub["expressions"].to_pylist()
        samples = sub["sample"].to_pylist()
        for j in range(len(keep_local)):
            g = np.asarray(genes[j], dtype=np.int64)
            e = np.asarray(exprs[j], dtype=np.float32)
            blocks.append(sp.csr_matrix((e, (np.zeros_like(g), g)), shape=(1, len(gene_names))))
            s = samples[j]
            obs_rows.append({
                "cell_line_id": cl_col[keep_local[j]],
                "cell_type": crc_ids[cl_col[keep_local[j]]],
                "condition": drugname_of.get(s, s),
                "dose": float(dose_of.get(s, np.nan)),
                "dataset": "tahoe",
                "batch": s,
                "sample": s,
            })
        for cid in crc_ids:
            counts[crc_ids[cid]] += sum(1 for i in keep_local if cl_col[i] == cid)
        log(f"分片 {shard_name}: 累计 {counts} （已下载 {n_downloaded} 片）")
        local.unlink(missing_ok=True)  # 提取后即删，防磁盘堆积

    X = sp.vstack(blocks).tocsr()
    obs = pd.DataFrame(obs_rows)
    adata = ad.AnnData(X)
    adata.obs = obs.reset_index(drop=True)
    adata.var["gene_name"] = gene_names
    adata.var["ensembl_id"] = gene_names
    adata.obs_names = [f"tahoe-{i}" for i in range(adata.n_obs)]
    adata.var_names_make_unique()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "tahoe_crc.h5ad"
    adata.write_h5ad(out, compression="gzip")
    log(f"契约写出 {out} {adata.shape}（各系计数 {counts}）")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "repro.py"), "sha256", str(out),
                    "--csv", str(ROOT / "checksums.csv"),
                    "--note", f"Tahoe-100M CRC subset ({n_downloaded} random shards seed {SEED})"], check=True)
    print("TAHOE_DONE", flush=True)


if __name__ == "__main__":
    main()