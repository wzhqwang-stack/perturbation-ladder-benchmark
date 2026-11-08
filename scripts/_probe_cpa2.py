import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\资料\课题\虚拟细胞\论文②\scripts")
from pathlib import Path
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import cpa

ROOT = Path(r"D:\资料\课题\虚拟细胞\论文②")
MD = r"D:\资料\课题\虚拟细胞\论文②\results\runs\cpa_tahoe_crc_T3-dose_s149_cpu20261031-131541-fe8c\cpa_model"

adata = ad.read_h5ad(ROOT / "data/processed/tahoe/tahoe_crc.h5ad")
so = pd.read_csv(ROOT / "data/processed/tahoe/tahoe_crc_splitobs.csv", index_col=0)
adata.obs["split"] = so.loc[adata.obs_names, "split"].to_numpy()
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000)
adata = adata[:, adata.var["highly_variable"]].copy()
adata.obs["dose_str"] = adata.obs["dose"].astype(float).map(lambda v: f"{v:g}")

model = cpa.CPA.load(MD, adata=adata)
print("模型加载 OK", flush=True)
ood = adata[adata.obs["split"] == "ood"].copy()[:20]
r = model.custom_predict(adata=ood, batch_size=64, return_mean=True)
print("predict 返回类型:", type(r).__name__, flush=True)
import numpy as np
if isinstance(r, dict):
    for k, v in r.items():
        arr = None
        if hasattr(v, "X"):
            Xp = v.X; arr = Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)
        else:
            try: arr = np.asarray(v)
            except Exception: arr = None
        print(f"KEY {k}: {type(v).__name__} shape={None if arr is None else arr.shape}", flush=True)
        if arr is not None and arr.size:
            print(f"样例: {np.round(arr.ravel()[:4], 3)}", flush=True)
elif hasattr(r, "X"):
    Xp = r.X; arr = Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)
    print("AnnData X shape:", arr.shape, flush=True)
else:
    arr = np.asarray(r); print("shape:", arr.shape, "dtype:", arr.dtype, flush=True)
print("PROBE_DONE", flush=True)