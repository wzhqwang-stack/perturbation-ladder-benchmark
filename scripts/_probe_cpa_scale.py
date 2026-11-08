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
ood = adata[adata.obs["split"] == "ood"].copy()[:200]
r = model.custom_predict(adata=ood, batch_size=64, return_mean=True)
pred = r["latent_x_pred"].X
pred = pred.toarray() if hasattr(pred, "toarray") else np.asarray(pred)
true = ood.X
true = true.toarray() if hasattr(true, "toarray") else np.asarray(true)
print("pred  min/median/mean/max:", np.round([pred.min(), np.median(pred), pred.mean(), pred.max()], 3), flush=True)
print("true  min/median/mean/max:", np.round([true.min(), np.median(true), true.mean(), true.max()], 3), flush=True)
from scipy import stats
pr = [stats.pearsonr(pred[i], true[i])[0] for i in range(min(50, len(pred)))]
print("逐细胞 Pearson 中位:", np.round(np.median(pr), 3), flush=True)
# 缩放假设检验：若 pred 与 true 仅差一个线性因子
ratio = true.mean() / max(pred.mean(), 1e-9)
scaled = pred * ratio
pr2 = [stats.pearsonr(scaled[i], true[i])[0] for i in range(min(50, len(pred)))]
print(f"线性缩放因子 {ratio:.3f}；缩放后逐细胞 Pearson 中位: {np.round(np.median(pr2), 3)}", flush=True)
print("PROBE_DONE", flush=True)