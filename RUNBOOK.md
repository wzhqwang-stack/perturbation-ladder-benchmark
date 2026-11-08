# RUNBOOK：论文② 全管线一页式入口

> 目标读者：第三人（审计/复现/续作）。目标：凭本文件 + scripts/ + requirements + checksums.csv 重跑全部数字与图。
> 环境：Windows，RTX 5060 Ti 16GB（GPU 走 gpu_guard 三信号判忙登记协议）；主环境 `venv`（Py3.12）、CPA 独立环境 `venv_cpa`（内含式 Py3.10）。

## 0. 环境准备（一次性）
```powershell
python -m venv venv; venv\Scripts\python -m pip install -r requirements.txt          # 主栈 torch 2.11.0+cu128/scanpy 1.12.4/scvi-tools 1.5.2/cell-gears 0.1.2/scgen 2.1.1
python -m venv venv_cpa; venv_cpa\Scripts\python -m pip install -r requirements_cpa.txt  # CPA 0.8.8 内含式（torch 2.1.2+cpu/PL 1.9.5/anndata 0.9.2）
```
- GPU 纪律：每个 GPU run 前 `runlib.begin(gpu=True)`（gpu_guard claim）→ 跑完 release；三信号判忙（python 进程/util≥15%/未到期登记）。
- 补丁：`scripts/patches/patch_gears_scipy118.py`（GEARS scipy 1.18 兼容，幂等+前后指纹）。

## 1. 数据获取与契约（→ checksums.csv 全程留痕）
| 步骤 | 命令 | 产物 |
|---|---|---|
| GEARS 原始数据 | `venv\Scripts\python scripts\fetch_gears_data.py`（PertData figshare 通道） | data/raw/gears/{norman,adamson,replogle_k562_essential,replogle_rpe1_essential}*.zip + pkl splits |
| 统一契约+T3 split JSON | `build_unified.py --name <ds> --split_t3 simulation --seed <s>`（逐数据集逐 seed） | data/processed/<ds>/<ds>.h5ad + splits/split_T3-simulation_s<seed>.json |
| GEARS pkl splits | `make_splits.py --datasets <ds...> --seeds 149 150 151`（pkl 已存在则跳过；⚠️契约 JSON 由 build_unified 生成，两脚本此处历史断链，勿混淆） | data/raw/gears/<ds>_simulation_<seed>_0.75.pkl |
| Tahoe CRC 契约 | `build_tahoe.py`（MAX_SHARDS=25 配额停止，seed149 随机分片+CVCL 过滤） | data/processed/tahoe/tahoe_crc.h5ad（SHA256=16d6913b…） |
| CKI 外部数据 | curl.exe --ssl-no-revoke 下载 GEO 补充文件（GSE124715/125933）；NCBI gene_info.gz | data/external/... |
| 指纹入库 | `repro.py sha256 <file> --csv checksums.csv --note <来源URL>` / `sha256dir <dir>` | checksums.csv |

## 2. 各层 run（每 run 自动 run_config.json：seed+库版本+输入 SHA256+起止时间）
| 层 | 脚本 | 说明 |
|---|---|---|
| T3 网格（3 数据集×3 seeds） | `run_gears.py --dataset <ds> --seed <s> --device cuda`；`run_scgen.py`；`baselines.py` | RPE1 GEARS s150/151 用 `--from-model <run>\gears_model` 补评测 |
| T3-剂量维（Tahoe） | `run_cpa_tahoe.py`（ext log）/ `run_cpa_tahoe_interp.py` / `run_cpa_tahoe_counts.py`（FROM_MODEL 常量复用）；`run_baselines_tahoe.py` | CPA 0.8.8：`custom_predict` 需完整派生字段（obsm perts/perts_doses+obs CPA_ctrl 等） |
| T1（K562↔RPE1） | scGen：`run_t1_scgen.py --source <src> --target <tgt> --seed <s> --device cuda`；GEARS：`run_gears_t1.py --dataset <src> --target <tgt> --level T1 --seed <s> --from-model <src的T3模型>`；CPA：`run_cpa_t1_generic.py --source <src> --target <tgt> --seed <s>`（venv_cpa） | GEARS 词表过滤对 `model.pert_list` 查（契约词表≠模型词表）；目标数据映射进源基因空间（缺失置 0）；T1 上下文细胞型=源系动态取值 |
| T2（norman↔adamson） | `run_gears_t1.py --level T2`（双向）；`run_baselines_t2.py [--seed <s>]` | 零公共条件协议（扰动基因宇宙 106∩87=∅） |
| T4（CKI 锚点） | `run_t4_cki.py`（双 GSE 一键） | 检索表=run 目录 t4_signature_retrieval.csv |

## 3. 聚合与图（确定性，重跑即重建）
```powershell
venv\Scripts\python scripts\aggregate.py            # T3 矩阵+Wilcoxon → docs/网格汇总_stage02.md
venv\Scripts\python scripts\aggregate_transfer.py   # T1/T2/T4 多种子 mean±std → docs/转移层汇总_T1T2T4.md
venv\Scripts\python scripts\make_figures.py         # P1-P6 → results/figures/（PNG+PDF+figures_index.md）
```
正典规则：同 (模型,层级,方向,seed) 取 metrics.json 最新；转移层 run 按 meta.level 与 T3 网格隔离。

## 4. 复现验收（已实测）
- 同 seed 重跑 T2 基线节点 → 12/12 统计量与存档比特级一致（Δ=0，2026-11-02 实测，见审查报告）。
- 训练节点（torch+CUDA）按「统计复现」口径：同协议+种子→结论一致；模型产物（gears_model×42/cpa_model×9/preds.pkl×30）支撑免训练评测重放。
- 数字总目：docs/简报_09_数字图注定稿_2026-11-01.md；审查结论：docs/审查_复现性逻辑创新性_2026-11-02.md。

## 5. 已知坑（务必先读）
1. PowerShell `curl` 别名劫持 → 一律 `curl.exe --ssl-no-revoke`；NCBI 对 python-urllib TLS 指纹限流 → 下载走 curl。
2. .ps1 含中文必须 UTF-8 带 BOM；多行代码替换用带 assert 的 python 补丁器（PowerShell 单引号串里 \n 是字面量）。
3. pip 长会话挂死 → chunked 8MB Range 短事务（scripts/chunked_dl.py / chunked_pip_fetch.py）；HF 走 hf-mirror.com。
4. CPA dose 域经 f"{v:g}" 是 "1"/"0" 非 "1.0"；CPA 数据底座必须非负；setup_anndata 先 filter_genes(min_cells=50)。
5. GEARS 各 seed 词表面板可略有差异 → 可行条件数随 seed 变（T2 前向 7→86），指标按 seed 内部口径一致。
6. 网络离线时 curl 证书吊销检查失败 rc35 → --ssl-no-revoke 必加。