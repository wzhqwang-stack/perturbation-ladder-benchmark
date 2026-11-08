#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scGen × scvi-tools 1.5.2 兼容垫片（论文②）。

根因（2026-10-29 定位）：scgen 2.1.1 的 SCGENVAE.inference 返回 {"qz_m","qz_v","z"}，
而 scvi-tools>=1.5 的 BaseVAE.get_latent_representation 读 MODULE_KEYS（"qz"/"qzm"/"qzv"），
键名取不到时 outputs.get() 返回 None → Normal(qzm, qzv.sqrt()) 抛 NoneType.sqrt。

垫片策略：运行时 monkey-patch，新旧键名并存；不改动 site-packages（保持 pip freeze 忠实），
本文件入库 checksums，并在每个 run 的 run_config.json 里记录 scgen_compat 已启用。

用法:
    import scgen_compat
    scgen_compat.apply()
"""
import torch
from torch.distributions import Normal
from scgen._scgenvae import SCGENVAE

_APPLIED = False
_orig_inference = SCGENVAE.inference


def _inference_compat(self, x):
    out = _orig_inference(self, x)
    if "qz" not in out:
        qzm = out.get("qz_m")
        qzv = out.get("qz_v")
        if qzm is not None and qzv is not None:
            out["qzm"] = qzm
            out["qzv"] = qzv
            out["qz"] = Normal(qzm, qzv.sqrt())
    return out


def apply():
    global _APPLIED
    if not _APPLIED:
        SCGENVAE.inference = _inference_compat
        _APPLIED = True
    return _APPLIED