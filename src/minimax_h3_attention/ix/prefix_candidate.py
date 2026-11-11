# Modified for minimax-h3-ROCm: package/build paths only; operator math preserved.
"""Opt-in diagnostic candidate: frozen Q/K/V encoding, native FP32 PV, BF16 output.

This composition was audited on the four real prefix head shards. It still
differs at PV rounding boundaries and has no full Standard qualification.
"""
import torch
from .._frozen.amd_sage_attention import quantize_qkv
from . import backend, prefix_native


def attention(q, k, v, profile='sageplus'):
    backend._check(q, k, v, profile)
    backend.load()
    with torch.cuda.device(q.device):
        return prefix_native.packed(quantize_qkv(q, k, v, profile), dtype=torch.bfloat16)
