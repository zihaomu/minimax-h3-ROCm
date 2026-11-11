"""Explicit native prologue/replay, including the existing direct-BF16 output ABI.

Original installed Python, vendor files and binaries remain unchanged.
"""
import importlib
import torch
import torch.nn.functional as F
from . import backend


def prepare(q, k, v, mean=None):
    package = backend.load()
    core = importlib.import_module('sageattention.core')
    native = core._require_gfx12_native()
    q, k, v = [x.contiguous() for x in (q, k, v)]
    n, h = q.shape[1:3]
    mean = k.mean(dim=1, keepdim=True) if mean is None else mean
    kp = (n + 63) // 64 * 64
    if kp != n:
        k = torch.cat((k, mean.expand(-1, kp - n, -1, -1)), dim=1)
        v = F.pad(v, (0, 0, 0, 0, 0, kp - n))
    ki = torch.empty_like(k, dtype=torch.int8)
    ks = torch.empty((1, h, kp // 64), dtype=torch.float32, device=q.device)
    core._quant_fused.quant_per_block_int8_fuse_sub_mean_cuda(k, mean.squeeze(1).contiguous(), ki, ks, 64, 0)
    v8t, vs = core._gfx12_fp8_value_native(native, v, 2.25, 'NHD')
    qi, qs = native.quant_q_nhd_per_warp(q)
    return {'q': q, 'ki': ki, 'ks': ks, 'v8t': v8t, 'vs': vs, 'mean': mean,
            'qi': qi, 'qs': qs, 'native': native, 'n': n}


def raw(prepared, dtype=torch.float16):
    p = prepared
    n, h = p['q'].shape[1:3]
    qpad = (n + 127) // 128 * 128
    output = torch.empty((1, qpad, h, 128), dtype=dtype, device=p['q'].device)
    p['native'].qk_rawq_int8_sv_f8_scaled_native_attn(p['q'], p['ki'], p['v8t'], output,
                                                    p['ks'], p['vs'], 0, 0, 128**-.5, n, 1, 0)
    return output[:, :n].contiguous()


def canonical(prepared):
    p, n = prepared, prepared['n']
    return (p['qi'].transpose(1, 2).contiguous(), p['ki'][:, :n].transpose(1, 2).contiguous(),
            p['v8t'][:, :, :, :n].transpose(2, 3).contiguous().view(torch.float8_e4m3fn),
            p['qs'][:, :, :(n + 31) // 32].contiguous(), p['ks'], p['vs'])


def packed(packed_values, dtype=torch.bfloat16):
    qi, ki, v8, qs, ks, vs = packed_values
    native = importlib.import_module('sageattention.core')._require_gfx12_native()
    b, h, n, d = qi.shape
    qpad, kpad = (n + 127) // 128 * 128, (n + 63) // 64 * 64
    qp = F.pad(qi, (0, 0, 0, qpad - n)).contiguous()
    kp = F.pad(ki, (0, 0, 0, kpad - n)).contiguous()
    vp = F.pad(v8.view(torch.uint8), (0, 0, 0, kpad - n)).transpose(2, 3).contiguous()
    qsp = F.pad(qs, (0, qpad // 32 - qs.shape[-1]), value=1e-7 / 127).contiguous()
    out = torch.empty((b, h, qpad, d), device=qi.device, dtype=dtype)
    native.qk_int8_sv_f8_scaled_native_attn(qp, kp, vp, out, qsp, ks, vs, 1, 0, 128**-.5, n)
    return out[:, :, :n].transpose(1, 2).contiguous()
