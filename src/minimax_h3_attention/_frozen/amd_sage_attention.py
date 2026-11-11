"""ROCm Sage semantic port: INT8 QK, E4M3FN P/V, FP32 online accumulation.

Implements the ordinary sm120/FP32 and SagePlus source profiles in
SageAttention 2.2.0. SagePlus uses explicit FP16 instruction-buffer rounding
around native FP8 WMMA. No CUDA or legacy HIP binaries are loaded.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quantize_i8(X, Mean, Y, Scale, N: tl.constexpr, H: tl.constexpr,
                 SB: tl.constexpr, SN: tl.constexpr, SH: tl.constexpr,
                 GROUP: tl.constexpr, CENTER: tl.constexpr):
    group, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = group * GROUP + tl.arange(0, GROUP)
    dim = tl.arange(0, 128)
    x = tl.load(X + batch * SB + row[:, None] * SN + head * SH + dim[None, :],
                row[:, None] < N, other=0).to(tl.float32)
    if CENTER:
        mean = tl.load(Mean + (batch * H + head) * 128 + dim).to(tl.float32)
        x = tl.where(row[:, None] < N, x - mean[None, :], 0.)
    amax = tl.maximum(tl.max(tl.max(tl.abs(x), 1), 0), 1.e-7)
    inverse = tl.div_rn(127., amax)
    quantized = libdevice.rint(x * inverse).to(tl.int8)
    tl.store(Y + ((batch * H + head) * N + row[:, None]) * 128 + dim[None, :],
             quantized, row[:, None] < N)
    tl.store(Scale + (batch * H + head) * tl.cdiv(N, GROUP) + group, amax / 127.)


@triton.jit
def _quantize_v(X, Amax, Y, Scale, N: tl.constexpr, H: tl.constexpr,
                SB: tl.constexpr, SN: tl.constexpr, SH: tl.constexpr,
                ROWS: tl.constexpr, VMAX: tl.constexpr):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = block * ROWS + tl.arange(0, ROWS)
    dim = tl.arange(0, 128)
    amax = tl.load(Amax + (batch * H + head) * 128 + dim)
    inverse = tl.where(amax > 0., tl.div_rn(VMAX, amax), 0.)
    x = tl.load(X + batch * SB + row[:, None] * SN + head * SH + dim[None, :],
                row[:, None] < N, other=0).to(tl.float32)
    x = tl.minimum(tl.maximum(x * inverse[None, :], -448.), 448.)
    tl.store(Y + ((batch * H + head) * N + row[:, None]) * 128 + dim[None, :],
             x.to(tl.float8e4nv), row[:, None] < N)
    if block == 0:
        tl.store(Scale + (batch * H + head) * 128 + dim, amax / VMAX)


@triton.jit
def _sage_attention(Q, K, V, QS, KS, VS, O, N: tl.constexpr, H: tl.constexpr,
                    LOG2_SCALE: tl.constexpr, BM: tl.constexpr, PLUS: tl.constexpr):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = block * BM + tl.arange(0, BM)
    col = tl.arange(0, 64)
    dim = tl.arange(0, 128)
    base = (batch * H + head) * N * 128
    q = tl.load(Q + base + row[:, None] * 128 + dim[None, :], row[:, None] < N, other=0)
    qs = tl.load(QS + (batch * H + head) * tl.cdiv(N, 32) + row // 32,
                 row < N, other=0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 128), tl.float32)
    for start in range(tl.cdiv(N, 64)):
        key_row = start * 64 + col
        k = tl.load(K + base + key_row[None, :] * 128 + dim[:, None],
                    key_row[None, :] < N, other=0)
        raw_score = tl.dot(q, k).to(tl.float32)
        ks = tl.load(KS + (batch * H + head) * tl.cdiv(N, 64) + start)
        scale = (qs * ks) * LOG2_SCALE
        valid_score = tl.where(key_row[None, :] < N, raw_score, -float("inf"))
        tile_max = tl.max(valid_score, 1) * scale - 8.807
        new_max = tl.maximum(maximum, tile_max)
        alpha = tl.exp2(maximum - new_max)
        p = tl.exp2(tl.fma(raw_score, scale[:, None], -new_max[:, None]))
        p = tl.where(key_row[None, :] < N, p, 0.)
        denominator = denominator * alpha + tl.sum(p, 1)
        p8 = tl.minimum(p, 448.).to(tl.float8e4nv)
        v8 = tl.load(V + base + key_row[:, None] * 128 + dim[None, :],
                     key_row[:, None] < N, other=0.)
        acc = acc * alpha[:, None]
        if PLUS:
            p0, p1 = tl.split(tl.reshape(p8, (BM, 2, 32)).trans(0, 2, 1))
            v0, v1 = tl.split(tl.reshape(v8, (2, 32, 128)).trans(1, 2, 0))
            partial = tl.dot(p0, v0).to(tl.float16).to(tl.float32)
            partial = tl.dot(p1, v1, partial).to(tl.float16).to(tl.float32)
            acc += partial
        else:
            acc = tl.dot(p8, v8, acc)
        maximum = new_max
    vs = tl.load(VS + (batch * H + head) * 128 + dim)
    output = acc / denominator[:, None] * vs[None, :]
    tl.store(O + ((batch * N + row[:, None]) * H + head) * 128 + dim[None, :],
             output.to(O.dtype.element_ty), row[:, None] < N)


def _validate(q, k, v):
    if torch.version.hip != "7.16.26385" or torch.__version__ != "2.14.0+rocm10.1.0":
        raise RuntimeError("AMD Sage is qualified only for the locked ROCm 10.1 Torch runtime")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or q.shape[1] < 1 or q.shape[2] < 1:
        raise ValueError("AMD Sage requires nonempty [1, sequence, heads, 128] NHD inputs")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K and V must have equal shapes")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("AMD Sage requires matching BF16 or FP16 inputs")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("Q, K and V must reside on the same ROCm device")
    if any(t.stride(-1) != 1 or any(s <= 0 for s in t.stride()) for t in (q, k, v)):
        raise ValueError("AMD Sage requires positive strides and a contiguous head dimension")
    if any(t.requires_grad for t in (q, k, v)):
        raise ValueError("AMD Sage is an inference operator")
    if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("AMD Sage has only been qualified on gfx1201")


def quantize_qkv(q, k, v, profile="fp32"):
    _validate(q, k, v)
    if profile not in ("fp32", "sageplus"):
        raise ValueError("AMD Sage profile must be fp32 or sageplus")
    batch, seq, heads, dim = q.shape
    with torch.cuda.device(q.device):
        # FP64 reduction followed by explicit FP32 rounding resolves BF16 mean
        # ties consistently with the independent oracle. The stored mean still
        # has the original input dtype; this preprocessing cost stays in scope.
        mean = k.double().mean(dim=1, keepdim=True).float().to(k.dtype).contiguous()
        amax = v.abs().amax(dim=1).float().contiguous()
        shape = (batch, heads, seq, dim)
        qi = torch.empty(shape, dtype=torch.int8, device=q.device)
        ki = torch.empty_like(qi)
        vf = torch.empty(shape, dtype=torch.float8_e4m3fn, device=q.device)
        qs = torch.empty((batch, heads, triton.cdiv(seq, 32)), dtype=torch.float32, device=q.device)
        ks = torch.empty((batch, heads, triton.cdiv(seq, 64)), dtype=torch.float32, device=q.device)
        vs = torch.empty((batch, heads, dim), dtype=torch.float32, device=q.device)
        for x, y, scale, group, center in ((q, qi, qs, 32, False), (k, ki, ks, 64, True)):
            _quantize_i8[(triton.cdiv(seq, group), heads, batch)](
                x, mean, y, scale, seq, heads, *x.stride()[:3], group, center)
        _quantize_v[(triton.cdiv(seq, 64), heads, batch)](
            v, amax, vf, vs, seq, heads, *v.stride()[:3], 64, 2.25 if profile == "sageplus" else 448.)
    return qi, ki, vf, qs, ks, vs


def _launch(packed, output, profile="fp32"):
    batch, heads, seq, _ = packed[0].shape
    return _sage_attention[(triton.cdiv(seq, 32), heads, batch)](
        *packed, output, seq, heads, 128**-.5 * 1.44269504, 32, profile == "sageplus",
        num_warps=4, num_stages=1, enable_fp_fusion=True)


def amd_sage_attention(q, k, v, profile="fp32"):
    """Noncausal, unmasked H3 self-attention; finite inputs are required."""
    packed = quantize_qkv(q, k, v, profile)
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    with torch.cuda.device(q.device):
        _launch(packed, output, profile)
    return output
