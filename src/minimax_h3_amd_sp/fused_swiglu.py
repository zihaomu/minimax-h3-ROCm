"""BF16 SwiGLU fusion with the original ConvRot and INT8 linear retained."""
import os

import torch
import triton
import triton.language as tl


def _flag(name, default):
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


OPTIONS = {"fused_swiglu": _flag("H3_FC2_FUSED_SWIGLU", "1"),
           "audit_swiglu": _flag("H3_SWIGLU_AUDIT", "0")}
STATE = {"fused_calls": 0, "eager_calls": 0, "audited_calls": 0,
         "audited_elements": 0}


@triton.jit
def _swiglu_bf16(X, Y, SIZE: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // K, offset % K
    gate = tl.load(X + row * (2 * K) + col, offset < SIZE, other=0).to(tl.float32)
    up = tl.load(X + row * (2 * K) + K + col, offset < SIZE, other=0).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    output = (silu.to(tl.float32) * up).to(tl.bfloat16)
    tl.store(Y + offset, output, offset < SIZE)


def swiglu_bf16(x):
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.ndim != 2 or not x.is_contiguous() or x.shape[1] % 2:
        raise ValueError("Fused SwiGLU requires contiguous 2D BF16 GPU gate/up halves")
    width = x.shape[1] // 2
    output = torch.empty((x.shape[0], width), device=x.device, dtype=x.dtype)
    _swiglu_bf16[(triton.cdiv(output.numel(), 1024),)](
        x, output, SIZE=output.numel(), K=width, BLOCK=1024)
    return output


def fc2(linear, x):
    import comfy.ops
    if not OPTIONS["fused_swiglu"]:
        STATE["eager_calls"] += 1
        return comfy.ops.linear_input_act(linear, x, "swiglu")
    weight = linear.weight
    if (comfy.model_management.in_training or type(weight).__name__ != "QuantizedTensor"
            or weight._layout_cls != "TensorWiseINT8Layout"
            or getattr(weight._params, "transposed", False)):
        raise ValueError("Fused fc2 requires the H3 inference INT8 weight layout")
    activated = swiglu_bf16(x)
    if OPTIONS["audit_swiglu"]:
        gate, up = x.chunk(2, dim=-1)
        reference = torch.nn.functional.silu(gate).mul_(up)
        if not torch.equal(activated.view(torch.uint8), reference.view(torch.uint8)):
            raise RuntimeError("Fused SwiGLU raw-bit audit failed")
        STATE["audited_calls"] += 1
        STATE["audited_elements"] += activated.numel()
    STATE["fused_calls"] += 1
    # Keep Comfy's original weight cast/unload lifecycle, rotation, row scale,
    # INT8 GEMM and BF16 epilogue. None bypasses only the already-applied act.
    return comfy.ops.linear_input_act(linear, activated, None)
