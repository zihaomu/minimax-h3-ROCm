"""Narrow ROCm 10 adapter for the frozen Comfy Kitchen NVFP4 decoder."""
import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import eager


def enable_nvfp4_eager():
    if torch.version.hip != "7.16.26385":
        raise RuntimeError("This adapter is qualified only against the locked ROCm 10.1 runtime")
    # Kitchen 0.2.26's Triton NVFP4 decoder contains NVIDIA-only inline PTX.
    # Register only this operation, preserving Triton dispatch for INT8 and RoPE.
    ck.registry.register("rocm10_nvfp4_eager", eager, {
        "dequantize_nvfp4": ck.registry.get_constraints("eager", "dequantize_nvfp4")})
    ck.set_backend_priority(["rocm10_nvfp4_eager", "triton", "eager"])
    ck.registry.disable("cuda")
    ck.registry.disable("hip")
