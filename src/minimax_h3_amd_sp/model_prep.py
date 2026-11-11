import gc

import torch

from .quant_shard import (
    attention_head_indices,
    shard_linear_rows,
)


def shard_dit_attention(dit, rank, world_size=2):
    if world_size != 2:
        raise ValueError("the optimized implementation currently supports two GPUs")

    for block in dit.blocks:
        attention = block.attn
        full_heads = attention.heads
        index, _, _ = attention_head_indices(
            full_heads,
            attention.head_dim,
            rank,
            world_size,
            device=attention.qkv_proj.weight.device,
        )
        attention.qkv_proj = shard_linear_rows(attention.qkv_proj, index)
        attention._amd_sp_full_heads = full_heads
        attention.heads = full_heads // world_size

    gc.collect()
    return dit


def prune_worker_model(dit):
    keep = {"blocks", "dtype", "hidden_size"}
    for name in list(vars(dit)):
        if name.startswith("_") or name in keep:
            continue
        value = getattr(dit, name)
        if isinstance(value, (torch.nn.Module, torch.nn.Parameter)):
            delattr(dit, name)

    for name in list(dit._modules):
        if name != "blocks":
            delattr(dit, name)
    for name in list(dit._parameters):
        delattr(dit, name)
    for name in list(dit._buffers):
        delattr(dit, name)
    gc.collect()
    return dit


def model_options_for_dtype(weight_dtype):
    options = {}
    if weight_dtype == "fp8_e4m3fn":
        options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        options["dtype"] = torch.float8_e4m3fn
        options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        options["dtype"] = torch.float8_e5m2
    elif weight_dtype != "default":
        raise ValueError(f"unsupported weight dtype: {weight_dtype}")
    return options
