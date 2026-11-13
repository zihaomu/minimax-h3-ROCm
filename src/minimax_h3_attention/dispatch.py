"""NHD attention ABI, with explicit selection and no automatic fallbacks."""
from importlib import import_module

from .config import AttentionConfig


def implementation(config):
    """Resolve lazily so directory checks never import Torch/Triton."""
    if not isinstance(config, AttentionConfig):
        raise TypeError("AttentionConfig required")
    if config.backend == "reduced_error":
        if config.query_groups == 1:
            return import_module("minimax_h3_attention.native_query_group.adapter").attention
        if config.query_tile == 512:
            module = import_module("minimax_h3_attention.native_tile.adapter")
            return lambda q, k, v: module.attention(q, k, v, 512)
        return import_module("minimax_h3_attention.ix.prefix_candidate").attention
    if config.backend in ("int8_qk_fp8_pv", "fp8_qk_fp8_pv"):
        return getattr(import_module("minimax_h3_attention.ix.backend"), config.backend)
    if config.backend == "sageplus_triton":
        frozen = import_module("minimax_h3_attention._frozen.amd_sage_attention")
        return lambda q, k, v: frozen.amd_sage_attention(q, k, v, "sageplus")
    if config.backend == "torch_sdpa":
        def sdpa(q, k, v):
            import torch.nn.functional as functional
            return functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False,
            ).transpose(1, 2).contiguous()
        return sdpa
    if config.backend in ("aiter_6390_int8", "aiter_4188_bf16", "aiter_4188_fp8"):
        return getattr(import_module("minimax_h3_attention.aiter.adapter"), config.backend)
    raise ValueError(config.backend)


def attention(q, k, v, config):
    """[1,N,H,128] BF16 -> same shape/dtype, with independent head groups."""
    import torch
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("H3 attention requires [1,sequence,heads,128]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V shapes must match")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("H3 attention input/output ABI is BF16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("Q/K/V must reside on the same ROCm device")
    selected = implementation(config)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    for start in range(0, q.shape[2], config.head_chunk):
        stop = min(start + config.head_chunk, q.shape[2])
        result = selected(q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop])
        if result.shape != q[:, :, start:stop].shape or result.dtype != q.dtype:
            raise RuntimeError("Attention backend violated the output ABI")
        output[:, :, start:stop].copy_(result)
    return output
