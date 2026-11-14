"""Shared rank configuration and ComfyUI adapter, independent of old experiments."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from .config import AttentionConfig

from .paths import ROOT
STATE = {"attention_calls": 0, "automatic_fallback": False,
         "standard_timing_qualified": False}
_configured = False


def source_identity(config):
    files = {}
    for folder in ("src/minimax_h3_amd_sp", "src/minimax_h3_attention"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".hip", ".json"):
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("nodes.py", "integrations/comfyui/nodes.py", "configs/runtime-lock.json"):
        files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    from minimax_h3_amd_sp.fused_swiglu import OPTIONS
    from minimax_h3_amd_sp.cache_dit import CacheDiTConfig
    record = {"configuration": config.identity(), "fc2_options": OPTIONS, "source_sha256": files,
              "cache_dit": CacheDiTConfig.from_environment().identity()}
    if config.query_groups == 1:
        from .native_query_group.adapter import build_identity
        record["native_query_group_build"] = build_identity()
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def configure(worker=False):
    global _configured
    if _configured:
        return
    import torch
    lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())["runtime"]
    if (torch.__version__ != lock["torch"] or torch.version.hip != lock["hip"]
            or importlib.metadata.version("triton") != lock["triton"]):
        raise RuntimeError("Use the pinned ROCm 10.1 Torch/Triton environment in configs/runtime-lock.json")
    if os.environ.get("DEEPINFER_IMAGE_ID") != lock["container_image_id"]:
        raise RuntimeError("Locked container identity is missing or mismatched; use the workspace ROCm10 launcher")
    if torch.cuda.device_count() != (1 if worker else 2):
        raise RuntimeError("Expose one worker GPU / exactly two primary GPUs")
    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(index).gcnArchName.split(":")[0] != "gfx1201":
            raise RuntimeError("This runtime migration is scoped to the qualified R9700/gfx1201")
    torch.cuda.set_device(0)
    torch.set_num_threads(8 if worker else 16)
    if worker:
        import comfy.options
        comfy.options.enable_args_parsing()
        original = sys.argv
        try:
            sys.argv = ["h3-rocm-worker", "--enable-triton-backend", "--disable-mmap",
                        "--preview-method", "latent2rgb"]
            importlib.import_module("comfy.cli_args")
        finally:
            sys.argv = original
    from ._frozen.rocm_backend import enable_nvfp4_eager
    enable_nvfp4_eager()
    _configured = True


def install_attention(module="minimax_h3_amd_sp.parallel_blocks", worker=False):
    import torch
    from .dispatch import attention
    config = AttentionConfig.from_environment()
    identity = source_identity(config)
    target = importlib.import_module(module)
    previous = getattr(target, "_h3_attention_identity", None)
    if previous:
        if previous != identity:
            raise RuntimeError("Restart both ranks before changing the attention configuration/source")
        return
    handshaken = False

    def selected(q, k, v, heads, mask=None, skip_reshape=False,
                 transformer_options=None, **kwargs):
        nonlocal handshaken
        if mask is not None or not skip_reshape or q.shape[1] != heads:
            raise ValueError("Only noncausal H3 packed self-attention is supported")
        if not handshaken:
            import torch.distributed as dist
            checks = [None, None]
            dist.all_gather_object(checks, identity)
            if checks[0] != checks[1]:
                raise RuntimeError("Rank attention configuration/source mismatch")
            handshaken = True
        result = attention(*(x.transpose(1, 2) for x in (q, k, v)), config=config)
        STATE["attention_calls"] += 1
        return result.reshape(result.shape[0], result.shape[1], -1)

    target.optimized_attention = selected
    target._h3_attention_identity = identity
    from minimax_h3_amd_sp.fused_swiglu import OPTIONS
    STATE.update(rank=1 if worker else 0, configuration=config.identity(), source_identity=identity,
                 fc2_options=OPTIONS)
    if config.query_groups == 1:
        from .native_query_group.adapter import STATE as QUERY_GROUP_STATE
        STATE["native_query_group"] = QUERY_GROUP_STATE
