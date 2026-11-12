"""Pinned upstream AITER PR paths; neither path can replace reduced_error implicitly."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from ..paths import ROOT

PINS = {
    "6390": "62019a492a433943abf6618fda38ced32d3aef52",
    "4188": "e4f40a683b52ef6ef09cf7d9e5851fea3e6a460e",
}
_loaded = {}


def _source(pr):
    source = ROOT / f".runtime/sources/aiter-{pr}"
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PINS[pr]:
        raise RuntimeError(f"AITER PR {pr} source commit mismatch: {actual}")
    return source


def _load(pr):
    if pr in _loaded:
        return _loaded[pr]
    import torch
    lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())["runtime"]
    if (torch.__version__ != lock["torch"] or torch.version.hip != lock["hip"]
            or importlib.metadata.version("triton") != lock["triton"]
            or os.environ.get("DEEPINFER_IMAGE_ID") != lock["container_image_id"]):
        raise RuntimeError("AITER control requires the pinned ROCm10.1 runtime")
    if "aiter" in sys.modules:
        raise RuntimeError("AITER was imported before its pinned control was selected")
    source = _source(pr)
    os.environ["AITER_AOT_IMPORT"] = "1"
    os.environ["GPU_ARCHS"] = "gfx1201"
    os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", str(ROOT / ".runtime/cache/aiter-flydsl"))
    if pr == "6390":
        dependency = ROOT / ".runtime/deps/aiter-build-6390"
        if not dependency.exists():
            raise RuntimeError("Pinned AITER HIP JIT build dependencies are missing")
        sys.path.insert(0, str(dependency))
        rocm_home = Path("/opt/rocm/core-10.1")
        if not (rocm_home / "lib/llvm/amdgcn/bitcode/ocml.bc").is_file():
            raise RuntimeError("Pinned ROCm 10.1 device libraries are missing")
        os.environ["ROCM_HOME"] = str(rocm_home)
        os.environ["HIP_DEVICE_LIB_PATH"] = str(rocm_home / "lib/llvm/amdgcn/bitcode")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1201"
        os.environ["MAX_JOBS"] = "6"
        os.environ["PATH"] = f"{rocm_home / 'bin'}:{dependency / 'bin'}:{os.environ['PATH']}"
        if importlib.metadata.version("pybind11") != "3.0.4":
            raise RuntimeError("AITER PR 6390 requires isolated pybind11 3.0.4")
    if pr == "4188":
        dependency = ROOT / ".runtime/deps/aiter-flydsl-0.4.0"
        if not dependency.exists():
            raise RuntimeError("Pinned FlyDSL 0.4.0 is missing from .runtime/deps")
        sys.path.insert(0, str(dependency))
        if importlib.metadata.version("flydsl") != "0.4.0":
            raise RuntimeError("AITER PR 4188 requires isolated FlyDSL 0.4.0")
    sys.path.insert(0, str(source))
    package = importlib.import_module("aiter")
    if Path(package.__file__).resolve().parent != source / "aiter":
        raise RuntimeError("Resolved a different AITER package")
    if pr == "6390":
        result = importlib.import_module("aiter.ops.gfx1201_sage_attention")
    else:
        result = importlib.import_module("aiter.ops.flydsl.fmha_kernels")
    _loaded[pr] = result
    return result


def aiter_6390_int8(q, k, v):
    """Use PR 6390's own Q/K/V prepare and HIP consumer as one control."""
    module = _load("6390")
    return module.gfx1201_sage_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), softmax_scale=128 ** -.5
    )


def aiter_4188_bf16(q, k, v):
    """Use PR 4188's dense BF16 FlyDSL attention on already normalized Q/K."""
    module = _load("4188")
    return module.flydsl_flash_attn_func(
        q.contiguous(), k.contiguous(), v.contiguous(), causal=False,
        softmax_scale=128 ** -.5,
    )


def aiter_4188_fp8(q, k, v):
    """Use PR 4188's FP8 producer and consumer, without a dense fallback."""
    module = _load("4188")
    quant = importlib.import_module("aiter.ops.flydsl.fp8_quant")
    q8, k8, v8, sq, sk, sv = quant.flydsl_fp8_quant(
        q.contiguous(), k.contiguous(), v.contiguous(), rotation=True,
    )
    return module.flydsl_flash_attn_func(
        q8, k8, v8, causal=False, softmax_scale=128 ** -.5,
        q_descale=sq, k_descale=sk, v_descale=sv,
    )
