"""Qualified QG1/BR256 HIP entry with frozen producer and direct BF16 output."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from ..paths import ROOT

STATE = {"core_calls": 0, "audited_calls": 0, "query_groups": 1, "query_tile": 256}
_extension = None
_audit = None


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_identity():
    value = os.environ.get("H3_ATTENTION_QGROUP_BUILD")
    if not value:
        raise RuntimeError("QG1 requires H3_ATTENTION_QGROUP_BUILD; no automatic fallback")
    path = Path(value).resolve() / "build.json"
    record = json.loads(path.read_text())
    extension = Path(record["extension"])
    audit = os.environ.get("H3_ATTENTION_QGROUP_AUDIT", "0")
    if audit not in ("0", "1"):
        raise ValueError("H3_ATTENTION_QGROUP_AUDIT must be 0 or 1")
    return {"manifest": str(path), "manifest_sha256": digest(path),
            "extension": str(extension), "extension_sha256": digest(extension),
            "audit": audit == "1"}


def extension():
    global _extension, _audit
    if _extension is None:
        identity = build_identity()
        record = json.loads(Path(identity["manifest"]).read_text())
        lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())["runtime"]
        if record["runtime"] != lock or os.environ.get("DEEPINFER_IMAGE_ID") != lock["container_image_id"]:
            raise RuntimeError("QG1 extension runtime/image identity mismatch")
        if record["variants"]["QG1_BR256"] != [256, 1]:
            raise RuntimeError("QG1 extension variant mismatch")
        for ref, expected in record["files_sha256"].items():
            if digest(ROOT / ref) != expected:
                raise RuntimeError(f"QG1 build/source changed: {ref}")
        spec = importlib.util.spec_from_file_location("h3_attention_spill_v1", identity["extension"])
        _extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_extension)
        _audit = identity["audit"]
        STATE["build_identity"] = identity
    return _extension


def attention(q, k, v):
    import torch
    from .._frozen.amd_sage_attention import quantize_qkv
    from ..ix import backend
    from ..native_tile.adapter import prepare, original
    backend._check(q, k, v, "sageplus")
    if q.shape != (1, 37787, 7, 128):
        raise ValueError("Qualified QG1 path requires Standard [1,37787,7,128]")
    backend.load()
    native = extension()
    with torch.cuda.device(q.device):
        packed = prepare(quantize_qkv(q, k, v, "sageplus"))
        if _audit:
            expected = original(packed).clone()
        native.tile(*packed, 3)
        STATE["core_calls"] += 1
        if _audit:
            if not torch.equal(packed[3].view(torch.uint8), expected.view(torch.uint8)):
                raise RuntimeError("QG1 real model core output differs from installed BR128 bits")
            STATE["audited_calls"] += 1
        return packed[3][:, :, :q.shape[1]].transpose(1, 2).contiguous()
