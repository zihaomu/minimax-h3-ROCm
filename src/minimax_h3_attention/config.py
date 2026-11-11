"""CPU-only configuration shared by both sequence-parallel ranks."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os

DEFAULT_BACKEND = "reduced_error"
BACKENDS = {
    "reduced_error": "Frozen Q/K INT8 + V FP8 encoding, Ix HIP FP32 PV, direct BF16 output",
    "int8_qk_fp8_pv": "Ix inherited PR368 native INT8 QK / FP8 PV control",
    "fp8_qk_fp8_pv": "Ix SK1 native FP8 QK / FP8 PV control",
    "sageplus_triton": "Frozen AMD SagePlus Triton semantic control",
    "torch_sdpa": "Explicit PyTorch SDPA control",
}


@dataclass(frozen=True)
class AttentionConfig:
    backend: str = DEFAULT_BACKEND
    head_chunk: int = 7
    query_tile: int = 128

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend {self.backend!r}; choose {tuple(BACKENDS)}")
        if type(self.head_chunk) is not int or not 1 <= self.head_chunk <= 28:
            raise ValueError("head_chunk must be an integer in [1,28]")
        if type(self.query_tile) is not int or self.query_tile not in (128, 512):
            raise ValueError("query_tile must be 128 or 512")
        if self.query_tile != 128 and self.backend != DEFAULT_BACKEND:
            raise ValueError("BR512 is implemented only for reduced_error")

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get("H3_ATTENTION_BACKEND", DEFAULT_BACKEND),
                   int(os.environ.get("H3_ATTENTION_HEAD_CHUNK", "7")),
                   int(os.environ.get("H3_ATTENTION_QUERY_TILE", "128")))

    def identity(self):
        return asdict(self)

    def digest(self):
        raw = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()
