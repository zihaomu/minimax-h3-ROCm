"""Opt-in Cache-DiT context adapter for the H3 two-rank block loop.

The upstream package owns DBCache policy, counters and residual application.
This consumer owns in-place-safe boundaries and sequence-shard-weighted gating.
No upstream imports or feature buffers are needed when disabled.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import sys

PACKAGE_VERSION = "1.5.3"
DEPENDENCY_DIR = Path(__file__).resolve().parents[2] / ".runtime/deps/cache-dit-1.5.3/site-packages"


def geometry_signature(total_rows, hidden, segments, rope):
    # Modulation-row indices change with timestep; token boundaries and tensor
    # geometry are the stable conditions for residual reuse.
    return (total_rows, hidden.shape[1], str(hidden.dtype),
            tuple((start, stop) for start, stop, _ in segments), tuple(rope.shape))


@dataclass(frozen=True)
class CacheDiTConfig:
    enabled: bool = False
    fn: int = 8
    bn: int = 0
    threshold: float = 0.12
    warmup: int = 3
    max_continuous: int = 1

    def __post_init__(self):
        if not (1 <= self.fn < 50 and 0 <= self.bn < 50 and self.fn + self.bn < 50):
            raise ValueError("Cache-DiT requires 1 <= Fn, 0 <= Bn and Fn+Bn < 50")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold < 1:
            raise ValueError("Cache-DiT threshold must be finite and in [0, 1)")
        if self.warmup < 1 or self.max_continuous < 1:
            raise ValueError("Cache-DiT warmup and max_continuous must be positive")

    @classmethod
    def from_environment(cls):
        flag = os.environ.get("H3_CACHE_DIT", "0")
        if flag not in ("0", "1"):
            raise ValueError("H3_CACHE_DIT must be 0 or 1")
        if flag == "0":
            return cls()  # Ignore inactive tuning knobs; no optional dependency.
        return cls(True, int(os.environ.get("H3_CACHE_DIT_FN", "8")),
                   int(os.environ.get("H3_CACHE_DIT_BN", "0")),
                   float(os.environ.get("H3_CACHE_DIT_THRESHOLD", "0.12")),
                   int(os.environ.get("H3_CACHE_DIT_WARMUP", "3")),
                   int(os.environ.get("H3_CACHE_DIT_MAX_CONTINUOUS", "1")))

    def identity(self):
        return asdict(self)


def _upstream():
    if DEPENDENCY_DIR.is_dir() and str(DEPENDENCY_DIR) not in sys.path:
        sys.path.insert(0, str(DEPENDENCY_DIR))
    try:
        version = importlib.metadata.version("cache-dit")
        if version != PACKAGE_VERSION:
            raise RuntimeError(f"Cache-DiT {PACKAGE_VERSION} required, found {version}")
        from cache_dit import DBCacheConfig
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    except ImportError as error:
        raise RuntimeError("Optional Cache-DiT dependencies missing; run tools/provision_cache_dit.py") from error
    import inspect
    files = {}
    for obj in (DBCacheConfig, CachedContextManager):
        path = Path(inspect.getfile(obj))
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return DBCacheConfig, CachedContextManager, {"version": version, "source_sha256": files}


def make_cache(config=None):
    config = config or CacheDiTConfig.from_environment()
    return H3DBCache(config) if config.enabled else None


class H3DBCache:
    """One serial sampling request per instance; no cross-request feature reuse."""
    def __init__(self, config):
        if not config.enabled:
            raise ValueError("Use make_cache() for disabled mode")
        import torch
        import torch.distributed as dist
        upstream_config, manager_type, self.package = _upstream()
        self.config = config

        class ShardedManager(manager_type):
            def similarity(manager, previous, current, *, threshold, parallelized=False, prefix="Fn"):
                # Sum elements, rather than average per-rank means: H3's two
                # token shards can differ by one row. FP32 is only the gate;
                # attention, ConvRot and residual tensor dtypes stay unchanged.
                a, b = previous.float(), current.float()
                terms = torch.stack(((a - b).abs().sum(), a.abs().sum()))
                if parallelized:
                    dist.all_reduce(terms, op=dist.ReduceOp.SUM)
                numerator, denominator = terms.tolist()
                difference = numerator / denominator if denominator else (0.0 if numerator == 0 else math.inf)
                manager.add_residual_diff(difference)
                return math.isfinite(difference) and difference < threshold

        self.manager = ShardedManager(name="h3-tp2")
        self.upstream_config = upstream_config(
            Fn_compute_blocks=config.fn, Bn_compute_blocks=config.bn,
            residual_diff_threshold=config.threshold, max_warmup_steps=config.warmup,
            max_continuous_cached_steps=config.max_continuous, enable_separate_cfg=False)
        self.rows = []
        self.signature = None
        self.request_index = 0
        self.reset()

    def identity(self):
        return {"config": self.config.identity(), "package": self.package,
                "gate": "global element-weighted relative L1, FP32 sums"}

    def reset(self):
        self.manager.clear_contexts()
        context = self.manager.new_context(name="h3-tp2", cache_config=self.upstream_config)
        self.manager.set_context(context)
        self.rows = []
        self.signature = None
        self.request_index += 1

    def release(self):
        # Keep scalar statistics for diagnostics; release every feature tensor.
        self.manager.get_context().clear_buffers()
        self.signature = None

    def run(self, hidden, run_range, block_count, signature):
        if block_count != 50:
            raise ValueError("The H3 Cache-DiT adapter supports exactly 50 main blocks")
        if self.signature is not None and self.signature != signature:
            self.reset()
        self.signature = signature
        c, manager = self.config, self.manager
        original = hidden.clone()  # The H3 block loop mutates hidden in place.
        hidden = run_range(hidden, 0, c.fn)
        prefix_residual = hidden - original
        del original
        manager.mark_step_begin()
        import torch.distributed as dist
        parallelized = dist.is_initialized() and dist.get_world_size() > 1
        hit = manager.can_cache(prefix_residual, parallelized=parallelized, prefix="H3_Fn_residual")
        if hit:
            manager.add_cached_step()
            hidden, _ = manager.apply_cache(hidden, prefix="H3_Bn_residual")
        else:
            manager.set_Fn_buffer(prefix_residual, prefix="H3_Fn_residual")
            middle_input = hidden.clone()
            hidden = run_range(hidden, c.fn, block_count - c.bn)
            manager.set_Bn_buffer(hidden - middle_input, prefix="H3_Bn_residual")
            del middle_input
        if c.bn:
            hidden = run_range(hidden, block_count - c.bn, block_count)
        step = manager.get_current_step()
        difference = manager.get_current_step_residual_diff()
        self.rows.append({"step": step, "cache_hit": hit,
                          "relative_l1": difference if difference is not None and math.isfinite(difference) else None,
                          "computed_blocks": c.fn + c.bn if hit else block_count,
                          "skipped_blocks": block_count - c.fn - c.bn if hit else 0})
        return hidden

    def summary(self):
        return {**self.identity(), "enabled": True, "approximate": True,
                "request_index": self.request_index, "steps": len(self.rows),
                "cache_hits": sum(r["cache_hit"] for r in self.rows),
                "computed_blocks": sum(r["computed_blocks"] for r in self.rows),
                "skipped_blocks": sum(r["skipped_blocks"] for r in self.rows),
                "retained_feature_tensors": len(self.manager.get_context().buffers),
                "decisions": list(self.rows)}
