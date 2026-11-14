"""Real upstream policy tests, including unequal-shard two-process collectives."""
from datetime import timedelta
import importlib.metadata
import json
import multiprocessing
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from minimax_h3_amd_sp.cache_dit import CacheDiTConfig, make_cache, DEPENDENCY_DIR, geometry_signature


def _upstream_available():
    if DEPENDENCY_DIR.is_dir():
        return True
    try:
        return importlib.metadata.version("cache-dit") == "1.5.3"
    except importlib.metadata.PackageNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _upstream_available(), reason="Provision optional Cache-DiT for policy tests")


def _runner(calls, prefix=1.0, fn=8):
    def run(hidden, first, last):
        calls.extend(range(first, last))
        for index in range(first, last):
            hidden.add_(prefix if index < fn else 0.125)
        return hidden
    return run


def test_no_hit_is_bitwise_identical_and_inplace_boundaries_survive():
    cache = make_cache(CacheDiTConfig(True, threshold=0))
    for step in range(5):
        x = torch.arange(15, dtype=torch.bfloat16).reshape(3, 5) + step
        expected = _runner([])(x.clone(), 0, 50)
        calls = []
        actual = cache.run(x.clone(), _runner(calls), 50, (3, 5))
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert calls == list(range(50))
    assert cache.summary()["cache_hits"] == 0
    assert torch.count_nonzero(cache.manager.get_Fn_buffer("H3_Fn_residual")) > 0
    assert torch.count_nonzero(cache.manager.get_Bn_buffer("H3_Bn_residual")) > 0


def test_hit_skips_middle_keeps_tail_and_request_reset_releases_features():
    cache = make_cache(CacheDiTConfig(True, fn=8, bn=2, warmup=1))
    for step in range(4):
        calls = []
        x = torch.ones(3, 5, dtype=torch.float64) * step
        result = cache.run(x, _runner(calls), 50, (3, 5))
        torch.testing.assert_close(result, torch.full_like(result, step + 8 + 42 * .125))
        assert calls == (list(range(8)) + [48, 49] if step % 2 else list(range(50)))
    assert cache.summary()["cache_hits"] == 2
    assert cache.summary()["skipped_blocks"] == 80
    cache.release()
    assert cache.summary()["retained_feature_tensors"] == 0
    cache.reset()
    calls = []
    cache.run(torch.zeros(3, 5), _runner(calls), 50, (3, 5))
    assert calls == list(range(50)) and cache.summary()["steps"] == 1


def test_geometry_change_invalidates_cache():
    cache = make_cache(CacheDiTConfig(True, warmup=1))
    cache.run(torch.zeros(3, 5), _runner([]), 50, (3, 5))
    cache.run(torch.zeros(3, 5), _runner([]), 50, (3, 5))
    assert cache.summary()["cache_hits"] == 1
    calls = []
    cache.run(torch.zeros(4, 5), _runner(calls), 50, (4, 5))
    assert calls == list(range(50)) and cache.summary()["cache_hits"] == 0


def test_timestep_modulation_rows_do_not_reset_geometry():
    hidden = torch.zeros(3, 5)
    rope = torch.zeros(3, 2)
    earlier = geometry_signature(3, hidden, [(0, 2, 0), (2, 3, 5)], rope)
    later = geometry_signature(3, hidden, [(0, 2, 3), (2, 3, 2)], rope)
    changed_tokens = geometry_signature(3, hidden, [(0, 1, 3), (1, 3, 2)], rope)
    assert earlier == later
    assert earlier != changed_tokens


def _gloo_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        cache = make_cache(CacheDiTConfig(True, fn=1, warmup=1, threshold=.36))
        for step in range(3):
            calls = []
            prefix = 1.8 if rank == 1 and step > 0 else 1.0
            x = torch.zeros(3 if rank == 0 else 2, 5)
            cache.run(x, _runner(calls, prefix=prefix, fn=1), 50, (5, 5))
            assert len(calls) == (1 if step == 1 else 50)
        result = cache.summary()
        # Local differences are 0 and .8; global element-weighted diff is .32,
        # while a mean of rank means would be .4 and choose the wrong branch.
        assert result["decisions"][1]["relative_l1"] == pytest.approx(.32)
        Path(directory, f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_two_ranks_agree_with_unequal_token_shards(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_gloo_worker, args=(rank, str(tmp_path))) for rank in (0, 1)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=60)
        assert all(p.exitcode == 0 for p in processes)
        a, b = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1)]
        assert a["decisions"] == b["decisions"]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
