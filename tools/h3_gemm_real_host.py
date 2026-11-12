"""Run a trace-off real H3 GEMM replay under the shared two-GPU ownership gate."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

from rdnascope_h3_host import ROOT, WORKSPACE, LEASES, Monitor, REGISTRY, choose, run, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    meta = json.loads((fixture / "fixture.json").read_text())
    assert sha(fixture / meta["file"]) == meta["sha256"]
    capture = json.loads((fixture.parents[2] / "hardware.json").read_text())
    session = ROOT / "runs" / args.session
    session.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        pair, snapshot = choose(stack, session, require_pair=capture["physical_gpus"])
        devices = [REGISTRY["gpu_map"][str(i)] for i in pair]
        write(session / "hardware.json", {"physical_gpus": pair, "devices": devices,
                                          "preflight": snapshot, "excluded": [3]})
        write(session / "source-manifest.json", {
            "fixture": {"path": str(fixture), "sha256": meta["sha256"]},
            "scripts": {str(path.relative_to(ROOT)): sha(path) for path in
                        (ROOT / "tools/h3_gemm_real_host.py", ROOT / "tools/h3_gemm_real_replay.py")}})
        lease = LEASES / f"gpu-reservation-h3-gemm-real-{os.getpid()}.json"
        row = {"owner": f"h3-gemm-real-{os.getpid()}", "status": "active",
               "physical_gpus": pair, "purpose": "trace-off real H3 block13 fc1 GEMM tile replay",
               "evidence": str(session), "updated_unix_ns": time.time_ns()}
        write(lease, row)
        cache = ROOT / ".runtime/cache/h3-gemm-real"
        cache.mkdir(parents=True, exist_ok=True)
        output = session / "replay"
        environment = {
            "ROCR_VISIBLE_DEVICES": ",".join(item["uuid"] for item in devices),
            "HIP_VISIBLE_DEVICES": "0,1", "H3_ATTENTION_BACKEND": "reduced_error",
            "H3_ATTENTION_HEAD_CHUNK": "7", "H3_ATTENTION_QUERY_TILE": "128",
            "H3_DIAG_HARDWARE": str(session / "hardware.json"),
            "PYTHONPATH": ":".join((str(ROOT / ".runtime/engines/ComfyUI-standard"),
                                   str(ROOT / ".runtime/deps/standard-rocm10"), str(ROOT / "src"))),
            "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCH_EXTENSIONS_DIR": str(cache / "torch"),
            "OMP_NUM_THREADS": "16", "TOKENIZERS_PARALLELISM": "false"}
        command = [str(WORKSPACE / "rocm10-launch.sh"), "env",
                   *[f"{key}={value}" for key, value in environment.items()],
                   "python", str(ROOT / "tools/h3_gemm_real_replay.py"),
                   "--fixture", str(fixture), "--output", str(output)]
        write(session / "launch.json", {"command": command, "trace_enabled": False,
                                        "timing_basis": "GPU event", "fixture": str(fixture)})
        try:
            monitor_dir = session / "monitor"
            with Monitor(ROOT, REGISTRY, list(map(str, pair)), monitor_dir):
                outcome = run(command, session / "console.log", monitor_dir, output / "container-id")
            write(session / "execution.json", outcome)
            if outcome["returncode"] or outcome["contamination"]:
                raise RuntimeError("GEMM replay failed or was contaminated; inspect console.log")
            assert (output / "replay.json").is_file()
            print("GEMM_REPLAY_COMPLETE", session, flush=True)
        finally:
            write(lease, {**row, "status": "released", "updated_unix_ns": time.time_ns()})


if __name__ == "__main__":
    main()
