"""Guard and run a short H3 TP2 oracle followed by one native RDNAScope capture."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import combinations
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LEGACY = ROOT / "archive/legacy-20261111"
REGISTRY = json.loads((LEGACY / "scripts/standard_opt/targets.json").read_text())
LEASES = WORKSPACE / "deepinfer_workspace/artifacts/amd/rocm10"
sys.path.insert(0, str(LEGACY / "scripts"))
spec = importlib.util.spec_from_file_location("h3_rdnascope_monitor", LEGACY / "minimax_h3_multi_gpu/host_monitor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Monitor = module.ExperimentMonitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reserved():
    active = set()
    for path in LEASES.glob("gpu-reservation*.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "active":
            active.update(row.get("physical_gpus", []))
    return active


def choose(stack, session):
    busy = reserved()
    ready = []
    for label, info in REGISTRY["gpu_map"].items():
        index = int(label)
        if index == 3 or index in busy:
            continue
        used = int((Path("/sys/bus/pci/devices") / info["pci"] / "mem_info_vram_used").read_text())
        if used < 1024**3:
            ready.append(index)
    preferred = [(0, 2), (0, 1), (1, 2), (5, 6), (6, 7)]
    for pair in preferred + [p for p in combinations(sorted(ready), 2) if p not in preferred]:
        if any(i not in ready for i in pair):
            continue
        held = ExitStack()
        try:
            for index in pair:
                handle = held.enter_context((ROOT / f".standard-opt-state/locks/gpu{index}.lock").open("a+"))
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            snapshot = Monitor(ROOT, REGISTRY, list(map(str, pair)), session / "preflight").snapshot()
            if (snapshot["errors"] or snapshot["external_pids"] or snapshot["processes"]
                    or any(x["vram_used_bytes"] >= 1024**3 for x in snapshot["devices"])
                    or any(i in reserved() for i in pair)):
                held.close()
                continue
            stack.enter_context(held)
            return pair, snapshot
        except BlockingIOError:
            held.close()
    raise RuntimeError("No eligible idle non-GPU3 pair")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_own_container(path):
    if path.is_file():
        value = path.read_text().strip()
        if len(value) >= 12 and all(c in "0123456789abcdef" for c in value):
            subprocess.run(["docker", "stop", "--time", "3", value],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(command, log, monitor_dir, container_id):
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        contamination = None
        while process.poll() is None:
            latest = json.loads((monitor_dir / "latest.json").read_text())
            if (latest["external_pids"] or latest["errors"]
                    or time.time_ns() - latest["unix_ns"] > 2_000_000_000):
                contamination = latest
                stop_own_container(container_id)
                process.terminate()
                break
            time.sleep(.25)
        code = process.wait(timeout=30)
    return {"returncode": code, "contamination": contamination}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", help="new session directory name under runs/")
    parser.add_argument("--oracle-only", action="store_true", help="trace-off A/A control only")
    args = parser.parse_args()
    name = args.session or "rdnascope-h3-tp2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("session must be a single new directory name")
    session = ROOT / "runs" / name
    session.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        pair, snapshot = choose(stack, session)
        devices = [REGISTRY["gpu_map"][str(i)] for i in pair]
        write(session / "hardware.json", {"physical_gpus": pair, "devices": devices,
                                          "preflight": snapshot, "excluded": [3]})
        source_paths = [ROOT / ".rdnascope.toml", ROOT / "configs/standard.json",
                        ROOT / "configs/runtime-lock.json", ROOT / "integrations/comfyui/nodes.py",
                        ROOT / "tools/rdnascope_h3_short.py", ROOT / "tools/rdnascope_h3_host.py",
                        ROOT / "tools/rdnascope_rocpd.sh",
                        *ROOT.joinpath("src/minimax_h3_amd_sp").rglob("*.py"),
                        *ROOT.joinpath("src/minimax_h3_attention").rglob("*.py")]
        write(session / "source-manifest.json", {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(source_paths))})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        oracle = session / f"oracle-{stamp}"
        trace = session / f"native-{stamp}"
        oracle.mkdir()
        lease = LEASES / f"gpu-reservation-h3-rdnascope-{os.getpid()}.json"
        row = {"owner": f"h3-rdnascope-{os.getpid()}", "status": "active",
               "physical_gpus": pair, "purpose": "H3 TP2 one-NFE native diagnostic",
               "evidence": str(session), "updated_unix_ns": time.time_ns()}
        write(lease, row)
        cache = ROOT / ".runtime/cache/h3-rdnascope-short"
        cache.mkdir(parents=True, exist_ok=True)
        shared = {"ROCR_VISIBLE_DEVICES": ",".join(x["uuid"] for x in devices),
                  "HIP_VISIBLE_DEVICES": "0,1", "H3_ATTENTION_BACKEND": "reduced_error",
                  "H3_ATTENTION_HEAD_CHUNK": "7", "H3_ATTENTION_QUERY_TILE": "128",
                  "AMD_H3_SP_PROFILE": "1", "H3_DIAG_HARDWARE": str(session / "hardware.json"),
                  "H3_DIAG_PORT": str(free_port()),
                  "PYTHONPATH": ":".join((str(WORKSPACE / "RDNAScope/src"),
                                           str(ROOT / ".runtime/engines/ComfyUI-standard"),
                                           str(ROOT / "src"), str(ROOT / ".runtime/deps/standard-rocm10"))),
                  "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
                  "TRITON_CACHE_DIR": str(cache / "triton"),
                  "TORCH_EXTENSIONS_DIR": str(cache / "torch"),
                  "OMP_NUM_THREADS": "16", "TOKENIZERS_PARALLELISM": "false"}
        launch = [str(WORKSPACE / "rocm10-launch.sh"), "env"]
        app = ["python", str(ROOT / "tools/rdnascope_h3_short.py")]
        command_oracle = [*launch, *[f"{k}={v}" for k, v in
                                      {**shared, "H3_DIAG_OUTPUT_DIR": str(oracle)}.items()], *app]
        command_trace = [*launch, *[f"{k}={v}" for k, v in
                                     {**shared, "H3_DIAG_OUTPUT_DIR": str(trace)}.items()],
                         "python", "-m", "rdnascope", "--config", str(ROOT / ".rdnascope.toml"),
                         "record", "--output", str(trace), "--", *app]
        write(session / "plan.json", {"oracle": oracle.name, "trace": trace.name,
                                      "commands": {"oracle": command_oracle, "trace": command_trace},
                                      "budget": {"capture_timeout_seconds": 300,
                                                 "capture_max_output_bytes": 1610612736},
                                      "scope": "one NFE from the Standard 10-step schedule; 50 TP2 blocks",
                                      "formal_standard_timing": False})
        try:
            monitor_dir = session / "monitor"
            with Monitor(ROOT, REGISTRY, list(map(str, pair)), monitor_dir):
                print("SELECTED", pair, "ORACLE", oracle, flush=True)
                first = run(command_oracle, session / "oracle.console.log", monitor_dir,
                            oracle / "container-id")
                write(session / "oracle-execution.json", first)
                if first["returncode"] or first["contamination"]:
                    raise RuntimeError("Trace-off oracle failed; inspect oracle.console.log")
                baseline = json.loads((oracle / "app/result.json").read_text())
                assert baseline["status"] == "complete" and baseline["attention_calls"] == 50
                print("ORACLE_COMPLETE", baseline["sampled_latents_sha256"], flush=True)
                if args.oracle_only:
                    return
                print("TRACE", trace, flush=True)
                second = run(command_trace, session / "trace.console.log", monitor_dir,
                             trace / "container-id")
                write(session / "trace-execution.json", second)
                if second["returncode"] or second["contamination"]:
                    raise RuntimeError("RDNAScope capture failed; inspect trace.console.log")
                measured = json.loads((trace / "app/result.json").read_text())
                same_input = baseline["input_checks"] == measured["input_checks"]
                same_source = baseline["source_identity"] == measured["source_identity"]
                same = (measured["status"] == "complete" and measured["attention_calls"] == 50
                        and baseline["sampled_latents_sha256"] == measured["sampled_latents_sha256"]
                        and same_input and same_source)
                write(session / "oracle-review.json", {"passed": same,
                      "oracle_latents_sha256": baseline["sampled_latents_sha256"],
                      "trace_latents_sha256": measured["sampled_latents_sha256"],
                      "same_input": same_input, "same_source": same_source})
                if not same:
                    raise RuntimeError("Short-request sampled latent oracle mismatch")
                print("TRACE_AND_ORACLE_COMPLETE", trace, flush=True)
        finally:
            write(lease, {**row, "status": "released", "updated_unix_ns": time.time_ns()})


if __name__ == "__main__":
    main()
