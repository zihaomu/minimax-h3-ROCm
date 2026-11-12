"""Guard a bounded ABBA control and two selected-region H3 TP2 captures."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time

from rdnascope_h3_host import ROOT, WORKSPACE, REGISTRY, LEASES, Monitor as BaseMonitor
from rdnascope_h3_host import choose, run, free_port, sha, write


class OwnedMonitor(BaseMonitor):
    container_path = None
    container_id = None
    full_id = None

    def snapshot(self):
        row = super().snapshot()
        path = self.container_path
        if path is None or not path.is_file():
            return row
        cid = path.read_text().strip()
        if len(cid) < 12 or not all(c in "0123456789abcdef" for c in cid):
            return row
        if cid != self.container_id:
            result = subprocess.run(["docker", "inspect", "--format", "{{.Id}}", cid],
                                    capture_output=True, text=True)
            if result.returncode:
                return row
            self.container_id, self.full_id = cid, result.stdout.strip()

        def owned(pid):
            try:
                return self.full_id in Path(f"/proc/{pid}/cgroup").read_text()
            except (FileNotFoundError, PermissionError):
                return False

        errors = []
        for error in row["errors"]:
            if error.get("pid") and owned(error["pid"]):
                row["processes"].append({"pid": error["pid"], "pci": [], "own_worktree": True,
                                         "ownership_proof": "owned_container_cgroup", "container": self.full_id})
            else:
                errors.append(error)
        row["errors"] = errors
        for process in row["processes"]:
            if not process["own_worktree"] and owned(process["pid"]):
                process.update(own_worktree=True, ownership_proof="owned_container_cgroup", container=self.full_id)
        row["external_pids"] = [p["pid"] for p in row["processes"] if not p["own_worktree"]]
        return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--pair", help="physical pair, otherwise select idle cards")
    parser.add_argument("--trace-only", action="store_true", help="retain prior ABBA controls and capture new run IDs only")
    args = parser.parse_args()
    if Path(args.session).name != args.session or args.session in (".", ".."):
        parser.error("session must be a single new directory name")
    pair_required = tuple(map(int, args.pair.split(","))) if args.pair else None
    if pair_required and (len(pair_required) != 2 or len(set(pair_required)) != 2 or 3 in pair_required):
        parser.error("pair needs two distinct non-GPU3 IDs")
    session = ROOT / "runs" / args.session
    session.mkdir(parents=True, exist_ok=False)
    config = session / "rdnascope.toml"
    config.write_text((ROOT / ".rdnascope.toml").read_text().replace(
        "/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/bin/rocprofv3",
        str(ROOT / "tools/rdnascope_rocprof_roi.sh")).replace(
        "minimax-h3-tp2-short-nfe1-gfx1201", "minimax-h3-swiglu-warm-first-nfe-ab"))
    sidecar = session / "sidecar"
    sidecar.mkdir()
    (sidecar / "sitecustomize.py").write_text("from rdnascope_h3_worker_roi import install\ninstall()\n")
    write(session / "plan.json", {
        "purpose": "diagnose sequential Standard +14.1s observation without relabeling it causal",
        "hypotheses": ["SwiGLU kernel regression", "weight loading or allocation changes", "rank readiness or host drift"],
        "workload": "Standard prompt/shape/noise/sigmas; one first-NFE warmup + one measured first NFE; fresh TE each",
        "order": (["native-E", "native-F"] if args.trace_only else ["off-E0", "off-F0", "off-F1", "off-E1", "native-E", "native-F"]),
        "expected_swiglu_budget_seconds_per_rank_per_nfe": 0.105,
        "AA_drift_screen_seconds": 0.1,
        "AA_screen_is_not_statistical_confidence": True,
        "budget": {"capture_seconds": 300, "export_seconds": 300, "capture_bytes": 1610612736,
                   "session_expected_bytes_upper": 6442450944},
        "trace_selection": "rank0 measured sampler including load; rank1 measured blocks; not equal CPU/GPU scopes",
        "unknown": ["cross-rank clock", "record loss", "writer independent completion", "normal-path observer overhead"]})
    lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())["runtime"]
    doctor = ["docker", "run", "--rm", "--network", "none", "--mount",
              f"type=bind,src={WORKSPACE},dst={WORKSPACE}", "--workdir", str(ROOT),
              "--env", f"PYTHONPATH={WORKSPACE}/RDNAScope/src", lock["container_image_id"],
              "python", "-m", "rdnascope", "--config", str(config), "doctor"]
    with (session / "doctor.json").open("x") as stream:
        subprocess.run(doctor, stdout=stream, stderr=subprocess.STDOUT, check=True)
    with ExitStack() as stack:
        pair, snapshot = choose(stack, session, require_pair=pair_required)
        devices = [REGISTRY["gpu_map"][str(i)] for i in pair]
        write(session / "hardware.json", {"physical_gpus": pair, "devices": devices,
                                          "preflight": snapshot, "excluded": [3]})
        paths = [config, *ROOT.joinpath("src").rglob("*.py"),
                 ROOT / "tools/rdnascope_h3_short.py", ROOT / "tools/rdnascope_h3_ab_app.py",
                 ROOT / "tools/rdnascope_h3_ab_host.py", ROOT / "tools/rdnascope_h3_worker_roi.py",
                 ROOT / "tools/rdnascope_record_safe.py",
                 ROOT / "tools/rdnascope_rocprof_roi.sh", ROOT / "tools/rdnascope_rocpd.sh",
                 ROOT / "configs/runtime-lock.json", ROOT / "configs/standard.json"]
        write(session / "source-manifest.json", {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))})
        lease = LEASES / f"gpu-reservation-h3-rdnascope-ab-{os.getpid()}.json"
        row = {"owner": f"h3-rdnascope-ab-{os.getpid()}", "status": "active", "physical_gpus": pair,
               "purpose": "bounded SwiGLU regression diagnostics", "evidence": str(session), "updated_unix_ns": time.time_ns()}
        write(lease, row)
        cache = ROOT / ".runtime/cache/h3-swiglu-standard"
        results = {}
        try:
            with OwnedMonitor(ROOT, REGISTRY, list(map(str, pair)), session / "monitor") as monitor:
                print("SELECTED", pair, flush=True)
                order = (("off-E0", 0, False), ("off-F0", 1, False),
                                              ("off-F1", 1, False), ("off-E1", 0, False),
                                              ("native-E", 0, True), ("native-F", 1, True))
                for name, enabled, traced in (order[-2:] if args.trace_only else order):
                    output = session / name
                    env = {"ROCR_VISIBLE_DEVICES": ",".join(d["uuid"] for d in devices), "HIP_VISIBLE_DEVICES": "0,1",
                           "H3_ATTENTION_BACKEND": "reduced_error", "H3_ATTENTION_HEAD_CHUNK": "7", "H3_ATTENTION_QUERY_TILE": "128",
                           "H3_FC2_FUSED_SWIGLU": str(enabled), "H3_SWIGLU_AUDIT": "0", "AMD_H3_SP_PROFILE": "0",
                           "H3_DIAG_HARDWARE": str(session / "hardware.json"), "H3_DIAG_OUTPUT_DIR": str(output),
                           "H3_DIAG_PORT": str(free_port()), "H3_AB_ROI": "1" if traced else "0",
                           "PYTHONPATH": ":".join(map(str, (sidecar, ROOT / "tools", WORKSPACE / "RDNAScope/src",
                               ROOT / ".runtime/engines/ComfyUI-standard", ROOT / "src", ROOT / ".runtime/deps/standard-rocm10"))),
                           "PYTHONPYCACHEPREFIX": str(cache / "pycache"), "TRITON_CACHE_DIR": str(cache / "triton"),
                           "TORCH_EXTENSIONS_DIR": str(cache / "torch"), "OMP_NUM_THREADS": "16", "TOKENIZERS_PARALLELISM": "false"}
                    app = ["python", str(ROOT / "tools/rdnascope_h3_ab_app.py")]
                    command = [str(WORKSPACE / "rocm10-launch.sh"), "env", *[f"{k}={v}" for k, v in env.items()]]
                    if traced:
                        command += ["python", str(ROOT / "tools/rdnascope_record_safe.py"), "--config", str(config), "record", "--output", str(output), "--"]
                    command += app
                    write(session / f"{name}-launch.json", {"command": command, "trace_enabled": traced})
                    monitor.container_path = output / "container-id"
                    print("RUN", name, flush=True)
                    outcome = run(command, session / f"{name}.console.log", session / "monitor", monitor.container_path)
                    write(session / f"{name}-execution.json", outcome)
                    if outcome["returncode"] or outcome["contamination"]:
                        raise RuntimeError(f"Failed {name}; retain logs")
                    result = json.loads((output / "app/result.json").read_text())
                    worker = json.loads((output / "app/worker-stats.json").read_text())
                    assert result["status"] == "complete" and worker["attention_calls"] == 100
                    key = "fused_calls" if enabled else "eager_calls"
                    assert worker["swiglu"][key] == 500 and worker["swiglu"]["audited_calls"] == 0
                    results[name] = {"requests": result["requests"], "load_calls": result["load_calls"], "worker_swiglu": worker["swiglu"]}
                    write(session / "completed.json", results)
                    print("COMPLETE", name, result["requests"][-1]["stages"], flush=True)
        finally:
            write(lease, {**row, "status": "released", "updated_unix_ns": time.time_ns()})


if __name__ == "__main__":
    main()
