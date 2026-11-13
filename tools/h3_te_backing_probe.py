"""Real fresh-TE backing qualification and short A/B/A; no Standard score."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(os.environ["H3_DIAG_OUTPUT_DIR"])
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "container-id").write_text(os.environ["HOSTNAME"])
sys.path[:0] = [str(ROOT / ".runtime/engines/ComfyUI-standard"),
                str(ROOT / ".runtime/deps/standard-rocm10"), str(ROOT / "src")]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = ["h3-te-backing-short", "--enable-triton-backend", "--disable-mmap",
            "--preview-method", "latent2rgb"]
import comfy.cli_args
from minimax_h3_attention.runtime import configure
configure(worker=False)
import torch
from safetensors.torch import save_file
import comfy.sd
import comfy.model_management as mm
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from minimax_h3_amd_sp.te_cpu_backing import CPUWeightReuse, parameter_parts, signature

GEN = json.loads((ROOT / "configs/standard.json").read_text())["generation"]
REPORT = {"status": "initializing", "scope": "fresh TE only; no DiT or Standard timing",
          "trace_enabled": False, "conditioning_reused": False, "runs": [], "qualifications": []}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(root):
    """All original packed bytes and every quantization metadata tensor."""
    rows = {}
    payload = 0
    for name, parameter in root.named_parameters(remove_duplicate=False):
        parts, _ = parameter_parts(parameter)
        values = {}
        for key, tensor in parts.items():
            value = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
            values[key] = {"sha256": hashlib.sha256(value.numpy()).hexdigest(), "bytes": value.numel()}
            payload += value.numel()
        rows[name] = {"representation": signature(parameter), "payloads": values}
    return {"parameters": rows, "parameter_payload_bytes_including_aliases": payload}


def main():
    import ctypes
    hardware = json.loads(Path(os.environ["H3_DIAG_HARDWARE"]).read_text())
    hip = ctypes.CDLL("libamdhip64.so.7")
    for i, expected in enumerate(hardware["devices"]):
        pci = ctypes.create_string_buffer(64)
        assert hip.hipDeviceGetPCIBusId(pci, 64, i) == 0
        assert pci.value.decode().lower() == expected["pci"]
    assert not any(k in os.environ for k in ("ROCP_TOOL_LIBRARIES", "RDNASCOPE_RUN_DIR"))
    REPORT.update(hardware=hardware, image_id=os.environ["DEEPINFER_IMAGE_ID"],
                  torch=torch.__version__, hip=torch.version.hip)
    write(DEST / "short.json", REPORT)
    with torch.inference_mode():
        clip = comfy.sd.load_clip(
            [str(ROOT / "models/Comfy-Org-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")],
            clip_type=comfy.sd.CLIPType.MINIMAX,
            model_options={"dtype": torch.bfloat16, "load_device": torch.device("cuda", 0),
                           "offload_device": torch.device("cpu"), "initial_device": torch.device("cpu")},
            disable_dynamic=True)
        model = clip.patcher.model
        assert clip.patcher.loaded_size() == 0
        original_to = model.to
        golden_weights = fingerprint(model)
        assert len(golden_weights["parameters"]) == 1003
        assert golden_weights["parameter_payload_bytes_including_aliases"] == 15686891868
        write(DEST / "initial-CPU-weights.json", golden_weights)
        cache = None
        reference = None
        counts = {"forward": 0, "load": 0, "load_seconds": 0.}
        last = [m for n, m in clip.cond_stage_model.named_modules() if n.endswith("layers.49")]
        assert len(last) == 1
        hook = last[0].register_forward_hook(lambda *_: counts.__setitem__("forward", counts["forward"] + 1))
        original_load = clip.load_model

        def load(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original_load(*args, **kwargs)
            finally:
                counts["load"] += 1
                counts["load_seconds"] += time.perf_counter() - start

        clip.load_model = load

        def execute(role, enabled, qualification=False):
            nonlocal cache, reference
            assert clip.patcher.loaded_size() == 0
            folder = DEST / role
            folder.mkdir()
            before = dict(counts)
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            if enabled and cache is None:
                cache = CPUWeightReuse(clip.patcher)
                cache.set_enabled(True)
            elif not enabled and cache is not None:
                cache.close()
                cache = None
            assert enabled or (cache is None and model.to == original_to)
            events_before = len(cache.rows) if cache else 0
            conditioning, latent = MiniMaxH3ImageToVideo.execute(
                clip, None, GEN["prompt"], GEN["width"], GEN["height"], GEN["frames"]).result
            torch.cuda.synchronize()
            encoded = time.perf_counter_ns()
            loaded_bytes = clip.patcher.loaded_size()
            if qualification:
                proof = fingerprint(model)
                write(folder / "device-weights.json", proof)
                assert proof == golden_weights, "Actual device weights/scales changed"
            offload_start = time.perf_counter_ns()
            mm.unload_model_and_clones(clip.patcher)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            actual = {"conditioning": conditioning[0][0].detach().cpu().contiguous(),
                      "token_tags": conditioning[0][1]["minimax_token_tags"].detach().cpu().contiguous()}
            if reference is None:
                reference = {k: v.clone() for k, v in actual.items()}
            exact = {k: bool(torch.equal(v.reshape(-1).view(torch.uint8),
                                        reference[k].reshape(-1).view(torch.uint8))) for k, v in actual.items()}
            save_file(actual, str(folder / "conditioning.safetensors"))
            weights_exact = None
            if qualification:
                proof = fingerprint(model)
                write(folder / "offloaded-CPU-weights.json", proof)
                weights_exact = proof == golden_weights
                assert weights_exact
            events = cache.rows[events_before:] if cache else []
            if enabled:
                assert len(events) == 1 and events[0]["parameters"] == 1003
                assert events[0]["device_parameter_payload_bytes_replaced"] == 15686891868
                assert events[0]["completion_barriers"] == 1
                assert not events[0]["D2H_parameter_copy_performed"]
            assert all(t.device.type == "cpu" for p in model.parameters() for t in parameter_parts(p)[0].values())
            assert all(t.device.type == "cpu" for t in model.buffers())
            row = {"role": role, "enabled": enabled, "qualification_only": qualification,
                   "timer_start_ns": start, "timer_stop_ns": stop,
                   "seconds": None if qualification else (stop - start) / 1e9,
                   "encode_and_load_seconds": (encoded - start) / 1e9,
                   "load_seconds": counts["load_seconds"] - before["load_seconds"],
                   "offload_seconds": (stop - offload_start) / 1e9,
                   "forward_calls": counts["forward"] - before["forward"],
                   "load_calls": counts["load"] - before["load"],
                   "loaded_bytes": loaded_bytes, "loaded_bytes_after_offload": clip.patcher.loaded_size(),
                   "CPU_backing_bytes": cache.retained_CPU_storage_bytes if cache else 0,
                   "backing_events": events, "conditioning_raw_equal": exact,
                   "offloaded_weights_exact": weights_exact}
            row["passed"] = (all(exact.values()) and row["forward_calls"] == row["load_calls"] == 1
                             and loaded_bytes > 0 and row["loaded_bytes_after_offload"] == 0)
            write(folder / "run.json", row)
            REPORT["qualifications" if qualification else "runs"].append(row)
            write(DEST / "short.json", REPORT)
            assert row["passed"], row
            print("TE_BACKING_SHORT", role, row["seconds"], row["offload_seconds"], flush=True)
            del conditioning, latent, actual

        try:
            for phase, enabled in (("baseline", False), ("reuse", True), ("baseline-repeat", False)):
                execute("qual-" + phase, enabled, qualification=True)
            for phase, enabled in (("baseline", False), ("reuse", True), ("baseline-repeat", False)):
                for index in range(4):
                    execute(phase + ("-warmup" if index == 0 else f"-run-{index}"), enabled)
            REPORT.update(status="complete_CPU_review_pending", medians_seconds={
                phase: statistics.median(r["seconds"] for r in REPORT["runs"]
                                         if r["role"].startswith(phase + "-run-"))
                for phase in ("baseline", "reuse", "baseline-repeat")})
            write(DEST / "short.json", REPORT)
        finally:
            if cache is not None:
                cache.close()
            hook.remove()
            clip.load_model = original_load
            assert model.to == original_to


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        REPORT.update(status="failed", traceback=traceback.format_exc())
        write(DEST / "short.json", REPORT)
        raise
