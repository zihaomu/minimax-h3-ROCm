"""One warmup + one first-NFE diagnostic; the Standard schedule/input is retained."""
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager
import traceback

import rdnascope_h3_short as b

SCOPES = []
PHASE = "initialization"


@contextmanager
def mark(name):
    start = time.perf_counter_ns()
    b.ROCTX.roctxRangePushA(name.encode())
    try:
        yield
    finally:
        b.ROCTX.roctxRangePop()
        SCOPES.append({"name": name, "start_ns": start,
                       "duration_ns": time.perf_counter_ns() - start})


def main():
    global PHASE
    b.ROCTX = b.ctypes.CDLL("/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib/librocprofiler-sdk-roctx.so.1")
    b.ROCTX.roctxRangePushA.argtypes = [b.ctypes.c_char_p]
    b.ROCTX.roctxRangePop.argtypes = []
    for name in ("roctxProfilerResume", "roctxProfilerPause"):
        function = getattr(b.ROCTX, name)
        function.argtypes = [b.ctypes.c_uint32]
        function.restype = b.ctypes.c_int
    traced = "RDNASCOPE_RUN_DIR" in os.environ
    hardware = b.device_identity()
    lock = json.loads((b.ROOT / "configs/runtime-lock.json").read_text())["runtime"]
    assert b.torch.__version__ == lock["torch"] and b.torch.version.hip == lock["hip"]
    assert os.environ["DEEPINFER_IMAGE_ID"] == lock["container_image_id"]
    assert not b.SWIGLU_OPTIONS["audit_swiglu"]
    os.environ["AMD_H3_SP_LOGDIR"] = str(b.APP)
    os.environ["H3_OUTPUT_DIR"] = str(b.APP)
    report = {"status": "running", "trace_enabled": traced, "fc2_options": b.SWIGLU_OPTIONS,
              "source_identity": b.source_identity(b.AttentionConfig.from_environment()),
              "image_id": lock["container_image_id"], "hardware": hardware,
              "scope": "two first-NFE requests (warmup then measure), fresh TE each; no VAE/media",
              "requests": [], "scopes": SCOPES, "load_calls": []}
    plugin = None
    hooks = []
    original_load = b.mm.load_models_gpu

    def loading(*args, **kwargs):
        models = args[0] if args else kwargs.get("models", [])
        row = {"phase": PHASE, "models": [type(getattr(m, "model", m)).__name__ for m in models]}
        start = time.perf_counter_ns()
        with mark(f"h3/mm/load_models_gpu/{PHASE}"):
            result = original_load(*args, **kwargs)
        row["seconds"] = (time.perf_counter_ns() - start) / 1e9
        report["load_calls"].append(row)
        return result

    b.mm.load_models_gpu = loading
    try:
        with b.torch.inference_mode():
            weights = b.ROOT / "models/Comfy-Org-MiniMax-H3"
            with mark("h3/initialize"):
                clip = b.comfy.sd.load_clip(
                    [str(weights / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")],
                    clip_type=b.comfy.sd.CLIPType.MINIMAX,
                    model_options={"dtype": b.torch.bfloat16, "load_device": b.torch.device("cuda", 0),
                                   "offload_device": b.torch.device("cpu"), "initial_device": b.torch.device("cpu")},
                    disable_dynamic=True)
                b.folder_paths.add_model_folder_path("diffusion_models", str(weights / "diffusion_models"))
                spec = b.importlib.util.spec_from_file_location("h3_ab_plugin", b.ROOT / "integrations/comfyui/nodes.py")
                plugin = b.importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plugin)
                model = plugin.AMDMinimaxH3SequenceParallelLoader.execute(
                    "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default",
                    ",".join(x["uuid"] for x in hardware["devices"]), int(os.environ["H3_DIAG_PORT"])).result[0]
            ranks = [{"rank": 0, "pid": os.getpid(), "device": hardware["devices"][0]["pci"], "proof": "exact"},
                     {"rank": 1, "pid": plugin._RUNTIME.process.pid, "device": hardware["devices"][1]["pci"], "proof": "exact"}]
            if traced:
                b.write(b.OUTPUT / "rank-map.json", ranks)
            report["ranks"] = ranks
            counts = {"te": 0, "dit": 0}
            layer = [(n, m) for n, m in clip.cond_stage_model.named_modules() if n.endswith("layers.49")]
            assert len(layer) == 1
            hooks.append(layer[0][1].register_forward_hook(lambda *_: counts.__setitem__("te", counts["te"] + 1)))
            core = model.get_model_object("diffusion_model")
            core_started = []
            def core_start(*_):
                counts["dit"] += 1
                core_started.append(time.perf_counter_ns())
                b.ROCTX.roctxRangePushA(f"h3/forward/{PHASE}".encode())
            # Explicit hook closure avoids labeling host submission as GPU time.
            def core_end(*_):
                start = core_started.pop()
                b.ROCTX.roctxRangePop()
                SCOPES.append({"name": f"h3/forward/{PHASE}", "start_ns": start,
                               "duration_ns": time.perf_counter_ns() - start})
            hooks.extend((core.register_forward_pre_hook(core_start), core.register_forward_hook(core_end)))
            reference = b.load_file(str(b.ROOT / "archive/legacy-20261111/doc/9700-standard/c1/inputs.safetensors"))
            for role in ("warmup", "measure"):
                row = {"role": role, "stages": {}}
                before = dict(counts)
                before_attention = b.STATE["attention_calls"]
                before_swiglu = dict(b.SWIGLU_STATE)
                PHASE = f"{role}/text_encoding"
                start = time.perf_counter_ns()
                with mark(f"h3/{PHASE}"):
                    b.mm.unload_model_and_clones(model)
                    conditioning, latent = b.MiniMaxH3ImageToVideo.execute(
                        clip, None, b.GEN["prompt"], b.GEN["width"], b.GEN["height"], b.GEN["frames"]).result
                row["stages"]["fresh_te"] = (time.perf_counter_ns() - start) / 1e9
                PHASE = f"{role}/te_offload"
                start = time.perf_counter_ns()
                with mark(f"h3/{PHASE}"):
                    b.mm.unload_model_and_clones(clip.patcher)
                row["stages"]["te_offload"] = (time.perf_counter_ns() - start) / 1e9
                full_sigmas = b.BasicScheduler.execute(model, b.GEN["scheduler"], b.GEN["steps"], b.GEN["denoise"])[0]
                assert b.torch.equal(full_sigmas.cpu(), reference["sigmas"])
                noise = b.RandomNoise.execute(b.GEN["seed"])[0]
                generated = noise.generate_noise(latent)
                assert all(b.torch.equal(x.cpu(), reference[name]) for x, name in
                           zip(generated.unbind(), ("video_noise", "audio_noise")))
                PHASE = f"{role}/denoise_first_nfe"
                start = time.perf_counter_ns()
                if traced and role == "measure":
                    row["resume_status"] = b.ROCTX.roctxProfilerResume(0)
                with mark(f"h3/{PHASE}"):
                    sampled = b.SamplerCustomAdvanced.execute(
                        noise, b.BasicGuider.execute(model, conditioning)[0],
                        b.KSamplerSelect.execute(b.GEN["sampler"])[0], full_sigmas[:2].contiguous(), latent)[0]
                    b.torch.cuda.synchronize()
                if traced and role == "measure":
                    row["pause_status"] = b.ROCTX.roctxProfilerPause(0)
                row["stages"]["denoise_first_nfe"] = (time.perf_counter_ns() - start) / 1e9
                row["counts"] = {key: counts[key] - before[key] for key in counts}
                row["attention_calls"] = b.STATE["attention_calls"] - before_attention
                row["swiglu"] = {key: b.SWIGLU_STATE[key] - before_swiglu[key] for key in b.SWIGLU_STATE}
                assert row["counts"] == {"te": 1, "dit": 1} and row["attention_calls"] == 50
                assert row["swiglu"]["fused_calls" if b.SWIGLU_OPTIONS["fused_swiglu"] else "eager_calls"] == 250
                tensors = dict(zip(("video", "audio"), sampled["samples"].unbind()))
                assert all(bool(b.torch.isfinite(x).all()) for x in tensors.values())
                path = b.APP / f"{role}-latents.safetensors"
                b.save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(path))
                row.update(latents_sha256=b.sha(path), finite=True, C1_noise_and_sigmas=True)
                report["requests"].append(row)
                print("H3_AB_REQUEST", json.dumps(row), flush=True)
                b.write(b.APP / "result.json", report)
                del sampled, conditioning, latent, generated, noise
            report["status"] = "complete"
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        for hook in hooks:
            hook.remove()
        b.mm.load_models_gpu = original_load
        if plugin is not None and plugin._RUNTIME is not None:
            plugin._RUNTIME.shutdown()
        b.write(b.APP / "result.json", report)
        if traced:
            b.write(b.OUTPUT / "annotations.json", {
                "schema_version": 1, "run_id": b.OUTPUT.name, "source": "app/result.json",
                "scopes": [{"name": row["name"], "kind": "h3_stage", "rank": 0,
                            "time_basis": "cpu_submission", "clock_domain": "host_monotonic_ns",
                            "start_ns": row["start_ns"], "duration_ns": row["duration_ns"],
                            "association": "exact", "gpu_interval_verified": False,
                            "locator": f"app/result.json#scopes[{i}]"} for i, row in enumerate(SCOPES)]})


if __name__ == "__main__":
    main()
