"""One-NFE TP2 H3 diagnostic with project-owned input and output checks.

This is intentionally not a Standard timing run. RDNAScope only wraps this
command and records native events; the H3 project owns the model and oracle.
"""
import ctypes
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("RDNASCOPE_RUN_DIR") or os.environ["H3_DIAG_OUTPUT_DIR"])
APP = OUTPUT / "app"
APP.mkdir(parents=True, exist_ok=True)
(OUTPUT / "container-id").write_text(os.environ["HOSTNAME"])
CONFIG = json.loads((ROOT / "configs/standard.json").read_text())
GEN = CONFIG["generation"]

sys.path[:0] = [str(ROOT / ".runtime/engines/ComfyUI-standard"),
                str(ROOT / ".runtime/deps/standard-rocm10"), str(ROOT / "src")]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = ["h3-rdnascope-short", "--enable-triton-backend", "--disable-mmap",
            "--preview-method", "latent2rgb"]
import comfy.cli_args
from minimax_h3_attention.runtime import STATE, configure, source_identity
from minimax_h3_attention.config import AttentionConfig

configure(worker=False)
import torch
from minimax_h3_amd_sp.fused_swiglu import STATE as SWIGLU_STATE, OPTIONS as SWIGLU_OPTIONS
from safetensors.torch import load_file, save_file
import comfy.sd
import comfy.quant_ops
import comfy.ops
import comfy.model_management as mm
import folder_paths
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
from comfy_extras.nodes_custom_sampler import BasicScheduler, BasicGuider, RandomNoise, KSamplerSelect, SamplerCustomAdvanced

ROCTX = ctypes.CDLL("/opt/rocm/core-10.1/lib/libroctx64.so")
ROCTX.roctxRangePushA.argtypes = [ctypes.c_char_p]
ROCTX.roctxRangePushA.restype = ctypes.c_int
ROCTX.roctxRangePop.argtypes = []
ROCTX.roctxRangePop.restype = ctypes.c_int
SCOPES = []


@contextmanager
def mark(name):
    start = time.perf_counter_ns()
    ROCTX.roctxRangePushA(name.encode())
    try:
        yield
    finally:
        ROCTX.roctxRangePop()
        SCOPES.append({"name": name, "time_basis": "cpu_submission",
                       "start_monotonic_ns": start, "end_monotonic_ns": time.perf_counter_ns()})


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def device_identity():
    expected = json.loads(Path(os.environ["H3_DIAG_HARDWARE"]).read_text())
    assert torch.cuda.device_count() == 2
    hip = ctypes.CDLL("libamdhip64.so.7")
    for rank, item in enumerate(expected["devices"]):
        pci = ctypes.create_string_buffer(64)
        assert hip.hipDeviceGetPCIBusId(pci, 64, rank) == 0
        assert pci.value.decode().lower() == item["pci"]
    return expected


def main():
    hardware = device_identity()
    lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())["runtime"]
    assert torch.__version__ == lock["torch"] and torch.version.hip == lock["hip"]
    assert os.environ["DEEPINFER_IMAGE_ID"] == lock["container_image_id"]
    assert AttentionConfig.from_environment().backend == "reduced_error"
    os.environ["AMD_H3_SP_LOGDIR"] = str(APP)
    os.environ["H3_OUTPUT_DIR"] = str(APP)
    weights = ROOT / "models/Comfy-Org-MiniMax-H3"
    record = {"status": "running", "purpose": "one of ten Standard NFE, unchanged 124-frame input",
              "fc2_options": SWIGLU_OPTIONS,
              "source_identity": source_identity(AttentionConfig.from_environment()),
              "image_id": lock["container_image_id"], "torch": torch.__version__,
              "hip": torch.version.hip, "hardware": hardware,
              "standard_config_sha256": sha(ROOT / "configs/standard.json"),
              "selected_sigmas": None, "scopes": SCOPES}
    write(APP / "result.json", record)
    plugin = None
    hooks = []
    original_linear_input_act = None
    with torch.inference_mode():
        try:
            with mark("h3/short/load_models"):
                clip = comfy.sd.load_clip(
                    [str(weights / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")],
                    clip_type=comfy.sd.CLIPType.MINIMAX,
                    model_options={"dtype": torch.bfloat16,
                                   "load_device": torch.device("cuda", 0),
                                   "offload_device": torch.device("cpu"),
                                   "initial_device": torch.device("cpu")},
                    disable_dynamic=True)
                folder_paths.add_model_folder_path("diffusion_models", str(weights / "diffusion_models"))
                spec = importlib.util.spec_from_file_location("h3_rdnascope_plugin", ROOT / "integrations/comfyui/nodes.py")
                plugin = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plugin)
                devices = ",".join(d["uuid"] for d in hardware["devices"])
                model = plugin.AMDMinimaxH3SequenceParallelLoader.execute(
                    "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default",
                    devices, int(os.environ["H3_DIAG_PORT"])).result[0]
            rank_map = [{"rank": 0, "pid": os.getpid(), "device": hardware["devices"][0]["pci"], "proof": "exact"},
                        {"rank": 1, "pid": plugin._RUNTIME.process.pid,
                         "device": hardware["devices"][1]["pci"], "proof": "exact"}]
            if "RDNASCOPE_RUN_DIR" in os.environ:
                write(OUTPUT / "rank-map.json", rank_map)
            record["rank_map"] = rank_map
            counts = {"te": 0, "dit": 0}
            layer = [(n, m) for n, m in clip.cond_stage_model.named_modules() if n.endswith("layers.49")]
            assert len(layer) == 1
            hooks.append(layer[0][1].register_forward_hook(lambda *_: counts.__setitem__("te", counts["te"] + 1)))
            core = model.get_model_object("diffusion_model")
            hooks.append(core.register_forward_pre_hook(lambda *_: counts.__setitem__("dit", counts["dit"] + 1)))
            fixture_dir = os.environ.get("H3_GEMM_FIXTURE_DIR")
            fixture_sites = tuple(os.environ.get("H3_GEMM_FIXTURE_SITES", "fc1").split(","))
            fixture = {"pending": False, "captured": False}
            if fixture_dir and fixture_sites != ("fc1",):
                destination = Path(fixture_dir)
                destination.mkdir(parents=True, exist_ok=False)
                targets = {"qkv": core.blocks[13].attn.qkv_proj,
                           "out": core.blocks[13].attn.out_proj,
                           "fc2": core.blocks[13].mlp.fc2}
                assert set(fixture_sites) <= targets.keys()
                captured = set()
                pending = {}

                def prepare(site, x, module, input_act):
                    if site in captured or site in pending:
                        return
                    weight = module.weight
                    assert type(weight).__name__ == "QuantizedTensor"
                    qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
                    assert qdata.dtype == torch.int8 and scale.numel() == qdata.shape[0]
                    values = {"x": x, "weight": qdata, "weight_scale": scale}
                    if module.bias is not None:
                        values["bias"] = module.bias
                    pre_scale = getattr(module, "pre_quant_scale", None)
                    if pre_scale is not None:
                        values["pre_quant_scale"] = pre_scale
                    pending[site] = ({key: value.detach().cpu().contiguous()
                                      for key, value in values.items()},
                                     {"convrot": bool(getattr(weight._params, "convrot", False)),
                                      "convrot_groupsize": int(getattr(weight._params, "convrot_groupsize", 256)),
                                      "input_act": input_act})

                def finish(site, output):
                    if site not in pending:
                        return
                    values, metadata = pending.pop(site)
                    values["output"] = output.detach().cpu().contiguous()
                    folder = destination / site
                    folder.mkdir()
                    path = folder / f"block13-{site}.safetensors"
                    save_file(values, str(path))
                    write(folder / "fixture.json", {
                        "source": f"rank0 block13 {site} first call; real TP2 one-NFE input",
                        "site": site, "file": path.name, "sha256": sha(path),
                        "shapes": {key: list(value.shape) for key, value in values.items()},
                        "dtypes": {key: str(value.dtype) for key, value in values.items()},
                        **metadata, "out_dtype": str(output.dtype)})
                    captured.add(site)
                    fixture["captured"] = captured == set(fixture_sites)

                for site in ("qkv", "out"):
                    if site not in fixture_sites:
                        continue
                    target = targets[site]
                    hooks.append(target.register_forward_pre_hook(
                        lambda module, arguments, label=site: prepare(label, arguments[0], module, None)))
                    hooks.append(target.register_forward_hook(
                        lambda module, arguments, output, label=site: finish(label, output)))

                if "fc2" in fixture_sites:
                    original_linear_input_act = comfy.ops.linear_input_act

                    def capture_linear_input_act(linear, x, input_act):
                        if linear is targets["fc2"] and "fc2" not in captured:
                            prepare("fc2", x, linear, input_act)
                            result = original_linear_input_act(linear, x, input_act)
                            finish("fc2", result)
                            return result
                        return original_linear_input_act(linear, x, input_act)

                    comfy.ops.linear_input_act = capture_linear_input_act
            elif fixture_dir:
                destination = Path(fixture_dir)
                destination.mkdir(parents=True, exist_ok=False)
                target = core.blocks[13].mlp.fc1

                def before_fc1(module, arguments):
                    if fixture["captured"]:
                        return
                    x = arguments[0]
                    assert x.ndim == 2 and x.shape == (4096, 5376)
                    weight = module.weight
                    assert type(weight).__name__ == "QuantizedTensor"
                    qdata, scale = comfy.quant_ops.TensorWiseINT8Layout.get_plain_tensors(weight)
                    assert qdata.dtype == torch.int8 and qdata.shape == (28672, 5376)
                    values = {"x": x, "weight": qdata, "weight_scale": scale}
                    if module.bias is not None:
                        values["bias"] = module.bias
                    pre_scale = getattr(module, "pre_quant_scale", None)
                    if pre_scale is not None:
                        values["pre_quant_scale"] = pre_scale
                    fixture["values"] = {key: value.detach().cpu().contiguous()
                                         for key, value in values.items()}
                    fixture["convrot"] = bool(getattr(weight._params, "convrot", False))
                    fixture["convrot_groupsize"] = int(getattr(weight._params, "convrot_groupsize", 256))
                    fixture["pending"] = True

                def after_fc1(module, arguments, output):
                    if not fixture["pending"] or fixture["captured"]:
                        return
                    values = {**fixture["values"], "output": output.detach().cpu().contiguous()}
                    path = destination / "block13-fc1.safetensors"
                    save_file(values, str(path))
                    write(destination / "fixture.json", {
                        "source": "rank0 block13 mlp.fc1 first 4096-row chunk; real TP2 one-NFE input",
                        "file": path.name, "sha256": sha(path),
                        "shapes": {name: list(value.shape) for name, value in values.items()},
                        "dtypes": {name: str(value.dtype) for name, value in values.items()},
                        "convrot": fixture["convrot"],
                        "convrot_groupsize": fixture["convrot_groupsize"],
                        "input_act": None, "out_dtype": str(output.dtype)})
                    fixture["captured"] = True
                    fixture["pending"] = False
                    del fixture["values"]

                hooks.append(target.register_forward_pre_hook(before_fc1))
                hooks.append(target.register_forward_hook(after_fc1))
            with mark("h3/short/fresh_te"):
                conditioning, latent = MiniMaxH3ImageToVideo.execute(
                    clip, None, GEN["prompt"], GEN["width"], GEN["height"], GEN["frames"]).result
            mm.unload_model_and_clones(clip.patcher)
            sigmas_full = BasicScheduler.execute(model, GEN["scheduler"], GEN["steps"], GEN["denoise"])[0]
            sigmas = sigmas_full[:2].contiguous()
            reference = load_file(str(ROOT / "archive/legacy-20261111/doc/9700-standard/c1/inputs.safetensors"))
            assert torch.equal(sigmas_full.cpu(), reference["sigmas"])
            noise = RandomNoise.execute(GEN["seed"])[0]
            generated = noise.generate_noise(latent)
            assert all(torch.equal(x.cpu(), reference[name]) for x, name in
                       zip(generated.unbind(), ("video_noise", "audio_noise")))
            record["selected_sigmas"] = [float(x) for x in sigmas.cpu()]
            record["input_checks"] = {"prompt_sha256": hashlib.sha256(GEN["prompt"].encode()).hexdigest(),
                                      "full_10_step_sigmas_match_C1": True,
                                      "noise_matches_C1": True, "frames": GEN["frames"],
                                      "width": GEN["width"], "height": GEN["height"]}
            with mark("h3/short/denoise_one_nfe"):
                sampled = SamplerCustomAdvanced.execute(
                    noise, BasicGuider.execute(model, conditioning)[0],
                    KSamplerSelect.execute(GEN["sampler"])[0], sigmas, latent)[0]
                torch.cuda.synchronize()
            assert counts == {"te": 1, "dit": 1}, counts
            assert STATE["attention_calls"] == 50, STATE
            if fixture_dir:
                assert fixture["captured"], "block13 fc1 INT8 fixture was not captured"
            parts = dict(zip(("video", "audio"), sampled["samples"].unbind()))
            for value in parts.values():
                assert bool(torch.isfinite(value).all())
            result_path = APP / "sampled-latents.safetensors"
            save_file({name: value.detach().cpu().contiguous() for name, value in parts.items()}, str(result_path))
            record.update(status="complete", text_encoder_calls=counts["te"],
                          model_forwards=counts["dit"], attention_calls=STATE["attention_calls"],
                          sampled_latents_sha256=sha(result_path),
                          sampled_shapes={name: list(value.shape) for name, value in parts.items()},
                          finite=True, scopes=SCOPES)
            record["swiglu"] = dict(SWIGLU_STATE)
            if "RDNASCOPE_RUN_DIR" in os.environ:
                write(OUTPUT / "annotations.json", {
                    "schema_version": 1, "run_id": OUTPUT.name, "source": "app/result.json",
                    "scopes": [{"name": item["name"], "kind": "h3_stage", "rank": 0,
                                "time_basis": "cpu_submission", "clock_domain": "host_monotonic_ns",
                                "start_ns": item["start_monotonic_ns"],
                                "duration_ns": item["end_monotonic_ns"] - item["start_monotonic_ns"],
                                "association": "exact", "gpu_interval_verified": False,
                                "locator": f"app/result.json#scopes[{index}]"}
                               for index, item in enumerate(SCOPES)]})
            print("H3_SHORT_COMPLETE", json.dumps({"attention_calls": 50,
                  "latents_sha256": record["sampled_latents_sha256"]}), flush=True)
            write(APP / "result.json", record)
        finally:
            if original_linear_input_act is not None:
                comfy.ops.linear_input_act = original_linear_input_act
            for hook in hooks:
                hook.remove()
            if plugin is not None and plugin._RUNTIME is not None:
                plugin._RUNTIME.shutdown()
            write(APP / "result.json", record)


if __name__ == "__main__":
    main()
