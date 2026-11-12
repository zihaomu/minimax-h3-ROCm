"""Trace-off, real TP2 block13 fc1 GEMM tile replay on locked ROCm 10.1."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
CLI_ARGS = sys.argv[1:]
sys.path[:0] = [str(ROOT / ".runtime/engines/ComfyUI-standard"),
                str(ROOT / ".runtime/deps/standard-rocm10"), str(ROOT / "src")]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = ["h3-gemm-replay", "--enable-triton-backend", "--disable-mmap"]
import comfy.cli_args
import torch
from safetensors.torch import load_file
import triton
from comfy_kitchen.backends.triton import quantization as q
from minimax_h3_attention.runtime import configure


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def elapsed_ms(action, warmups=3, repeats=9):
    for _ in range(warmups):
        action()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        action()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop))
    return {"samples_ms": values, "median_ms": statistics.median(values),
            "minimum_ms": min(values), "maximum_ms": max(values)}


def compare(actual, expected):
    delta = actual.float() - expected.float()
    return {"exact": bool(torch.equal(actual, expected)),
            "finite": bool(torch.isfinite(actual).all()),
            "relative_l2": float(torch.linalg.vector_norm(delta) /
                                 torch.linalg.vector_norm(expected.float())),
            "max_abs": float(delta.abs().max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(CLI_ARGS)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "container-id").write_text(os.environ["HOSTNAME"])
    configure(worker=False)
    hardware = json.loads(Path(os.environ["H3_DIAG_HARDWARE"]).read_text())
    assert torch.cuda.device_count() == 2
    hip = ctypes.CDLL("libamdhip64.so.7")
    pci = ctypes.create_string_buffer(64)
    assert hip.hipDeviceGetPCIBusId(pci, 64, 0) == 0
    assert pci.value.decode().lower() == hardware["devices"][0]["pci"]
    meta = json.loads((args.fixture / "fixture.json").read_text())
    source = args.fixture / meta["file"]
    assert sha(source) == meta["sha256"]
    values = {k: v.to("cuda:0") for k, v in load_file(str(source)).items()}
    x, w, scale, expected = (values[k] for k in ("x", "weight", "weight_scale", "output"))
    if "pre_quant_scale" in values:
        x = x * values["pre_quant_scale"].to(dtype=x.dtype)
    bias = values.get("bias")
    assert x.shape == (4096, 5376) and w.shape == (28672, 5376)
    assert w.dtype == torch.int8 and scale.numel() == w.shape[0]
    dtype = expected.dtype
    import comfy.quant_ops
    ck = comfy.quant_ops.ck
    assert not ck.registry.is_available("hip")

    def full():
        return ck.int8_linear(x, w, scale, bias, dtype,
                              meta["convrot"], meta["convrot_groupsize"],
                              meta["input_act"])

    baseline = full()
    baseline_check = compare(baseline, expected)
    assert baseline_check["exact"] and baseline_check["finite"], baseline_check
    baseline_a = elapsed_ms(full)

    x2d = q._apply_input_act(x, meta["input_act"]).reshape(-1, x.shape[-1])
    if meta["convrot"]:
        h = q._build_hadamard(meta["convrot_groupsize"], device=x.device, dtype=x.dtype)
        rotated = q._rotate_activation(x2d, h, meta["convrot_groupsize"])
        a, a_scale = q.triton_quantize_rowwise(rotated)
    else:
        a, a_scale = q.triton_quantize_rowwise(x2d)
    m, k = a.shape
    n = w.shape[0]
    output = torch.empty((m, n), device=x.device, dtype=dtype)
    autotuned = q._int8_matmul_dequant_per_row_kernel

    def invoke(kernel, config=None):
        kwargs = {} if config is None else config.kwargs
        bm, bn = kwargs.get("block_m", 128), kwargs.get("block_n", 128)
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),) if config else (
            lambda meta: (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),))
        params = dict(a_ptr=a, b_ptr=w, c_ptr=output, a_scale_ptr=a_scale,
                      b_scale_ptr=scale.reshape(-1), bias_ptr=bias if bias is not None else x,
                      m=m, n=n, k=k, stride_am=a.stride(0), stride_ak=a.stride(1),
                      stride_bk=w.stride(1), stride_bn=w.stride(0),
                      stride_cm=output.stride(0), stride_cn=output.stride(1),
                      has_bias=bias is not None)
        if config is None:
            return kernel[grid](**params)
        return kernel.fn[grid](**params, **kwargs,
                               num_warps=config.num_warps, num_stages=config.num_stages)

    invoke(autotuned)
    direct_check = compare(output, expected)
    assert direct_check["exact"], direct_check
    direct_a = elapsed_ms(lambda: invoke(autotuned))
    chosen = autotuned.best_config
    candidates = [
        triton.Config({"block_m": 64, "block_n": 256, "block_k": 64, "group_size_m": 8}, num_warps=4, num_stages=3),
        triton.Config({"block_m": 64, "block_n": 128, "block_k": 64, "group_size_m": 8}, num_warps=4, num_stages=3),
        triton.Config({"block_m": 128, "block_n": 128, "block_k": 64, "group_size_m": 8}, num_warps=8, num_stages=3),
        triton.Config({"block_m": 32, "block_n": 256, "block_k": 64, "group_size_m": 8}, num_warps=4, num_stages=3),
    ]
    rows = []
    for config in candidates:
        row = {"config": config.kwargs, "num_warps": config.num_warps,
               "num_stages": config.num_stages}
        try:
            invoke(autotuned, config)
            row["check"] = compare(output, expected)
            if row["check"]["exact"]:
                row["gpu_event"] = elapsed_ms(lambda: invoke(autotuned, config))
            else:
                row["status"] = "numerical_gate_failed"
        except Exception as exc:
            row.update(status="compile_or_launch_failed", error=repr(exc))
        rows.append(row)
        print("CANDIDATE", json.dumps(row), flush=True)
    direct_b = elapsed_ms(lambda: invoke(autotuned))
    baseline_b = elapsed_ms(full)
    report = {"fixture_sha256": meta["sha256"], "source": meta["source"],
              "hardware": hardware, "torch": torch.__version__, "hip": torch.version.hip,
              "triton": triton.__version__, "image_id": os.environ["DEEPINFER_IMAGE_ID"],
              "output_check": baseline_check, "direct_check": direct_check,
              "autotune_config": {"kwargs": chosen.kwargs, "num_warps": chosen.num_warps,
                                  "num_stages": chosen.num_stages},
              "full_linear_A": baseline_a, "full_linear_AA": baseline_b,
              "kernel_A": direct_a, "kernel_AA": direct_b,
              "candidates": rows, "timing_basis": "GPU event, same rank0 input, trace disabled"}
    (args.output / "replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("REPLAY_COMPLETE", args.output / "replay.json", flush=True)


if __name__ == "__main__":
    main()
