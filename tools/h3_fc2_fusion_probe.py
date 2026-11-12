"""Isolated fc2 SwiGLU/ConvRot256/row-INT8 fusion gates on a real TP2 input."""
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
sys.argv = ["h3-fc2-fusion", "--enable-triton-backend", "--disable-mmap"]
import comfy.cli_args
import torch
from safetensors.torch import load_file
import triton
import triton.language as tl
from triton.language.extra import libdevice
from comfy_kitchen.backends.triton import quantization as tq
from comfy_kitchen.backends import hip as hq
from minimax_h3_attention.runtime import configure


@triton.jit
def _silu_bf16(gate, MODE: tl.constexpr):
    if MODE == 0:
        result = gate / (1.0 + tl.exp(-gate))
    elif MODE == 1:
        result = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    elif MODE == 2:
        result = gate * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    else:
        result = gate * (1.0 / (1.0 + tl.exp(-gate)))
    return result.to(tl.bfloat16)


@triton.jit
def _swiglu_values(X, Y, SIZE: tl.constexpr, K: tl.constexpr,
                    MODE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // K, offset % K
    gate = tl.load(X + row * (2 * K) + col, offset < SIZE, other=0).to(tl.float32)
    up = tl.load(X + row * (2 * K) + K + col, offset < SIZE, other=0).to(tl.float32)
    silu = _silu_bf16(gate, MODE)
    result = (silu.to(tl.float32) * up).to(tl.bfloat16)
    tl.store(Y + offset, result, offset < SIZE)


@triton.jit
def _swiglu_rot_groups(X, H, Y, M: tl.constexpr, K: tl.constexpr,
                       MODE: tl.constexpr, PREACT: tl.constexpr):
    group_id = tl.program_id(0) * 16 + tl.arange(0, 16)
    row = group_id // (K // 256)
    group = group_id % (K // 256)
    col = group[:, None] * 256 + tl.arange(0, 256)[None, :]
    if PREACT:
        activated = tl.load(X + row[:, None] * K + col, row[:, None] < M, other=0)
    else:
        gate = tl.load(X + row[:, None] * (2 * K) + col, row[:, None] < M, other=0).to(tl.float32)
        up = tl.load(X + row[:, None] * (2 * K) + K + col, row[:, None] < M, other=0).to(tl.float32)
        silu = _silu_bf16(gate, MODE)
        activated = (silu.to(tl.float32) * up).to(tl.bfloat16)
    hs = tl.arange(0, 256)
    outputs = tl.arange(0, 64)
    for part in range(4):
        h = tl.load(H + hs[:, None] * 256 + part * 64 + outputs[None, :])
        rotated = tl.dot(activated, h).to(tl.bfloat16)
        out_col = group[:, None] * 256 + part * 64 + outputs[None, :]
        tl.store(Y + row[:, None] * K + out_col, rotated, row[:, None] < M)


@triton.jit
def _swiglu_rot_quant_row(X, H, Scratch, Q, Scale,
                          K: tl.constexpr, GROUPS: tl.constexpr):
    """One CTA per row; BF16 boundaries mirror eager, scratch holds rotated BF16."""
    row = tl.program_id(0)
    groups = tl.arange(0, 16)
    keys = tl.arange(0, 256)
    hs = tl.arange(0, 256)
    outputs = tl.arange(0, 64)
    max_abs = tl.full((), 0, tl.float32)
    for tile in range(tl.cdiv(GROUPS, 16)):
        group = tile * 16 + groups[:, None]
        col = group * 256 + keys[None, :]
        valid = col < K
        gate = tl.load(X + row * (2 * K) + col, valid, other=0).to(tl.float32)
        up = tl.load(X + row * (2 * K) + K + col, valid, other=0).to(tl.float32)
        silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
        activated = (silu.to(tl.float32) * up).to(tl.bfloat16)
        for part in range(4):
            h = tl.load(H + hs[:, None] * 256 + part * 64 + outputs[None, :])
            rotated = tl.dot(activated, h).to(tl.bfloat16)
            out_col = group * 256 + part * 64 + outputs[None, :]
            mask = out_col < K
            rotated = tl.where(mask, rotated, 0)
            tl.store(Scratch + row * K + out_col, rotated, mask)
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(rotated).to(tl.float32)))
    scale = tl.maximum(max_abs / 127.0, 1e-30)
    tl.store(Scale + row, scale)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(K, 1024)):
        col = tile * 1024 + offsets
        rotated = tl.load(Scratch + row * K + col, col < K, other=0)
        quot = (rotated / scale.to(tl.bfloat16)).to(tl.bfloat16)
        quant = tl.clamp(libdevice.rint(quot.to(tl.float32)), -128.0, 127.0).to(tl.int8)
        tl.store(Q + row * K + col, quant, col < K)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def elapsed_ms(action, warmups=3, repeats=9):
    for _ in range(warmups):
        action()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        action()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def tensor_check(candidate, reference):
    delta = candidate.float() - reference.float()
    raw = candidate.contiguous().view(torch.uint8).reshape(candidate.numel(), candidate.element_size())
    ref_raw = reference.contiguous().view(torch.uint8).reshape(reference.numel(), reference.element_size())
    return {"exact": bool(torch.equal(raw, ref_raw)),
            "mismatches": int(torch.count_nonzero(candidate != reference)),
            "bitwise_mismatches": int(torch.count_nonzero(torch.any(raw != ref_raw, dim=1))),
            "relative_l2": float(torch.linalg.vector_norm(delta) /
                                 torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)),
            "max_abs": float(delta.abs().max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-only", action="store_true")
    parser.add_argument("--swiglu-only", action="store_true")
    args = parser.parse_args(CLI_ARGS)
    if args.parallel_only and args.swiglu_only:
        parser.error("choose one fusion variant")
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
    assert meta["site"] == "fc2" and meta["input_act"] == "swiglu"
    source = args.fixture / meta["file"]
    assert sha(source) == meta["sha256"]
    data = {key: value.to("cuda:0") for key, value in load_file(str(source)).items()}
    x, weight, weight_scale, expected = (data[key] for key in ("x", "weight", "weight_scale", "output"))
    assert x.shape == (4096, 28672) and weight.shape == (5376, 14336)
    assert meta["convrot"] and meta["convrot_groupsize"] == 256
    assert "pre_quant_scale" not in data
    k = weight.shape[1]
    h = tq._build_hadamard(256, device=x.device, dtype=x.dtype)
    bias = data.get("bias")

    def baseline_producer():
        activated = tq._apply_input_act(x, "swiglu")
        rotated = tq._rotate_activation(activated, h, 256)
        return tq.triton_quantize_rowwise(rotated)

    q_reference, s_reference = baseline_producer()

    def gemm(qdata, row_scale):
        output = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=expected.dtype)
        grid = lambda cfg: (triton.cdiv(x.shape[0], cfg["block_m"]) *
                            triton.cdiv(weight.shape[0], cfg["block_n"]),)
        tq._int8_matmul_dequant_per_row_kernel[grid](
            a_ptr=qdata, b_ptr=weight, c_ptr=output,
            a_scale_ptr=row_scale, b_scale_ptr=weight_scale.reshape(-1),
            bias_ptr=bias if bias is not None else x,
            m=x.shape[0], n=weight.shape[0], k=k,
            stride_am=qdata.stride(0), stride_ak=qdata.stride(1),
            stride_bk=weight.stride(1), stride_bn=weight.stride(0),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            has_bias=bias is not None)
        return output

    assert torch.equal(gemm(q_reference, s_reference), expected)
    report = {"fixture_sha256": meta["sha256"], "hardware": hardware,
              "image_id": os.environ["DEEPINFER_IMAGE_ID"],
              "torch": torch.__version__, "hip": torch.version.hip,
              "triton": triton.__version__, "trace_enabled": False,
              "baseline_producer": elapsed_ms(baseline_producer), "candidates": {}}
    report["oracle_sources"] = {str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / ".runtime/deps/standard-rocm10/comfy_kitchen/backends/triton/quantization.py",
        ROOT / ".runtime/deps/standard-rocm10/comfy_kitchen/backends/_activations.py",
        ROOT / ".runtime/deps/standard-rocm10/comfy_kitchen/tensor/int8_utils.py")}
    import comfy.quant_ops
    def baseline_full():
        return comfy.quant_ops.ck.int8_linear(
            x, weight, weight_scale, bias, expected.dtype, True, 256, "swiglu")
    report["baseline_full_A"] = elapsed_ms(baseline_full)

    def assess(name, action):
        entry = {}
        try:
            quant, scale = action()
            scale = scale.reshape(-1, 1)
            entry["qdata"] = tensor_check(quant, q_reference)
            entry["row_scale"] = tensor_check(scale, s_reference)
            output = gemm(quant, scale)
            entry["output"] = tensor_check(output, expected)
            entry["producer"] = elapsed_ms(action)
            entry["complete_fc2"] = elapsed_ms(lambda: gemm(*action()))
        except Exception as exc:
            entry["error"] = repr(exc)
        report["candidates"][name] = entry
        (args.output / "fusion.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("CANDIDATE", name, json.dumps(entry)[:800], flush=True)

    initial = not args.parallel_only and not args.swiglu_only
    if initial and hq._convrot_supported(k, 256, x.device, x.dtype):
        assess("hip_rotate_quant_after_eager_swiglu",
               lambda: hq._rotate_quant_int8(tq._apply_input_act(x, "swiglu"), 256))
    elif initial:
        report["candidates"]["hip_rotate_quant_after_eager_swiglu"] = {"status": "unsupported_shape"}

    def fused_three():
        scratch = torch.empty((x.shape[0], k), device=x.device, dtype=x.dtype)
        quant = torch.empty((x.shape[0], k), device=x.device, dtype=torch.int8)
        scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        _swiglu_rot_quant_row[(x.shape[0],)](x, h, scratch, quant, scale,
                                             K=k, GROUPS=k // 256, num_warps=4)
        return quant, scale

    if initial:
        assess("triton_single_kernel_three_stage", fused_three)
    elif args.parallel_only:
        activated_reference = tq._apply_input_act(x, "swiglu")
        rotated_reference = tq._rotate_activation(activated_reference, h, 256)
        groups = triton.cdiv(x.shape[0] * (k // 256), 16)

        def group_rotate(value, mode=1, preact=False):
            result = torch.empty((x.shape[0], k), device=x.device, dtype=x.dtype)
            _swiglu_rot_groups[(groups,)](value, h, result, M=x.shape[0], K=k,
                                          MODE=mode, PREACT=preact, num_warps=4)
            return result

        report["rotation_only_check"] = tensor_check(
            group_rotate(activated_reference, preact=True), rotated_reference)
        report["activation_checks"] = {}
        for mode in range(4):
            activated = torch.empty_like(activated_reference)
            _swiglu_values[(triton.cdiv(activated.numel(), 1024),)](
                x, activated, SIZE=activated.numel(), K=k, MODE=mode, BLOCK=1024)
            check = tensor_check(activated, activated_reference)
            report["activation_checks"][str(mode)] = check
            if not check["exact"]:
                continue
            rotated = group_rotate(x, mode=mode)
            report.setdefault("fused_rotation_checks", {})[str(mode)] = tensor_check(rotated, rotated_reference)
            assess(f"parallel_swiglu_rotation_then_row_quant_mode{mode}",
                   lambda selected=mode: tq.triton_quantize_rowwise(group_rotate(x, mode=selected)))
        report["baseline_producer_AA"] = elapsed_ms(baseline_producer)
    elif args.swiglu_only:
        def fused_activation(value=x, mode=0):
            width = value.shape[1] // 2
            activated = torch.empty((value.shape[0], width), device=value.device, dtype=value.dtype)
            _swiglu_values[(triton.cdiv(activated.numel(), 1024),)](
                value, activated, SIZE=activated.numel(), K=width, MODE=mode, BLOCK=1024)
            return activated

        report["activation_check"] = tensor_check(
            fused_activation(), tq._apply_input_act(x, "swiglu"))
        bits = torch.arange(65536, device=x.device, dtype=torch.int32).to(torch.int16)
        gates = bits.view(torch.bfloat16)
        gates = gates[torch.isfinite(gates)]
        multipliers = torch.tensor([1.0, -1.0, 0.125, 0.333984375],
                                   device=x.device, dtype=x.dtype)
        domain = torch.cat((gates.expand(4, -1),
                            multipliers[:, None].expand(-1, gates.numel())), dim=1).contiguous()
        domain_reference = tq._apply_input_act(domain, "swiglu")
        report["finite_bf16_gate_domain"] = {
            "finite_gate_count": gates.numel(), "up_multipliers": multipliers.tolist(), "modes": {}}
        for mode in range(4):
            actual = fused_activation(domain, mode=mode)
            report["finite_bf16_gate_domain"]["modes"][str(mode)] = {
                "exact": bool(torch.equal(actual.view(torch.int16), domain_reference.view(torch.int16))),
                "bitwise_mismatches": int(torch.count_nonzero(actual.view(torch.int16) != domain_reference.view(torch.int16)))}

        def partial_producer():
            activated = fused_activation()
            rotated = tq._rotate_activation(activated, h, 256)
            return tq.triton_quantize_rowwise(rotated)

        assess("fused_swiglu_original_rotation_quant", partial_producer)
        report["tail_rows"] = {}
        for rows in (1, 2509, 2510):
            value = x[:rows]
            ref_act = tq._apply_input_act(value, "swiglu")
            ref_rot = tq._rotate_activation(ref_act, h, 256)
            ref_q, ref_s = tq.triton_quantize_rowwise(ref_rot)
            act = fused_activation(value)
            rot = tq._rotate_activation(act, h, 256)
            quant, scale = tq.triton_quantize_rowwise(rot)
            report["tail_rows"][str(rows)] = {
                "activation_exact": bool(torch.equal(act.view(torch.int16), ref_act.view(torch.int16))),
                "qdata_exact": bool(torch.equal(quant, ref_q)),
                "scale_exact": bool(torch.equal(scale.view(torch.int32), ref_s.view(torch.int32)))}
        report["paired_full_runs"] = []
        for _ in range(3):
            report["paired_full_runs"].append({
                "A": elapsed_ms(baseline_full),
                "B": elapsed_ms(lambda: gemm(*partial_producer())),
                "AA": elapsed_ms(baseline_full)})
        report["baseline_producer_AA"] = elapsed_ms(baseline_producer)
    report["baseline_full_fc2"] = elapsed_ms(baseline_full)
    (args.output / "fusion.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("FUSION_PROBE_COMPLETE", args.output / "fusion.json", flush=True)


if __name__ == "__main__":
    main()
