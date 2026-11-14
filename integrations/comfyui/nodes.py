import logging
import os
from pathlib import Path
import sys

# Modified for minimax-h3-ROCm: one canonical core package on both ranks.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

import comfy.sd
import folder_paths
from comfy_api.latest import ComfyExtension, io

from minimax_h3_amd_sp.model_prep import (
    model_options_for_dtype,
    shard_dit_attention,
)
from minimax_h3_amd_sp.runtime import DualGpuRuntime


_RUNTIME = None
_GIB = 1024**3


def _resolve_devices(value):
    if value and value.strip().lower() != "auto":
        devices = [item.strip() for item in value.split(",") if item.strip()]
    else:
        visible = (
            os.environ.get("ROCR_VISIBLE_DEVICES")
            or os.environ.get("HIP_VISIBLE_DEVICES")
            or os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        devices = (
            [item.strip() for item in visible.split(",")]
            if visible
            else ["0", "1"]
        )
    if len(devices) != 2:
        raise ValueError(f"expected exactly two devices, got {devices}")
    if "3" in devices:
        raise ValueError("Physical GPU3 is excluded in this workspace")
    if devices[0] == devices[1]:
        raise ValueError("Choose two distinct physical GPUs")
    return devices


def _reserve_activation_memory(model):
    reserve_gib = float(os.environ.get("AMD_H3_SP_ACTIVATION_RESERVE_GB", "8"))
    original = model.model.memory_required

    def memory_required(input_shape, cond_shapes=None):
        return original(
            input_shape,
            cond_shapes={} if cond_shapes is None else cond_shapes,
        ) + reserve_gib * _GIB

    model.model.memory_required = memory_required
    logging.info(
        "[amd-h3-sp] reserving %.1f GiB for sequence-parallel activations",
        reserve_gib,
    )
    return int(reserve_gib * _GIB)


class AMDMinimaxH3SequenceParallelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AMDMinimaxH3SequenceParallelLoader",
            display_name="MiniMax H3 AMD Dual-GPU Sequence Parallel",
            category="advanced/amd/multigpu",
            description=(
                "Runs one MiniMax H3 DiT forward across two GPUs using exact "
                "head and sequence parallelism."
                " Optional approximate Cache-DiT is disabled unless H3_CACHE_DIT=1."
            ),
            inputs=[
                io.Combo.Input(
                    "unet_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=[
                        "default",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                    ],
                ),
                io.String.Input(
                    "devices",
                    default="auto",
                    tooltip="Two physical GPU ids or UUIDs. auto uses visible-device environment variables.",
                ),
                io.Int.Input("port", default=29541, min=1024, max=65535),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        unet_name,
        weight_dtype,
        devices="auto",
        port=29541,
    ):
        global _RUNTIME
        from minimax_h3_attention.config import AttentionConfig
        from minimax_h3_attention.runtime import configure, install_attention
        AttentionConfig.from_environment()
        configure()
        install_attention()

        if torch.cuda.device_count() < 2:
            raise RuntimeError("two visible GPUs are required")
        resolved_devices = _resolve_devices(devices)
        model_path = folder_paths.get_full_path_or_raise(
            "diffusion_models",
            unet_name,
        )
        model = comfy.sd.load_diffusion_model(
            model_path,
            model_options=model_options_for_dtype(weight_dtype),
        )
        dit = model.get_model_object("diffusion_model")
        if dit.blocks[0].attn.heads % 2:
            raise ValueError("MiniMax H3 attention heads must be divisible by two")

        shard_dit_attention(dit, rank=0, world_size=2)
        activation_reserve = _reserve_activation_memory(model)
        runtime_key = (
            model_path,
            weight_dtype,
            tuple(resolved_devices),
            port,
        )
        if _RUNTIME is None:
            _RUNTIME = DualGpuRuntime(
                model_path,
                weight_dtype,
                resolved_devices,
                port,
            )
        else:
            active_key = (
                _RUNTIME.model_path,
                _RUNTIME.weight_dtype,
                _RUNTIME.devices,
                _RUNTIME.port,
            )
            if active_key != runtime_key:
                raise RuntimeError(
                    "a sequence-parallel runtime with different settings is active; "
                    "restart ComfyUI to change model, dtype, devices, or port"
                )
            _RUNTIME._check_worker()
            logging.info("[amd-h3-sp] reusing the active rank-1 worker")

        def distributed_forward(
            x,
            timestep,
            context,
            transformer_options=None,
            minimax_payload=None,
            **kwargs,
        ):
            del kwargs
            current_timestep = float(timestep.flatten()[0])
            new_prompt = (
                _RUNTIME.last_timestep is None
                or current_timestep >= _RUNTIME.last_timestep
            )
            if new_prompt:
                comfy.model_management.load_models_gpu(
                    [model],
                    memory_required=activation_reserve,
                    minimum_memory_required=activation_reserve,
                    force_full_load=True,
                )
            return _RUNTIME.forward(
                dit,
                x,
                timestep,
                context,
                transformer_options or {},
                minimax_payload,
            )

        model.add_object_patch("diffusion_model._forward", distributed_forward)
        if _RUNTIME.cache_dit is not None:
            from comfy.patcher_extension import WrappersMP

            def cache_sample(executor, *args, **kwargs):
                _RUNTIME.last_timestep = None
                try:
                    return executor(*args, **kwargs)
                finally:
                    _RUNTIME.finish_cache_request()

            model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "h3_cache_dit", cache_sample)
        logging.info(
            "[amd-h3-sp] model sharded across devices %s",
            resolved_devices,
        )
        return io.NodeOutput(model)


class AMDMinimaxH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [AMDMinimaxH3SequenceParallelLoader]


async def comfy_entrypoint():
    return AMDMinimaxH3Extension()
