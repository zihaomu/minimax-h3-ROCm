import argparse
import gc
import logging
import os
import sys
from datetime import timedelta

from minimax_h3_attention.runtime import configure, install_attention
configure(worker=True)

import torch
import torch.distributed as dist


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--weight-dtype", default="default")
    parser.add_argument("--comfy-root", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    sys.path.insert(0, args.comfy_root)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{args.port}",
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(minutes=40),
        device_id=torch.device("cuda", 0),
    )
    object_group = dist.new_group(
        backend="gloo",
        timeout=timedelta(minutes=40),
    )

    import comfy.model_management
    import comfy.sd

    from minimax_h3_amd_sp.model_prep import (
        model_options_for_dtype,
        prune_worker_model,
        shard_dit_attention,
    )

    from minimax_h3_amd_sp.parallel_blocks import run_blocks
    from minimax_h3_amd_sp.partition import SequencePartition
    from minimax_h3_amd_sp.peer_ops import receive_tensor, send_tensor

    install_attention(worker=True)
    patcher = comfy.sd.load_diffusion_model(
        args.model_path,
        model_options=model_options_for_dtype(args.weight_dtype),
    )
    dit = patcher.model.diffusion_model
    shard_dit_attention(dit, rank=args.rank, world_size=args.world_size)
    prune_worker_model(dit)
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    logging.info(
        "rank %d ready, %.2f GiB allocated",
        args.rank,
        torch.cuda.memory_allocated() / 2**30,
    )
    dist.barrier()

    profile = os.environ.get("AMD_H3_SP_PROFILE") == "1"
    while True:
        message = [None]
        dist.broadcast_object_list(message, src=0, group=object_group)
        metadata = message[0]
        if metadata is None or metadata.get("op") == "shutdown":
            break
        if metadata.get("op") != "forward":
            raise RuntimeError(f"unknown worker operation: {metadata}")
        if metadata.get("reset_cache"):
            gc.collect()
            torch.cuda.empty_cache()
            logging.info(
                "rank %d prompt reset, %.2f GiB allocated",
                args.rank,
                torch.cuda.memory_allocated() / 2**30,
            )

        partition = SequencePartition(
            metadata["total_rows"],
        )
        local_hidden = torch.empty(
            (partition.size(args.rank), metadata["hidden_width"]),
            dtype=getattr(torch, metadata["hidden_dtype"]),
            device="cuda",
        )
        receive_tensor(local_hidden, peer=0)
        time_embedding = torch.empty(
            metadata["time_shape"],
            dtype=getattr(torch, metadata["time_dtype"]),
            device="cuda",
        )
        rope = torch.empty(
            metadata["rope_shape"],
            dtype=getattr(torch, metadata["rope_dtype"]),
            device="cuda",
        )
        dist.broadcast(time_embedding, src=0)
        dist.broadcast(rope, src=0)

        with torch.inference_mode():
            local_hidden = run_blocks(
                dit.blocks,
                local_hidden,
                time_embedding,
                metadata["segments"],
                rope,
                metadata["total_rows"],
                rank=args.rank,
                transformer_options={},
                profile=profile,
            )
        send_tensor(local_hidden, peer=0)
        del local_hidden, time_embedding, rope

    import json
    from minimax_h3_attention.runtime import STATE
    from minimax_h3_amd_sp.fused_swiglu import STATE as SWIGLU_STATE
    STATE["swiglu"] = dict(SWIGLU_STATE)
    output_dir = os.environ.get("H3_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "runs"))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "worker-stats.json"), "w") as handle:
        json.dump(STATE, handle, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
