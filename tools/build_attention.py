"""Build isolated attention extensions; never runs a GPU workload."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("ix", "native-tile"))
    args = parser.parse_args()
    import torch
    lock = json.loads((Path(__file__).resolve().parents[1] / "configs/runtime-lock.json").read_text())["runtime"]
    if (torch.__version__ != lock["torch"] or torch.version.hip != lock["hip"]
            or importlib.metadata.version("triton") != lock["triton"]
            or os.environ.get("DEEPINFER_IMAGE_ID") != lock["container_image_id"]):
        raise RuntimeError("Build only in the locked ROCm10.1 Torch/Triton container")
    if torch.cuda.is_available():
        raise RuntimeError("Build with no GPU devices mounted; no-GPU compilation is required")
    if args.target == "ix":
        from minimax_h3_attention.ix.build import main as build
    else:
        from minimax_h3_attention.native_tile.build import main as build
    build()


if __name__ == "__main__":
    main()
