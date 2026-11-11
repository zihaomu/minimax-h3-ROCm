"""Read-only local model presence/size/header checks, no full tensors or GPUs."""
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "configs/model-manifest.json").read_text())
    results = []
    for record in manifest["files"]:
        path = ROOT / "models/Comfy-Org-MiniMax-H3" / record["file"]
        row = {"file": record["file"], "exists": path.is_file(), "passed": False}
        if path.is_file():
            row["bytes"] = path.stat().st_size
            with path.open("rb") as stream:
                count = struct.unpack("<Q", stream.read(8))[0]
                if not 0 < count < 16 * 1024**2:
                    raise ValueError(f"Invalid safetensors header size: {path}")
                header = json.loads(stream.read(count))
            row["tensors"] = len(header) - int("__metadata__" in header)
            row["passed"] = row["bytes"] == record["bytes"]
        results.append(row)
    passed = all(row["passed"] for row in results)
    print(json.dumps({"passed": passed, "checks": results,
                      "full_file_sha_recomputed": False, "GPU_work": False}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
