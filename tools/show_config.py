"""Display the exact selected backend without importing Torch or using GPUs."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minimax_h3_attention import AttentionConfig, BACKENDS

if __name__ == "__main__":
    selected = AttentionConfig.from_environment()
    print(json.dumps({"selected": selected.identity(), "configuration_sha256": selected.digest(),
                      "available": BACKENDS, "automatic_fallback": False,
                      "standard_timing_qualified": False}, indent=2))
