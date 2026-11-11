"""Print the current location of an old workspace path; never changes files."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "archive/legacy-20261111"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Old repository-relative or absolute path")
    args = parser.parse_args()
    path = Path(args.path)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            parser.error("The path must belong to this project")
    if ".." in path.parts or not path.parts:
        parser.error("Provide a project path without parent traversal")
    first, *rest = path.parts
    if first in ("scripts", "minimax_h3_multi_gpu", "outputs", "results", "logs",
                 "references", "build", ".worktrees", ".research", ".rocprofv3", "README.txt"):
        dest = LEGACY / first
    else:
        dest = {"doc": ROOT / "vibe", "docs": ROOT / "vibe/project", "examples": ROOT / "configs/workflows",
                "provenance": ROOT / "configs/provenance", "licenses": ROOT / "third_party/licenses",
                "minimax_h3_amd_sp": ROOT / "src/minimax_h3_amd_sp",
                "minimax_h3_attention": ROOT / "src/minimax_h3_attention",
                "engines": ROOT / ".runtime/engines"}.get(first, ROOT / first)
    target = dest.joinpath(*rest)
    print(target)
    return 0 if target.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
