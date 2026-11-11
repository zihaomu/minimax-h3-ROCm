"""Print the current location of an old workspace path; never changes files."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_legacy_path(value, root=None):
    root = Path(root) if root is not None else ROOT
    legacy = root / "archive/legacy-20261111"
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            raise ValueError("The path must belong to this project") from None
    if ".." in path.parts or not path.parts:
        raise ValueError("Provide a project path without parent traversal")
    first, *rest = path.parts
    if first in ("scripts", "minimax_h3_multi_gpu", "outputs", "results", "logs",
                 "references", "build", ".worktrees", ".research", ".rocprofv3", "README.txt"):
        dest = legacy / first
    else:
        dest = {"doc": root / "vibe", "docs": root / "vibe/project", "examples": root / "configs/workflows",
                "provenance": root / "configs/provenance", "licenses": root / "third_party/licenses",
                "minimax_h3_amd_sp": root / "src/minimax_h3_amd_sp",
                "minimax_h3_attention": root / "src/minimax_h3_attention",
                "engines": root / ".runtime/engines"}.get(first, root / first)
    target = dest.joinpath(*rest)
    if not target.exists() and first in ("doc", "vibe"):
        # vibe is a curated working set; the complete historical tree is archived.
        historical = rest[1:] if rest and rest[0] == "history" else rest
        archived = (legacy / "doc").joinpath(*historical)
        if archived.exists():
            return archived
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Old repository-relative or absolute path")
    args = parser.parse_args()
    try:
        target = resolve_legacy_path(args.path)
    except ValueError as error:
        parser.error(str(error))
    print(target)
    return 0 if target.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
