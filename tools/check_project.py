"""CPU-only public-tree, provenance, import and documentation checks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = ("src", "integrations", "configs", "docs", "tools", "tests", "third_party")
PUBLIC_DOCS = {"docs/RESULTS.md", "docs/results.json"}
PRIVATE_DIRS = {"vibe", "doc", "archive", "models", ".runtime", "runs",
                "logs", "outputs", "results", "references", "build",
                ".cache", ".deps", ".tools", ".venv"}
PUBLIC_FILES = ("README.md", "LICENSE", "AGENTS.md", ".gitignore",
                "__init__.py", "nodes.py", "pyproject.toml", "requirements.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-git", action="store_true", help="For an exported source tree without .git")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from minimax_h3_attention import AttentionConfig, BACKENDS
    assert AttentionConfig().backend == "reduced_error" and len(BACKENDS) == 8
    assert "torch" not in sys.modules, "Structure checks must not import GPU frameworks"
    assert (ROOT / "docs").is_dir(), "Curated public results are required"
    if (ROOT / "vibe").exists():
        assert (ROOT / "vibe").is_dir() and not (ROOT / "vibe").is_symlink(), "The local vibe entry must be a real directory"
    provenance = json.loads((ROOT / "configs/provenance/upstreams.json").read_text())
    checked = 0
    for name, expected in provenance["vendor_files_sha256"].items():
        assert hashlib.sha256((ROOT / "third_party/sageattention_rdna4" / name).read_bytes()).hexdigest() == expected, name
        checked += 1
    for entry in provenance["derived_files"]:
        assert hashlib.sha256((ROOT / entry["destination"]).read_bytes()).hexdigest() == entry["destination_sha256"], entry["destination"]
        checked += 1
    parsed = 0
    for folder in ("src", "integrations", "tools", "tests"):
        for p in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(p.read_text(), filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert not module.startswith(("standard_opt", "optimization", "runtime_setup")), (p, module)
            parsed += 1
    for name in ("nodes.py", "__init__.py"):
        ast.parse((ROOT / name).read_text(), filename=name)
        parsed += 1
    links = 0
    for p in [ROOT / "README.md", ROOT / "third_party/README.md", *(ROOT / "docs").glob("*.md")]:
        for _, target in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", p.read_text()):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            resolved = (p.parent / target.split("#")[0]).resolve()
            assert resolved.exists(), (p, target)
            assert resolved.is_relative_to(ROOT), (p, target)
            assert not any(resolved.is_relative_to(ROOT / folder) for folder in PRIVATE_DIRS), (p, target)
            links += 1
    files = None
    if not args.skip_git:
        files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        files = [name for name in files if name]
        for name in files:
            assert name in PUBLIC_FILES or name.split("/")[0] in PUBLIC_DIRS, name
            if name.startswith("docs/"):
                assert name in PUBLIC_DOCS, f"Only curated result pages are public: {name}"
                assert (ROOT / name).stat().st_size < 64 * 1024, name
            p = ROOT / name
            assert not p.is_symlink(), f"Public source must not depend on local symlinks: {name}"
            assert p.suffix not in (".so", ".hsaco", ".gguf", ".safetensors", ".pt", ".pth", ".bin"), name
            assert p.stat().st_size < 10 * 1024**2, name
        for name in ("models/local-probe", "archive/local-probe", ".runtime/local-probe", "vibe/local-probe", "doc/local-probe"):
            r = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", name], cwd=ROOT)
            assert r.returncode == 0, name
        # A clean tip alone is insufficient: ancestors are uploaded by a normal push.
        commits = subprocess.check_output(["git", "rev-list", "HEAD"], cwd=ROOT, text=True).split()
        for commit in commits:
            historical = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True).splitlines()
            for name in historical:
                assert name.split("/")[0] not in PRIVATE_DIRS, f"Private path in public ancestry: {name}"
                assert Path(name).suffix not in (".so", ".hsaco", ".gguf", ".safetensors", ".pt", ".pth", ".bin"), name
                if name.startswith("docs/"):
                    assert name in PUBLIC_DOCS, f"Full research documents in public ancestry: {name}"
    print(json.dumps({"passed": True, "GPU_work": False, "backends": list(BACKENDS),
                      "source_hashes_checked": checked, "python_files_parsed": parsed,
                      "public_doc_links_checked": links, "tracked_public_files": len(files) if files else None,
                      "private_research_in_public_history": False if not args.skip_git else "not checked without Git",
                      "standard_timing_qualified": False}, indent=2))


if __name__ == "__main__":
    main()
