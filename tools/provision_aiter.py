"""Fetch pinned AITER PR sources and isolate their build dependencies without GPUs."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minimax_h3_attention.aiter.adapter import PINS  # noqa: E402

SOURCE = ROOT / ".runtime/sources/aiter-prs"
UPSTREAM = "https://github.com/ROCm/aiter.git"
DEPS = {
    "6390": ("aiter-build-6390", {"pybind11": "3.0.4", "ninja": "1.13.0"}),
    "4188": ("aiter-flydsl-0.4.0", {"flydsl": "0.4.0"}),
}


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sources():
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not (SOURCE / ".git").exists():
        SOURCE.mkdir()
        subprocess.run(["git", "init", "-q", str(SOURCE)], check=True)
        subprocess.run(["git", "-C", str(SOURCE), "remote", "add", "origin", UPSTREAM], check=True)
    assert command("git", "-C", str(SOURCE), "remote", "get-url", "origin") == UPSTREAM
    for pr, revision in PINS.items():
        destination = SOURCE.parent / f"aiter-{pr}"
        if not destination.exists():
            try:
                command("git", "-C", str(SOURCE), "cat-file", "-e", revision + "^{commit}")
            except subprocess.CalledProcessError:
                subprocess.run(["git", "-C", str(SOURCE), "-c", "protocol.version=2", "fetch",
                                "--filter=blob:none", "--depth=1", "origin", revision], check=True)
            subprocess.run(["git", "-C", str(SOURCE), "worktree", "add", "--detach",
                            str(destination), revision], check=True)
        assert command("git", "-C", str(destination), "rev-parse", "HEAD") == revision
        if command("git", "-C", str(destination), "status", "--porcelain"):
            raise RuntimeError(f"Pinned AITER source has local modifications: {destination}")


def installed(path, packages):
    if not path.exists():
        return False
    for name, version in packages.items():
        try:
            found = importlib.metadata.distribution(name, path=[str(path)]).version
        except TypeError:
            # Python 3.12 metadata API lacks a path parameter on distribution().
            found = next((d.version for d in importlib.metadata.distributions(path=[str(path)])
                          if d.metadata["Name"].lower() == name.lower()), None)
        if found != version:
            raise RuntimeError(f"Incomplete isolated dependency {name}: {found}, expected {version}")
    return True


def dependencies():
    lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())["runtime"]
    root = ROOT / ".runtime/deps"
    root.mkdir(parents=True, exist_ok=True)
    for _, (directory, packages) in DEPS.items():
        target = root / directory
        if installed(target, packages):
            continue
        with tempfile.TemporaryDirectory(prefix=f".{directory}-", dir=root) as temporary:
            command_line = ["docker", "run", "--rm", "--network", "bridge",
                            "--user", f"{os.getuid()}:{os.getgid()}",
                            "--mount", f"type=bind,src={ROOT.parent},dst={ROOT.parent}",
                            "--workdir", str(ROOT), "--env", "PIP_DISABLE_PIP_VERSION_CHECK=1",
                            lock["container_image_id"], "python", "-m", "pip", "install",
                            "--no-deps", "--target", temporary,
                            *[f"{name}=={version}" for name, version in packages.items()]]
            subprocess.run(command_line, check=True)
            if not installed(Path(temporary), packages):
                raise RuntimeError(f"Dependency installation incomplete: {packages}")
            Path(temporary).rename(target)


def main():
    sources()
    dependencies()
    record = {"source_commits": PINS, "dependencies": DEPS,
              "runtime_image": json.loads((ROOT / "configs/runtime-lock.json").read_text())["runtime"]["container_image_id"]}
    (ROOT / ".runtime/aiter-provision.json").write_text(json.dumps(record, indent=2) + "\n")
    print("AITER_SOURCES_AND_DEPS_READY", json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
