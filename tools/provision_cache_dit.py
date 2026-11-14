"""Provision only optional pure-Python Cache-DiT wheels; never replace Torch."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".runtime/deps/cache-dit-1.5.3"
PACKAGES = ("cache-dit==1.5.3", "diffusers==0.40.0")
WHEEL_SHA256 = {
    "cache_dit-1.5.3-py3-none-any.whl": "f4564e68b5f6206d25712b06b4d9e0321ef96aadcd00e0ed4645be92d3fd9e75",
    "diffusers-0.40.0-py3-none-any.whl": "5b5da7c3ddb62152fa4afc577f02e050af688c797375c34fb2d01006da3f3541",
}


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
                    "--no-deps", "--dest", str(TARGET), *PACKAGES], check=True)
    wheels = []
    for pattern in ("cache_dit-1.5.3-*.whl", "diffusers-0.40.0-*.whl"):
        matches = list(TARGET.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one wheel for {pattern}, found {matches}")
        path = matches[0]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if WHEEL_SHA256.get(path.name) != digest:
            raise RuntimeError(f"Unexpected wheel hash for {path.name}: {digest}")
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not (TARGET / "site-packages" / member).resolve().is_relative_to((TARGET / "site-packages").resolve()):
                    raise RuntimeError("Unsafe wheel path")
            archive.extractall(TARGET / "site-packages")
        wheels.append({"file": path.name, "sha256": digest})
    record = {"packages": list(PACKAGES), "wheels": wheels,
              "Torch_Triton_ROCm_installed_or_modified": False,
              "scope": "Optional package provisioning; runtime/GPU/quality qualification is separate"}
    (TARGET / "provision.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
