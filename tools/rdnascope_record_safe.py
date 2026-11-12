"""Consumer compatibility shim for the thin alpha's disappearing-file size race."""
from pathlib import Path
import runpy
import stat

import rdnascope.capture


def stable_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        try:
            info = path.lstat()
        except FileNotFoundError:
            # SQLite may delete a rollback journal between enumeration and stat.
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Producer created a symlink in the run directory")
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


if __name__ == "__main__":
    rdnascope.capture._bytes = stable_bytes
    runpy.run_module("rdnascope", run_name="__main__")
