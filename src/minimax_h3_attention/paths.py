"""Shared source-checkout paths. No framework imports or directory creation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
RUNTIME_ROOT = ROOT / ".runtime"
CACHE_ROOT = RUNTIME_ROOT / "cache"
CONFIG_ROOT = ROOT / "configs"
