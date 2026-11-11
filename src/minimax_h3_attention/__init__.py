"""Explicit MiniMax H3 attention backends; importing this package uses no GPU."""
from .config import AttentionConfig, BACKENDS, DEFAULT_BACKEND

__all__ = ["AttentionConfig", "BACKENDS", "DEFAULT_BACKEND"]
