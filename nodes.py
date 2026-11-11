"""Compatibility entry point for ComfyUI's source-extension loader."""
from .integrations.comfyui.nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
