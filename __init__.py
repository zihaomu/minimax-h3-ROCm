async def comfy_entrypoint():
    from .nodes import comfy_entrypoint as load_extension

    return await load_extension()
