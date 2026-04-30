"""rAIdio aimdo reset — recover GPU virtual-address-space fragmentation.

Exposes `POST /aimdo_reset` on ComfyUI's HTTP server. The handler
unloads all currently loaded models, clears the PyTorch CUDA cache,
then deinit / reinit the `comfy_aimdo` C library so its VBAR
(Virtual BAR / GPU virtual address) mappings get fully released.

Why this exists:
  ComfyUI 0.14.x with `comfy_aimdo` allocates `model_size * 10` bytes
  of contiguous virtual address space every time it stages a model
  for inference (via `partially_load → _vbar_get(create=True)`).
  Address space released by aimdo on model eviction is not always
  reclaimed cleanly, so after enough cumulative model-load cycles
  (~25 generations of mixed Music / Voice / Stem work on RTX 5090
  with 32 GB) the next `cudaMemAddressReserve` fails with
  `MemoryError: VBAR allocation failed`. The HTTP server stays
  alive but every queued prompt fails silently.

  Restarting Python clears the address space — but full restart
  costs ~30 s of model reload on the next generation. Calling
  `comfy_aimdo.control.deinit()` then `init(0)` achieves the same
  address-space reset *in-process*, dropping the cost to a few
  hundred ms. Models reload on the next generation as if after a
  fresh start.

Triggered proactively by the rAIdio.bot frontend every N successful
generations (currently 15) to pre-empt the failure, AND by users
clicking "Restart Backend" in the status bar after the failure
already fired.

License: GPL-3.0 (matches the patched ecosystem).
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""

import logging

from aiohttp import web
from server import PromptServer

# `comfy_aimdo.control` is the public Python entry to the underlying
# `aimdo.dll` / `aimdo.so` C library. Both `init` and `init_device`
# are exposed; calling them in this order rebuilds the library state
# and re-binds device 0 (the only GPU rAIdio.bot uses today —
# multi-GPU support would extend this loop).
try:
    import comfy_aimdo.control as aimdo_ctl
    AIMDO_AVAILABLE = True
except Exception as e:  # pragma: no cover — runtime-only path
    aimdo_ctl = None
    AIMDO_AVAILABLE = False
    logging.warning("[rAIdio aimdo_reset] comfy_aimdo not importable: %s", e)


@PromptServer.instance.routes.post("/aimdo_reset")
async def aimdo_reset(request):
    """Unload models + reset comfy_aimdo's VBAR address space."""
    if not AIMDO_AVAILABLE:
        return web.json_response(
            {"ok": False, "error": "comfy_aimdo not available in this build"},
            status=501,
        )

    try:
        # 1. Unload all model patches so no live reference holds a VBAR
        # mapping. Without this, deinit could leave dangling pointers
        # inside ComfyUI's model_management — next generation segfaults.
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache(True)

        # 2. Tear down + rebuild the aimdo native lib. This releases
        # all VBAR mappings to the CUDA driver, defragmenting the
        # virtual address space for the next model load.
        aimdo_ctl.deinit()
        if not aimdo_ctl.init():
            return web.json_response(
                {"ok": False, "error": "aimdo init failed after deinit"},
                status=500,
            )
        if not aimdo_ctl.init_device(0):
            logging.warning(
                "[rAIdio aimdo_reset] init_device(0) returned False; "
                "next model load may still struggle."
            )

        logging.info("[rAIdio aimdo_reset] aimdo reset complete; VBAR address space defragmented.")
        return web.json_response({"ok": True})

    except Exception as e:
        logging.error("[rAIdio aimdo_reset] handler raised: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=500)


# No node classes exposed — this is server-route-only.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
