#!/usr/bin/env python3
"""Memescreamer Memory Cleaner — POST /clear_memory route for ComfyUI.

First-party node by Creative Mayhem for the memescreamer platform
(rAIdio.bot / vAIdeo.bot).

License: GPL-3.0 (matches the patched ecosystem). This node imports ComfyUI's
GPL-3.0 API (`server.PromptServer`, `comfy.model_management`), so it is a
derivative work of the GPL-3.0 ComfyUI ecosystem and is licensed GPL-3.0 to
match — exactly like the sibling `rAIdio_aimdo_reset` node. See LICENSE.

WHY THIS EXISTS
  ComfyUI's `/free` endpoint only evicts models registered with
  `comfy.model_management` (e.g. ACE-Step). It cannot reach:
    1. `rAIdio_whisper_patch._WHISPER_CACHE` — a module-level dict that
       holds the faster-whisper model across Sing-Along transcriptions
       (cleared by the node itself only when unload_models=true, default
       false). A real persistent VRAM leak that survives `/free`.
    2. SeedVC (voice) / Demucs (stems) models, which load fresh into a
       local var each call and drop their Python ref on return — but the
       freed blocks stay in PyTorch's CUDA *caching allocator* (reserved,
       not returned to the driver), so the OS / Task Manager still shows
       them pinned until an explicit `empty_cache()`.

  On memory-tight cards (16 GB 5060 Ti) that residue is the difference
  between "GPU memory falls back to ~3 GB after a gen" and "pinned at
  15.5/16 GB, next render spills to system RAM and crawls." This route
  evicts the whisper cache, unloads managed models, then forces the
  allocator to return everything to the driver.

  Mirrors `rAIdio_aimdo_reset`'s server-route-only shape: no workflow
  node, just `POST /clear_memory`. Triggered by the rAIdio.bot backend
  (`free_vram`) after a generation on <24 GB cards. Big cards skip it to
  keep models warm.
"""

import gc
import logging
import os
import sys
import time

from aiohttp import web
from server import PromptServer

try:
    import torch
    _TORCH_OK = True
except Exception as e:  # pragma: no cover — runtime-only path
    torch = None
    _TORCH_OK = False
    logging.warning("[memescreamer_memory_cleaner] torch not importable: %s", e)


def _vram_gb():
    """(allocated_gb, reserved_gb) on the current CUDA device, or (0, 0)."""
    if not (_TORCH_OK and torch.cuda.is_available()):
        return (0.0, 0.0)
    return (
        torch.cuda.memory_allocated() / 1024 ** 3,
        torch.cuda.memory_reserved() / 1024 ** 3,
    )


def _evict_whisper_cache():
    """Clear the rAIdio whisper patch's module-level model cache.

    ComfyUI registers each directory custom node in `sys.modules` under its
    full filesystem path (`nodes.py::load_custom_node`), NOT its bare name —
    so `import rAIdio_whisper_patch` would bind a *fresh* module with its own
    empty `_WHISPER_CACHE`, freeing nothing. We must mutate the SAME module
    object ComfyUI loaded, so we scan the live `sys.modules` for the cache.

    Returns the number of cached whisper models evicted.
    """
    evicted = 0
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        cache = getattr(mod, "_WHISPER_CACHE", None)
        if isinstance(cache, dict) and cache:
            evicted += len(cache)
            cache.clear()
    return evicted


def _unload_managed_models():
    """Unload models registered with ComfyUI's model_management (e.g. ACE-Step)."""
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache(True)
        return True
    except Exception as e:
        logging.warning("[memescreamer_memory_cleaner] unload_all_models failed: %s", e)
        return False


def _trim_working_set():
    """Windows-only: return trimmed pages to the OS so Task Manager reflects the
    freed VRAM-staging host memory. No-op / best-effort elsewhere."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetCurrentProcess()
        kernel32.SetProcessWorkingSetSize(h, -1, -1)
        try:
            ctypes.windll.psapi.EmptyWorkingSet(h)
        except Exception:
            pass
    except Exception as e:  # pragma: no cover — platform-dependent
        logging.debug("[memescreamer_memory_cleaner] working-set trim skipped: %s", e)


def clear_memory():
    """Evict rAIdio node caches + managed models, then return all freed CUDA
    blocks to the driver. Returns a dict of before/after VRAM and what was
    evicted (also used as the HTTP response body)."""
    start = time.time()
    before_alloc, before_res = _vram_gb()

    # 1. Evict the one true persistent strong ref (whisper) so its tensors
    #    become free for the allocator to reclaim.
    whisper_evicted = _evict_whisper_cache()

    # 2. Unload model_management-tracked models (belt-and-suspenders; /free
    #    already does this, but the route may be called standalone).
    _unload_managed_models()

    # 3. Drop dangling Python refs (SeedVC / Demucs locals already out of
    #    scope after the node returned; this collects their cycles).
    for _ in range(4):
        gc.collect()

    # 4. Return freed-but-reserved blocks to the CUDA driver. This is the step
    #    that makes Task Manager's "Dedicated GPU memory" actually fall on the
    #    non-model-managed (SeedVC / Demucs) residue.
    if _TORCH_OK and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception as e:
            logging.warning("[memescreamer_memory_cleaner] cuda cleanup partial: %s", e)

    # 5. Hand trimmed host pages back to the OS (Windows working set).
    _trim_working_set()

    after_alloc, after_res = _vram_gb()
    elapsed = time.time() - start
    result = {
        "ok": True,
        "whisper_evicted": whisper_evicted,
        "vram_allocated_gb": {"before": round(before_alloc, 3), "after": round(after_alloc, 3)},
        "vram_reserved_gb": {"before": round(before_res, 3), "after": round(after_res, 3)},
        "freed_reserved_gb": round(before_res - after_res, 3),
        "elapsed_s": round(elapsed, 3),
    }
    logging.info(
        "[memescreamer_memory_cleaner] cleared: whisper=%d, reserved %.2f->%.2f GB "
        "(freed %.2f GB) in %.3fs",
        whisper_evicted, before_res, after_res, before_res - after_res, elapsed,
    )
    return result


@PromptServer.instance.routes.post("/clear_memory")
async def clear_memory_route(request):
    """Evict node caches + managed models + return CUDA blocks to the driver."""
    try:
        return web.json_response(clear_memory())
    except Exception as e:
        logging.error("[memescreamer_memory_cleaner] handler raised: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=500)


# Server-route-only — no workflow node classes exposed.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

logging.info("[memescreamer_memory_cleaner] POST /clear_memory route registered")
