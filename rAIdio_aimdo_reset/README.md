# rAIdio_aimdo_reset

A ComfyUI custom node that recovers GPU virtual-address-space
fragmentation by unloading models and re-initialising the
`comfy_aimdo` native library *in-process* — much faster than a
full ComfyUI restart.

## Endpoint

```
POST /aimdo_reset
→ 200 { "ok": true }
→ 5xx { "ok": false, "error": "<reason>" }
```

The handler:

1. Calls `comfy.model_management.unload_all_models()` so no model
   patch holds a live VBAR mapping.
2. Calls `comfy.model_management.soft_empty_cache(True)` to release
   the PyTorch CUDA cache.
3. Calls `comfy_aimdo.control.deinit()` then `init()` + `init_device(0)`
   to release every VBAR mapping the C library was tracking.

After this returns, the next model load gets a fresh, contiguous
virtual address range — the same outcome as a process restart at a
small fraction of the cost.

## Why

ComfyUI 0.14.x with `comfy_aimdo` allocates `model_size × 10` bytes
of contiguous virtual address space on every `partially_load → _vbar_get(create=True)`
call. Address space released on model eviction is not always
reclaimed cleanly, so after enough cumulative model-load cycles
the next `cudaMemAddressReserve` fails with:

```
aimdo: src/model-vbar.c:195:ERROR:Could not reserve Virtual Address space for VBAR
MemoryError: VBAR allocation failed
```

The HTTP listener stays up but every queued prompt errors out
silently, leaving users wondering why generation suddenly stopped
working. This node is the surgical fix.

rAIdio.bot calls this proactively every N successful generations
to pre-empt the failure, and reactively when the failure has
already fired.

## License

GPL-3.0 — matches the GPL-3.0 ecosystem this patches into.
