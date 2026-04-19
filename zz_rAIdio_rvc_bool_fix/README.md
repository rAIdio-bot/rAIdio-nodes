# zz_rAIdio RVC Bool Fix

ComfyUI custom node that fixes ComfyUI-RVC's boolean-to-i18n coercion bug
in `RVC_Train`.

## What it does

ComfyUI-RVC defines its train flags as BOOLEAN inputs (`if_save_latest`,
`if_cache_gpu`, `if_save_every_weights`) but then compares them to a
localized Chinese string inside `rvc/train.py`'s `click_train()`:

```python
1 if if_save_every_weights18 == i18n("是") else 0,
```

A Python `True` never equals the string `"是"` (or its English
translation), so the comparison is always `False` and `click_train`
always emits `-sw 0 -l 0 -c 0` regardless of what the caller requested.
The consequence is that no inference `.pth` ever lands in
`rvc/assets/weights/<name>.pth`, so the Custom Voice dropdown never
picks up a trained model. The bug also disables `if_save_latest` (no
latest checkpoint written) and `if_cache_gpu` (GPU caching always off).

This patch wraps `RVC_Train.train` and coerces incoming booleans to the
exact string `i18n("是")` returns in the current locale. The upstream
comparison then matches and the CLI flags are built correctly. No
third-party code is edited.

## Load order

This package is named `zz_...` so ComfyUI's alphabetical custom-node
loader runs it **after** ComfyUI-RVC has registered `RVC_Train`. The
monkey-patch needs the class to exist in `NODE_CLASS_MAPPINGS` at the
time it applies; if it runs before ComfyUI-RVC the patch logs a warning
and skips.

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

Copyright (c) 2025-2026 Creative Mayhem UG
