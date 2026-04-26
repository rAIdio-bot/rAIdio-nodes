# rAIdio Safe Load Patch

ComfyUI custom node that closes the malicious-pickle-via-shared-model-file
attack vector across every `torch.load` call in the bundled backend.

## What it does

Wraps `torch.load` at custom-node import time with two layers:

1. **`weights_only=True` default.** PyTorch's safe-unpickle mode. Blocks
   `__reduce__`-based code execution at the pickle level. Callers can
   still pass `weights_only=False` explicitly when they genuinely need
   it (e.g. legacy checkpoints with custom classes), and that path is
   still gated by layer 2.
2. **`picklescan` gate.** Refuses loads whose pickle contains globals
   known to indicate code execution intent — `os.system`,
   `subprocess.*`, `builtins.eval`, `builtins.exec`, `pty.spawn`, etc.

If `picklescan` is not installed, the patch logs a warning and falls
back to `weights_only=True` alone — the bundled install never crashes
for lack of the optional defense-in-depth scanner.

The `00_` filename prefix forces ComfyUI to load this BEFORE any
node that touches `torch.load` at import time, so the rebind is in
place before the first call.

## Threat model

Defends against: a user downloads a "voice model.pth" from the
internet and drops it into rAIdio's RVC voice folder. Without this
patch, the next generation runs `torch.load(file)` and arbitrary
Python from the malicious pickle executes on the user's machine.
With this patch, `weights_only=True` blocks the deserialisation path
entirely, and `picklescan` provides a second-opinion refusal for
files the upstream library happened to load with `weights_only=False`.

Full Phase 2a/2b/2c scope:
<https://github.com/rAIdio-bot/sbom/blob/main/docs/security/model-loading.md>

## Compatibility

Designed against PyTorch ≥ 2.6 where `weights_only=True` is the safe
path for state-dict-only loads. Older torch versions silently accept
the kwarg as a no-op; the picklescan gate still applies, so the patch
remains useful in transitional environments.

Tested against the rAIdio.bot bundled Python (CPython 3.12) and the
RVC, SeedVC, QwenTTS, and ACE-Step custom nodes that account for the
23 `torch.load` sites in the bundled backend.

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

Copyright (c) 2026 Creative Mayhem UG
