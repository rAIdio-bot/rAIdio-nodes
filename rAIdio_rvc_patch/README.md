# rAIdio RVC Patch

ComfyUI custom node that fixes ComfyUI-RVC's argparse crash.

## What it does

ComfyUI-RVC's `config.py` calls `argparse.ArgumentParser.parse_args()` which
crashes with `SystemExit` when ComfyUI passes its own CLI arguments (e.g.
`--listen`, `--output-directory`, `--fast`). This patch wraps `parse_args` to
catch `SystemExit` and fall back to `parse_known_args`, ignoring unrecognized
arguments.

Applied at import time when ComfyUI loads custom nodes.

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

Copyright (c) 2025-2026 Creative Mayhem UG
