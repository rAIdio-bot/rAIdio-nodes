# rAIdio torchaudio Patch

ComfyUI custom node that fixes torchaudio.save() for environments without
FFmpeg shared libraries.

## What it does

torchaudio 2.10.0 delegates `save()` to torchcodec, which requires FFmpeg
shared DLLs (avcodec, avformat, etc.) on the system PATH. Static FFmpeg
builds (like ours) don't include these DLLs.

This patch replaces `torchaudio.save()` with a `soundfile.write()`
implementation at import time. soundfile uses libsndfile which is bundled
and works without FFmpeg.

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

2025-2026 Creative Mayhem UG
