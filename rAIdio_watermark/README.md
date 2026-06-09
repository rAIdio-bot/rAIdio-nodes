# rAIdio_watermark

ComfyUI custom nodes that embed and detect an in-signal neural audio watermark
(SilentCipher) on rAIdio.bot's AI-generated audio — the strip-resistant half of
a two-layer AI-provenance mark (the other layer is a C2PA cryptographic manifest).

The watermark runs **in ComfyUI's process** so the model shares ComfyUI's CUDA
context and memory allocator — it is loaded, offloaded and reclaimed by the same
system that manages the generator, rather than holding VRAM in a separate process.

## Nodes

- **`RaidioWatermarkEmbed`** (`AUDIO → AUDIO`) — idempotent embed; normalizes the
  watermarked output to 44.1 kHz (the watermark's native rate) so it survives.
- **`RaidioWatermarkDetect`** (`AUDIO → result`) — read-only detection.
- **`RaidioWatermarkFile`** (`path + mode`) — watermark/detect a WAV file in place;
  submitted by the app as a one-node prompt.

Each node returns its result under `ui.raidio_watermark` so the host can read the
method / SDR / confidence back from the prompt history.

## Licensing

GPL-3.0. These nodes import ComfyUI's API (`comfy.model_management`,
`comfy.model_patcher`) and are therefore a derivative of the GPL-3.0 ComfyUI
ecosystem. See `LICENSE`.

SilentCipher itself is a separate component: the package is MIT-licensed
(github.com/sony/silentcipher) and the model weights are MIT (sony/silentcipher
on Hugging Face). They are bundled with the node at build time and are **not**
included in this source repository — only rAIdio.bot's own orchestration code is.
