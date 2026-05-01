# zz_rAIdio RVC Infer Audio Fix

ComfyUI custom node that bridges the V1↔V3 audio shape mismatch around
ComfyUI-RVC's `RVC_Infer.inference`.

## What it does

ComfyUI 0.14.x ships V3 audio nodes (e.g. Qwen3 TTS, Voice Clone) that
output the modern AUDIO dict `{"waveform": tensor, "sample_rate": int}`.
ComfyUI-RVC's `RVC_Infer.inference` was written against the V1 audio
API where the audio input is a string filename. Its first line is:

```python
audio_path = folder_paths.get_annotated_filepath(audio)
```

When a V3-output node feeds RVC_Infer, this crashes immediately with:

```
AttributeError: 'dict' object has no attribute 'endswith'
```

(inside `folder_paths.annotated_filepath`, which calls
`name.endswith(...)`).

This patch wraps `RVC_Infer.inference` so that when `audio` is a V3
dict, the wrapper:

1. **On input**: writes the waveform to ComfyUI's input directory as a
   short-lived `.wav`, then hands the basename to the original
   inference. The original then resolves it via `get_annotated_filepath`
   as it always wanted to. The temp file is removed after inference
   returns.
2. **On output**: RVC_Infer returns `(out_file_path,)` — a V1
   string-path tuple. Downstream nodes (e.g. our `SaveAudioWAV`) expect
   a V3 dict. The wrapper reads the output file with `soundfile` and
   re-emits as `{"waveform": tensor (1, C, T), "sample_rate": int}`.

If `audio` is already a string, both bridges are no-ops and the
original inference runs unchanged.

## Load order

`zz_` prefix ensures ComfyUI's alphabetical custom-node loader runs
this after ComfyUI-RVC has registered `RVC_Infer`. The monkey-patch
needs the class to exist in `NODE_CLASS_MAPPINGS` when it applies; if
it runs before ComfyUI-RVC the patch logs a warning and skips.

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

Copyright (c) 2025-2026 Creative Mayhem UG
