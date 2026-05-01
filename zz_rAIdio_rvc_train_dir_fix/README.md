# zz_rAIdio RVC Train Directory Fix

ComfyUI custom node that re-enables directory-mode training in
ComfyUI-RVC's `RVC_Train`.

## What it does

ComfyUI-RVC's `RVC_Train.INPUT_TYPES` only declares a single `audio`
input. The training body (`nodes.py:188`) does:

```python
shutil.copy(wav_path, os.path.join(trainset_dir, basename(wav_path)))
```

…and then runs `preprocess_dataset` over `trainset_dir`. So upstream
RVC only ever trains on **one** audio file, regardless of how much
voice data the user has. Our frontend supports a "Train from
directory" mode that uploads N audio files and submits the workflow
with `audio_files = "file1.wav;file2.wav;..."`. But RVC_Train doesn't
have an `audio_files` parameter, so ComfyUI silently drops it and
only one file makes it into the trainset. Result: tiny dataset,
low-quality voice models even when 10+ minutes of clean audio is
provided.

This patch:

1. **Extends `RVC_Train.INPUT_TYPES`** to add `audio_files` as an
   optional STRING input (default empty) — semicolon-separated list
   of basenames in ComfyUI's input directory.
2. **Wraps `RVC_Train.train`**. When `audio_files` is non-empty, it
   monkey-patches `shutil.copy` for the duration of the call so the
   FIRST copy inside `train()` (the trainset copy at `nodes.py:188`)
   ALSO copies every extra file to the same destination directory.
   `preprocess_dataset` then scans all of them naturally. Subsequent
   `shutil.copy` calls (final-weight copy, etc.) are unaffected
   because the wrapper restores after the first invocation.

If `audio_files` is empty (single-file mode), the patch is a
pass-through.

## Layering

This node loads alphabetically after `zz_rAIdio_rvc_bool_fix` and
wraps the already-wrapped `train()`. Final call chain:

```
ComfyUI → train_dir_fix.train(*args, **kw)
        → bool_fix.train(*args, **kw)      # bool coercion
        → orig.train(*args, **kw)           # real RVC_Train logic
                                            # (with shutil.copy patched)
```

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

Copyright (c) 2025-2026 Creative Mayhem UG
