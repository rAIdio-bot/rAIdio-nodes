# `_python_embeded_overrides/`

Files here are NOT ComfyUI custom nodes. They land in the embedded
Python interpreter's `site-packages/` directory, where Python's
`site.py` auto-loads them at every interpreter startup — including
the multiprocessing children that RVC's training subprocess spawns
on Windows.

## Install paths

| Source path here | Install path in shipped Backend |
|---|---|
| `site-packages/sitecustomize.py` | `Backend/comfyui_portable/python_embeded/Lib/site-packages/sitecustomize.py` |
| `site-packages/fairseq/` | `Backend/comfyui_portable/python_embeded/Lib/site-packages/fairseq/` |

The release pipeline's `steam/prepare_depots.ps1` copies the entire
`Backend/` tree, so a file dropped at the dev tree's install path is
included automatically.

## What's here

### `sitecustomize.py`

Three runtime patches to PyTorch + matplotlib that re-enable
single-rank RVC training on Windows. All three use a
`MetaPathFinder` so torch is **never imported at sitecustomize
load time** (an earlier version that eagerly imported torch broke
ComfyUI's argparse pipeline; see the [previous file's git
history](https://github.com/rAIdio-bot/rAIdio-nodes/commit/c698bdf)).

1. **`torch.distributed.init_process_group` no-op for `world_size ≤ 1`.**
   ComfyUI-RVC's training script calls `init_process_group` even for
   single-GPU setups. Gloo on Windows can't construct a `Device` for
   single-rank — neither hostname nor interface lookup works — so it
   crashes with `makeDeviceForHostname(): unsupported gloo device`.
   Skipping the call is correct (single rank has nothing to coordinate)
   and lets training continue.
2. **`torch.nn.parallel.DistributedDataParallel` passthrough for
   single rank.** Subclass with `__new__` that returns the bare model
   when `world_size ≤ 1`. Multi-rank case still uses real DDP. This
   preserves `class X(DistributedDataParallel)` declarations elsewhere
   in torch (e.g. `DistributedDataParallelCPU`).
3. **`matplotlib.backends.backend_agg.FigureCanvasAgg.tostring_rgb`
   restored.** matplotlib 3.8+ removed this method but ComfyUI-RVC's
   `utils.py:238` still calls it during per-epoch spectrogram
   logging. We re-implement it as a thin wrapper around
   `buffer_rgba()`.

Renamed from `usercustomize.py` because ComfyUI launches python with
`-s` (which disables user-site loading); `sitecustomize.py` still
loads under `-s`.

License: GPL-3.0 (matches PyTorch's BSD-3-Clause and the broader
GPL-3.0 ecosystem this patches into).

### `fairseq/`

Replacement package for the always-raise stub that previously stood
in for the unbuildable real fairseq (the embedded Python distribution
has no `Python.h` and no MSVC build environment, so
`pip install fairseq` doesn't work).

ComfyUI-RVC's `extract_feature_print.py:89` calls
`fairseq.checkpoint_utils.load_model_ensemble_and_task([hubert_base.pt])`
to load HuBERT for feature extraction during voice training.

This package re-implements that surface on top of
`transformers.HubertModel` (which IS installable in the embed):

- `checkpoint_utils.load_model_ensemble_and_task(paths)` loads the
  fairseq HuBERT-base checkpoint, remaps fairseq state-dict keys to
  HuggingFace `HubertModel` keys (the well-known mapping from
  HuggingFace's official conversion script), constructs a
  `HubertConfig` matching the checkpoint, loads the weights, and
  wraps in a `_HubertFairseqWrapper` that exposes the fairseq-style
  `extract_features(source, padding_mask, output_layer)` API
  ComfyUI-RVC expects.
- `data/dictionary.py`, `dataclass/configs.py`,
  `tasks/hubert_pretraining.py`, `models/hubert/hubert.py`,
  `modules/grad_multiply.py` — minimal `_StubBase` shells whose only
  job is to let `torch.load(weights_only=False)` resolve fairseq
  class names referenced inside the checkpoint pickle. None of
  those classes are actually instantiated for inference; the
  unpickler just needs to find them by name.

License: GPL-3.0 — matches fairseq upstream and the GPL-3.0 PyTorch
ecosystem this replaces into.
