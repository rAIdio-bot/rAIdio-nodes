# `_python_embeded_overrides/`

Files here are NOT ComfyUI custom nodes. They land in the embedded
Python interpreter's `site-packages/` directory, where
`site.py` auto-loads them at every interpreter startup.

## Install paths

| Source path here | Install path in shipped Backend |
|---|---|
| `site-packages/usercustomize.py` | `Backend/comfyui_portable/python_embeded/Lib/site-packages/usercustomize.py` |

The Backend depot push (`tools/prepare_depots.ps1` from the
`rAIdio.bot-rust` repo) copies the entire `Backend/` tree, so a
file dropped at the dev tree's install path is included
automatically.

## What's here

### `usercustomize.py`

Disables `torch.distributed.init_process_group` for single-rank
invocations. Pins `GLOO_SOCKET_IFNAME=lo` + `MASTER_ADDR=127.0.0.1`
as defence-in-depth.

ComfyUI-RVC's training script (`rvc/infer/modules/train/train.py`)
calls `init_process_group` unconditionally even for single-GPU
setups. The call would open a TCP socket on loopback for rank
coordination — unnecessary for us, alarming to users who don't
expect any network code path during AI training. Patching at the
interpreter level rather than at the RVC source means we never
edit third-party code, and the patch reaches every interpreter
including subprocess-spawned multiprocessing children.

License: GPL-3.0 (matches PyTorch's BSD-3-Clause and the broader
GPL-3.0 ecosystem this patches into).
