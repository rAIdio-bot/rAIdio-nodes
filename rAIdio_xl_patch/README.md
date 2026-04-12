# rAIdio XL Patch

ComfyUI custom node that enables ACE-Step 1.5 XL model loading on ComfyUI 0.14.x.

## What it does

ComfyUI 0.14.x auto-detects ACE-Step 1.5 models with `hidden_size=2048`. The XL
variant uses `hidden_size=2560` for the decoder but `2048` for the encoder. This
node monkey-patches `AceStepConditionGenerationModel` at import time to accept
separate `encoder_hidden_size`, `encoder_num_heads`, and `encoder_intermediate_size`
parameters, then provides a `RaidioXLLoader` node that loads XL models with the
correct architecture.

The patch is forward-compatible — it checks if the parameters already exist (as in
ComfyUI 0.18+) and skips itself if so.

## Nodes

**RaidioXLLoader** — Drop-in replacement for UNETLoader when loading ACE-Step XL
models. Uses the same KSampler pipeline as the standard model.

## Model

Requires `acestep_v1.5_xl_turbo_bf16.safetensors` (9.3 GB) in
`ComfyUI/models/diffusion_models/`. Available from
[Comfy-Org/ace_step_1.5_ComfyUI_files](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files).

## License

GPL-3.0 — derivative work of [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

Copyright (c) 2025-2026 Creative Mayhem UG
