"""rAIdio RVC LoadAudio alias — sidestep the V1/V3 LoadAudio collision.

ComfyUI 0.14.x ships a V3 `LoadAudio` in `comfy_extras/nodes_audio.py`
that returns the modern AUDIO dict `{"waveform": tensor, "sample_rate": int}`.
ComfyUI-RVC has its own V1 `LoadAudio` returning a string filename
tuple `(folder_paths.get_annotated_filepath(name),)`. Both register
under `"LoadAudio"` in `NODE_CLASS_MAPPINGS` (`nodes.py:2248`,
unconditional last-write-wins). When core's V3 wins the slot,
`RVC_Train.train` dies with `'dict' object has no attribute 'endswith'`
because it calls `folder_paths.get_annotated_filepath(audio)` on what
it expected to be a string filename.

This patch registers a NEW node, `RVCLoadAudio`, that takes a STRING
input and returns a string path tuple — exactly the shape RVC_Train
needs. Workflows that need that behaviour use `class_type: "RVCLoadAudio"`;
the original `"LoadAudio"` slot is left alone.

We don't try to import ComfyUI-RVC's V1 LoadAudio class itself —
ComfyUI-RVC's `nodes.py:9` does `from .rvc.train import ...`, a
relative import that requires the parent package context. Loading
that module via `importlib.util.spec_from_file_location` doesn't set
the package context, so the relative import fails. Sidestepped by
defining our own tiny shim — it's GPL-3.0 either way.

Licensed under GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""

import logging

import folder_paths


class RVCLoadAudio:
    """Minimal shim: takes a STRING filename, returns a string path tuple
    typed as ("AUDIO",) so it can wire into RVC_Train (which declares
    audio as ("AUDIO",) but actually expects a string filename inside
    its train() function)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "rAIdio"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        # `get_annotated_filepath` resolves "input/foo.wav" → absolute path
        # and handles the "[output]" / "[temp]" annotation suffixes ComfyUI
        # users sometimes write. RVC_Train.train calls the same function
        # again on whatever we hand it; passing the resolved string-path
        # through is the right shape for that downstream call.
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)


NODE_CLASS_MAPPINGS = {"RVCLoadAudio": RVCLoadAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"RVCLoadAudio": "Load Audio (RVC, string filename)"}

logging.info(
    "[rAIdio rvc_loadaudio_alias] registered RVCLoadAudio "
    "(STRING input → AUDIO string-path tuple) for RVC_Train workflows"
)
