"""rAIdio RVC LoadAudio alias — sidestep the V1/V3 LoadAudio collision.

ComfyUI core's `comfy_extras/nodes_audio.py:LoadAudio` (V3 IO.ComfyNode)
returns the modern AUDIO dict format `{"waveform": tensor, "sample_rate": int}`.
ComfyUI-RVC's `nodes.py:LoadAudio` (V1) returns a string filename tuple
`(folder_paths.get_annotated_filepath(name),)`.

Both register under the same `"LoadAudio"` name in `NODE_CLASS_MAPPINGS`.
ComfyUI's loader (`nodes.py:2248`) does an unconditional overwrite, so
"last wins" — and depending on load order or registration timing the
wrong class can end up bound to that name. The symptom is RVC_Train.train
receiving a dict where it expects a string filename, dying with
`'dict' object has no attribute 'endswith'` from
`folder_paths.get_annotated_filepath`.

This patch sidesteps the collision: it dynamically imports ComfyUI-RVC's
V1 LoadAudio by file path and re-registers it under a NEW name,
`RVCLoadAudio`. Workflows that need the V1 string-filename behaviour use
class_type `RVCLoadAudio`; everything else continues using `LoadAudio`
(and gets whichever flavour the load-order resolved). No third-party
code is edited; the original `LoadAudio` registration is untouched.

The `zz_` prefix makes ComfyUI's alphabetical loader run this AFTER
`ComfyUI-RVC` so the import-by-path can find the file. Without that,
the path-relative module locator might race against ComfyUI-RVC not
yet being on disk in fresh-install scenarios (rare, but cheap to
defend).

Licensed under GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""

import importlib.util
import logging
import os


def _register_rvc_loadaudio_alias():
    try:
        # Locate ComfyUI-RVC's nodes.py relative to our own __init__.py.
        # Custom-nodes dir layout:
        #   custom_nodes/
        #     ComfyUI-RVC/nodes.py
        #     zz_rAIdio_rvc_loadaudio_alias/__init__.py  <-- us
        custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rvc_nodes_path = os.path.join(custom_nodes_dir, "ComfyUI-RVC", "nodes.py")
        if not os.path.exists(rvc_nodes_path):
            logging.warning(
                "[rAIdio rvc_loadaudio_alias] ComfyUI-RVC/nodes.py not found at %s; "
                "RVCLoadAudio alias not registered. Voice training will not work.",
                rvc_nodes_path,
            )
            return

        # Hyphens in directory names break Python's normal `import ComfyUI-RVC`.
        # Use spec_from_file_location to bypass the module-name parser.
        spec = importlib.util.spec_from_file_location(
            "rAIdio_rvc_v1_loadaudio_module", rvc_nodes_path
        )
        if spec is None or spec.loader is None:
            logging.warning("[rAIdio rvc_loadaudio_alias] could not build import spec")
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rvc_load_audio = getattr(mod, "LoadAudio", None)
        if rvc_load_audio is None:
            logging.warning(
                "[rAIdio rvc_loadaudio_alias] ComfyUI-RVC/nodes.py has no LoadAudio class"
            )
            return

        # Register under a NEW name. Workflows can address it via
        # `"class_type": "RVCLoadAudio"`. The original `"LoadAudio"` slot
        # in NODE_CLASS_MAPPINGS is left untouched — other consumers
        # (audio editing nodes, etc.) that expect the modern AUDIO dict
        # format keep working.
        import nodes as _comfy_nodes

        _comfy_nodes.NODE_CLASS_MAPPINGS["RVCLoadAudio"] = rvc_load_audio
        if hasattr(_comfy_nodes, "NODE_DISPLAY_NAME_MAPPINGS"):
            _comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS["RVCLoadAudio"] = (
                "Load Audio (RVC, V1 string filename)"
            )
        logging.info(
            "[rAIdio rvc_loadaudio_alias] registered RVCLoadAudio "
            "(V1 string-filename loader) for RVC_Train workflows"
        )
    except Exception as e:
        logging.exception(
            "[rAIdio rvc_loadaudio_alias] alias registration failed: %s", e
        )


_register_rvc_loadaudio_alias()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
