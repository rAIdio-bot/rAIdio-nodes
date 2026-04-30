# zz_rAIdio_rvc_loadaudio_alias

Sidesteps the ComfyUI core / ComfyUI-RVC `LoadAudio` name collision.

## The problem

Both `comfy_extras/nodes_audio.py` (ComfyUI core, V3) and
`custom_nodes/ComfyUI-RVC/nodes.py` (V1) declare `class LoadAudio`
and register it under `"LoadAudio"` in `NODE_CLASS_MAPPINGS`.
ComfyUI's loader uses unconditional overwrite (`nodes.py:2248`),
so whichever class registers last wins.

The two implementations are NOT compatible:

- **Core V3** returns the modern AUDIO dict
  `{"waveform": tensor, "sample_rate": int}`.
- **ComfyUI-RVC V1** returns a string filename tuple
  `(folder_paths.get_annotated_filepath(name),)`.

`RVC_Train.train` calls `folder_paths.get_annotated_filepath(audio)`
on its `audio` input, which only works when `audio` is a string. If
core's V3 wins the registration, RVC_Train dies with:

```
AttributeError: 'dict' object has no attribute 'endswith'
```

## The fix

This patch dynamically imports ComfyUI-RVC's V1 `LoadAudio` and
registers it under a separate name, `RVCLoadAudio`. Workflows that
need the V1 string-filename behaviour set `"class_type": "RVCLoadAudio"`
instead of `"LoadAudio"`. The original `"LoadAudio"` slot is left
untouched — workflows that want the modern AUDIO dict still get it.

## Usage

In `workflows/rvc_train.json`:

```json
{
  "1": {
    "class_type": "RVCLoadAudio",
    "inputs": { "audio": "{{AUDIO_PATH}}" }
  },
  "2": {
    "class_type": "RVC_Train",
    "inputs": { "audio": ["1", 0], "...": "..." }
  }
}
```

## Why a `zz_` prefix

ComfyUI loads custom nodes alphabetically. `zz_` ensures this patch
runs after `ComfyUI-RVC/__init__.py` so the file-path-based import
of `ComfyUI-RVC/nodes.py` succeeds.

## License

GPL-3.0 — matches the GPL-3.0 ecosystem this patches into.
