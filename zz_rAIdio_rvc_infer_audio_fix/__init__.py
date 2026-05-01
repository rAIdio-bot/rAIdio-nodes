"""rAIdio RVC_Infer audio-shape fix.

ComfyUI-RVC's `RVC_Infer.inference()` was written for the V1 audio API
where the audio input is a string filename. Its first line is

    audio_path = folder_paths.get_annotated_filepath(audio)

ComfyUI 0.14.x core ships V3 audio nodes (Qwen TTS, voice clone, etc.)
that output the modern AUDIO dict `{waveform, sample_rate}`. Plugging
ANY V3-output node into RVC_Infer's audio input crashes immediately with

    AttributeError: 'dict' object has no attribute 'endswith'

inside `folder_paths.annotated_filepath` (which calls `name.endswith(...)`).

This patch wraps RVC_Infer.inference so that when audio is a dict it gets
saved to ComfyUI's input directory as a short-lived .wav and the basename
is handed to the original inference. The original then resolves it via
get_annotated_filepath as it always wanted to. The temp file is removed
after inference returns.

The zz_ prefix makes ComfyUI's alphabetical loader run this AFTER
ComfyUI-RVC so RVC_Infer is registered by the time we patch.

Licensed under GPL-3.0.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""

import logging
import os
import time

import numpy as np
import torch
from scipy.io.wavfile import write as wavwrite

import folder_paths


def _patch_rvc_infer():
    """Locate ComfyUI-RVC's RVC_Infer class via NODE_CLASS_MAPPINGS and
    swap its `inference` method for our wrapper. Idempotent."""
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except Exception as e:
        logging.warning("[rAIdio rvc_infer_audio_fix] cannot import NODE_CLASS_MAPPINGS: %r", e)
        return

    cls = NODE_CLASS_MAPPINGS.get("RVC_Infer")
    if cls is None:
        logging.warning("[rAIdio rvc_infer_audio_fix] RVC_Infer not registered yet")
        return

    orig = getattr(cls, "inference", None)
    if orig is None or getattr(orig, "_raidio_audio_patched", False):
        return

    def _patched_inference(self, audio, *args, **kwargs):
        # If audio is the V3 AUDIO dict (waveform+sample_rate), save it
        # to ComfyUI's input dir as a temp wav so the original inference
        # (which expects a string filename) can resolve it normally.
        temp_path = None
        if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
            wf = audio["waveform"]
            sr = int(audio["sample_rate"])

            if isinstance(wf, torch.Tensor):
                wf = wf.detach().cpu().numpy()
            wf = np.squeeze(np.asarray(wf))
            # scipy.wavwrite accepts (T,) mono or (T, C) multi-channel
            if wf.ndim > 1:
                wf = wf.T

            input_dir = folder_paths.get_input_directory()
            os.makedirs(input_dir, exist_ok=True)
            name = f"raidio_rvc_in_{int(time.time() * 1000)}.wav"
            temp_path = os.path.join(input_dir, name)

            # scipy.wavwrite expects integer or float PCM; float32 is fine
            wavwrite(temp_path, sr, wf.astype(np.float32))
            logging.info(
                "[rAIdio rvc_infer_audio_fix] V3 AUDIO dict bridged to %s "
                "(sr=%d, shape=%s)",
                name,
                sr,
                wf.shape,
            )
            audio = name  # original inference does get_annotated_filepath(audio)

        try:
            result = orig(self, audio, *args, **kwargs)
        except Exception:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        # cleanup temp input
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        # Bridge output back to V3 AUDIO dict if it's a string-path tuple.
        # RVC_Infer.inference returns `(out_file_path,)` (V1 shape) but
        # downstream SaveAudioWAV expects {waveform, sample_rate} (V3).
        if (
            isinstance(result, tuple)
            and len(result) == 1
            and isinstance(result[0], str)
            and os.path.exists(result[0])
        ):
            try:
                import soundfile as sf
                wav_np, sr_out = sf.read(result[0], always_2d=True)
                # soundfile gives (T, C); torch convention for AUDIO is (B, C, T)
                wav_t = torch.from_numpy(wav_np.T).float().unsqueeze(0)
                logging.info(
                    "[rAIdio rvc_infer_audio_fix] V1 string-path output "
                    "bridged to V3 AUDIO dict (sr=%d, shape=%s)",
                    sr_out,
                    tuple(wav_t.shape),
                )
                return ({"waveform": wav_t, "sample_rate": int(sr_out)},)
            except Exception as e:
                logging.warning(
                    "[rAIdio rvc_infer_audio_fix] output bridge failed: %r — "
                    "passing through original return", e,
                )
                # fall through to return original
        return result

    _patched_inference._raidio_audio_patched = True
    cls.inference = _patched_inference
    logging.info(
        "[rAIdio rvc_infer_audio_fix] RVC_Infer.inference patched to bridge V3 AUDIO dicts"
    )


_patch_rvc_infer()

# This file registers no new nodes; ComfyUI requires these at module level.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
