"""torchaudio.save compatibility patch.

torchaudio 2.10.0 delegates save() to torchcodec which requires FFmpeg shared
DLLs. This monkey-patch replaces torchaudio.save with a soundfile-based
implementation that works without torchcodec.

Licensed under GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""
import logging

def _apply_torchaudio_save_patch():
    try:
        import torchaudio
        import soundfile as sf
        import numpy as np

        _orig_save = torchaudio.save

        def _patched_save(filepath, src, sample_rate, *args, **kwargs):
            try:
                # Convert torch tensor to numpy for soundfile
                if hasattr(src, 'numpy'):
                    audio_np = src.cpu().numpy()
                else:
                    audio_np = np.array(src)
                # soundfile expects (samples, channels), torchaudio gives (channels, samples)
                if audio_np.ndim == 2:
                    audio_np = audio_np.T
                sf.write(str(filepath), audio_np, sample_rate, subtype='PCM_16')
            except Exception:
                # Fall back to original if soundfile fails
                _orig_save(filepath, src, sample_rate, *args, **kwargs)

        torchaudio.save = _patched_save
        logging.info("[rAIdio] torchaudio.save patched to use soundfile (bypasses torchcodec)")
    except Exception as e:
        logging.warning("[rAIdio] torchaudio.save patch failed (non-fatal): %s", e)

_apply_torchaudio_save_patch()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
