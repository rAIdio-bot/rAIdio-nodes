"""Whisper STT word-timestamp patch for ComfyUI-QwenTTS.

Registers a replacement AILab_Qwen3TTSWhisperSTT node class that uses
faster-whisper with word_timestamps=True, returning full segment/word
timing JSON alongside plain text.

ComfyUI loads custom_nodes/ alphabetically. This package name sorts after
ComfyUI-QwenTTS, so our NODE_CLASS_MAPPINGS entry overwrites theirs.
The original FOSS file on disk is never modified.

Licensed under GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""
import json
import logging

import numpy as np
import torch

_WHISPER_CACHE = {}


def _resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_download_root():
    """Cache faster-whisper models in ComfyUI's models dir instead of ~/.cache."""
    import os
    import folder_paths
    models_dir = folder_paths.models_dir
    cache_dir = os.path.join(models_dir, "faster-whisper")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_model(model_size, device):
    compute = "float16" if device == "cuda" else "int8"
    key = (model_size, device, compute)
    model = _WHISPER_CACHE.get(key)
    if model is not None:
        return model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Please install it in your ComfyUI environment."
        )
    import os
    download_root = _get_download_root()
    # Check if model is pre-shipped as a local directory
    local_model_dir = os.path.join(download_root, f"faster-whisper-{model_size}")
    if os.path.isdir(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "model.bin")):
        model_path = local_model_dir
        print(
            f"[rAIdio.bot] Loading faster-whisper model from local: {model_path} "
            f"({compute} on {device})...",
            flush=True,
        )
    else:
        model_path = model_size  # Downloads from Systran HF repo
        print(
            f"[rAIdio.bot] Loading faster-whisper model: {model_size} "
            f"({compute} on {device}, cache: {download_root})...",
            flush=True,
        )
    model = WhisperModel(model_path, device=device, compute_type=compute,
                         download_root=download_root)
    print(f"[rAIdio.bot] faster-whisper model loaded: {model_size}", flush=True)
    _WHISPER_CACHE[key] = model
    return model


class Qwen3TTSWhisperSTT:
    """Drop-in replacement for ComfyUI-QwenTTS's Whisper STT node.

    Uses faster-whisper with word_timestamps=True and returns both plain
    text and a JSON string containing segments with word-level timing.
    """

    WHISPER_MODELS = [
        "tiny", "base", "small", "medium",
        "large", "large-v2", "large-v3", "large-v3-turbo",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to transcribe."}),
                "model_size": (
                    cls.WHISPER_MODELS,
                    {
                        "default": "small",
                        "tooltip": "Whisper model size. Larger = slower but more accurate.",
                    },
                ),
                "language": (
                    [
                        "auto", "en", "ja", "es", "pt", "ko", "zh", "fr", "de",
                        "it", "ru", "ar", "hi", "th", "vi", "id", "tr", "pl",
                        "uk", "nl", "sv",
                    ],
                    {"default": "auto", "tooltip": "Force language or auto-detect."},
                ),
            },
            "optional": {
                "task": (
                    ["transcribe", "translate"],
                    {
                        "default": "transcribe",
                        "tooltip": "transcribe = keep original language, "
                                   "translate = output English",
                    },
                ),
                "unload_models": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Unload cached models after transcription"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "timestamps")
    OUTPUT_NODE = True
    FUNCTION = "transcribe"
    CATEGORY = "\U0001f9ea AILab/\U0001f399\ufe0fQwenTTS"

    def transcribe(
        self,
        audio,
        model_size="large-v3-turbo",
        language="auto",
        task="transcribe",
        unload_models=False,
    ):
        device = _resolve_device()
        print(f"[Whisper STT] Using device: {device} (faster-whisper)", flush=True)
        model = _get_model(model_size, device)

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        # Convert to mono float32 numpy at 16 kHz
        wf = waveform.cpu().float()
        if wf.shape[0] > 1:
            wf = wf.mean(dim=0, keepdim=True)
        wf = wf.squeeze(0)
        if sample_rate != 16000:
            import torchaudio.functional as F
            wf = F.resample(wf, sample_rate, 16000)
        audio_np = wf.numpy().astype(np.float32)

        try:
            lang_arg = None if language == "auto" else language
            segments_iter, info = model.transcribe(
                audio_np,
                beam_size=5,
                language=lang_arg,
                task=task,
                word_timestamps=True,
            )

            if language == "auto":
                print(
                    f"[Whisper STT] Detected: {info.language} "
                    f"(conf: {info.language_probability:.2f})",
                    flush=True,
                )

            # Consume iterator and build output
            full_text_parts = []
            segments_out = []
            for seg in segments_iter:
                text = seg.text.strip()
                full_text_parts.append(text)
                words_out = []
                if seg.words:
                    for w in seg.words:
                        words_out.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                        })
                segments_out.append({
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": text,
                    "words": words_out,
                })

            full_text = " ".join(full_text_parts)
            timestamps_json = json.dumps(
                {
                    "text": full_text,
                    "language": info.language or "",
                    "segments": segments_out,
                },
                ensure_ascii=False,
            )
            return {
                "ui": {"text": [full_text], "timestamps": [timestamps_json]},
                "result": (full_text, timestamps_json),
            }
        finally:
            if unload_models:
                try:
                    _WHISPER_CACHE.clear()
                    import gc
                    gc.collect()
                    gc.collect()
                except Exception:
                    pass


NODE_CLASS_MAPPINGS = {
    "AILab_Qwen3TTSWhisperSTT": Qwen3TTSWhisperSTT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Qwen3TTSWhisperSTT": "Whisper STT (rAIdio patched)",
}

logging.info("[rAIdio] Whisper STT patch loaded — word timestamps enabled")
