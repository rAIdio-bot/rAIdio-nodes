"""Chord detection and MIDI transcription nodes for rAIdio.bot.

Uses librosa (already installed) for audio analysis and midiutil for MIDI export.
No tensorflow or large ML models required.
"""

import json
import os

import folder_paths
import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Chord templates: 12 pitch classes x 9 chord types = 108 templates
# Each template is a 12-element binary vector (C, C#, D, ..., B)
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Intervals relative to root (semitones)
_CHORD_TYPES = {
    "":     [0, 4, 7],           # major
    "m":    [0, 3, 7],           # minor
    "dim":  [0, 3, 6],           # diminished
    "aug":  [0, 4, 8],           # augmented
    "7":    [0, 4, 7, 10],       # dominant 7th
    "m7":   [0, 3, 7, 10],      # minor 7th
    "maj7": [0, 4, 7, 11],      # major 7th
    "sus2": [0, 2, 7],          # suspended 2nd
    "sus4": [0, 5, 7],          # suspended 4th
}


def _build_templates():
    """Build all chord template vectors and their labels."""
    templates = []
    labels = []
    for root_idx, root_name in enumerate(_NOTE_NAMES):
        for suffix, intervals in _CHORD_TYPES.items():
            vec = np.zeros(12, dtype=np.float32)
            for interval in intervals:
                vec[(root_idx + interval) % 12] = 1.0
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            templates.append(vec)
            labels.append(f"{root_name}{suffix}")
    return np.array(templates), labels


_TEMPLATES, _TEMPLATE_LABELS = _build_templates()


def _detect_chords(audio_np, sr, hop_length=4096, min_duration=0.5,
                   confidence_threshold=0.6):
    """Detect chord progression from audio using chroma features.

    Returns list of dicts: [{"start": float, "end": float, "label": str}, ...]
    """
    # Compute chroma using CQT (better frequency resolution for chords)
    chroma = librosa.feature.chroma_cqt(y=audio_np, sr=sr, hop_length=hop_length)
    # chroma shape: (12, n_frames)

    n_frames = chroma.shape[1]
    frame_duration = hop_length / sr

    # Classify each frame
    frame_labels = []
    for i in range(n_frames):
        frame_vec = chroma[:, i].astype(np.float32)
        norm = np.linalg.norm(frame_vec)
        if norm < 1e-6:
            frame_labels.append("N.C.")
            continue
        frame_vec /= norm

        # Cosine similarity against all templates
        similarities = _TEMPLATES @ frame_vec
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= confidence_threshold:
            frame_labels.append(_TEMPLATE_LABELS[best_idx])
        else:
            frame_labels.append("N.C.")

    # Group consecutive same-label frames into regions
    if not frame_labels:
        return []

    regions = []
    current_label = frame_labels[0]
    region_start = 0
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != current_label:
            start_sec = region_start * frame_duration
            end_sec = i * frame_duration
            regions.append({"start": start_sec, "end": end_sec,
                            "label": current_label})
            current_label = frame_labels[i]
            region_start = i

    # Final region
    start_sec = region_start * frame_duration
    end_sec = n_frames * frame_duration
    regions.append({"start": start_sec, "end": end_sec, "label": current_label})

    # Merge short regions into neighbors
    if min_duration > 0:
        merged = []
        for r in regions:
            dur = r["end"] - r["start"]
            if dur < min_duration and merged:
                # Extend previous region to absorb this short one
                merged[-1]["end"] = r["end"]
            else:
                merged.append(r)
        regions = merged

    return regions


class ChordDetectNode:
    """Detect chord progression from audio. Returns JSON array of chord events."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hop_length": ("INT", {
                    "default": 4096, "min": 512, "max": 16384, "step": 512,
                    "tooltip": "Analysis hop size in samples. Larger = faster but less precise."
                }),
                "min_duration": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Minimum chord duration in seconds. Shorter chords are merged."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("chord_json",)
    FUNCTION = "detect_chords"
    OUTPUT_NODE = True
    CATEGORY = "rAIdio.bot"

    def detect_chords(self, audio, hop_length=4096, min_duration=0.5):
        # audio is a dict with "waveform" (tensor) and "sample_rate" (int)
        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        # Convert to numpy, mono
        audio_np = waveform.squeeze().numpy()
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=0)

        chords = _detect_chords(audio_np, sr, hop_length=hop_length,
                                min_duration=min_duration)

        # Round floats for cleaner JSON
        for c in chords:
            c["start"] = round(c["start"], 3)
            c["end"] = round(c["end"], 3)

        result_json = json.dumps(chords, ensure_ascii=True)
        return {"ui": {"text": [result_json]}, "result": (result_json,)}


class MidiTranscribeNode:
    """Transcribe audio to MIDI using Spotify basic-pitch neural model.

    Uses a purpose-built polyphonic music transcription model (ONNX).
    Handles multiple simultaneous notes, pitch bends, and velocity.
    Best results on isolated stems.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "min_note_duration": ("FLOAT", {
                    "default": 0.06, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum note duration in seconds. Shorter notes are discarded."
                }),
                "confidence": ("FLOAT", {
                    "default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": "Note confidence threshold. Higher = fewer but more certain notes."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("midi_filename",)
    FUNCTION = "transcribe_midi"
    OUTPUT_NODE = True
    CATEGORY = "rAIdio.bot"

    def transcribe_midi(self, audio, min_note_duration=0.06, confidence=0.3):
        import tempfile
        import soundfile as sf
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        # Convert to mono numpy
        audio_np = waveform.squeeze().numpy()
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=0)

        # basic-pitch reads from file, so write a temp WAV
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_np, sr)

            # Run basic-pitch inference (ONNX model, polyphonic)
            _, midi_data, _ = predict(
                tmp_path,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5,
                frame_threshold=confidence,
                minimum_note_length=min_note_duration * 1000.0,
                melodia_trick=True,
                midi_tempo=120.0,
            )
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # Save MIDI to output directory
        output_dir = folder_paths.get_output_directory()
        midi_subdir = os.path.join(output_dir, "Midi")
        os.makedirs(midi_subdir, exist_ok=True)

        # Find next available counter
        base = "rAIdio.bot_midi"
        counter = 1
        existing = [f for f in os.listdir(midi_subdir)
                     if f.startswith(base) and f.endswith(".mid")]
        if existing:
            nums = []
            for f in existing:
                mid = f[len(base):-4].strip("_")
                if mid.isdigit():
                    nums.append(int(mid))
            if nums:
                counter = max(nums) + 1

        filename = f"{base}_{counter:05d}.mid"
        filepath = os.path.join(midi_subdir, filename)

        midi_data.write(filepath)

        # Return result with UI metadata (same pattern as SaveAudio)
        return {"ui": {"midi": [{"filename": filename,
                                  "subfolder": "Midi",
                                  "type": "output"}]},
                "result": (filename,)}
