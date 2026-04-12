# ComfyUI-ChordMidi

ComfyUI custom nodes for chord detection and audio-to-MIDI transcription.

No large ML models required for chord detection — uses librosa's chroma features.
MIDI transcription uses Spotify basic-pitch (ONNX, Apache 2.0).

## Nodes

**ChordDetect** — Detects chord progression from audio. Returns a JSON array of timed chord events `[{"start": 0.0, "end": 2.3, "label": "Am"}, ...]`. Uses CQT chroma analysis with cosine similarity against 108 chord templates (12 roots × 9 types).

**MidiTranscribe** — Transcribes audio to a MIDI file using Spotify basic-pitch. Best results on isolated stems (melody, bass). Saves to `ComfyUI/output/Midi/`.

## Requirements

```
basic-pitch>=0.4.0
pretty-midi>=0.2.10
mir-eval>=0.7
librosa  (usually already installed in ComfyUI environments)
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright (c) 2025-2026 Creative Mayhem UG
