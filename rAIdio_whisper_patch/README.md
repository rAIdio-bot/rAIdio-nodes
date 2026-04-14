# rAIdio Whisper STT Patch

ComfyUI custom node that replaces the QwenTTS Whisper STT node with a
faster-whisper implementation that returns word-level timestamps.

## What it does

The upstream `AILab_Qwen3TTSWhisperSTT` node returns only plain text from
Whisper transcription, discarding segment and word timing data.

This patch registers a replacement node class under the same name. Because
ComfyUI loads custom nodes alphabetically, this package (`rAIdio_whisper_patch`)
loads after `ComfyUI-QwenTTS` and overwrites the node registration.

The replacement node:
- Uses faster-whisper instead of openai-whisper
- Enables `word_timestamps=True`
- Returns two outputs: plain text and a JSON string with segments/words
- Supports transcribe and translate tasks
- Preprocesses audio to 16kHz mono numpy (no temp file needed)

## Output format

The `timestamps` output is a JSON string:

```json
{
  "text": "full transcription",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "segment text",
      "words": [
        {"word": "segment", "start": 0.0, "end": 1.2},
        {"word": "text", "start": 1.3, "end": 2.5}
      ]
    }
  ]
}
```

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

2025-2026 Creative Mayhem UG
