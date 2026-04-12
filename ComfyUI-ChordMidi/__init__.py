"""rAIdio.bot Chord Detection and MIDI Transcription nodes for ComfyUI."""

from .chord_midi_nodes import ChordDetectNode, MidiTranscribeNode

NODE_CLASS_MAPPINGS = {
    "ChordDetect": ChordDetectNode,
    "MidiTranscribe": MidiTranscribeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChordDetect": "Chord Detect",
    "MidiTranscribe": "MIDI Transcribe",
}
