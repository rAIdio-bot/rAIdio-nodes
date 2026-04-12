"""rAIdio.bot utility nodes for ComfyUI."""
import os
import json
import numpy as np
import folder_paths


class SaveAudioWAV:
    """Save audio as 16-bit PCM WAV (Unity-compatible)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
            "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save_audio_wav"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save_audio_wav(self, audio, filename_prefix="ComfyUI"):
        import soundfile as sf

        output_dir = folder_paths.get_output_directory()
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename_base = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = os.path.join(output_dir, subfolder)
        os.makedirs(full_output_folder, exist_ok=True)

        # Find next available counter
        counter = 1
        existing = [f for f in os.listdir(full_output_folder)
                     if f.startswith(filename_base) and f.endswith(".wav")]
        if existing:
            nums = []
            for f in existing:
                mid = f[len(filename_base):-4]  # strip prefix and .wav
                mid = mid.strip("_")
                if mid.isdigit():
                    nums.append(int(mid))
            if nums:
                counter = max(nums) + 1

        filename = f"{filename_base}_{counter:05d}_.wav"
        filepath = os.path.join(full_output_folder, filename)

        waveform = audio["waveform"].squeeze(0).cpu()
        sample_rate = audio["sample_rate"]

        # Convert to 16-bit PCM WAV via soundfile (torchaudio.save
        # requires torchcodec in torch >=2.10 which may not be installed)
        wav_np = waveform.numpy().T  # (samples, channels) for soundfile
        sf.write(filepath, wav_np, sample_rate, subtype="PCM_16")

        results = [{
            "filename": filename,
            "subfolder": subfolder,
            "type": "output",
        }]
        return {"ui": {"audio": results}}


NODE_CLASS_MAPPINGS = {
    "SaveAudioWAV": SaveAudioWAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudioWAV": "Save Audio (WAV)",
}
