# rAIdio.bot in-signal watermark — ComfyUI nodes (SilentCipher, MIT, by Sony).
#
# GPL-3.0: this node imports ComfyUI's API (comfy.model_management) and is a
# derivative of the GPL-3.0 ComfyUI ecosystem. It runs SilentCipher IN-PROCESS so
# the ~12 MB model shares ComfyUI's CUDA context + allocator pool (no separate
# process => no VRAM squat; transient tensors are reabsorbed and reused by the
# generator). See docs/BASELINE.md + EU_AI_ACT.md for the provenance role.
#
# Two real AUDIO nodes:
#   RaidioWatermarkEmbed  (AUDIO -> AUDIO + ui result)  — idempotent embed
#   RaidioWatermarkDetect (AUDIO -> ui result)          — read-only
# Both emit {"ui": {"raidio_watermark": [<result dict>]}} so the app reads the
# method/schema/sdr/confidence/sample_rate/channels back from GET /history.
import os
import sys

# The fixed rAIdio identity payload (5 bytes / 40 bits): 'R','A','I',1,0.
RAIDIO_PAYLOAD = [82, 65, 73, 1, 0]
MODEL_TYPE = "44.1k"
METHOD = "SilentCipher-44.1k"
SCHEMA = 1
WM_SR = 44100  # SilentCipher's native rate; we normalize all watermarked output to this.

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = None  # lazy singleton: the wrapped SilentCipher model (kept resident; ~12 MB)


def _resolve_libs():
    """Find the vendored silentcipher overlay. In-package (Phase 4 layout) first,
    then the current app-depot bundle location for dev."""
    for c in (
        os.path.join(_HERE, "libs"),
        os.environ.get("RAIDIO_WM_LIBS", ""),
        os.path.join(_HERE, "..", "..", "..", "raidio_watermark", "libs"),  # dev Backend bundle
    ):
        if c and os.path.isdir(os.path.join(c, "silentcipher")):
            return os.path.abspath(c)
    return None


def _resolve_model_dir():
    for c in (
        os.path.join(_HERE, "model"),
        os.environ.get("RAIDIO_WM_MODEL", ""),
        os.path.join(_HERE, "..", "..", "..", "raidio_watermark", "model"),
    ):
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return None


class _SCWrap:
    """Holds the SilentCipher Model and a torch.nn.Module that registers its four
    submodule groups as children, so .to()/.modules() move them natively. Resyncs
    the plain Model.device (read at encode time) after every device move."""

    def __init__(self, sc_model, nn):
        self.sc = sc_model

        class _Mod(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc_c = sc_model.enc_c
                s.dec_c = sc_model.dec_c
                s.stft = sc_model.stft
                s.dec_m = nn.ModuleList(sc_model.dec_m)

        self.mod = _Mod()

    def to(self, dev):
        import torch
        d = dev if isinstance(dev, torch.device) else torch.device(dev)
        self.mod.to(d)
        self.sc.device = d
        self.sc.enc_c = self.mod.enc_c
        self.sc.dec_c = self.mod.dec_c
        self.sc.stft = self.mod.stft
        self.sc.dec_m = list(self.mod.dec_m)
        return self


def _get_model(prefer_cuda=True):
    """Lazy-build + cache the wrapped SilentCipher model, resident in-process."""
    global _MODEL
    libs = _resolve_libs()
    if libs and libs not in sys.path:
        sys.path.insert(0, libs)
    import torch
    import torch.nn as nn  # noqa: F401  (used by _SCWrap via closure)
    import silentcipher

    want_cuda = prefer_cuda and torch.cuda.device_count() > 0
    dev = "cuda" if want_cuda else "cpu"

    if _MODEL is None:
        mdir = _resolve_model_dir()
        ckpt = os.path.join(mdir, "44_1_khz", "73999_iteration") if mdir else None
        cfg = os.path.join(ckpt, "hparams.yaml") if ckpt else None
        if ckpt and os.path.isfile(os.path.join(ckpt, "enc_c.ckpt")) and os.path.isfile(cfg):
            sc = silentcipher.get_model(model_type=MODEL_TYPE, ckpt_path=ckpt, config_path=cfg, device="cpu")
        else:
            sc = silentcipher.get_model(model_type=MODEL_TYPE, device="cpu")
        # Assert the (singleton) STFT matches our 44.1k config — the metaclass caches
        # by class and ignores ctor args, so a foreign STFT would be silently wrong.
        assert int(getattr(sc.stft, "filter_length")) == int(sc.config.N_FFT), "STFT/config mismatch"
        _MODEL = _SCWrap(sc, nn)
    _MODEL.to(dev)
    return _MODEL.sc


def _to_mono_channels(waveform):
    """ComfyUI AUDIO waveform [B, C, T] (B=1) -> numpy [C, T] float32."""
    import numpy as np
    wf = waveform[0].detach().cpu().numpy().astype("float32")  # [C, T]
    if wf.ndim == 1:
        wf = wf[None, :]
    return np.ascontiguousarray(wf)


def _detect_payload(sc, mono, sr):
    """Decode (phase-off, whole-file) at 44.1k; return (match, payload, conf)."""
    import numpy as np
    import librosa
    s = mono if sr == WM_SR else librosa.resample(np.ascontiguousarray(mono), orig_sr=sr, target_sr=WM_SR)
    res = sc.decode_wav(np.ascontiguousarray(s), WM_SR, phase_shift_decoding=False)
    if isinstance(res, list):
        res = res[0]
    msgs = res.get("messages") or []
    pl = list(msgs[0]) if msgs else None
    conf = float((res.get("confidences") or [0.0])[0])
    return (pl == RAIDIO_PAYLOAD), pl, conf


def _embed(sc, chans, sr):
    """Per-channel embed; ALWAYS output at 44.1k (sub-44.1k save strips the mark).
    Returns (out_channels[list of np arrays @44.1k], min_sdr)."""
    import numpy as np
    import librosa
    outs, sdrs = [], []
    for ch in chans:
        x44 = ch if sr == WM_SR else librosa.resample(np.ascontiguousarray(ch), orig_sr=sr, target_sr=WM_SR)
        import torch
        enc, sdr = sc.encode_wav(np.ascontiguousarray(x44), WM_SR, RAIDIO_PAYLOAD, calc_sdr=True, disable_checks=False)
        enc = enc.detach().cpu().numpy() if torch.is_tensor(enc) else np.asarray(enc, dtype="float32")
        outs.append(enc.astype("float32"))
        sdrs.append(float(sdr))
    n = min(len(o) for o in outs)
    outs = [o[:n] for o in outs]
    return outs, (min(sdrs) if sdrs else 0.0)


class RaidioWatermarkEmbed:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "embed"
    OUTPUT_NODE = True   # so the ui result surfaces in history
    CATEGORY = "audio/watermark"

    def embed(self, audio):
        import numpy as np
        import torch
        sr = int(audio["sample_rate"])
        chans = list(_to_mono_channels(audio["waveform"]))  # [C][T]
        nch = len(chans)

        def _run(prefer_cuda):
            sc = _get_model(prefer_cuda=prefer_cuda)
            with torch.no_grad():
                # Idempotent: if our mark is already present, don't stack.
                mono = np.mean(np.stack(chans, 0), axis=0)
                det, _, conf0 = _detect_payload(sc, mono, sr)
                if det:
                    out44 = [ch if sr == WM_SR else __import__("librosa").resample(
                        np.ascontiguousarray(ch), orig_sr=sr, target_sr=WM_SR) for ch in chans]
                    return out44, None, round(conf0, 4), "already_present"
                outs, sdr = _embed(sc, chans, sr)
                # Verify at the OUTPUT rate (44.1k) — fail-loud contract.
                mono_out = np.mean(np.stack(outs, 0), axis=0)
                det2, _, conf2 = _detect_payload(sc, mono_out, WM_SR)
                if not det2:
                    raise RuntimeError("embed verification failed")
                return outs, round(sdr, 1), round(conf2, 4), "embedded"

        try:
            outs, sdr, conf, action = _run(prefer_cuda=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # CUDA-OOM => CPU retry (C2PA-every-output)
            print(f"[rAIdio_watermark] GPU embed failed ({e}); CPU retry", file=sys.stderr)
            outs, sdr, conf, action = _run(prefer_cuda=False)

        out_wf = torch.from_numpy(np.stack(outs, 0)).unsqueeze(0).float()  # [1, C, T]
        result = {"ok": True, "action": action, "method": METHOD, "schema": SCHEMA,
                  "sdr": sdr, "confidence": conf, "sample_rate": WM_SR, "channels": nch}
        return {"ui": {"raidio_watermark": [result]}, "result": ({"waveform": out_wf, "sample_rate": WM_SR},)}


class RaidioWatermarkDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ()
    FUNCTION = "detect"
    OUTPUT_NODE = True
    CATEGORY = "audio/watermark"

    def detect(self, audio):
        import numpy as np
        import torch
        sr = int(audio["sample_rate"])
        chans = _to_mono_channels(audio["waveform"])
        sc = _get_model(prefer_cuda=True)
        with torch.no_grad():
            mono = np.mean(chans, axis=0)
            match, payload, conf = _detect_payload(sc, mono, sr)
        result = {"ok": True, "detected": bool(match), "payload_match": bool(match),
                  "payload": payload, "confidence": round(conf, 4), "method": METHOD,
                  "schema": SCHEMA, "sample_rate": sr, "channels": int(chans.shape[0])}
        return {"ui": {"raidio_watermark": [result]}}


class RaidioWatermarkFile:
    """Watermark/detect a WAV file IN PLACE by absolute path. The app submits this
    as a 1-node /prompt (replacing the old python subprocess), so it runs in
    ComfyUI's process — the ~12 MB model shares the CUDA context + allocator pool
    (no separate context => no VRAM squat). `ensure` writes the file back at 44.1k
    (sub-44.1k saves strip the mark). Result shape mirrors the old WatermarkResult."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path": ("STRING", {"default": ""}),
            "mode": (["ensure", "detect"], {"default": "ensure"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "process"
    OUTPUT_NODE = True
    CATEGORY = "audio/watermark"

    def process(self, path, mode):
        import numpy as np
        import torch
        import soundfile as sf

        def ui(res):
            return {"ui": {"raidio_watermark": [res]}}

        if not path or not os.path.isfile(path):
            return ui({"ok": False, "error": f"file not found: {path}"})
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
            subtype = sf.info(path).subtype
        except Exception as e:  # noqa: BLE001
            return ui({"ok": False, "error": f"read failed: {e}"})
        chans = [np.ascontiguousarray(data[:, c]) for c in range(data.shape[1])]
        nch = len(chans)
        mono = np.ascontiguousarray(data.mean(axis=1))

        sc = _get_model(prefer_cuda=True)
        with torch.no_grad():
            if mode == "detect":
                match, payload, conf = _detect_payload(sc, mono, sr)
                return ui({"ok": True, "detected": bool(match), "payload_match": bool(match),
                           "payload": payload, "confidence": round(conf, 4), "method": METHOD,
                           "schema": SCHEMA, "sample_rate": int(sr), "channels": nch})
            # ensure: idempotent detect-then-embed
            det, _, conf0 = _detect_payload(sc, mono, sr)
            if det:
                return ui({"ok": True, "action": "already_present", "confidence": round(conf0, 4),
                           "method": METHOD, "schema": SCHEMA, "sample_rate": int(sr), "channels": nch})
            try:
                outs, sdr = _embed(sc, chans, sr)  # @44.1k
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"[rAIdio_watermark] GPU embed failed ({e}); CPU retry", file=sys.stderr)
                sc = _get_model(prefer_cuda=False)
                outs, sdr = _embed(sc, chans, sr)
            out = np.stack(outs, axis=1).astype("float32")  # [T, C] @44.1k
            det2, _, conf2 = _detect_payload(sc, np.ascontiguousarray(out.mean(axis=1)), WM_SR)
            if not det2:
                return ui({"ok": False, "error": "embed verification failed"})
            try:
                sf.write(path, out, WM_SR, subtype=subtype)
            except Exception as e:  # noqa: BLE001
                return ui({"ok": False, "error": f"write failed: {e}"})
            return ui({"ok": True, "action": "embedded", "sdr": round(sdr, 1), "confidence": round(conf2, 4),
                       "method": METHOD, "schema": SCHEMA, "sample_rate": WM_SR, "channels": nch})


NODE_CLASS_MAPPINGS = {
    "RaidioWatermarkEmbed": RaidioWatermarkEmbed,
    "RaidioWatermarkDetect": RaidioWatermarkDetect,
    "RaidioWatermarkFile": RaidioWatermarkFile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RaidioWatermarkEmbed": "rAIdio Watermark (Embed)",
    "RaidioWatermarkDetect": "rAIdio Watermark (Detect)",
    "RaidioWatermarkFile": "rAIdio Watermark (File)",
}
