"""Safe torch.load patch — defaults weights_only=True + picklescan gate.

Closes the malicious-pickle-via-shared-model-file attack vector. Every
torch.load call in the ComfyUI process — bundled or user-supplied —
goes through a wrapper that:

  1. Defaults weights_only=True (PyTorch's safe-unpickle mode) when the
     caller didn't pass it explicitly. Blocks __reduce__ execution at
     the pickle level.
  2. Runs picklescan against the file path before deserialising and
     refuses on dangerous globals (os.system, subprocess.*, eval,
     exec, posix.system).

Applied at ComfyUI custom-node import time. The 00_ prefix forces
ComfyUI to load this BEFORE any node that calls torch.load at import,
so the rebind is in place before the first usage.

Failure of the patch (missing picklescan, weird torch version, etc.) is
logged as a warning but never raises — the bundled install must keep
working even if a future torch version breaks the wrapper.

Threat model + Phase 2a/2b/2c scope:
  https://github.com/neitzert/rAIdio-rust/blob/master/docs/security/model-loading.md

Licensed under GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""

import logging
import os

# Globals that, if present in a pickle, indicate the file is trying to
# execute arbitrary code at deserialisation. weights_only=True already
# blocks these on a clean PyTorch 2.6+; picklescan is defense-in-depth
# for older torch and for the (rare) case the caller passes
# weights_only=False intentionally.
_DANGEROUS_GLOBALS = {
    "os.system",
    "os.popen",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.run",
    "posix.system",
    "nt.system",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.open",
    "builtins.__import__",
    "shutil.rmtree",
    "pty.spawn",
    "platform.popen",
}


def _apply_safe_load_patch():
    try:
        import torch
    except Exception as e:
        logging.warning("[rAIdio] safe_load patch: torch import failed: %s", e)
        return

    try:
        from picklescan.scanner import scan_file_path  # noqa: F401
        _picklescan_available = True
    except Exception:
        _picklescan_available = False
        logging.warning(
            "[rAIdio] safe_load patch: picklescan not installed; relying on "
            "weights_only=True default only. Run `python -m pip install "
            "picklescan` to enable defense-in-depth."
        )

    _orig_load = torch.load

    def _safe_load(f, *args, **kwargs):
        # 1. Default-safe: weights_only=True unless the caller explicitly
        #    set it. Some legacy code paths in custom nodes need
        #    weights_only=False (they pickle full nn.Module instances or
        #    custom dataclasses); those callers retain control.
        kwargs.setdefault("weights_only", True)

        # 2. Picklescan defense-in-depth — only when given a path. File
        #    objects, BytesIO, etc. skip this branch (their content
        #    typically comes from controlled in-memory sources).
        if _picklescan_available and isinstance(f, (str, os.PathLike)):
            try:
                from picklescan.scanner import scan_file_path
                result = scan_file_path(str(f))
                if getattr(result, "scan_err", False):
                    raise RuntimeError(
                        "[rAIdio.safe_load] picklescan rejected %s: "
                        "scan errors during analysis (likely malformed pickle)"
                        % f
                    )
                bad = []
                for g in (getattr(result, "globals", None) or []):
                    fq = "%s.%s" % (getattr(g, "module", ""), getattr(g, "name", ""))
                    if fq in _DANGEROUS_GLOBALS:
                        bad.append(fq)
                if bad:
                    raise RuntimeError(
                        "[rAIdio.safe_load] refusing to load %s: "
                        "dangerous globals in pickle: %s" % (f, sorted(set(bad)))
                    )
            except RuntimeError:
                raise
            except Exception as scan_e:
                # Picklescan can crash on truly malformed inputs. Log + continue
                # to weights_only torch.load — the kernel-level safe unpickle
                # still protects us.
                logging.warning(
                    "[rAIdio.safe_load] picklescan errored on %s (%s); "
                    "falling back to weights_only=True only", f, scan_e
                )

        return _orig_load(f, *args, **kwargs)

    torch.load = _safe_load
    logging.info(
        "[rAIdio] torch.load patched: weights_only=True default + "
        "%spicklescan gate", "" if _picklescan_available else "(no-picklescan-fallback) "
    )


try:
    _apply_safe_load_patch()
except Exception as e:
    logging.warning("[rAIdio] safe_load patch failed (non-fatal): %s", e)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
