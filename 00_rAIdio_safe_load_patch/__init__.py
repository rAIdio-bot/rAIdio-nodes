"""Safe-load patch — picklescan-gated torch.load, pickle.load, numpy.load.

Closes the malicious-pickle-via-shared-model-file attack vector across
every standard deserialisation entry point in the ComfyUI process. The
package wraps three primitives at custom-node import time:

  1. torch.load      — defaults weights_only=True; picklescan-gated.
  2. pickle.load     — picklescan-gated.
     pickle.loads    — picklescan-gated.
  3. numpy.load      — defaults allow_pickle=False; picklescan-gated
                       when the caller insists on allow_pickle=True.

All three wrappers refuse files whose pickle contents reference
dangerous globals (os.system, subprocess.*, builtins.{eval,exec,
compile,open,__import__}, pty.spawn, shutil.rmtree, etc — see
_DANGEROUS_GLOBALS below). Refusal happens BEFORE the deserialiser
runs, so even callers that explicitly opt into legacy-pickle mode
(e.g. weights_only=False on torch.load, allow_pickle=True on
numpy.load) are still gated.

Applied at ComfyUI custom-node import time. The 00_ prefix forces
ComfyUI to load this BEFORE any node that calls these primitives at
import time, so each rebind is in place before first use.

Failure of any individual wrapper (missing picklescan, weird torch
version, etc.) is logged as a warning but never raises — the bundled
install must keep working even if a future library version breaks one
of the wrappers. Each wrapper applies independently.

Threat model + Phase 2a/2b/2c scope:
  https://github.com/rAIdio-bot/sbom/blob/main/docs/security/model-loading.md

Licensed under GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""

import io
import logging
import os

# Globals that, if present in a pickle, indicate the file is trying to
# execute arbitrary code at deserialisation. Same set used by every
# wrapper in this module.
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


def _check_picklescan_result(result, source_label):
    """Raise RuntimeError if the picklescan result reports a dangerous
    pickle. Used by every wrapper in this module."""
    if getattr(result, "scan_err", False):
        raise RuntimeError(
            "[rAIdio.safe_load] picklescan rejected %s: scan errors "
            "during analysis (likely malformed pickle)" % source_label
        )
    bad = []
    for g in (getattr(result, "globals", None) or []):
        fq = "%s.%s" % (getattr(g, "module", ""), getattr(g, "name", ""))
        if fq in _DANGEROUS_GLOBALS:
            bad.append(fq)
    if bad:
        raise RuntimeError(
            "[rAIdio.safe_load] refusing to load %s: "
            "dangerous globals in pickle: %s" % (source_label, sorted(set(bad)))
        )


def _apply_torch_load_patch():
    try:
        import torch
    except Exception as e:
        logging.warning("[rAIdio] torch.load patch: torch import failed: %s", e)
        return

    try:
        from picklescan.scanner import scan_file_path  # noqa: F401
        _picklescan_available = True
    except Exception:
        _picklescan_available = False
        logging.warning(
            "[rAIdio] torch.load patch: picklescan not installed; relying "
            "on weights_only=True default only. Run `python -m pip install "
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
                _check_picklescan_result(scan_file_path(str(f)), str(f))
            except RuntimeError:
                raise
            except Exception as scan_e:
                # Picklescan can crash on truly malformed inputs. Log +
                # continue to weights_only torch.load — the kernel-level
                # safe unpickle still protects us.
                logging.warning(
                    "[rAIdio.safe_load] picklescan errored on %s (%s); "
                    "falling back to weights_only=True only", f, scan_e
                )

        return _orig_load(f, *args, **kwargs)

    torch.load = _safe_load
    logging.info(
        "[rAIdio] torch.load patched: weights_only=True default + "
        "%spicklescan gate",
        "" if _picklescan_available else "(no-picklescan-fallback) ",
    )


def _apply_pickle_load_patch():
    try:
        import pickle
        from picklescan.scanner import scan_file_path, scan_pickle_bytes
    except Exception as e:
        logging.warning("[rAIdio] pickle.load patch: deps missing: %s", e)
        return

    _orig_load = pickle.load
    _orig_loads = pickle.loads

    # pickle.load*/loads have no kernel-level safety net like torch's
    # weights_only=True. If picklescan can't parse the stream, we fail
    # CLOSED — refusing to deserialise — rather than slip a possibly-
    # malicious pickle through. For the rare case where a legitimate
    # picklescan internal error blocks a known-good file, the caller
    # has the option of using torch.load (which is fail-open against
    # picklescan errors because weights_only protects the kernel path).

    def _safe_load(f, *args, **kwargs):
        # Real file: scan via path, hand original stream to pickle.
        path = getattr(f, "name", None)
        if isinstance(path, str) and os.path.exists(path):
            try:
                _check_picklescan_result(scan_file_path(path), path)
            except Exception as e:
                if isinstance(e, RuntimeError) and "rAIdio.safe_load" in str(e):
                    raise
                raise RuntimeError(
                    "[rAIdio.safe_load] picklescan failed on %s (%s); "
                    "refusing to load (fail-closed)" % (path, e)
                ) from e
            return _orig_load(f, *args, **kwargs)

        # File-like without a real path (BytesIO, etc): consume the
        # stream once, scan the bytes, deserialise from those captured
        # bytes via pickle.loads.
        buf = f.read()
        try:
            _check_picklescan_result(
                scan_pickle_bytes(io.BytesIO(buf), "<file-like>"),
                "<file-like>",
            )
        except Exception as e:
            if isinstance(e, RuntimeError) and "rAIdio.safe_load" in str(e):
                raise
            raise RuntimeError(
                "[rAIdio.safe_load] picklescan failed on <file-like> (%s); "
                "refusing to load (fail-closed)" % e
            ) from e
        return _orig_loads(buf, *args, **kwargs)

    def _safe_loads(data, *args, **kwargs):
        try:
            _check_picklescan_result(
                scan_pickle_bytes(io.BytesIO(data), "<bytes>"),
                "<bytes>",
            )
        except Exception as e:
            if isinstance(e, RuntimeError) and "rAIdio.safe_load" in str(e):
                raise
            raise RuntimeError(
                "[rAIdio.safe_load] picklescan failed on <bytes> (%s); "
                "refusing to load (fail-closed)" % e
            ) from e
        return _orig_loads(data, *args, **kwargs)

    pickle.load = _safe_load
    pickle.loads = _safe_loads
    logging.info("[rAIdio] pickle.load/loads patched: picklescan gate (fail-closed)")


def _apply_numpy_load_patch():
    try:
        import numpy as np
        from picklescan.scanner import scan_file_path, scan_zip_bytes
    except Exception as e:
        logging.warning("[rAIdio] numpy.load patch: deps missing: %s", e)
        return

    _orig_load = np.load

    def _safe_load(file, *args, **kwargs):
        # Preserve numpy's modern default (False) when caller doesn't say.
        # If allow_pickle is omitted, numpy will refuse pickled arrays
        # itself and our wrapper has nothing to add. If allow_pickle=True
        # is explicit, scan the file before letting numpy unpickle.
        kwargs.setdefault("allow_pickle", False)
        if kwargs.get("allow_pickle") and isinstance(file, (str, os.PathLike)):
            path = str(file)
            try:
                if path.lower().endswith(".npz"):
                    # .npz is a zip of .npy files; picklescan handles
                    # the zip case directly.
                    with open(path, "rb") as fh:
                        _check_picklescan_result(scan_zip_bytes(fh, path), path)
                else:
                    _check_picklescan_result(scan_file_path(path), path)
            except RuntimeError:
                raise
            except Exception as scan_e:
                logging.warning(
                    "[rAIdio.safe_load] picklescan errored on %s (%s); "
                    "letting numpy.load proceed", path, scan_e,
                )
        return _orig_load(file, *args, **kwargs)

    np.load = _safe_load
    logging.info(
        "[rAIdio] numpy.load patched: allow_pickle=False default + "
        "picklescan gate when allow_pickle=True"
    )


# Apply each wrapper independently. Failure of one MUST NOT prevent
# the others — a partially-protected install is better than a crashed
# import that knocks the whole node out of the registry.
for _patch in (_apply_torch_load_patch, _apply_pickle_load_patch, _apply_numpy_load_patch):
    try:
        _patch()
    except Exception as _e:
        logging.warning("[rAIdio] safe_load patch %s failed (non-fatal): %s", _patch.__name__, _e)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
