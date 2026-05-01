"""rAIdio RVC_Train directory-mode fix.

ComfyUI-RVC's RVC_Train.INPUT_TYPES only declares a single `audio` input.
The training body (nodes.py:188) does
    shutil.copy(wav_path, os.path.join(trainset_dir, basename(wav_path)))
and then runs preprocess_dataset over trainset_dir. So upstream RVC only
ever trains on ONE audio file, regardless of how much voice data the
user has.

The rAIdio frontend supports a "Train from directory" mode that uploads
N audio files to ComfyUI's input dir and then submits the workflow with
audio_files = "file1.wav;file2.wav;...". But RVC_Train doesn't have an
audio_files parameter, so ComfyUI silently drops it and only one file
makes it into trainset_dir. Result: tiny dataset, low-quality voice
models even when the user provides 10+ minutes of clean source audio.

This patch:
  1. Extends RVC_Train.INPUT_TYPES to add `audio_files` as an optional
     STRING input (default empty) — semicolon-separated list of basenames
     in ComfyUI's input directory.
  2. Wraps RVC_Train.train. When audio_files is non-empty, monkey-patches
     shutil.copy for the duration of the call so the FIRST copy inside
     train() (the trainset copy at nodes.py:188) ALSO copies every extra
     file to the same destination directory. preprocess_dataset then
     scans all of them naturally. Subsequent shutil.copy calls (for
     final-weight copy, etc.) are unaffected because we restore on the
     first invocation.

Layers cleanly on top of zz_rAIdio_rvc_bool_fix:
    ComfyUI → dir_fix.train(*args, **kw)
            → bool_fix.train(*args, **kw)   # bool coercion
            → orig.train(*args, **kw)        # real RVC_Train logic

Licensed under GPL-3.0.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""
import logging
import os
import shutil

import folder_paths


def _apply_rvc_train_dir_patch():
    try:
        import nodes as _comfy_nodes
        rvc_train_cls = _comfy_nodes.NODE_CLASS_MAPPINGS.get("RVC_Train")
        if rvc_train_cls is None:
            logging.warning(
                "[rAIdio rvc_train_dir_fix] RVC_Train not in NODE_CLASS_MAPPINGS; skipped"
            )
            return

        # ── 1. Extend INPUT_TYPES so ComfyUI accepts audio_files ─────────
        orig_input_types = rvc_train_cls.INPUT_TYPES
        if not getattr(orig_input_types, "_raidio_dir_patched", False):
            @classmethod
            def _patched_input_types(cls):
                base = (
                    orig_input_types.__func__(cls)
                    if hasattr(orig_input_types, "__func__")
                    else orig_input_types()
                )
                base.setdefault("optional", {})["audio_files"] = (
                    "STRING",
                    {"default": ""},
                )
                return base
            _patched_input_types.__func__._raidio_dir_patched = True  # type: ignore[attr-defined]
            rvc_train_cls.INPUT_TYPES = _patched_input_types
            logging.info(
                "[rAIdio rvc_train_dir_fix] added optional audio_files to RVC_Train.INPUT_TYPES"
            )

        # ── 2. Wrap train() to splat audio_files into trainset_dir ──────
        orig_train = rvc_train_cls.train
        if getattr(orig_train, "_raidio_dir_patched", False):
            return

        def _patched_train(self, *args, **kwargs):
            audio_files = kwargs.pop("audio_files", "") or ""
            extra_paths = [p.strip() for p in audio_files.split(";") if p.strip()]

            if not extra_paths:
                # No directory mode — pass through unchanged.
                return orig_train(self, *args, **kwargs)

            # Resolve extra paths early. folder_paths.get_annotated_filepath
            # returns the absolute path under ComfyUI's input dir.
            resolved = []
            for ep in extra_paths:
                try:
                    src = folder_paths.get_annotated_filepath(ep)
                    if os.path.exists(src):
                        resolved.append(src)
                    else:
                        logging.warning(
                            "[rAIdio rvc_train_dir_fix] missing extra file: %s", ep
                        )
                except Exception as e:
                    logging.warning(
                        "[rAIdio rvc_train_dir_fix] could not resolve %s: %r", ep, e
                    )

            if not resolved:
                return orig_train(self, *args, **kwargs)

            real_copy = shutil.copy
            fired = [False]

            def _wrapped_copy(src, dst, *a, **kw):
                # Always perform the original copy first.
                result = real_copy(src, dst, *a, **kw)
                if not fired[0]:
                    # The FIRST copy after our patch installed is the
                    # nodes.py:188 trainset copy. Splat extras to the same
                    # destination directory now.
                    dst_dir = dst if os.path.isdir(dst) else os.path.dirname(dst)
                    primary_basename = os.path.basename(src)
                    copied_count = 0
                    for extra_src in resolved:
                        bn = os.path.basename(extra_src)
                        if bn == primary_basename:
                            continue  # dedupe
                        try:
                            real_copy(extra_src, os.path.join(dst_dir, bn))
                            copied_count += 1
                        except Exception as e:
                            logging.warning(
                                "[rAIdio rvc_train_dir_fix] copy %s failed: %r",
                                extra_src,
                                e,
                            )
                    fired[0] = True
                    logging.info(
                        "[rAIdio rvc_train_dir_fix] staged %d extra file(s) into %s",
                        copied_count,
                        dst_dir,
                    )
                return result

            shutil.copy = _wrapped_copy
            try:
                return orig_train(self, *args, **kwargs)
            finally:
                shutil.copy = real_copy

        _patched_train._raidio_dir_patched = True  # type: ignore[attr-defined]
        rvc_train_cls.train = _patched_train
        logging.info(
            "[rAIdio rvc_train_dir_fix] RVC_Train.train wrapped for audio_files directory mode"
        )

    except Exception as e:
        logging.exception("[rAIdio rvc_train_dir_fix] patch failed: %s", e)


_apply_rvc_train_dir_patch()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
