"""RVC argparse compatibility patch for ComfyUI.

Monkey-patches ComfyUI-RVC's config.py to use parse_known_args() instead of
parse_args(), preventing crashes when ComfyUI passes its own CLI arguments.

Licensed under GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""
import logging

def _apply_rvc_argparse_patch():
    try:
        import argparse
        _orig_parse_args = argparse.ArgumentParser.parse_args

        def _safe_parse_args(self, args=None, namespace=None):
            try:
                return _orig_parse_args(self, args, namespace)
            except SystemExit:
                known, _ = self.parse_known_args(args, namespace)
                return known

        argparse.ArgumentParser.parse_args = _safe_parse_args
        logging.info("[rAIdio] RVC argparse patch applied")
    except Exception as e:
        logging.warning("[rAIdio] RVC argparse patch failed (non-fatal): %s", e)

_apply_rvc_argparse_patch()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
