"""RVC boolean-to-i18n coercion patch for ComfyUI-RVC.

Upstream ComfyUI-RVC defines its train flags as BOOLEAN inputs (if_save_latest,
if_cache_gpu, if_save_every_weights) but then compares them to a localized
Chinese string inside rvc/train.py's click_train():

    1 if if_save_every_weights18 == i18n("是") else 0,

A Python True never equals the string "是" (or its English translation), so the
comparison is always False and click_train always emits `-sw 0 -l 0 -c 0`
regardless of what the caller requested. Consequence: no inference .pth ever
lands in rvc/assets/weights/<name>.pth, so the Custom Voice dropdown never
picks up a trained model. The bug also disables if_save_latest (no latest
checkpoint written) and if_cache_gpu (GPU caching always off).

Fix: wrap RVC_Train.train and coerce incoming booleans to the exact string
i18n("是") returns in the current locale. That way the upstream comparison
matches and the CLI flags are built correctly. No third-party code is edited.

This package is named `zz_...` so ComfyUI's alphabetical custom-node loader
runs it AFTER ComfyUI-RVC has registered RVC_Train.

Licensed under GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""
import logging

_BOOL_ARGS = ("if_save_latest", "if_cache_gpu", "if_save_every_weights")


def _apply_rvc_train_bool_patch():
    try:
        import nodes as _comfy_nodes
        rvc_train_cls = _comfy_nodes.NODE_CLASS_MAPPINGS.get("RVC_Train")
        if rvc_train_cls is None:
            logging.warning("[rAIdio] RVC bool-fix: RVC_Train not in NODE_CLASS_MAPPINGS; patch skipped")
            return

        from rvc.i18n.i18n import I18nAuto
        _i18n = I18nAuto()
        yes_str = _i18n("是")
        no_str = "__rAIdio_no__"

        _orig_train = rvc_train_cls.train
        if getattr(_orig_train, "_raidio_bool_patched", False):
            return

        def _coerce(v):
            if v is True:
                return yes_str
            if v is False:
                return no_str
            return v

        def _patched_train(self, *args, **kwargs):
            for k in _BOOL_ARGS:
                if k in kwargs:
                    kwargs[k] = _coerce(kwargs[k])
            return _orig_train(self, *args, **kwargs)

        _patched_train._raidio_bool_patched = True
        rvc_train_cls.train = _patched_train
        logging.info("[rAIdio] RVC_Train bool-coercion patch applied (yes=%r)", yes_str)
    except Exception as e:
        logging.exception("[rAIdio] RVC bool-fix patch failed: %s", e)


_apply_rvc_train_bool_patch()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
