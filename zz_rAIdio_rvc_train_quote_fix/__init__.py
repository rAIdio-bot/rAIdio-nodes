"""rAIdio RVC train spaces-in-path quoting fix.

ComfyUI-RVC's `rvc/train.py` builds shell command strings for the
training subprocesses (preprocess, f0 extract, feature extract, train)
in the form:

    cmd = '"%s" %s/infer/modules/train/preprocess.py "%s" ...' % (
        config.python_cmd, now_dir, ...)

The python.exe path is wrapped in quotes; the script path
(`<now_dir>/infer/modules/train/<name>.py`) is NOT. When `now_dir`
contains spaces, e.g. the Steam install path
`C:\\Program Files (x86)\\Steam\\steamapps\\common\\rAIdio.bot\\...`,
Windows tokenises the unquoted script path at the first space and
reports:

    python.exe: can't open file 'C:\\Program': [Errno 2]

Result: every training subprocess fails silently, `0_gt_wavs/` never
gets created, and `rvc/train.py:160` raises FileNotFoundError. Voice
training is completely broken on any Windows install path with spaces.

This patch wraps `rvc.train.Popen` so that, before the subprocess is
launched, the cmd string is rewritten to quote the unquoted script
path. The fix is targeted (`rvc.train` only) so unrelated subprocess
work in the rest of ComfyUI is untouched.

Licensed under GPL-3.0.
Source: https://github.com/rAIdio-bot/rAIdio-nodes
"""
import logging
import re
import subprocess

# Match the closing quote of the python.exe path, whitespace, then a
# captured non-quoted path ending in `.py` followed by whitespace or
# end-of-string. Non-greedy on the path body so we stop at the FIRST
# `.py` boundary after the exe quote.
_QUOTED_EXE_RE = re.compile(r'^("[^"]+\.exe")\s+([^"]+?\.py)(?=\s|$)')

# DML/AMD branch in train.py uses `config.python_cmd + ' ...py ...'`,
# i.e. the exe is unquoted too. We don't ship DML, but be defensive.
_BARE_EXE_RE = re.compile(r'^(\S+?\.exe)\s+(\S+?\.py)(?=\s|$)')


def _fix_cmd_quoting(cmd):
    m = _QUOTED_EXE_RE.match(cmd)
    if m:
        return cmd[: m.start(2)] + '"' + m.group(2) + '"' + cmd[m.end(2) :]
    m = _BARE_EXE_RE.match(cmd)
    if m:
        return (
            '"' + m.group(1) + '" "' + m.group(2) + '"' + cmd[m.end(2) :]
        )
    return cmd


class _PatchedPopen(subprocess.Popen):
    def __init__(self, args, *posargs, **kwargs):
        if isinstance(args, str) and kwargs.get("shell"):
            new_args = _fix_cmd_quoting(args)
            if new_args != args:
                logging.info(
                    "[rAIdio rvc_train_quote_fix] re-quoted script path in cmd"
                )
                args = new_args
        super().__init__(args, *posargs, **kwargs)


def _apply_quote_patch():
    try:
        from rvc import train as _rvc_train
    except Exception as e:
        logging.warning(
            "[rAIdio rvc_train_quote_fix] could not import rvc.train: %r", e
        )
        return

    if getattr(_rvc_train.Popen, "_raidio_quote_patched", False):
        return
    _PatchedPopen._raidio_quote_patched = True  # type: ignore[attr-defined]
    _rvc_train.Popen = _PatchedPopen
    logging.info(
        "[rAIdio rvc_train_quote_fix] rvc.train.Popen wrapped — script paths re-quoted for spaces-in-path installs"
    )


_apply_quote_patch()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
