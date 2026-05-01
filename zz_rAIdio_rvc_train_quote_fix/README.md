# zz_rAIdio RVC Train Quote Fix

ComfyUI custom node that fixes a spaces-in-path quoting bug in
upstream ComfyUI-RVC's training subprocess invocations.

## What it fixes

`rvc/train.py` builds shell command strings for every training
subprocess (preprocess, f0 extract, feature extract, train) in this
form:

```python
cmd = '"%s" %s/infer/modules/train/preprocess.py "%s" %s %s ...' % (
    config.python_cmd, now_dir, trainset_dir, sr, n_p, ...)
```

The Python executable path is wrapped in quotes; the script path
(`<now_dir>/infer/modules/train/<name>.py`) is **not**. When
`now_dir` contains spaces — e.g. an install at
`C:\Program Files (x86)\Steam\steamapps\common\rAIdio.bot\...` —
Windows tokenises the unquoted script path at the first space:

```
python.exe: can't open file 'C:\Program': [Errno 2]
```

Every training subprocess fails. `0_gt_wavs/` never gets created.
`rvc/train.py:160` then raises `FileNotFoundError`. Voice training is
completely broken on any Windows install path that contains spaces.

The bug is invisible on dev installs at paths like `C:\dev\...` and
only surfaces in distribution layouts (`Program Files`, user-name
directories with spaces, etc.).

## What it does

Replaces `rvc.train.Popen` with a subclass that, when called with a
string `cmd` and `shell=True`, runs the cmd through a regex re-quoter
before launching the subprocess. The re-quoter detects the pattern
`"<exe>" <unquoted-path>.py` and wraps the script path in quotes.

Targeted scope (`rvc.train` only) — unrelated subprocess work
elsewhere in ComfyUI and the rest of the embedded Python is left
alone.

## Layering

This node only patches `rvc.train.Popen`; it does not interact with
`RVC_Train.train` or `INPUT_TYPES`, so it composes cleanly with
`zz_rAIdio_rvc_bool_fix` and `zz_rAIdio_rvc_train_dir_fix`.

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

Copyright (c) 2025-2026 Creative Mayhem UG
