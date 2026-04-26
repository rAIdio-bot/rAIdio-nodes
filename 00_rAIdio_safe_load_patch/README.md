# rAIdio Safe Load Patch

ComfyUI custom node that closes the malicious-pickle-via-shared-model-file
attack vector across every standard pickle deserialisation entry point
in the bundled backend.

## What it does

Wraps three primitives at custom-node import time:

| Primitive | Wrapper behaviour |
|-----------|-------------------|
| `torch.load` | Defaults `weights_only=True` (PyTorch's safe-unpickle mode). Picklescan-gates the file path. Dangerous globals → refuse. Picklescan errors → warn and fall through to weights_only-only (kernel-level fallback). |
| `pickle.load`, `pickle.loads` | Picklescan-gates the file/bytes. Dangerous globals → refuse. Picklescan errors → **fail closed** (refuse). No kernel fallback exists for raw pickle. |
| `numpy.load` | Defaults `allow_pickle=False`. If the caller insists on `allow_pickle=True`, picklescan-gates the file before deserialisation (handles `.npy` raw and `.npz` zip-of-arrays). |

All three layers refuse when a file's pickle contents reference globals
known to indicate code-execution intent — `os.system`, `nt.system`,
`posix.system`, `subprocess.{Popen,call,check_call,check_output,run}`,
`builtins.{eval,exec,compile,open,__import__}`, `pty.spawn`,
`shutil.rmtree`, `platform.popen`, `os.popen`. Refusal happens BEFORE
the deserialiser runs, so callers that explicitly opt into legacy
pickle (e.g. `weights_only=False`, `allow_pickle=True`) are still
gated.

The `00_` filename prefix forces ComfyUI to load this BEFORE any
node that touches these primitives at import time, so each rebind
is in place before first use.

If `picklescan` is not installed, `torch.load` falls back to
`weights_only=True`-only mode (still safe for state-dict loads); the
`pickle.load` and `numpy.load` wrappers don't apply because they have
no kernel-level safety net to fall back on.

## Threat model

Defends against: a user downloads a "voice model.pth" / "embedding.pt"
/ "preset.npy" from the internet and drops it into rAIdio's model
folder. Without this patch, the next generation runs `torch.load(file)`
or `pickle.load(file)` and arbitrary Python from the malicious pickle
executes on the user's machine. With this patch, the wrapper refuses
the file at scan time, before any deserialiser opcode runs.

Full Phase 2a/2b/2c scope:
<https://github.com/rAIdio-bot/sbom/blob/main/docs/security/model-loading.md>

## Compatibility

Designed against PyTorch ≥ 2.6 where `weights_only=True` is the safe
path for state-dict loads. Older torch silently accepts the kwarg as
a no-op; the picklescan gate still applies.

Tested in the rAIdio.bot bundled Python (CPython 3.12 + torch 2.10 +
numpy 2.x + picklescan 1.0.4). Smoke covers torch.load (default and
weights_only=False), pickle.load with real-file and BytesIO sources,
pickle.loads with raw bytes, numpy.load with default and
allow_pickle=True modes — 11 tests, all passing.

## Failure posture

| Scenario | Behaviour |
|----------|-----------|
| Patch import fails entirely | Logged warning, ComfyUI continues. Node still loads (empty NODE_CLASS_MAPPINGS). |
| Picklescan import fails | torch.load wrapper falls back to weights_only-only. pickle.load and numpy.load wrappers are silently not applied. |
| Picklescan parse error on a torch.load file | Warning logged. weights_only=True kernel safety still applies. |
| Picklescan parse error on a pickle.load* file or bytes | **Fail closed** — RuntimeError raised. No deserialisation. |
| Picklescan parse error on a numpy.load file | Warning logged. numpy still loads (allow_pickle handling unchanged from default). |

## License

GPL-3.0 — applied to GPL-3.0 licensed ComfyUI ecosystem.

Copyright (c) 2026 Creative Mayhem UG
