# Memescreamer Memory Cleaner

First-party ComfyUI custom node for the memescreamer platform (rAIdio.bot / vAIdeo.bot).
Copyright (c) 2025 Creative Mayhem. Free for non-commercial use under CC BY-NC-SA 4.0;
commercial use requires a paid license (license@creativemayhem.com). rAIdio.bot ships
under that commercial grant.

## What it does

Exposes a single server route — **`POST /clear_memory`** — that returns GPU memory to
the driver after a generation. It reaches the two places ComfyUI's built-in `/free`
endpoint cannot:

1. **Whisper cache** — `rAIdio_whisper_patch._WHISPER_CACHE`, a module-level dict that
   holds the faster-whisper model across Sing-Along transcriptions. The route scans
   `sys.modules` for the live module and clears it (importing by name would bind a fresh,
   empty cache and free nothing).
2. **Caching-allocator residue** — SeedVC (voice) and Demucs (stems) load fresh each call
   and drop their Python ref on return, but the freed blocks stay *reserved* in PyTorch's
   CUDA caching allocator. The route forces `gc.collect()` + `empty_cache()` so those
   blocks are returned to the driver and the OS reflects the drop.

It also unloads `model_management`-tracked models (belt-and-suspenders) and trims the
Windows working set.

## Who calls it

The rAIdio.bot backend's `free_vram` command calls `POST /clear_memory` after a generation
on memory-tight cards (< 24 GB VRAM). Larger cards skip it to keep models warm.

Server-route-only: no workflow node is registered.
