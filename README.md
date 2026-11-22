# Inference Console

One web app for running local models. It manages three inference engines,
**llama.cpp**, **Strata**, and **Avalon**, plus your GGUF library, live
telemetry, downloads, and chat, all from a single FastAPI file at
`http://127.0.0.1:8817`.

## Quick start

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
./venv/bin/python panel.py            # then open http://127.0.0.1:8817
```

Login is your system (sudo) password. For always-on operation, install the
systemd user unit in `deploy/` and `systemctl --user enable --now inference-console.service`
(manages the panel at `:8817`, one llama.cpp child on ports 8830-8839, and the
`avalon-flashnext.service` eGPU server).

> Formerly `lfm-panel`. Config and data migrate automatically from the old
> paths on first run; `LFM_PANEL_PORT` still works as a fallback for
> `INFER_CONSOLE_PORT`.

## Engines

Each engine gets its own tab with status, launch controls, presets, and logs.
Whichever one is serving becomes the chat backend and the coding-agent
registry target automatically.

- **llama.cpp**, launches and supervises a `llama-server` child. Presets for
  dense GPU, MoE CPU/GPU split, vision, and iGPU (Vulkan `--device` resolved
  via `--list-devices`, with a fit guard against free VRAM).
- **Strata**, wraps the `strata launch` CLI: launch form (preset, family,
  quant, context, KV format, parallel, vision, MTP, drafting, GPU, access
  mode), read-only plan preview before any download, preset save/delete,
  `doctor --live`, `update --check`, masked API key. Empty fields keep
  Strata's own defaults.
- **Avalon**, profiles (`flagship`, `prefast`, `compact`) with fit-aware
  launches (`-fit on` places dense + KV on the GPU, MoE experts in system
  memory; no hardcoded full offload), per-profile config optimizer with
  device/offload/`--n-cpu-moe`/FA/threads switches and recommendation hints,
  auto-target: newest complete Flash-Next pack enters the tab on its own
  (selectable from a dropdown, pinnable via `avalon_model`), one-click saved
  launch snapshots, LAN/API-key access, GPU-conflict guard against the other
  engines.

## Launch tuning

The launch dialog exposes per-model knobs with a live VRAM recalculation and a
CLI preview:

- Context, offload (`-ngl`), threads, flash attention, KV quant (`-ctk`/`-ctv`),
  `--parallel`, batches, `--n-cpu-moe`, `--mlock`, `--no-mmap`, free-form extra flags
- Speculative decoding: `--model-draft` with `--draft-max`/`--draft-min`;
  MTP/draft GGUFs sitting beside the model are auto-detected
- `-ot` tensor overrides, `--tensor-split`, `--threads-batch`, `--keep`, `--jinja`
- One-click optimizer with `balanced` / `speed` / `max_ctx` / `low_vram` targets
  (`speed` enables speculation when a draft file exists)

Every flag is gated on what your binary advertises in `--help`, so older
builds never receive flags they'd reject.

## Model library

- GGUF scan with metadata: arch, quant, params, ctx_len, blocks, experts,
  imatrix, mmproj
- Multi-shard packs group into one collapsible entry (sizes summed; deleting a
  pack removes all shards)
- VRAM/RAM fit meter on every card before you launch
- Pinning and recents
- Hugging Face downloads: resumable `.part` transfers, pause/resume, disk
  space guard

## Day to day

- **Telemetry**: live tok/s (generation & prefill), KV-cache fill, active
  parallel slots, idle auto-unload watchdog
- **Benchmarks**: one-click `standard` / `code` / `short` suites with saved
  history for comparison
- **Chat**: sampling controls, edit & regenerate, Markdown/JSON export,
  backed by whichever engine is up
- **Coding-agent registry**: writes the running model into omp, pi, opencode,
  codex, aider, zed, continue, claude-code, hermes, and kimi configs
- **Setup**: Settings installs the latest official llama.cpp release for your
  platform (Metal / Vulkan / cpu / cuda), the Strata CLI, or builds Avalon
  from source, each behind an explicit confirm, with a streamed log
- **Panel restart**: Settings > Panel process, or the command palette, re-execs
  the panel in place. Same PID, engines keep serving, and the tab reconnects
  itself
- **⌘K / Ctrl+K** command palette

## Layout

```
panel.py                          the whole app (UI is an inline HTML string)
deploy/inference-console.service  systemd user unit
tests/test_panel_metrics.py       stubbed unit tests
tests/test_panel_agents.py        coding-agent config writers
tests/test_panel_restart.py       panel restart, argv rebuild, session mirror
~/.config/inference-console/      config.json
~/.local/share/inference-console/ runtime state (logs, benchmarks, chats, sessions.json)
```

## Tests

```bash
python3 -m pytest tests/
```

Host probes (nvidia-smi, /proc, llama-server health, Strata CLI) are stubbed,
so the suite runs on any machine, GPU or not.
