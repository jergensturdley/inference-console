# Inference Console

One web app for running local models. It manages four inference engines,
**llama.cpp**, **Strata**, **Avalon**, and **Project Maya**, plus your GGUF
library, a config-optimization agent, live telemetry, downloads, and chat, all
from a single FastAPI app at `http://127.0.0.1:8817`.

## Quick start

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
./venv/bin/python panel.py            # then open http://127.0.0.1:8817
```

Login is your system (sudo) password. When a session expires, the page sends
you back to the sign-in screen instead of quietly freezing on stale data. For
always-on operation, install the
systemd user unit in `deploy/` and `systemctl --user enable --now inference-console.service`
(manages the panel at `:8817`, one llama.cpp child on ports 8830-8839, and the
`avalon-flashnext.service` eGPU server).

> Formerly `lfm-panel`. Config and data migrate automatically from the old
> paths on first run; `LFM_PANEL_PORT` still works as a fallback for
> `INFER_CONSOLE_PORT`.

## Engines

Each engine gets its own tab with status, launch controls, presets, and logs.
Every stopped tab has a one-click way to start: Strata starts its last-used
preset, Avalon and Maya start the selected profile or config, and the
llama.cpp tab links into the model library. Whichever engine is serving
becomes the chat backend and the coding-agent registry target automatically.
Engines are GPU-exclusive: starting one is refused while another holds VRAM,
so there is never a silent OOM.

- **llama.cpp**: launches and supervises a `llama-server` child. Presets for
  dense GPU, MoE CPU/GPU split, vision, and iGPU (Vulkan `--device` resolved
  via `--list-devices`, with a fit guard against free VRAM).
- **Strata**: wraps the `strata launch` CLI. A Start button launches the
  last-used preset with the settings it saved, in one click. The launch form
  (preset, family, quant, context, KV format, parallel, vision, MTP,
  drafting, GPU, access mode) pre-fills from that preset; empty fields keep
  Strata's own defaults. The read-only plan runs before any launch, and the
  confirm says whether the model files are already local or a download is
  coming, so nothing implies a download that is not. Preset save/delete,
  `doctor --live`, `update --check`, masked API key. A failed
  `--list-presets` run shows its error in the presets card instead of
  pretending no presets exist.
- **Avalon**: profiles (`flagship`, `prefast`, `compact`) with fit-aware
  launches (`-fit on` places dense + KV on the GPU, MoE experts in system
  memory; no hardcoded full offload), per-profile config optimizer with
  device/offload/`--n-cpu-moe`/FA/threads switches and recommendation hints,
  auto-target: newest complete Flash-Next pack enters the tab on its own
  (selectable from a dropdown, pinnable via `avalon_model`), one-click saved
  launch snapshots, LAN/API-key access, GPU-conflict guard against the other
  engines.
- **Project Maya** (`~/project-maya`): serves GLM and HY4 models from your own
  Maya checkout with its HIP/ROCm build (`maya-rocm`). JSON model configs are
  discovered automatically (port, quant, context, MTP, dir size, key state
  shown per config), started through the bundled `maya.py` launcher in its
  venv, one config at a time, with fast Switch between configs, log-parsed
  telemetry (prefill/generation tok/s, VRAM, routed experts), masked API key
  forwarding, and register-in-coding-agents support. Paths are configurable
  via `maya_dir` / `maya_rocm_path`.

## Optimization agent

A hybrid agent (`optimizer_agent.py`) that tunes any of the four engines:

- **Dry-run advisor**: instant config recommendations from baked-in MoE
  mechanics (attention/router/KV on GPU vs routed experts in RAM, never
  dismisses a model just because its file size exceeds VRAM), documentation
  grounding (`--help`, `strata doctor`, bundled guides), and your hardware.
  Zero restarts.
- **Active auto-tune**: bounded launch, benchmark, iterate loop that searches
  for the best config per goal: `fastest`, `low_latency`, `max_context`,
  `balanced`, `low_vram`, `multi_user`.
- **Checkpoints and rollback**: every optimization snapshots the prior config;
  a dud config is one click to restore. Crash/OOM self-healing included.
- **Persistent memory** (`agent_memory.py`): what worked best per
  (model, engine, goal) is remembered across restarts, along with reusable
  presets (built-ins included) and real-world VRAM calibration.
- **Error diagnosis**: paste a launch failure and get a grounded diagnosis
  of the offending flag.
- **LLM layer (optional)**: provider presets for OpenAI, Anthropic,
  OpenRouter, Gemini, DeepSeek, xAI, Groq, Together, Fireworks, Mistral,
  Moonshot, Z.ai, Nous, Ollama, LM Studio, vLLM, or any custom
  OpenAI-compatible endpoint, with always-on model polling and a connection
  test. Falls back to the currently running local engine when no cloud
  provider is set.
- **Speculation tuning**: picks `--model-draft` / draft-length flags when a
  draft or MTP GGUF sits beside the model, and can recommend HF draft models.
- **Thermal guard** (`thermal_guard.py`): graduated escalation, alert-only or
  auto-acting. Level 1 ramps fans and trims the GPU power cap, Level 2
  checkpoints and downshifts to a cooler profile, Level 3 is an emergency
  killswitch, with hysteresis to restore hardware state once cooled, plus
  external VRAM-pressure alerts when another app shares the GPU.

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

- **Workbench UI**: left-rail workbench with a top command bar and command
  palette (**⌘K / Ctrl+K**), tabbed engine views with per-tab state
- **Telemetry**: live tok/s (generation & prefill), KV-cache fill, active
  parallel slots, idle auto-unload watchdog
- **Benchmarks**: one-click `standard` / `code` / `short` suites with saved
  history for comparison
- **Chat**: sampling controls, edit & regenerate, Markdown/JSON export,
  backed by whichever engine is up
- **Coding-agent registry**: writes the running model into omp, pi, opencode,
  codex, aider, zed, continue, claude-code, hermes, and kimi configs, with
  API-key forwarding for engines that use one
- **Setup**: Settings installs the latest official llama.cpp release for your
  platform (Metal / Vulkan / cpu / cuda), the Strata CLI, or builds Avalon
  from source, each behind an explicit confirm, with a streamed log
- **Panel restart**: Settings > Panel process, or the command palette, re-execs
  the panel in place. Same PID, engines keep serving, and the tab reconnects
  itself

## Layout

```
panel.py                          the app and inline HTML UI
optimizer_agent.py                hybrid engine-config optimization agent
agent_memory.py                   agent presets, checkpoints, persistent memory
thermal_guard.py                  graduated GPU thermal & VRAM guard
deploy/inference-console.service  systemd user unit
tests/test_panel_metrics.py       stubbed unit tests
tests/test_panel_agents.py        coding-agent config writers
tests/test_panel_restart.py       panel restart, argv rebuild, session mirror
tests/test_panel_maya.py          Maya discovery, start/stop, telemetry parsing
tests/test_optimizer_agent.py     advisor, auto-tune, memory, provider presets
~/.config/inference-console/      config.json
~/.local/share/inference-console/ runtime state (logs, benchmarks, chats, agent/, sessions.json)
```

## Tests

```bash
python3 -m pytest tests/
```

Host probes (nvidia-smi, /proc, llama-server health, Strata CLI) are stubbed,
so the suite runs on any machine, GPU or not.
