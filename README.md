# LFM Panel

Single-file FastAPI app: local model library, engine managers (llama.cpp, Strata,
Avalon), system vitals, chat.
Runs at `:8817` as the `lfm-panel.service` systemd user unit; manages one llama.cpp
child server (ports 8830-8839), the Strata CLI (`strata launch`), and the
`avalon-flashnext.service` eGPU llama-server.

## Run

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
systemctl --user enable --now ~/.config/systemd/user/lfm-panel.service
# http://127.0.0.1:8817  (login = your sudo password)
```

## Layout

- `panel.py` — the whole app (UI is an inline HTML string)
- `deploy/lfm-panel.service` — systemd user unit
- Config: `~/.config/lfm-panel/config.json`
- Runtime state (git-ignored): `venv/`, `bin/` (llama-server build), `chats/`, `*.json`, `*.bak.*`

## Features

- **Engine tabs**: llama.cpp, Strata, and Avalon each have their own tab with status, launch controls, presets, and logs. Whichever engine is serving becomes the chat backend and the coding-agent registry target automatically; the header shows all live engines.
- **Model library & VRAM fit**: GGUF scan with metadata (`arch`, `quant`, `params`, `ctx_len`, `blocks`, `experts`, `imatrix`, `mmproj`), pre-launch VRAM/RAM fit meter on every model card, model pinning & recents
- **Launch dialog & config optimizer**: Interactive per-model tuning (`ctx`, `-ngl`, `-t`, `-fa`, `-ctk`/`-ctv`, `--parallel`, `-b`/`-ub`, `--n-cpu-moe`, `--mlock`, `--no-mmap`, extra CLI flags) plus speculative decoding (`--model-draft` with `--draft-max`/`--draft-min`, auto-detects an MTP/draft GGUF sitting beside the model), tensor overrides (`-ot`), `--tensor-split`, `--threads-batch`, `--keep` (system-prompt cache), and a `--jinja` toggle. Live VRAM recalculation, CLI preview, and an auto-config optimizer (`balanced`, `speed`, `max_ctx`, `low_vram` targets; `speed` turns on speculative decoding when a draft file is found). Every flag is gated on what the binary advertises in `--help`, so old builds never receive flags they reject.
- **Strata manager**: Detects the Strata CLI (Settings path, `$STRATA_BIN`, `PATH`, `~/.strata/bin`, `/usr/local/bin`, the app bundle helper). Status card from `strata launch --status`, launch form covering preset, family, quant, model path, context, KV format, parallel, vision, MTP, drafting mode, GPU selection, access (local/lan/cloudflare/tailscale), port, `--set` pairs, and preset save; read-only plan preview before any download; preset list with use/delete; `strata doctor --live` and `strata update --check`; masked API key with click-to-reveal; copyable install command when Strata is missing. Only documented flags are emitted; an empty field keeps Strata's own default.
- **Server & live telemetry**: Live `tok/s` (generation & prefill), KV-cache fill %, active parallel slots, one-click benchmark suite (`standard`, `code`, `short`) with persistent comparison history, and configurable idle auto-unload watchdog
- **Server presets**: `dense-gpu`, `moe-split`, `vision`, plus iGPU presets `igpu-full` / `igpu-vision` (Vulkan `--device` resolved via `--list-devices`, 780M RADV; fit guard refuses models larger than free iGPU memory + KV headroom)
- **Avalon manager**: Profiles (`flagship`, `prefast`, `compact`), interactive Avalon Config Optimizer (`ctx`, `batch`, `ubatch`, `ctk`, `ctv`, MTP `draft_n`, `ubd`, `parallel`), LAN/API-key access, GPU-conflict guard
- **Engine setup (one-click)**: Settings opens with an Engine setup card — installs the latest official llama.cpp release build for your platform (macOS Metal / Linux & Windows Vulkan, with cpu/cuda variants) and points `server_binary` at it, runs the official Strata installer, or builds Avalon from `~/avalon` with cmake. Update detection compares the installed build number against the newest release. Jobs run single-flight in the background with a streamed log, and every install requires an explicit confirmation first.
- **Coding-agent registry**: "Register in coding agents…" writes the running model into other tools' configs — omp, pi, opencode, codex, aider, zed, continue, claude-code, hermes, kimi (`agent-registry.json`). Works for llama.cpp, Strata, and Avalon launches alike.
- **Chat, HF pull & settings**: Sampling controls (`temperature`, `top_p`, `min_p`, `repeat_penalty`, `max_tokens`), message edit & regenerate, Markdown/JSON chat export, resumable `.part` Hugging Face downloads with pause/resume & disk space guard, dedicated Settings tab (model dirs, binaries, Strata CLI path/alias/API-key override, ports, idle watchdog, VRAM reserve, HF token), and `⌘K` / `Ctrl+K` command palette

## Tests

```bash
python3 -m pytest tests/          # or: python3 tests/test_panel_metrics.py
```

Host probes (nvidia-smi, /proc, llama-server health, Strata CLI) are stubbed; the suite
runs on any machine without a GPU.
