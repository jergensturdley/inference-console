# LFM Panel

Single-file FastAPI app: local model library, llama-server manager, system vitals, chat.
Runs at `:8817` as the `lfm-panel.service` systemd user unit; manages one llama.cpp
child server (ports 8830–8839) and the `avalon-flashnext.service` eGPU llama-server.

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

- **Model library & VRAM fit**: GGUF scan with metadata (`arch`, `quant`, `params`, `ctx_len`, `blocks`, `experts`, `imatrix`, `mmproj`), pre-launch VRAM/RAM fit meter on every model card, model pinning & recents
- **Launch dialog & config optimizer**: Interactive per-model tuning (`ctx`, `-ngl`, `-t`, `-fa`, `-ctk`/`-ctv`, `--parallel`, `-b`/`-ub`, `--n-cpu-moe`, `--mlock`, `--no-mmap`, extra CLI flags) with live VRAM recalculation, CLI preview, and auto-config optimizer (`balanced`, `speed`, `max_ctx`, `low_vram` targets)
- **Server & live telemetry**: Live `tok/s` (generation & prefill), KV-cache fill %, active parallel slots, one-click benchmark suite (`standard`, `code`, `short`) with persistent comparison history, and configurable idle auto-unload watchdog
- **Server presets**: `dense-gpu`, `moe-split`, `vision`, plus iGPU presets `igpu-full` / `igpu-vision` (Vulkan `--device` resolved via `--list-devices`, 780M RADV; fit guard refuses models larger than free iGPU memory + KV headroom)
- **Avalon manager**: Profiles (`flagship`, `prefast`, `compact`), interactive Avalon Config Optimizer (`ctx`, `batch`, `ubatch`, `ctk`, `ctv`, MTP `draft_n`, `ubd`, `parallel`), LAN/API-key access, GPU-conflict guard
- **Coding-agent registry**: "Register in coding agents…" writes the running model into other tools' configs — omp, pi, opencode, codex, aider, zed, continue, claude-code, hermes, kimi (`agent-registry.json`)
- **Chat, HF pull & settings**: Sampling controls (`temperature`, `top_p`, `min_p`, `repeat_penalty`, `max_tokens`), message edit & regenerate, Markdown/JSON chat export, resumable `.part` Hugging Face downloads with pause/resume & disk space guard, dedicated Settings tab, and `⌘K` / `Ctrl+K` command palette
