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

- Model library: GGUF scan with metadata, quant/params tags, HF search + download
- Server presets: `dense-gpu`, `moe-split`, `vision`, plus iGPU presets `igpu-full` /
  `igpu-vision` (Vulkan `--device` resolved via `--list-devices`, 780M RADV; fit
  guard refuses models larger than free iGPU memory + KV headroom)
- Avalon manager: profiles, LAN/API-key access, GPU-conflict guard, MTP spec decode
- **Coding-agent registry**: "Agents…" button writes the running model into other
  tools' configs — omp, pi, opencode, codex, aider, zed, continue, claude-code,
  hermes, kimi. Registry state: `agent-registry.json`; re-sync updates port/alias.
- Chat, vitals (amdgpu sysfs), downloads
