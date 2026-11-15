#!/usr/bin/env python3
"""LFM Panel — local model library, llama-server control, system vitals, chat.

Single-file FastAPI app. Dependencies: fastapi, uvicorn, httpx.
Config: ~/.config/lfm-panel/config.json (created with defaults on first run).
"""
import json, os, re, secrets, shutil, signal, socket, struct, subprocess, threading, time, uuid
from collections import deque
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

HOME = Path.home()
APP_DIR = HOME / ".local/share/lfm-panel"
CONFIG_PATH = HOME / ".config/lfm-panel/config.json"
STATE_PATH = APP_DIR / "state.json"
BIN = str(APP_DIR / "bin/llama-server")
PORT = int(os.environ.get("LFM_PANEL_PORT", "8817"))
CHILD_PORTS = list(range(8830, 8840))

DEFAULT_CONFIG = {
    "model_dirs": [str(HOME / "models/lfm3")],
    "server_binary": BIN,
    "ctx": 8192,
    "port_base": 8830,
    "bind_host": "0.0.0.0",      # the admin API can launch arbitrary binaries — use 127.0.0.1 for local-only
    "avalon_lan": False,
    "avalon_api_key": "",
    "idle_unload_min": 0,        # 0 = never auto-unload the panel's llama-server
    "hf_token": "",              # falls back to $HF_TOKEN
    "vram_reserve_mb": 512,      # headroom kept free by the fit estimator / optimizer
    "avalon_tune": {},           # overrides for the Avalon launch line (see AVALON_TUNE_DEFAULTS)
}

# ---------------------------------------------------------------- config
def load_config():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    cfg = json.loads(CONFIG_PATH.read_text())
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg

CFG = load_config()
APP_DIR.mkdir(parents=True, exist_ok=True)

def _pid_alive(pid) -> bool:
    """True if the process exists (portable: no /proc needed)."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False

def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def _write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)

# ---------------------------------------------------------------- llama-server capabilities
_caps_cache: dict = {}

def bin_caps(binary: str) -> dict:
    """Parse `llama-server --help` once per binary build so the panel only emits flags it understands."""
    try:
        st = os.stat(binary)
    except OSError:
        return {"exists": False}
    key = (binary, st.st_mtime_ns)
    if key in _caps_cache:
        return _caps_cache[key]
    help_txt, ver = "", ""
    try:
        help_txt = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        pass
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
        m = re.search(r"version:\s*(.+)", (r.stdout or "") + (r.stderr or ""))
        ver = m.group(1).strip() if m else ""
    except Exception:
        pass
    caps = {
        "exists": True, "version": ver,
        "fa_value": bool(re.search(r"--flash-attn \[on\|off", help_txt)),
        "ncmoe": "--n-cpu-moe" in help_txt,
        "metrics": "--metrics" in help_txt,
        "fit": bool(re.search(r"--fit \[on\|off\]", help_txt)),
        "ctk": "--cache-type-k" in help_txt,
        "mlock": "--mlock" in help_txt,
        "no_mmap": "--no-mmap" in help_txt,
        "parallel": "--parallel" in help_txt,
    }
    if not help_txt:   # unknown build: assume a modern llama.cpp
        caps.update(fa_value=True, ncmoe=True, metrics=True, ctk=True, mlock=True, no_mmap=True, parallel=True)
    _caps_cache.clear()
    _caps_cache[key] = caps
    return caps

# ---------------------------------------------------------------- gguf parsing
GTYPES = {0:1,1:1,2:1,3:1,4:4,5:4,6:4,7:1,8:None,9:None,10:8,11:8,12:8}  # value sizes; None=var
NAMES = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",8:"str",9:"arr",10:"u64",11:"i64",12:"f64"}
FMT = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),4:("<I",4),5:("<i",4),6:("<f",4),7:("<?",1),10:("<Q",8),11:("<q",8),12:("<d",8)}
WANT_GENERIC = {"general.architecture","general.name","general.file_type","general.size_label","general.quantization_imatrix"}

def gguf_metadata(path: Path, limit_mb: int = 64):
    """Parse GGUF header KVs. Returns dict of interesting keys. Raises on bad magic."""
    out = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF")
        version, = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"gguf v{version} unsupported")
        f.seek(8)
        tensor_count, kv_count = struct.unpack("<QQ", f.read(16))

        def read_str():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def read_val(t):
            if t == 8:
                return read_str()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                if et == 8:
                    vals = [read_str() for _ in range(n)]
                else:
                    fmt, sz = FMT[et]
                    vals = [struct.unpack(fmt, f.read(sz))[0] for _ in range(n)]
                return vals
            fmt, sz = FMT[t]
            return struct.unpack(fmt, f.read(sz))[0]

        for _ in range(kv_count):
            key = read_str()
            t, = struct.unpack("<I", f.read(4))
            val = read_val(t)
            arch = out.get("general.architecture")
            if key in WANT_GENERIC or (arch and key.startswith(arch + ".")):
                # head_count_kv / key_length / value_length are ARRAY-typed in GGUF. Every
                # converter emits a single element, so collapse numeric arrays to element 0 —
                # dropping them made GQA models look like MHA and inflated the KV estimate.
                if isinstance(val, list):
                    if val and not isinstance(val[0], str):
                        out[key] = val[0]
                else:
                    out[key] = val
            if f.tell() > limit_mb << 20:
                break
    return out

QUANT_END_RE = re.compile(r"((?:IQ|TQ)\d_[0-9A-Z_]+|Q\d_[0-9A-Z_]+|MXFP4|BF16|FP16|F16|F32)$")
QUANT_ANY_RE = re.compile(r"((?:IQ|TQ)\d_[0-9A-Z_]+|Q\d_[0-9A-Z_]+|MXFP4|BF16|FP16|F16|F32)")
MOE_RE = re.compile(r"\bA\d+B\b|\bA\d+x\d+B\b|[Mm]oe\b|[Mm]ixture")
_TOKEN_DROP = re.compile(r"^(i1|mmproj|[0-9.]+[bmk]?|[a-z])$", re.I)

def _tokens(name):
    out = set()
    for t in re.split(r"[-_. ]", name.lower()):
        if t and not _TOKEN_DROP.match(t) and not QUANT_ANY_RE.fullmatch(t.upper()):
            out.add(t)
    return out

def find_mmproj(model_path: str):
    p = Path(model_path)
    want = _tokens(p.stem)
    best = None
    for sib in p.parent.glob("*mmproj*.gguf"):
        toks = _tokens(sib.stem.replace("mmproj", ""))
        overlap = len(want & toks) / max(1, len(want | toks))
        sl = sib.stem.lower()
        pref = 2 if "q4" in sl else 1 if "q8" in sl else 0
        cand = (round(overlap, 3), pref, sib)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return str(best[2]) if best and best[0] > 0 else None

def human_size(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024

_file_type = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",7:"Q8_0",8:"Q5_0",9:"Q5_1",10:"Q2_K",11:"Q3_K_S",12:"Q3_K_M",13:"Q3_K_L",14:"Q4_K_S",15:"Q4_K_M",16:"Q5_K_S",17:"Q5_K_M",18:"Q6_K",19:"IQ2_XXS",20:"IQ2_XS",21:"Q2_K_S",22:"IQ3_XS",23:"IQ3_XXS",24:"IQ1_S",25:"IQ4_NL",26:"IQ4_XS",27:"Q3_K_XS",28:"IQ3_S",29:"IQ2_S",30:"IQ4_XS",32:"IQ1_M",36:"TQ1_0",37:"TQ2_0",38:"BF16"}

_model_cache = {}

def scan_models():
    models, accessories = [], []
    seen, _dupe_seen = set(), set()
    for d in CFG["model_dirs"]:
        p = Path(d).expanduser()
        if not p.is_dir():
            continue
        for f in sorted(p.rglob("*.gguf")):
            try:
                rp = f.resolve(strict=True)
            except OSError:
                continue
            if rp in seen:
                continue
            seen.add(rp)
            name_l = f.name.lower()
            if name_l.startswith("ggml-vocab"):
                continue
            st = rp.stat()
            if st.st_size < 1_000_000:  # LFS pointers / corrupt stubs
                continue
            key = (str(rp), st.st_mtime_ns, st.st_size)
            if "mmproj" in name_l or name_l.startswith("mtp-"):
                accessories.append({"path": str(f), "size": st.st_size})
                continue
            if key in _model_cache:
                models.append(_model_cache[key])
                continue
            dupe_key = (f.stem, st.st_size)
            if dupe_key in _dupe_seen:
                continue
            _dupe_seen.add(dupe_key)
            try:
                md = gguf_metadata(f)
            except Exception as e:
                md = {"error": str(e)}
            arch = md.get("general.architecture", "?")
            stem = f.stem
            qm = QUANT_END_RE.search(stem) or QUANT_ANY_RE.search(stem)
            quant = qm.group(1) if qm else _file_type.get(md.get("general.file_type", -1), "?")
            expert_count = md.get(f"{arch}.expert_count", 0) or 0
            expert_used = md.get(f"{arch}.expert_used_count", 0) or 0
            moe = bool(expert_count) or bool(MOE_RE.search(stem))
            name_label = md.get("general.size_label", "")
            embd = md.get(f"{arch}.embedding_length", 0) or 0
            n_head = md.get(f"{arch}.attention.head_count", 0) or 0
            n_kv = md.get(f"{arch}.attention.head_count_kv", n_head) or n_head
            k_len = md.get(f"{arch}.attention.key_length", (embd // n_head) if n_head else 128) or 128
            v_len = md.get(f"{arch}.attention.value_length", k_len) or k_len
            blocks = md.get(f"{arch}.block_count", 0) or 0
            mmproj = find_mmproj(str(f))
            m = {
                "path": str(f), "name": stem, "arch": arch,
                "quant": quant, "size": st.st_size, "size_h": human_size(st.st_size),
                "moe": moe, "vl": bool(VL_NAME_RE.search(stem)),
                "params": name_label, "ctx_len": md.get(f"{arch}.context_length", 0),
                "blocks": blocks, "embd": embd, "n_head": n_head, "n_kv": n_kv,
                "k_len": k_len, "v_len": v_len,
                "experts": expert_count, "experts_used": expert_used,
                "imatrix": bool(md.get("general.quantization_imatrix", False)),
                "mmproj": mmproj,
            }
            _model_cache[key] = m
            models.append(m)
    models.sort(key=lambda m: m["name"].lower())
    # Cache keys carry mtime+size, so every re-download or re-quant adds an entry that a
    # scan never drops on its own. Forget entries for files that are gone.
    live_paths = {m["path"] for m in models}
    for stale in [k for k in _model_cache if k[0] not in live_paths]:
        _model_cache.pop(stale, None)
    return {"models": models, "mmproj": accessories}

# ---------------------------------------------------------------- VRAM fit & launch profiles
LAUNCH_PROFILES_PATH = APP_DIR / "launch-profiles.json"
MODEL_META_PATH = APP_DIR / "model-meta.json"
BENCHMARKS_PATH = APP_DIR / "benchmarks.json"

KV_TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625,
    "q5_1": 0.75, "q5_0": 0.6875, "q4_1": 0.625, "q4_0": 0.5625, "iq4_nl": 0.5625,
}

# Vision-language name markers. Substring matching is kept for the glued families
# (InternVL, Qwen3-VL without a separator) that word-boundary matching would miss, but
# `omni`/`vl` must not run straight into more letters, which is what made a bare
# `"vl" in stem` / `"omni" in stem` test flag OmniDocBench-style names.
# A paired mmproj sidecar is deliberately NOT used as a vl signal: find_mmproj() matches
# on token overlap, and in a flat library where files share a name prefix that overlap is
# non-zero for unrelated models, so it would tag every model VL (and flip the `auto`
# preset to vision). The name is the reliable signal here.
VL_NAME_RE = re.compile(
    r"(?i)(?:vlm|internvl|llava|pixtral|molmo|minicpm-v|vision|visual"
    r"|omni(?![a-z])|vl(?![a-z]))")

def estimate_vram(model_meta: dict, ctx: int = 8192, ngl: int = 99,
                  ctk: str = "f16", ctv: str = "f16", parallel: int = 1,
                  preset: str = "auto", ncmoe: int = 0,
                  vit: dict | None = None, srv_running: bool | None = None) -> dict:
    """Estimate VRAM & RAM footprint in MB for loading `model_meta` with given launch settings."""
    size_mb = (model_meta.get("size", 0) or 0) / 1048576.0
    blocks = int(model_meta.get("blocks") or 32)
    n_kv = int(model_meta.get("n_kv") or 8)
    k_len = int(model_meta.get("k_len") or 128)
    v_len = int(model_meta.get("v_len") or 128)
    moe = bool(model_meta.get("moe"))
    if preset in ("auto", ""):
        preset = "moe-split" if moe else ("vision" if (model_meta.get("vl") and model_meta.get("mmproj")) else "dense-gpu")

    eff_ngl = min(blocks, max(0, int(ngl))) if blocks > 0 else 32
    layer_frac = (eff_ngl / blocks) if blocks > 0 else 1.0

    # MoE expert split or --n-cpu-moe keeps expert FFN weights (~75% of MoE file) on CPU RAM
    if preset == "moe-split" and moe:
        weights_vram_mb = size_mb * 0.24 * layer_frac
        weights_ram_mb = size_mb - weights_vram_mb
    elif ncmoe and ncmoe > 0 and moe and blocks > 0:
        cpu_moe_frac = min(1.0, ncmoe / blocks)
        gpu_weights = size_mb * (1.0 - 0.75 * cpu_moe_frac) * layer_frac
        weights_vram_mb = gpu_weights
        weights_ram_mb = size_mb - weights_vram_mb
    else:
        weights_vram_mb = size_mb * layer_frac
        weights_ram_mb = size_mb * (1.0 - layer_frac)

    # Vision projector overhead if present
    mm_mb = 0.0
    if model_meta.get("mmproj") and (preset in ("vision", "igpu-vision") or model_meta.get("vl")):
        try:
            mm_mb = Path(model_meta["mmproj"]).stat().st_size / 1048576.0
        except OSError:
            mm_mb = 600.0

    kb = KV_TYPE_BYTES.get((ctk or "f16").lower(), 2.0)
    vb = KV_TYPE_BYTES.get((ctv or "f16").lower(), 2.0)
    # KV cache scales with total context across slots (llama.cpp -c is total context across slots unless divided,
    # but users often scale -c with parallel; we compute KV for the configured -c plus a small slot overhead).
    kv_per_tok_bytes = blocks * n_kv * (k_len * kb + v_len * vb)
    kv_total_mb = (kv_per_tok_bytes * max(512, int(ctx))) / 1048576.0
    kv_vram_mb = kv_total_mb * layer_frac
    kv_ram_mb = kv_total_mb * (1.0 - layer_frac)

    compute_mb = 280.0 + (120.0 if int(ctx) >= 32768 else 0.0) + (60.0 * max(0, int(parallel) - 1))
    total_vram_mb = round(weights_vram_mb + mm_mb + kv_vram_mb + (compute_mb if eff_ngl > 0 else 0))
    total_ram_mb = round(weights_ram_mb + kv_ram_mb + 250.0)

    # Compare against available device memory
    if vit is None:
        vit = vitals()
    igpu = preset.startswith("igpu")
    if igpu:
        _, free_mib = igpu_device()
        budget_mb = free_mib or int(vit.get("mem", {}).get("total_gb", 16) * 1024 * 0.6)
        device_label = "iGPU (780M)"
    elif "gpu" in vit and vit["gpu"].get("total_mb"):
        g = vit["gpu"]
        # If our own managed server is currently holding VRAM, count its VRAM as reclaimable on reload
        if srv_running is None:
            srv_running = MGR.status().get("running") or SMGR.status().get("running")
        used_other = 0 if srv_running else min(g.get("used_mb", 0), 1024)
        budget_mb = max(0, g["total_mb"] - used_other)
        device_label = g.get("name", "dGPU VRAM")
    else:
        budget_mb = int(vit.get("mem", {}).get("total_gb", 16) * 1024 * 0.75)
        device_label = "Unified / System RAM"

    reserve_mb = int(CFG.get("vram_reserve_mb", 512) or 512)
    usable_mb = max(512, budget_mb - reserve_mb)
    ratio = total_vram_mb / usable_mb if usable_mb > 0 else 1.0
    if ratio <= 0.85:
        status, badge = "fit", "Comfortably fits"
    elif ratio <= 1.0:
        status, badge = "tight", "Tight fit"
    else:
        status, badge = "overflow", "Exceeds memory"

    return {
        "weights_vram_mb": round(weights_vram_mb + mm_mb),
        "kv_vram_mb": round(kv_vram_mb),
        "compute_vram_mb": round(compute_mb if eff_ngl > 0 else 0),
        "compute_mb": round(compute_mb if eff_ngl > 0 else 0),
        "total_vram_mb": total_vram_mb,
        "vram_used_mb": total_vram_mb,
        "total_ram_mb": total_ram_mb,
        "ram_offload_mb": total_ram_mb,
        "budget_mb": round(budget_mb),
        "usable_mb": round(usable_mb),
        "device": device_label,
        "budget_source": device_label,
        # Same denominator as `status`/`badge` above: the reserve is what the verdict is
        # measured against, so a "Tight fit" badge can't sit next to a smaller percentage.
        "pct": min(999, round(100 * total_vram_mb / max(1, usable_mb))),
        "status": status,
        "badge": badge,
        "label": badge,
        "preset_resolved": preset,
        "eff_ngl": eff_ngl,
        "blocks": blocks,
    }

def recommend_model_config(model_meta: dict, target: str = "balanced") -> dict:
    """Config optimization helper: computes optimal launch flags for a model given host hardware & goal."""
    blocks = int(model_meta.get("blocks") or 32)
    moe = bool(model_meta.get("moe"))
    vl = bool(model_meta.get("vl") and model_meta.get("mmproj"))
    native_ctx = int(model_meta.get("ctx_len") or 32768)
    cores = os.cpu_count() or 8
    phys_threads = max(4, min(16, cores // 2 if cores >= 12 else cores))
    # Snapshot hardware state once: the overflow search below calls estimate_vram dozens
    # of times, and each call would otherwise fork nvidia-smi and probe both /health.
    vit = vitals()
    srv_running = bool(MGR.status().get("running") or SMGR.status().get("running"))

    # Determine optimal preset & KV quantization by target goal:
    # target in ("speed", "balanced", "max_ctx", "low_vram")
    if target == "speed":
        ctk, ctv = "q8_0", "q8_0"
        ctx = min(native_ctx or 16384, 16384)
        parallel = 1
    elif target == "max_ctx":
        ctk, ctv = "q8_0", "q4_0"
        ctx = min(native_ctx or 131072, 65536)
        parallel = 1
    elif target == "low_vram":
        ctk, ctv = "q4_0", "q4_0"
        ctx = min(native_ctx or 8192, 8192)
        parallel = 1
    else:  # balanced
        ctk, ctv = "q8_0", "q8_0"
        ctx = min(native_ctx or 32768, 32768 if model_meta.get("size", 0) < 16 * 1024**3 else 16384)
        parallel = 2 if model_meta.get("size", 0) < 12 * 1024**3 else 1

    preset = "vision" if vl else ("moe-split" if moe else "dense-gpu")
    ngl = 99
    ncmoe = 0
    fit = estimate_vram(model_meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset="dense-gpu" if not vl else "vision", vit=vit, srv_running=srv_running)

    notes = []
    if not moe and fit["status"] == "fit":
        preset = "vision" if vl else "dense-gpu"
        notes.append(f"Full GPU offload (-ngl 99) fits in {fit['device']} ({fit['total_vram_mb']} / {fit['budget_mb']} MB).")
    elif moe:
        dense_fit = estimate_vram(model_meta, ctx=ctx, ngl=99, ctk=ctk, ctv=ctv, parallel=parallel, preset="dense-gpu", vit=vit, srv_running=srv_running)
        if dense_fit["status"] == "fit" and target == "speed":
            preset = "dense-gpu"
            notes.append("All MoE experts fit in VRAM — using full GPU offload for maximum token generation speed.")
        else:
            preset = "moe-split"
            phys_threads = max(8, min(24, cores - 2))
            notes.append("MoE architecture detected — keeping attention & router on GPU and offloading FFN experts to CPU RAM.")
    elif fit["status"] == "overflow":
        # Binary search highest ngl or ctx that fits
        if ctk != "q8_0" or ctv != "q4_0":
            ctk, ctv = "q8_0", "q4_0"
            fit = estimate_vram(model_meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, srv_running=srv_running)
            notes.append("Enabled Q8_0 / Q4_0 KV cache quantization to halve KV memory footprint.")
        if fit["status"] == "overflow" and ctx > 8192:
            while ctx > 4096 and estimate_vram(model_meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, srv_running=srv_running)["status"] == "overflow":
                ctx //= 2
            notes.append(f"Scaled context window to {ctx:,} tokens to fit available VRAM.")
        fit = estimate_vram(model_meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, srv_running=srv_running)
        if fit["status"] == "overflow" and blocks > 0:
            best_ngl = 0
            for cand in range(blocks, -1, -1):
                if estimate_vram(model_meta, ctx=ctx, ngl=cand, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, srv_running=srv_running)["status"] != "overflow":
                    best_ngl = cand
                    break
            ngl = best_ngl
            notes.append(f"Partial layer offload: {ngl}/{blocks} layers on GPU, remaining {blocks - ngl} in system RAM.")
            fit = estimate_vram(model_meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, srv_running=srv_running)

    notes.append("Flash Attention enabled (-fa on) for faster prefill and reduced KV overhead.")
    return {
        "preset": preset, "ctx": ctx, "ngl": ngl, "threads": phys_threads,
        "fa": "on", "ctk": ctk, "ctv": ctv, "parallel": parallel,
        "batch": 2048, "ubatch": 512, "ncmoe": ncmoe, "extra_args": "",
        "fit": fit, "notes": notes,
    }

_igpu_cache = {"checked": 0.0, "tag": None, "free": 0}

def igpu_device(refresh=False):
    """Resolve the 780M iGPU's Vulkan device tag + free MiB via llama-server --list-devices."""
    now = time.time()
    if not refresh and _igpu_cache["tag"] and now - _igpu_cache["checked"] < 30:
        return _igpu_cache["tag"], _igpu_cache["free"]
    tag, free = None, 0
    try:
        out = subprocess.run([CFG["server_binary"], "--list-devices"],
                             capture_output=True, text=True, timeout=20).stdout
        for ln in out.splitlines():
            m = re.match(r"\s*(Vulkan\d+):", ln)
            if not m:
                continue
            mi = re.search(r"\((\d+) MiB, (\d+) MiB free\)", ln)
            if ("780M" in ln or "PHOENIX" in ln.upper()) and mi:
                tag, free = m.group(1), int(mi.group(2))
                break
    except Exception:
        pass
    _igpu_cache.update(checked=now, tag=tag, free=free)
    return tag, free

# llama.cpp counter names differ between builds, so match the family rather than one
# suffix, and record what a build actually exposed: an unrecognised counter then shows up
# in /api/server/telemetry's metric_keys instead of needing a hand scrape of /metrics.
_METRIC_TOK_TOTAL_RE = re.compile(r"^(?:llama[:_])?tokens_(predicted|prompted)_total$")
_METRIC_KEYS: dict = {}    # port -> counter names that build emitted (first scrape wins)

_SECRET_ARG_RE = re.compile(r"(--api-key[ =])(\S+)")

def _redact_cmd(text: str) -> str:
    """Mask launch secrets before a command string reaches the UI, the log buffer, or the
    state file. The real argv still carries the key: on Linux /proc/<pid>/cmdline is
    world-readable, so this only stops the panel from re-publishing it. Nothing parses
    these strings, so masking is safe at every exposure point."""
    return _SECRET_ARG_RE.sub(r"\1***", text)

class ServerManager:
    def __init__(self):
        self.proc = None
        self.info = {}     # alias, model, port, preset, started_at, pid, extra, ctx, cmd
        self.logbuf = deque(maxlen=600)
        self.last_active = time.time()
        self.last_tok_s = 0.0
        self.last_prompt_s = 0.0
        self._load_state()
        threading.Thread(target=self._idle_watchdog, daemon=True).start()

    def touch(self):
        self.last_active = time.time()

    def _idle_watchdog(self):
        while True:
            time.sleep(15)
            try:
                mins = int(CFG.get("idle_unload_min", 0) or 0)
                if mins <= 0 or not self.info:
                    continue
                idle = time.time() - self.last_active
                if idle >= mins * 60:
                    st = self.status()
                    if st.get("running"):
                        # Don't unload if a slot is currently processing
                        tel = self.telemetry(st["port"])
                        if tel.get("active_slots", 0) > 0:
                            self.touch()
                            continue
                        self.log(f"idle watchdog: unloading {st.get('alias')} after {mins}m of inactivity")
                        self.stop()
            except Exception:
                pass

    def _load_state(self):
        if STATE_PATH.exists():
            try:
                st = json.loads(STATE_PATH.read_text())
                pid, port = st.get("pid"), st.get("port")
                if pid and port and self._health(port, timeout=1.0):
                    self.proc = None
                    self.info = {**st, "adopted": True}
                    self.last_active = time.time()
                    self.log("adopted running server pid=%s port=%s" % (pid, port))
                elif pid and not _pid_alive(pid):
                    self.log("discarding stale state: pid %s is gone" % pid)
                    STATE_PATH.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _discover_running():
        """Find every running llama-server process via /proc; parse model, port, preset."""
        if not Path("/proc").is_dir():
            return []
        clk = os.sysconf("SC_CLK_TCK") or 100
        btime = 0
        try:
            with open("/proc/stat") as f:
                for ln in f:
                    if ln.startswith("btime"):
                        btime = int(ln.split()[1]); break
        except OSError:
            pass
        out = []
        for pdir in Path("/proc").iterdir():
            if not pdir.name.isdigit():
                continue
            try:
                argv = [a.decode(errors="replace")
                        for a in pdir.joinpath("cmdline").read_bytes().split(b"\0") if a]
            except OSError:
                continue
            if not argv or os.path.basename(argv[0]) != "llama-server":
                continue
            model = port = mmproj = ngl = dev = None
            ot = False
            i = 1
            while i < len(argv):
                a = argv[i]
                if a in ("-m", "--mmproj", "-ngl", "--device") and i + 1 < len(argv):
                    v = argv[i + 1]; i += 1
                    if a == "-m": model = v
                    elif a == "--mmproj": mmproj = v
                    elif a == "--device": dev = v
                    else: ngl = v
                elif a == "--port" and i + 1 < len(argv):
                    try: port = int(argv[i + 1])
                    except ValueError: pass
                    i += 1
                elif a == "-ot":
                    ot = True
                i += 1
            if not (model and port):
                continue
            if ot:
                preset = "moe-split"
            elif mmproj:
                preset = "vision"
            elif ngl not in (None, "0"):
                preset = "dense-gpu"
            else:
                preset = "external"
            if dev and dev == igpu_device()[0] and preset != "external":
                preset = "igpu-vision" if mmproj else "igpu-full"
            started_at = time.time()
            try:
                with open(pdir / "stat") as f:
                    starttime = float(f.read().split(")", 1)[1].split()[19])
                if btime:
                    started_at = btime + starttime / clk
            except Exception:
                pass
            out.append({"pid": int(pdir.name), "port": port, "model": model,
                        "alias": Path(model).stem, "preset": preset,
                        "started_at": started_at})
        return out

    def log(self, line):
        self.logbuf.append(f"[{time.strftime('%H:%M:%S')}] {_redact_cmd(line)}")

    @staticmethod
    def _health(port, timeout=1.5):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def telemetry(self, port=None, api_key: str = "") -> dict:
        """Fetch live slots & metrics from a running llama-server."""
        if port is None:
            st = self.status()
            if not st.get("running") or not st.get("ready"):
                return {"ready": False}
            port = st["port"]
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        out = {
            "ready": True, "port": port,
            "active_slots": 0, "total_slots": 0,
            "kv_used_tokens": 0, "kv_total_tokens": 0, "kv_pct": 0.0,
            "tok_s": round(self.last_tok_s, 1),
            "prompt_s": round(self.last_prompt_s, 1),
            "requests_processing": 0, "requests_deferred": 0,
            "tokens_predicted_total": 0, "tokens_prompted_total": 0,
            "metric_keys": list(_METRIC_KEYS.get(port, [])),
            "slots": [],
        }
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/slots", headers=headers, timeout=1.2)
            if r.status_code == 200 and isinstance(r.json(), list):
                slots = r.json()
                out["total_slots"] = len(slots)
                used_tok = 0
                tot_tok = 0
                slim = []
                for s in slots:
                    proc = bool(s.get("is_processing") or s.get("state", 0) != 0)
                    if proc:
                        out["active_slots"] += 1
                    n_ctx = int(s.get("n_ctx") or 0)
                    # Different llama.cpp builds report KV position in n_past or next_token[0].n_decoded
                    nt = s.get("next_token")
                    if isinstance(nt, list) and nt:
                        nt = nt[0]
                    n_past = int(s.get("n_past") or (nt.get("n_decoded", 0) if isinstance(nt, dict) else 0) or 0)
                    used_tok += max(0, n_past)
                    tot_tok += max(0, n_ctx)
                    slim.append({
                        "id": s.get("id", 0),
                        "busy": proc,
                        "n_ctx": n_ctx,
                        "n_past": max(0, n_past),
                    })
                out["slots"] = slim
                out["kv_used_tokens"] = used_tok
                out["kv_total_tokens"] = tot_tok
                if tot_tok > 0:
                    out["kv_pct"] = round(100.0 * used_tok / tot_tok, 1)
        except Exception:
            pass
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/metrics", headers=headers, timeout=1.2)
            if r.status_code == 200:
                seen = set()
                for ln in r.text.splitlines():
                    if ln.startswith("#") or not ln.strip():
                        continue
                    parts = ln.split()
                    if len(parts) < 2:
                        continue
                    k, v = parts[0], parts[-1]
                    try:
                        fv = float(v)
                    except ValueError:
                        continue
                    if k.startswith("llama") or "token" in k:
                        seen.add(k)
                    tok_total = _METRIC_TOK_TOTAL_RE.match(k)
                    if "predicted_seconds_total" in k or k.endswith("predicted_tokens_seconds"):
                        if fv > 0:
                            out["tok_s"] = round(fv, 1)
                            self.last_tok_s = fv
                    elif "prompted_seconds_total" in k or "prompt_tokens_seconds" in k:
                        if fv > 0:
                            out["prompt_s"] = round(fv, 1)
                            self.last_prompt_s = fv
                    elif tok_total:
                        if tok_total.group(1) == "predicted":
                            out["tokens_predicted_total"] = int(fv)
                        else:
                            out["tokens_prompted_total"] = int(fv)
                    elif "kv_cache_usage_ratio" in k:
                        out["kv_pct"] = max(out["kv_pct"], round(fv * 100.0, 1))
                    elif "requests_processing" in k:
                        out["requests_processing"] = int(fv)
                    elif "requests_deferred" in k:
                        out["requests_deferred"] = int(fv)
                if seen:
                    _METRIC_KEYS[port] = sorted(seen)[:80]
                out["metric_keys"] = _METRIC_KEYS.get(port, [])
        except Exception:
            pass
        if out["active_slots"] > 0 or out["requests_processing"] > 0:
            self.touch()
        out["running"] = True
        out["gen_tps"] = out["tok_s"]
        out["prompt_tps"] = out["prompt_s"]
        out["kv_used"] = out["kv_used_tokens"]
        out["kv_max"] = out["kv_total_tokens"]
        out["slots_busy"] = out["active_slots"]
        out["slots_total"] = out["total_slots"]
        return out

    def status(self):
        i = self.info
        idle_min = int(CFG.get("idle_unload_min", 0) or 0)
        if i:
            pid = i.get("pid")
            alive = (self.proc and self.proc.poll() is None) or _pid_alive(pid)
            if alive:
                idle_sec = int(max(0, time.time() - self.last_active))
                return {
                    "running": True, "pid": pid, "alias": i.get("alias"), "model": i.get("model"),
                    "port": i.get("port"), "preset": i.get("preset"), "ctx": i.get("ctx"),
                    "cmd": i.get("cmd", ""),
                    "uptime": int(time.time() - i.get("started_at", time.time())),
                    "idle_sec": idle_sec,
                    "idle_unload_in": max(0, idle_min * 60 - idle_sec) if idle_min > 0 else None,
                    "idle_unload_min": idle_min,
                    "idle_remaining_s": max(0, idle_min * 60 - idle_sec) if idle_min > 0 else None,
                    "ready": self._health(i["port"], timeout=1.0), "adopted": i.get("adopted", False),
                }
            self.log("server pid %s exited" % pid)
            self.info = {}
            STATE_PATH.unlink(missing_ok=True)
            i = {}
        # Nothing tracked: reflect any llama-server actually running on this host.
        for cand in self._discover_running():
            if self._health(cand["port"], timeout=1.0):
                self.proc = None
                self.info = {"pid": cand["pid"], "port": cand["port"],
                             "alias": cand["alias"], "model": cand["model"],
                             "preset": cand["preset"], "started_at": cand["started_at"],
                             "adopted": True}
                self.last_active = time.time()
                self._save_state()
                self.log("adopted running server pid=%s port=%s model=%s (discovered)"
                         % (cand["pid"], cand["port"], cand["alias"]))
                i = self.info
                break
        if not i:
            return {"running": False}
        idle_sec = int(max(0, time.time() - self.last_active))
        return {
            "running": True, "pid": i.get("pid"), "alias": i.get("alias"), "model": i.get("model"),
            "port": i.get("port"), "preset": i.get("preset"), "ctx": i.get("ctx"),
            "cmd": i.get("cmd", ""),
            "uptime": int(time.time() - i.get("started_at", time.time())),
            "idle_sec": idle_sec,
            "idle_unload_in": max(0, idle_min * 60 - idle_sec) if idle_min > 0 else None,
            "idle_unload_min": idle_min,
            "idle_remaining_s": max(0, idle_min * 60 - idle_sec) if idle_min > 0 else None,
            "ready": self._health(i["port"], timeout=1.0), "adopted": i.get("adopted", False),
        }

    def build_args(self, model: str, preset: str = "auto", extra_args: str = "",
                   ctx: int | None = None, ngl: int | None = None, threads: int | None = None,
                   fa: str = "auto", ctk: str = "", ctv: str = "", parallel: int | None = None,
                   batch: int | None = None, ubatch: int | None = None,
                   ncmoe: int | None = None, mlock: bool = False, no_mmap: bool = False,
                   port: int | None = None):
        mp = Path(model)
        meta = next((m for m in scan_models()["models"] if m["path"] == model), None)
        igpu = preset.startswith("igpu")
        dev = None
        if igpu:
            dev, free_mib = igpu_device()
            if not dev:
                raise HTTPException(400, "iGPU (AMD 780M) not found via llama-server --list-devices — Vulkan backend missing?")
            if free_mib and mp.stat().st_size + 4*1024*1024*1024 > free_mib*1024*1024:
                raise HTTPException(400,
                    f"model ({human_size(mp.stat().st_size)}) does not fully fit in iGPU memory "
                    f"({free_mib} MiB free, KV needs headroom) — use a dGPU preset instead")
        if preset in ("auto", ""):
            preset = "moe-split" if (meta and meta["moe"]) else ("vision" if (meta and meta["vl"] and find_mmproj(model)) else "dense-gpu")

        preset_defaults = {
            "dense-gpu": {"ngl": 99, "t": 8, "ot": None},
            "moe-split": {"ngl": 99, "t": 16, "ot": ".ffn_.*_exps.=CPU"},
            "vision":    {"ngl": 12, "t": 8, "ot": None},
            "cpu-only":  {"ngl": 0,  "t": max(4, (os.cpu_count() or 8)), "ot": None},
        }
        if igpu:
            preset_defaults["igpu-full"] = {"ngl": 99, "t": 8, "ot": None, "dev": dev}
            preset_defaults["igpu-vision"] = {"ngl": 99, "t": 8, "ot": None, "dev": dev}
        if preset not in preset_defaults:
            raise HTTPException(400, f"unknown preset {preset}")
        pd = preset_defaults[preset]
        eff_ctx = int(ctx or CFG.get("ctx", 8192) or 8192)
        eff_ngl = pd["ngl"] if ngl is None else int(ngl)
        # The UI's -t field is labelled "0 = auto", so 0 means "use the preset default".
        eff_t = int(threads) if threads else pd["t"]

        binary = CFG.get("server_binary", BIN)
        caps = bin_caps(binary)

        args = [binary, "-m", model, "--host", "127.0.0.1", "--jinja",
                "-c", str(eff_ctx), "--alias", mp.stem,
                "-ngl", str(eff_ngl), "-t", str(eff_t)]
        if pd.get("dev"):
            args += ["--device", pd["dev"]]
        if pd.get("ot") and not (ncmoe and ncmoe > 0):
            args += ["-ot", pd["ot"]]
        if ncmoe and int(ncmoe) > 0 and caps.get("ncmoe"):
            args += ["--n-cpu-moe", str(int(ncmoe))]
        if fa and fa in ("on", "off"):
            if caps.get("fa_value"):
                args += ["-fa", fa]
            elif fa == "on":
                args += ["-fa"]
        if ctk and ctk != "f16" and caps.get("ctk"):
            args += ["-ctk", ctk]
        if ctv and ctv != "f16" and caps.get("ctk"):
            args += ["-ctv", ctv]
        if parallel and int(parallel) > 1 and caps.get("parallel"):
            args += ["--parallel", str(int(parallel))]
        if batch and int(batch) > 0:
            args += ["-b", str(int(batch))]
        if ubatch and int(ubatch) > 0:
            args += ["-ub", str(int(ubatch))]
        if mlock and caps.get("mlock"):
            args += ["--mlock"]
        if no_mmap and caps.get("no_mmap"):
            args += ["--no-mmap"]
        if caps.get("metrics"):
            args += ["--metrics"]
        mm = find_mmproj(model)
        if preset in ("vision", "igpu-vision") and not mm:
            raise HTTPException(400, "vision preset needs an mmproj file next to the model")
        if mm:
            args += ["--mmproj", mm]
        if extra_args:
            args += extra_args.split()
        if port is not None:
            args += ["--port", str(port)]
        return args, preset, eff_ctx

    def start(self, model, preset="auto", extra_args="", ctx=None, ngl=None, threads=None,
              fa="auto", ctk="", ctv="", parallel=None, batch=None, ubatch=None,
              ncmoe=None, mlock=False, no_mmap=False, save_profile=True):
        if self.status()["running"]:
            self.stop()
        mp = Path(model)
        if not mp.exists():
            raise HTTPException(404, f"model not found: {model}")
        base_port = int(CFG.get("port_base", 8830) or 8830)
        ports = list(range(base_port, base_port + 10))
        for port in ports:
            if not self._port_busy(port):
                break
        else:
            raise HTTPException(500, "no free child port")
        args, resolved_preset, eff_ctx = self.build_args(
            model, preset=preset, extra_args=extra_args, ctx=ctx, ngl=ngl, threads=threads,
            fa=fa, ctk=ctk, ctv=ctv, parallel=parallel, batch=batch, ubatch=ubatch,
            ncmoe=ncmoe, mlock=mlock, no_mmap=no_mmap, port=port,
        )
        if save_profile:
            profs = _read_json(LAUNCH_PROFILES_PATH, {})
            profs[model] = {
                "preset": preset, "ctx": eff_ctx, "ngl": ngl, "threads": threads,
                "fa": fa, "ctk": ctk, "ctv": ctv, "parallel": parallel,
                "batch": batch, "ubatch": ubatch, "ncmoe": ncmoe,
                "mlock": mlock, "no_mmap": no_mmap, "extra_args": extra_args,
                "updated_at": time.time(),
            }
            _write_json(LAUNCH_PROFILES_PATH, profs)
        # Track recent launches in model-meta
        mmeta = _read_json(MODEL_META_PATH, {"pins": [], "recents": {}})
        mmeta.setdefault("recents", {})[model] = time.time()
        _write_json(MODEL_META_PATH, mmeta)

        cmd_str = " ".join(args)
        self.log("starting: " + cmd_str)
        try:
            self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, errors="replace")
        except FileNotFoundError:
            raise HTTPException(503, f"llama-server binary not found at {args[0]} — check Settings")
        threading.Thread(target=self._pump_logs, daemon=True).start()
        self.last_active = time.time()
        self.last_tok_s = 0.0
        self.last_prompt_s = 0.0
        self.info = {"pid": self.proc.pid, "port": port, "alias": mp.stem, "model": model,
                     "preset": resolved_preset, "ctx": eff_ctx, "cmd": _redact_cmd(cmd_str),
                     "started_at": time.time(), "adopted": False}
        self._save_state()
        return self.status()

    @staticmethod
    def _port_busy(port):
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _pump_logs(self):
        proc = self.proc
        for line in proc.stdout:
            s = _redact_cmd(line.rstrip())
            self.logbuf.append(s)
            # Parse tok/s from llama-server log lines when /metrics isn't scraped yet
            m = re.search(r"eval time\s*=.*?\(\s*([0-9.]+)\s*tokens per second\)", s)
            if m:
                try:
                    v = float(m.group(1))
                    if "prompt eval" in s:
                        self.last_prompt_s = v
                    else:
                        self.last_tok_s = v
                except ValueError:
                    pass

    def stop(self):
        i = self.info
        if not i:
            return {"stopped": False}
        pid = i.get("pid")
        if pid and _pid_alive(pid):
            try: os.kill(pid, signal.SIGTERM)
            except ProcessLookupError: pass
            except PermissionError:
                # Adopted server owned by another user: stop tracking it rather than
                # failing the request, so the panel never believes it owns the process.
                self.log(f"pid {pid} is owned by another user — cannot signal it")
                self.proc, self.info = None, {}
                STATE_PATH.unlink(missing_ok=True)
                raise HTTPException(403, f"pid {pid} is owned by another user; stop it outside the panel")
            for _ in range(50):
                if not _pid_alive(pid): break
                time.sleep(0.1)
            else:
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
                except PermissionError:
                    self.log(f"pid {pid} survived SIGTERM and cannot be SIGKILLed (foreign owner)")
        self.log(f"stopped {i.get('alias')} (pid {pid})")
        self.proc, self.info = None, {}
        STATE_PATH.unlink(missing_ok=True)
        return {"stopped": True}

    def _save_state(self):
        STATE_PATH.write_text(json.dumps(self.info))

MGR = ServerManager()

# ---------------------------------------------------------------- avalon
AVALON_BIN = str(HOME / "avalon/build/bin/llama-server")
AVALON_MODEL = HOME / "models/qwen3.8-flash-next-avalon/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf"
AVALON_DRAFT = HOME / "models/qwen3.8-flash-next-avalon/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf"
AVALON_STATE_PATH = APP_DIR / "avalon-state.json"
AVALON_PORT = 8086
AVALON_PROFILES = {
    "flagship": {
        "label": "Flash-Next 262k (flagship)",
        "desc": "125B MoE IQ3KS · 262k context · MTP drafting on",
        "speed": "≈98 tok/s gen · needs the LFM server stopped (VRAM)",
        "exclusive": True,
        "draft": True,
        "defaults": {"ctx": 262144, "batch": 9536, "ubatch": 9536, "ctk": "q8_0", "ctv": "q4_0", "draft_n": 2, "ubd": 256, "parallel": 1},
    },
    "prefast": {
        "label": "Fast ingestion",
        "desc": "125B MoE IQ3KS · 262k context · drafting off · prompt reading ≈2× faster",
        "speed": "needs the LFM server stopped (VRAM)",
        "exclusive": True,
        "draft": False,
        "defaults": {"ctx": 262144, "batch": 9536, "ubatch": 9536, "ctk": "q8_0", "ctv": "q4_0", "draft_n": 0, "ubd": 256, "parallel": 1},
    },
    "compact": {
        "label": "Balanced 65k + MTP",
        "desc": "125B MoE IQ3KS · 65k context · MTP drafting on · ~7.5 GB less KV VRAM",
        "speed": "Fastest startup & generous headroom on 24–32 GB GPUs",
        "exclusive": True,
        "draft": True,
        "defaults": {"ctx": 65536, "batch": 4096, "ubatch": 4096, "ctk": "q8_0", "ctv": "q8_0", "draft_n": 2, "ubd": 256, "parallel": 1},
    },
}

class AvalonManager:
    """Runs the Avalon llama-server (Qwen3.8-Flash-Next 125B) on port 8086."""

    def __init__(self):
        self.proc = None
        self.info = {}   # profile, pid, started_at, adopted, cmd
        self.logbuf = deque(maxlen=400)
        self._load_state()

    def _load_state(self):
        if not AVALON_STATE_PATH.exists():
            return
        try:
            st = json.loads(AVALON_STATE_PATH.read_text())
            pid = st.get("pid")
            if pid and _pid_alive(pid):
                self.proc = None
                self.info = {**st, "adopted": True}
                self.log(f"adopted running avalon pid={pid} profile={st.get('profile')}")
        except Exception:
            pass

    def log(self, line):
        self.logbuf.append(f"[{time.strftime('%H:%M:%S')}] {_redact_cmd(line)}")

    def _health(self, timeout=1.5):
        """Returns 'down' | 'loading' | 'ready' plus detail dict."""
        try:
            r = httpx.get(f"http://127.0.0.1:{AVALON_PORT}/health", timeout=timeout)
            if r.status_code == 200:
                try:
                    return "ready", r.json()
                except Exception:
                    return "ready", {}
            if r.status_code == 503:
                return "loading", {}
            return "down", {}
        except Exception:
            return "down", {}

    def status(self):
        st = self.info
        if st:
            pid = st.get("pid")
            alive = (self.proc and self.proc.poll() is None) or _pid_alive(pid)
            if not alive:
                return {"running": False, "last_exit": True, "profile": st.get("profile")}
        state, detail = self._health(timeout=1.0)
        if state == "down" and not st:
            return {"running": False}
        return {
            "running": True, "state": state,
            "pid": st.get("pid"), "profile": st.get("profile"), "port": AVALON_PORT,
            "uptime": int(time.time() - st.get("started_at", time.time())),
            "model": detail.get("model"), "max_context": detail.get("max_context"),
            "cmd": st.get("cmd", ""),
            "adopted": st.get("adopted", False),
        }

    def build_args(self, profile: str, tune_override: dict | None = None) -> list[str]:
        p = AVALON_PROFILES.get(profile) or AVALON_PROFILES["flagship"]
        defs = dict(p.get("defaults", {}))
        saved_tune = (CFG.get("avalon_tune") or {}).get(profile, {})
        if saved_tune:
            defs.update({k: v for k, v in saved_tune.items() if v is not None})
        if tune_override:
            defs.update({k: v for k, v in tune_override.items() if v is not None})
        ctx = int(defs.get("ctx", 262144))
        batch = int(defs.get("batch", 9536))
        ubatch = int(defs.get("ubatch", batch))
        ctk = str(defs.get("ctk", "q8_0"))
        ctv = str(defs.get("ctv", "q4_0"))
        parallel = int(defs.get("parallel", 1))
        draft_n = int(defs.get("draft_n", 2 if p.get("draft") else 0))
        ubd = int(defs.get("ubd", 256))

        args = [AVALON_BIN, "-m", str(AVALON_MODEL),
                "--device", "ROCm0", "-ngl", "99", "-fit", "off",
                "-fa", "on", "-ctk", ctk, "-ctv", ctv,
                "-b", str(batch), "-ub", str(ubatch), "-c", str(ctx),
                "--parallel", str(parallel), "--jinja", "--temp", "0.6",
                "--metrics", "--port", str(AVALON_PORT)]
        if draft_n > 0 and AVALON_DRAFT.exists():
            args += ["-md", str(AVALON_DRAFT), "--spec-type", "draft-mtp",
                     "--spec-draft-n-max", str(draft_n), "-ubd", str(ubd), "-ctkd", "q8_0", "-ctvd", "q8_0"]
        if CFG.get("avalon_lan"):
            args += ["--host", "0.0.0.0"]
            if CFG.get("avalon_api_key"):
                args += ["--api-key", CFG["avalon_api_key"]]
        return args

    def start(self, profile: str, tune: dict | None = None):
        p = AVALON_PROFILES.get(profile)
        if not p:
            raise HTTPException(400, f"unknown avalon profile: {profile}")
        if not Path(AVALON_BIN).exists():
            raise HTTPException(503, "Avalon build not found — build ~/avalon first (cmake --build build --target llama-server).")
        if not AVALON_MODEL.exists():
            raise HTTPException(404, "model shards missing — finish the Qwen3.8-Flash-Next download first.")
        if p["exclusive"] and self._gpu_conflict():
            raise HTTPException(409, "Stop the LFM server first — this profile needs its VRAM.")
        if self.status()["running"] or not self._port_free():
            self.stop()
        for _ in range(150):
            if self._port_free():
                break
            time.sleep(0.1)
        else:
            raise HTTPException(503, f"port {AVALON_PORT} is still busy — stop the running Avalon server first.")
        if tune:
            CFG.setdefault("avalon_tune", {})[profile] = tune
            _save_config_secure()
        args = self.build_args(profile, tune)
        cmd_str = " ".join(args)
        self.log(f"starting profile {profile}{' [lan]' if CFG.get('avalon_lan') else ''}: {cmd_str}")
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", start_new_session=True)
        threading.Thread(target=self._pump_logs, daemon=True).start()
        self.info = {"profile": profile, "pid": self.proc.pid, "cmd": _redact_cmd(cmd_str),
                     "started_at": time.time(), "adopted": False}
        self._save_state()
        return self.status()

    def _pump_logs(self):
        proc = self.proc
        for line in proc.stdout:
            self.logbuf.append(_redact_cmd(line.rstrip()))

    def stop(self):
        st = self.info
        pid = st.get("pid")
        killed = False
        if pid and _pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try: os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError): pass
            for _ in range(100):
                if not (_pid_alive(pid) and not self._port_free()):
                    killed = True
                    break
                time.sleep(0.1)
            if not killed:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try: os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError): pass
            killed = True
        self.log(f"stopped avalon (pid {pid})" if killed else "avalon not running")
        self.proc, self.info = None, {}
        AVALON_STATE_PATH.unlink(missing_ok=True)
        if not self._port_free():          # untracked instance still holding the port
            self._kill_orphans()
        stopped = self._port_free()
        self.log("stopped avalon" if stopped else f"stop failed: port {AVALON_PORT} still busy")
        return {"stopped": stopped}

    @staticmethod
    def _kill_orphans():
        """Kill any avalon serve/engine processes the panel didn't spawn."""
        if not Path("/proc").is_dir():
            return False
        victims = []
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except Exception:
                continue
            if "lfm-panel" in cmd or "panel.py" in cmd:
                continue
            if "avalon/build/bin/llama-server" in cmd:
                victims.append(int(d.name))
        for v in victims:
            try: os.kill(v, signal.SIGTERM)
            except (ProcessLookupError, PermissionError): pass
        for _ in range(50):
            if AvalonManager._port_free():
                break
            time.sleep(0.1)
        for v in victims:
            if _pid_alive(v):
                try: os.kill(v, signal.SIGKILL)
                except (ProcessLookupError, PermissionError): pass
        for _ in range(30):
            if AvalonManager._port_free():
                break
            time.sleep(0.1)
        return bool(victims)

    @staticmethod
    def _gpu_conflict():
        """True if the eGPU holds a meaningful VRAM reservation (another server resident)."""
        try:
            used = int(Path("/sys/bus/pci/devices/0000:03:00.0/mem_info_vram_used").read_text())
        except Exception:
            return False
        return used > 3 * 1024**3

    @staticmethod
    def _port_free():
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", AVALON_PORT)) != 0

    def _save_state(self):
        AVALON_STATE_PATH.write_text(json.dumps(self.info))

SMGR = AvalonManager()

# ---------------------------------------------------------------- vitals
_prev_cpu = None
_gpu_last = {"g": None, "ts": 0.0}

def _read_amdgpu(hp):
    dev = hp / "device"
    def rd(rel):
        last = None
        for _ in range(3):
            try:
                return (dev / rel).read_text().strip()
            except Exception as e:
                last = e; time.sleep(0.03)
        raise OSError(f"{rel}: {last}")
    g = dict(_gpu_last["g"]) if _gpu_last["g"] else {}
    vram_t, vram_u = int(rd("mem_info_vram_total")), int(rd("mem_info_vram_used"))
    if vram_t <= 0:
        raise OSError("no vram")
    g["total_mb"] = vram_t // 1048576
    # temps/fan/power/clocks live on the hwmon dir; util/mem_info on the device dir
    hw = lambda rel: (lambda f: f.read_text().strip())(hp / rel)
    failures = 0
    for key, rel, div in (("temp", "temp2_input", 1000), ("temp_mem", "temp3_input", 1000),
                          ("power_w", "power1_average", 1_000_000),
                          ("fan_rpm", "fan1_input", 1), ("sclk_mhz", "freq1_input", 1e6),
                          ("mclk_mhz", "freq2_input", 1e6)):
        try:
            val = int(hw(rel))
            g[key] = round(val / div) if div > 1 else val
        except Exception:
            failures += 1  # keep last-known for this field
    used_mb = vram_u // 1048576
    if failures >= 3 and used_mb < g.get("used_mb", used_mb) // 4:
        used_mb = g["used_mb"]  # asleep: VRAM counter resets; keep last-known usage
    g["used_mb"] = used_mb
    try:
        g["util"] = int(rd("gpu_busy_percent"))
    except Exception:
        pass
    if "temp" not in g:
        raise OSError("no base fields")
    return g

def _discrete_gpu_candidates():
    """Yield amdgpu hwmon paths for discrete GPUs; attrs may be EBUSY while the card sleeps."""
    for hp in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            nm = (hp / "name").read_text().strip()
        except FileNotFoundError:
            continue
        except Exception:
            nm = "amdgpu"  # asleep: assume candidate, let the read fail later
        if nm != "amdgpu":
            continue
        try:
            (hp / "fan1_input").read_text()
            yield hp                       # awake discrete GPU (has a fan)
        except FileNotFoundError:
            continue                       # iGPU: no fan attr at all
        except Exception:
            yield hp                       # asleep discrete GPU: attrs EBUSY

def _gpu_prime():
    import threading
    def work():
        while not _gpu_last["g"]:
            try:
                for hp in _discrete_gpu_candidates():
                    _gpu_last["g"] = _read_amdgpu(hp)
                    _gpu_last["ts"] = time.time()
                    return
            except Exception:
                pass
            time.sleep(15)  # sensors are EBUSY while the card sleeps; wait for it to wake
    threading.Thread(target=work, daemon=True).start()
_gpu_prime()

_VITALS_TTL = 1.0
_vitals_cache = {"ts": 0.0, "data": None}

def vitals(refresh: bool = False) -> dict:
    """Memoized vitals(): every miss forks nvidia-smi, and estimate_vram() runs per model,
    so without this a single /api/models page load forks one subprocess per model."""
    now = time.time()
    if not refresh and _vitals_cache["data"] is not None and now - _vitals_cache["ts"] < _VITALS_TTL:
        return _vitals_cache["data"]
    v = _vitals_now()
    _vitals_cache["ts"] = now
    _vitals_cache["data"] = v
    return v

def _vitals_now():
    global _prev_cpu, _gpu_last
    v = {}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                                       "--format=csv,noheader,nounits"], text=True, timeout=2)
        mu, mt, util, temp = [x.strip() for x in out.splitlines()[0].split(",")]
        v["gpu"] = {"used_mb": int(mu), "total_mb": int(mt), "util": int(util), "temp": int(temp)}
    except Exception:
        pass
    if "gpu" not in v:
        last_err = None
        for hp in _discrete_gpu_candidates():
            try:
                v["gpu"] = _read_amdgpu(hp)
                _gpu_last = {"g": v["gpu"], "ts": time.time()}
                break
            except Exception as e:
                last_err = e
        if "gpu" not in v and last_err is not None and _gpu_last["g"]:
            # card asleep (attrs EBUSY): serve last-known reading, flagged
            g = dict(_gpu_last["g"])
            g["stale"] = time.time() - _gpu_last["ts"] > 60
            v["gpu"] = g
    if Path("/proc/meminfo").exists():
        mi = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, val = line.partition(":")
            mi[k] = int(val.split()[0])
        total, avail = mi["MemTotal"], mi["MemAvailable"]
        v["mem"] = {"used_gb": round((total - avail) / 1048576, 1), "total_gb": round(total / 1048576, 1)}
    else:
        try:
            tot_b = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip())
            vm_out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
            ps_m = re.search(r"page size of (\d+) bytes", vm_out)
            pg = int(ps_m.group(1)) if ps_m else 16384
            free_p = 0
            for k in ("Pages free", "Pages speculative"):
                m = re.search(rf"{k}:\s+(\d+)", vm_out)
                if m:
                    free_p += int(m.group(1))
            used_gb = max(0.5, (tot_b - free_p * pg) / (1024**3))
            v["mem"] = {"used_gb": round(used_gb, 1), "total_gb": round(tot_b / (1024**3), 1)}
        except Exception:
            v["mem"] = {"used_gb": 8.0, "total_gb": 16.0}
    if Path("/proc/stat").exists():
        cpu_now = Path("/proc/stat").read_text().splitlines()[0].split()[1:8]
        cur = list(map(int, cpu_now))
        if _prev_cpu:
            d = [c - p for c, p in zip(cur, _prev_cpu)]
            tot = sum(d)
            v["cpu"] = {"percent": round(100 * d[0] / tot) if tot else 0, "cores": os.cpu_count()}
        else:
            # first sample after start: no delta window yet — report honest idle
            v["cpu"] = {"percent": 0, "cores": os.cpu_count()}
        _prev_cpu = cur
    else:
        try:
            la = os.getloadavg()[0]
            cores = os.cpu_count() or 8
            v["cpu"] = {"percent": min(100, round(100 * la / cores)), "cores": cores}
        except Exception:
            v["cpu"] = {"percent": 0, "cores": os.cpu_count() or 8}
    for name, path in (("home", str(HOME)), ("root", "/")):
        try:
            u = shutil.disk_usage(path)
            v.setdefault("disk", {})[name] = {
                "used_gb": round(u.used / 2**30, 1),
                "total_gb": round(u.total / 2**30, 1),
                "free_gb": round(u.free / 2**30, 1),
            }
        except Exception:
            pass
    return v

# ---------------------------------------------------------------- hf search & pull
# ---------------------------------------------------------------- chat persistence
CHATS_DIR = APP_DIR / "chats"
_CID_RE = re.compile(r"^[0-9a-f]{8,64}$")

def _chat_file(cid: str) -> Path:
    if not _CID_RE.match(cid):
        raise HTTPException(400, "bad chat id")
    return CHATS_DIR / f"{cid}.json"

app = FastAPI(title="LFM Panel")

# ---------------------------------------------------------------- auth (sudo password)
# Password is validated live against sudo (same credential as terminal sudo).
# Every failed check counts toward PAM faillock (deny=10, unlock=120s), so the
# web limiter below must trip first: 5 failures / 5 min per IP.
SESSION_COOKIE = "lfm_sid"
SESSION_TTL = 7 * 24 * 3600
_sessions: dict = {}   # token -> expiry (unix time)
_fails: dict = {}      # ip -> deque(failure unix times)

LOGIN_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8"><title>LFM Panel — Sign in</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<script>try{const t=localStorage.getItem('lfm.theme');if(t==='light'||t==='dark'){document.documentElement.dataset.theme=t;document.querySelector('meta[name="color-scheme"]').content=t}}catch(_){}</script>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%2311161f'/><path d='M8 9h16M8 16h10M8 23h13' stroke='%2322d3ee' stroke-width='2.6' stroke-linecap='round'/><circle cx='23' cy='16' r='2.6' fill='%23f59e0b'/></svg>">
<style>
:root{
  color-scheme:light dark;
  --bg:#090b10;--bg-2:#0d1017;--panel:#11161f;--panel-2:#161c27;--raise:#1c2432;
  --bd:#222b3a;--bd-2:#2e3a4e;
  --fg:#edf2f8;--fg-2:#c4cfde;--mut:#8695aa;--faint:#59677a;
  --acc:#22d3ee;--acc-2:#f59e0b;--acc-ink:#041318;
  --err:#f87171;--ok:#34d399;
}
:root:has(> head > meta[name="color-scheme"][content="dark"]){color-scheme:dark}
:root:has(> head > meta[name="color-scheme"][content="light"]){color-scheme:light}
[data-theme=light]{
  --bg:#f4f6f9;--bg-2:#eaeef4;--panel:#ffffff;--panel-2:#f9fafc;--raise:#edf1f7;
  --bd:#d9e1ec;--bd-2:#c2cedd;
  --fg:#0f172a;--fg-2:#334155;--mut:#5b6b82;--faint:#8796ab;
  --acc:#0284c7;--acc-2:#d97706;--acc-ink:#ffffff;
  --err:#dc2626;--ok:#059669;
}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:20px;
  background-color:var(--bg);
  background-image:radial-gradient(rgba(34,211,238,.07) 1px,transparent 1px),radial-gradient(900px 420px at 80% -10%,rgba(34,211,238,.09),transparent 65%);
  background-size:24px 24px,100% 100%;
  color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px;border-radius:6px}
.card{width:min(380px,94vw);background:var(--panel);border:1px solid var(--bd-2);
  border-radius:12px;padding:26px;box-shadow:0 4px 12px rgba(0,0,0,.35)}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:22px;padding-bottom:16px;border-bottom:1px solid var(--bd)}
.mark{width:34px;height:34px;border-radius:9px;background:var(--bg-2);border:1px solid var(--bd-2);display:grid;place-items:center;color:var(--acc)}
.mark svg{width:19px;height:19px}
.brand b{font-size:16px;letter-spacing:-.02em;display:block}
.brand span.sub{display:block;font-size:12px;color:var(--mut)}
label{display:block;color:var(--fg-2);font-size:12.5px;font-weight:600;margin-bottom:6px}
input{width:100%;padding:10px 12px;border-radius:8px;border:1px solid var(--bd-2);background:var(--bg-2);color:var(--fg);
  font:inherit;font-size:14px;transition:border-color .15s}
input:focus{border-color:var(--acc);outline:2px solid var(--acc);outline-offset:1px}
button.sub{width:100%;margin-top:14px;padding:10px 14px;border:1px solid color-mix(in srgb,var(--acc) 55%,transparent);border-radius:8px;
  background:var(--acc);color:var(--acc-ink);font:inherit;font-weight:650;font-size:14px;cursor:pointer;transition:filter .15s,opacity .15s}
button.sub:hover{filter:brightness(1.06)}button.sub:active{transform:translateY(1px)}button.sub[disabled]{opacity:.55;cursor:progress}
#msg{color:var(--err);font-size:12.5px;min-height:18px;margin-top:10px}
.hint{color:var(--mut);font-size:12px;margin:14px 0 0;text-align:center}
.theme{position:fixed;top:14px;right:14px;width:34px;height:34px;display:grid;place-items:center;border-radius:8px;
  border:1px solid var(--bd);background:var(--panel);color:var(--mut);cursor:pointer;margin:0;padding:0}
.theme:hover{color:var(--fg);background:var(--raise)}
.theme svg{width:16px;height:16px}
@media (prefers-reduced-motion:reduce){*{transition-duration:.01ms !important}}
</style></head><body>
<button class="theme" id="theme" aria-label="Toggle color theme"></button>
<div class="card">
 <div class="brand">
  <span class="mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 7h14M5 12h9M5 17h11"/><circle cx="18" cy="12" r="2" fill="currentColor"/></svg></span>
  <div><b>LFM Panel</b><span class="sub">Local inference &amp; telemetry console</span></div>
 </div>
 <form>
  <label for="pw">Host sudo password</label>
  <input id="pw" type="password" placeholder="Enter your system login password" autocomplete="current-password" autofocus>
  <button type="submit" class="sub" id="go">Unlock console</button>
  <div id="msg" role="alert"></div>
 </form>
 <p class="hint">Authenticated via PAM sudo · 5 attempts per 5 minutes</p>
</div>
<script>
const SUN='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/></svg>';
const MOON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 14a8 8 0 01-9.5-9.5A8 8 0 1020 14z"/></svg>';
function $(s){return document.querySelector(s)}
function sysScheme(){return matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'}
function applyTheme(t){
 document.documentElement.dataset.theme=t;
 $('meta[name="color-scheme"]').content=localStorage.getItem('lfm.theme')||'light dark';
 $('#theme').innerHTML=t==='dark'?SUN:MOON;
}
applyTheme(localStorage.getItem('lfm.theme')||sysScheme());
$('#theme').onclick=()=>{
 const cur=document.documentElement.dataset.theme;
 const next=cur==='dark'?'light':'dark';
 if(next===sysScheme())localStorage.removeItem('lfm.theme');
 else localStorage.setItem('lfm.theme',next);
 applyTheme(next);
};
async function go(ev){ev.preventDefault();
 const m=$('#msg'),b=$('#go'),pw=$('#pw');m.textContent='';
 b.disabled=true;b.textContent='Verifying…';
 try{const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({password:pw.value})});
  if(r.ok){location.replace('/');return}
  m.textContent=(r.status===429)?'Too many attempts — wait 5 minutes.':'Incorrect password.';
  pw.select();
 }catch(e){m.textContent='Failed: '+e}
 b.disabled=false;b.textContent='Unlock console'}
document.querySelector('form').addEventListener('submit',go);
</script></body></html>"""

def _sudo_ok(pw: str) -> bool:
    if not pw:
        # Under NOPASSWD sudo, `sudo -S -k -v` never reads stdin and exits 0 for any input,
        # so an empty password must never open the panel.
        return False
    try:
        r = subprocess.run(["sudo", "-S", "-k", "-v"], input=pw + "\n",
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            detail = (r.stderr or "").strip().splitlines()
            print(f"sudo auth failed rc={r.returncode}: {detail[-1][:200] if detail else '(no output)'}",
                  flush=True)
        return r.returncode == 0
    except Exception as e:
        print(f"sudo auth error: {e!r}", flush=True)
        return False

class LoginBody(BaseModel):
    password: str

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    tok = request.cookies.get(SESSION_COOKIE)
    if tok and _sessions.get(tok, 0) > time.time():
        return RedirectResponse("/", status_code=302)
    return HTMLResponse(content=LOGIN_HTML, headers={"Cache-Control": "no-store"})

@app.post("/api/login")
def api_login(b: LoginBody, request: Request):
    ip = request.client.host if request.client else "?"
    now = time.time()
    q = _fails.setdefault(ip, deque())
    while q and now - q[0] > 300:
        q.popleft()
    if len(q) >= 5:
        raise HTTPException(429, "too many attempts; wait 5 minutes")
    if not _sudo_ok(b.password):
        q.append(now)
        raise HTTPException(401, "wrong password")
    q.clear()
    tok = secrets.token_urlsafe(32)
    _sessions[tok] = now + SESSION_TTL
    resp = JSONResponse({"ok": True})
    resp.set_cookie(SESSION_COOKIE, tok, max_age=SESSION_TTL, httponly=True,
                    samesite="lax", path="/")
    return resp

@app.post("/api/logout")
def api_logout(request: Request):
    _sessions.pop(request.cookies.get(SESSION_COOKIE, ""), None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp

@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    tok = request.cookies.get(SESSION_COOKIE)
    ok = tok is not None and _sessions.get(tok, 0) > time.time()
    if ok or request.url.path in ("/login", "/api/login"):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=302)


@app.get("/api/chats")
def api_chats_list():
    if not CHATS_DIR.exists():
        return []
    out = []
    for f in CHATS_DIR.glob("*.json"):
        try:
            c = json.loads(f.read_text())
            out.append({"id": c["id"], "title": c.get("title", "(untitled)"),
                        "created": c.get("created", 0), "model": c.get("model", ""),
                        "updated": c.get("updated", c.get("created", 0)),
                        "msgs": len(c.get("messages", []))})
        except Exception:
            continue
    return sorted(out, key=lambda c: -c["updated"])

@app.get("/api/chats/{cid}")
def api_chat_get(cid: str):
    f = _chat_file(cid)
    if not f.exists():
        raise HTTPException(404)
    return json.loads(f.read_text())

class AppendBody(BaseModel):
    id: str | None = None
    model: str = ""
    role: str
    content: str = ""
    reasoning: str = ""

@app.post("/api/chats/append")
def api_chat_append(b: AppendBody):
    if b.role not in ("user", "assistant", "system"):
        raise HTTPException(400, "bad role")
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chat = None
    if b.id:
        f = _chat_file(b.id)
        if f.exists():
            chat = json.loads(f.read_text())
    if chat is None:
        cid = uuid.uuid4().hex
        chat = {"id": cid, "title": "", "created": time.time(), "model": b.model, "messages": []}
    else:
        cid = chat["id"]
    msg = {"role": b.role, "content": b.content, "ts": time.time()}
    if b.reasoning:
        msg["reasoning"] = b.reasoning
    chat["messages"].append(msg)
    chat["updated"] = time.time()
    if b.model:
        chat["model"] = b.model
    if not chat["title"]:
        first = next((m["content"] for m in chat["messages"] if m["role"] == "user"), "")
        chat["title"] = first[:70] or "(untitled)"
    (CHATS_DIR / f"{cid}.json").write_text(json.dumps(chat))
    return {"id": cid, "title": chat["title"], "msgs": len(chat["messages"])}

@app.delete("/api/chats/{cid}")
def api_chat_delete(cid: str):
    _chat_file(cid).unlink(missing_ok=True)
    return {"deleted": cid}

class RenameChatBody(BaseModel):
    title: str

@app.post("/api/chats/{cid}/rename")
def api_chat_rename(cid: str, b: RenameChatBody):
    f = _chat_file(cid)
    if not f.exists():
        raise HTTPException(404)
    c = json.loads(f.read_text())
    c["title"] = (b.title or "").strip()[:100] or "(untitled)"
    c["updated"] = time.time()
    f.write_text(json.dumps(c))
    return {"id": cid, "title": c["title"]}

class TruncateChatBody(BaseModel):
    keep: int
    content: str | None = None

@app.post("/api/chats/{cid}/truncate")
def api_chat_truncate(cid: str, b: TruncateChatBody):
    """Truncate a conversation to `keep` messages (optionally updating the last user message) for edit/regenerate."""
    f = _chat_file(cid)
    if not f.exists():
        raise HTTPException(404)
    c = json.loads(f.read_text())
    msgs = c.get("messages", [])
    keep = max(0, min(len(msgs), int(b.keep)))
    msgs = msgs[:keep]
    if b.content is not None and msgs and msgs[-1].get("role") == "user":
        msgs[-1]["content"] = b.content
        msgs[-1]["ts"] = time.time()
    c["messages"] = msgs
    c["updated"] = time.time()
    f.write_text(json.dumps(c))
    return {"id": cid, "msgs": len(msgs)}

def _hf_headers():
    tok = CFG.get("hf_token") or os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}

@app.get("/api/hf/search")
def api_hf_search(q: str):
    try:
        r = httpx.get("https://huggingface.co/api/models", params={"search": q, "limit": 20, "sort": "downloads", "direction": -1}, headers=_hf_headers(), timeout=15)
        r.raise_for_status()
        res = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"HF search failed: {e}")
    if not isinstance(res, list):
        raise HTTPException(502, "HF search returned an unexpected shape")
    return [{"id": m["modelId"], "downloads": m.get("downloads", 0), "likes": m.get("likes", 0)}
            for m in res if m.get("modelId")]

@app.get("/api/hf/files")
def api_hf_files(repo: str):
    r = httpx.get(f"https://huggingface.co/api/models/{repo}/tree/main", headers=_hf_headers(), timeout=15)
    if r.status_code != 200:
        raise HTTPException(404, f"repo not found: {repo}")
    tree = r.json()
    if not isinstance(tree, list):
        raise HTTPException(502, "unexpected HF response")
    out = []
    for t in tree:
        pl = t["path"].lower()
        if not pl.endswith(".gguf") or pl.startswith("ggml-vocab"):
            continue
        size = (t.get("lfs") or {}).get("size") or t.get("size", 0)
        out.append({"path": t["path"], "size": size, "size_h": human_size(size)})
    return out

DOWNLOADS = {}
# FastAPI runs sync endpoints on a threadpool, so pull/cancel/resume/list can all touch
# DOWNLOADS concurrently: a bare `sorted(DOWNLOADS.values())` can raise "dictionary
# changed size during iteration" while a pull registers a job.
_dl_lock = threading.Lock()


def _dl_live(d: dict) -> bool:
    """True while the job's worker thread is still running. Status strings alone are
    not enough: a thread can die with status left at "downloading"."""
    t = d.get("_thread")
    return bool(t and t.is_alive())


def _download_worker(did: str):
    with _dl_lock:
        d = DOWNLOADS.get(did)
    if d is None:          # cancelled and dropped before we got scheduled
        return
    d["status"] = "downloading"
    dest = Path(d["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    d["done"] = resume_from
    headers = dict(_hf_headers())
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
    try:
        with httpx.stream("GET", d["url"], headers=headers, follow_redirects=True,
                          timeout=httpx.Timeout(30, read=120)) as r:
            if r.status_code == 416:  # range already complete
                if d["cancel"]:
                    raise RuntimeError("paused")
                tmp.rename(dest)
                d["status"] = "done"
                return
            r.raise_for_status()
            cl = int(r.headers.get("content-length", 0))
            if r.status_code == 206 and resume_from > 0:
                d["total"] = resume_from + cl if cl else d["total"]
                mode = "ab"
            else:
                d["total"] = cl or d["total"]
                d["done"] = 0
                mode = "wb"
            with open(tmp, mode) as fh:
                for chunk in r.iter_bytes(1 << 20):
                    if d["cancel"]:
                        raise RuntimeError("paused")
                    fh.write(chunk)
                    d["done"] = fh.tell()
        if d["cancel"]:
            # All chunks arrived before the cancel landed; don't publish a file the
            # user just cancelled.
            raise RuntimeError("paused")
        tmp.rename(dest)
        d["status"] = "done"
        _model_cache.clear()
        MGR.log(f"pulled {dest.name}")
    except Exception as e:
        cancelled = bool(d["cancel"])
        if cancelled:
            d["status"] = "paused"
            # A cancel that arrives while this thread holds the .part open is deferred
            # to here: unlinking from the request thread would leave the worker writing
            # to an unlinked inode and then failing its rename.
            if d.get("delete_part"):
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                with _dl_lock:
                    DOWNLOADS.pop(did, None)
        else:
            d["status"] = "error"
            d["error"] = str(e)[:300]

class PullBody(BaseModel):
    repo: str
    files: list[str]
    sizes: dict[str, int] = {}

@app.post("/api/hf/pull")
def api_hf_pull(b: PullBody):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", b.repo)
    root_dir = Path(CFG["model_dirs"][0]).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)
    try:
        free_b = shutil.disk_usage(str(root_dir)).free
        need_b = sum(int(b.sizes.get(f, 0) or 0) for f in b.files)
        if need_b > 0 and need_b + 512 * 1024**2 > free_b:
            raise HTTPException(400, f"Insufficient disk space: {human_size(need_b)} required, only {human_size(free_b)} free on {root_dir}")
    except HTTPException:
        raise
    except Exception:
        pass
    started = []
    for f in b.files:
        fname = f.split("/")[-1]
        if not fname.lower().endswith(".gguf"):
            continue
        dest = root_dir / "hf-pulls" / safe / fname
        did = f"{b.repo}:{fname}"
        part = dest.with_suffix(dest.suffix + ".part")
        done_b = part.stat().st_size if part.exists() else 0
        with _dl_lock:
            existing = DOWNLOADS.get(did)
            if existing and _dl_live(existing):
                # did is derived from repo+file, so one did is always one dest: attach to
                # the running job rather than racing a second writer onto the same .part.
                started.append(did)
                continue
            job = {"id": did, "repo": b.repo, "file": fname, "path": f,
                   "url": f"https://huggingface.co/{b.repo}/resolve/main/{f}",
                   "dest": str(dest), "done": done_b, "total": int(b.sizes.get(f, 0) or 0),
                   "status": "queued", "cancel": False, "delete_part": False, "error": ""}
            t = threading.Thread(target=_download_worker, args=(did,), daemon=True)
            job["_thread"] = t
            DOWNLOADS[did] = job
        t.start()
        started.append(did)
    if not started:
        raise HTTPException(400, "no .gguf files in selection")
    return {"started": started}

@app.get("/api/downloads")
def api_downloads():
    with _dl_lock:
        jobs = list(DOWNLOADS.values())
    # _thread is bookkeeping, not API surface (a Thread is not JSON-serialisable).
    return sorted(({k: v for k, v in d.items() if k != "_thread"} for d in jobs),
                  key=lambda d: d["id"])

class CancelBody(BaseModel):
    id: str
    delete_part: bool = False

@app.post("/api/downloads/cancel")
def api_cancel(b: CancelBody):
    with _dl_lock:
        d = DOWNLOADS.get(b.id)
        if not d:
            raise HTTPException(404)
        d["cancel"] = True
        d["delete_part"] = bool(b.delete_part)
        live = _dl_live(d)
        if b.delete_part and not live:
            # No worker holds the file open, so unlink and forget it here. A live worker
            # does that itself once its handle closes.
            try:
                Path(d["dest"] + ".part").unlink(missing_ok=True)
            except OSError:
                pass
            DOWNLOADS.pop(b.id, None)
    return {"ok": True, "deferred_to_worker": live}

@app.post("/api/downloads/resume")
def api_resume(b: CancelBody):
    t = None
    with _dl_lock:
        d = DOWNLOADS.get(b.id)
        if not d:
            raise HTTPException(404)
        # Clear cancel first: a live worker re-reads it per chunk, so resuming a running
        # download is just that. The old code returned early on status=="downloading"
        # without clearing it, leaving the job stuck showing "downloading".
        d["cancel"] = False
        d["delete_part"] = False
        d["error"] = ""
        live = _dl_live(d)
        if not live:
            d["status"] = "queued"
            t = threading.Thread(target=_download_worker, args=(b.id,), daemon=True)
            d["_thread"] = t
    if t is not None:
        t.start()
    return {"ok": True, "resumed_live": live}

# ---------------------------------------------------------------- delete
class DeleteBody(BaseModel):
    path: str

@app.post("/api/models/delete")
def api_model_delete(b: DeleteBody):
    p = Path(b.path)
    if p.suffix.lower() != ".gguf":
        raise HTTPException(400, "not a gguf")
    roots = [Path(d).expanduser().resolve() for d in CFG["model_dirs"]]
    rp = p.resolve()
    if not any(rp.is_relative_to(root) for root in roots):
        raise HTTPException(400, "path outside configured model dirs")
    target = rp
    # inside an HF hub cache? delete the whole models--repo dir (frees blob storage)
    for root in roots:
        if rp.is_relative_to(root):
            rel = rp.relative_to(root).parts
            repo_dirs = [x for x in rel if x.startswith("models--")]
            if repo_dirs:
                cand = root.joinpath(*rel[: rel.index(repo_dirs[0]) + 1])
                if cand.name.startswith("models--"):
                    target = cand
            break
    if target.is_dir():
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        shutil.rmtree(target)
    else:
        size = target.stat().st_size
        target.unlink(missing_ok=True)
    _model_cache.clear()
    MGR.log(f"deleted {target.name} ({human_size(size)} freed)")
    return {"deleted": str(target), "freed_h": human_size(size)}

@app.get("/api/models")
def api_models():
    data = scan_models()
    profs = _read_json(LAUNCH_PROFILES_PATH, {})
    mmeta = _read_json(MODEL_META_PATH, {"pins": [], "recents": {}})
    benches = _read_json(BENCHMARKS_PATH, [])
    pins = set(mmeta.get("pins", []))
    recents = mmeta.get("recents", {})
    # One hardware snapshot for the whole list: estimate_vram() would otherwise fork
    # nvidia-smi and probe both servers once per model.
    host_vit = vitals()
    srv_running = bool(MGR.status().get("running") or SMGR.status().get("running"))
    # Compute latest benchmark per model path
    best_bench = {}
    for b in benches:
        mp = b.get("model_path") or b.get("model")
        if mp and mp not in best_bench:
            tps = b.get("gen_tps") or b.get("tok_s", 0)
            ptps = b.get("prompt_tps") or b.get("prompt_s", 0)
            best_bench[mp] = {"tok_s": tps, "gen_tps": tps, "prompt_s": ptps, "prompt_tps": ptps, "ts": b.get("ts", 0)}
    for m in data["models"]:
        p = m["path"]
        saved = profs.get(p, {})
        m["pinned"] = p in pins
        m["last_used"] = recents.get(p, 0)
        m["saved_profile"] = saved or None
        m["bench"] = best_bench.get(p)
        m["fit"] = estimate_vram(
            m,
            ctx=int(saved.get("ctx") or CFG.get("ctx", 8192) or 8192),
            ngl=99 if saved.get("ngl") is None else int(saved.get("ngl")),
            ctk=saved.get("ctk") or "f16",
            ctv=saved.get("ctv") or "f16",
            parallel=int(saved.get("parallel") or 1),
            preset=saved.get("preset") or "auto",
            ncmoe=int(saved.get("ncmoe") or 0),
            vit=host_vit, srv_running=srv_running,
        )
    data["caps"] = bin_caps(CFG.get("server_binary", BIN))
    data["default_ctx"] = int(CFG.get("ctx", 8192) or 8192)
    return data

class PinBody(BaseModel):
    path: str
    pinned: bool | None = None

@app.post("/api/models/pin")
def api_model_pin(b: PinBody):
    mmeta = _read_json(MODEL_META_PATH, {"pins": [], "recents": {}})
    pins = list(mmeta.get("pins", []))
    if b.pinned is True or (b.pinned is None and b.path not in pins):
        if b.path not in pins:
            pins.append(b.path)
        now_pinned = True
    else:
        pins = [x for x in pins if x != b.path]
        now_pinned = False
    mmeta["pins"] = pins
    _write_json(MODEL_META_PATH, mmeta)
    return {"path": b.path, "pinned": now_pinned, "pins": pins}

class FitQueryBody(BaseModel):
    model: str
    preset: str = "auto"
    ctx: int = 8192
    ngl: int = 99
    threads: int = 8
    fa: str = "auto"
    ctk: str = "f16"
    ctv: str = "f16"
    parallel: int = 1
    batch: int = 2048
    ubatch: int = 512
    ncmoe: int = 0
    mlock: bool = False
    no_mmap: bool = False
    extra_args: str = ""

@app.post("/api/models/fit")
def api_model_fit(b: FitQueryBody):
    meta = next((m for m in scan_models()["models"] if m["path"] == b.model), None)
    if not meta:
        raise HTTPException(404, "model not found")
    fit = estimate_vram(
        meta, ctx=b.ctx, ngl=b.ngl, ctk=b.ctk, ctv=b.ctv,
        parallel=b.parallel, preset=b.preset, ncmoe=b.ncmoe,
    )
    try:
        args, resolved_preset, eff_ctx = MGR.build_args(
            b.model, preset=b.preset, extra_args=b.extra_args, ctx=b.ctx,
            ngl=b.ngl, threads=b.threads, fa=b.fa, ctk=b.ctk, ctv=b.ctv,
            parallel=b.parallel, batch=b.batch, ubatch=b.ubatch,
            ncmoe=b.ncmoe, mlock=b.mlock, no_mmap=b.no_mmap,
            port=int(CFG.get("port_base", 8830) or 8830),
        )
        cmd = " ".join(args)
    except HTTPException as e:
        cmd = f"# {e.detail}"
    return {"fit": fit, "cmd": cmd}

class OptimizeBody(BaseModel):
    model: str = ""
    path: str = ""
    target: str = "balanced"   # speed | balanced | max_ctx | low_vram

@app.post("/api/models/optimize")
def api_model_optimize(b: OptimizeBody):
    target_path = b.path or b.model
    meta = next((m for m in scan_models()["models"] if m["path"] == target_path), None)
    if not meta:
        raise HTTPException(404, "model not found")
    rec = recommend_model_config(meta, b.target)
    try:
        args, _, _ = MGR.build_args(
            target_path, preset=rec["preset"], extra_args=rec["extra_args"],
            ctx=rec["ctx"], ngl=rec["ngl"], threads=rec["threads"],
            fa=rec["fa"], ctk=rec["ctk"], ctv=rec["ctv"],
            parallel=rec["parallel"], batch=rec["batch"], ubatch=rec["ubatch"],
            ncmoe=rec["ncmoe"], port=int(CFG.get("port_base", 8830) or 8830),
        )
        rec["cmd"] = " ".join(args)
    except Exception:
        rec["cmd"] = ""
    rec["recommended"] = dict(rec)
    return rec

@app.get("/api/vitals")
def api_vitals():
    return vitals()

@app.get("/api/server/status")
def api_status():
    return MGR.status()

@app.get("/api/server/telemetry")
def api_telemetry():
    st = MGR.status()
    if st.get("running") and st.get("ready") and st.get("port") != AVALON_PORT:
        tel = MGR.telemetry(st["port"])
        tel["source"] = "panel"
        tel["alias"] = st.get("alias")
        return tel
    av = SMGR.status()
    if av.get("running") and av.get("state") == "ready":
        tel = MGR.telemetry(AVALON_PORT, CFG.get("avalon_api_key", ""))
        tel["source"] = "avalon"
        tel["alias"] = av.get("model") or "qwen3.8-flash-next"
        return tel
    if st.get("running") and st.get("ready"):
        tel = MGR.telemetry(AVALON_PORT, CFG.get("avalon_api_key", ""))
        tel["source"] = "avalon"
        tel["alias"] = "qwen3.8-flash-next"
        return tel
    return {"ready": False}

@app.get("/api/server/log")
def api_log():
    return {"log": "\n".join(list(MGR.logbuf)[-250:])}

# ---------------------------------------------------------------- benchmarks
BENCH_PROMPTS = {
    "standard": (
        "Explain the architecture of a modern Transformer language model, comparing multi-head attention, "
        "grouped-query attention, and Mixture-of-Experts routing, and provide a concise Python implementation "
        "of scaled dot-product attention with causal masking."
    ),
    "code": (
        "Write a complete, dependency-free Python 3 LRU cache class with TTL expiration, thread-safety via RLock, "
        "O(1) get/put operations using a doubly-linked list, and unit tests demonstrating eviction order."
    ),
    "short": (
        "Summarize the trade-offs between KV-cache quantization (Q8_0 vs Q4_0) and Flash Attention during "
        "long-context local LLM inference in five crisp bullet points."
    ),
}

class BenchRunBody(BaseModel):
    prompt_key: str = "standard"
    max_tokens: int = 160

@app.get("/api/benchmarks")
def api_benchmarks():
    return _read_json(BENCHMARKS_PATH, [])

@app.delete("/api/benchmarks")
def api_benchmarks_clear():
    _write_json(BENCHMARKS_PATH, [])
    return {"cleared": True}

@app.post("/api/benchmarks/run")
def api_benchmark_run(b: BenchRunBody):
    tgt = _chat_target()
    if not tgt:
        raise HTTPException(409, "No server ready — load a model or start Avalon first")
    port, key = tgt
    cur = _current_llm()
    alias = cur[0] if cur else "model"
    source = cur[4] if cur else "panel"
    srv = MGR.status()
    model_path = srv.get("model", "") if source == "panel" else str(AVALON_MODEL)
    preset = srv.get("preset", "avalon") if source == "panel" else (SMGR.status().get("profile") or "avalon")

    prompt_txt = BENCH_PROMPTS.get(b.prompt_key, BENCH_PROMPTS["standard"])
    max_toks = max(32, min(512, int(b.max_tokens or 160)))
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    MGR.touch()

    t_start = time.perf_counter()
    t_first = None
    gen_chars = 0
    usage = {}
    timings = {}
    try:
        with httpx.stream(
            "POST", f"http://127.0.0.1:{port}/v1/chat/completions",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": prompt_txt}],
                "max_tokens": max_toks,
                "temperature": 0.2,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=httpx.Timeout(15.0, read=120.0),
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw or not raw.startswith("data: "):
                    continue
                dat = raw[6:].strip()
                if dat == "[DONE]":
                    break
                try:
                    j = json.loads(dat)
                except Exception:
                    continue
                if "timings" in j and isinstance(j["timings"], dict):
                    timings = j["timings"]
                if "usage" in j and isinstance(j["usage"], dict):
                    usage = j["usage"]
                ch = (j.get("choices") or [{}])[0]
                dl = ch.get("delta") or {}
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    gen_chars += len(piece)
    except Exception as e:
        raise HTTPException(502, f"Benchmark request failed: {e}")

    t_end = time.perf_counter()
    total_s = max(0.001, t_end - t_start)
    ttft_ms = round((t_first - t_start) * 1000) if t_first else round(total_s * 250)
    gen_s = max(0.001, t_end - (t_first or t_start))

    comp_toks = int(timings.get("predicted_n") or usage.get("completion_tokens") or max(1, round(gen_chars / 3.8)))
    prompt_toks = int(timings.get("prompt_n") or usage.get("prompt_tokens") or max(1, round(len(prompt_txt) / 3.8)))
    tok_s = round(float(timings.get("predicted_per_second") or (comp_toks / gen_s)), 1)
    prompt_s = round(float(timings.get("prompt_per_second") or (prompt_toks / max(0.005, (t_first - t_start) if t_first else total_s * 0.2))), 1)

    MGR.last_tok_s = tok_s
    MGR.last_prompt_s = prompt_s
    vit = vitals(refresh=True)   # post-run VRAM must not come from a cached sample
    vram_mb = (vit.get("gpu") or {}).get("used_mb", 0)

    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "alias": alias,
        "model": alias,
        "model_path": model_path,
        "preset": preset,
        "source": source,
        "prompt_key": b.prompt_key,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "tok_s": tok_s,
        "gen_tps": tok_s,
        "prompt_s": prompt_s,
        "prompt_tps": prompt_s,
        "ttft_ms": ttft_ms,
        "total_s": round(total_s, 2),
        "vram_mb": vram_mb,
    }
    benches = _read_json(BENCHMARKS_PATH, [])
    benches.insert(0, entry)
    _write_json(BENCHMARKS_PATH, benches[:100])
    return entry

# ---------------------------------------------------------------- avalon api
@app.get("/api/avalon/profiles")
def api_avalon_profiles():
    out = []
    saved_all = CFG.get("avalon_tune") or {}
    for key, p in AVALON_PROFILES.items():
        defs = dict(p.get("defaults", {}))
        if key in saved_all and isinstance(saved_all[key], dict):
            defs.update(saved_all[key])
        cmd_str = " ".join(SMGR.build_args(key))
        out.append({
            "key": key, "label": p["label"], "desc": p["desc"],
            "speed": p["speed"], "exclusive": p["exclusive"],
            "draft": p.get("draft", False),
            "defaults": defs,
            # What the user explicitly saved (empty = inherit profile defaults) — the tune
            # form restores this, not `defaults`, so unset fields stay unset.
            "saved": saved_all.get(key) if isinstance(saved_all.get(key), dict) else {},
            "cmd": _redact_cmd(cmd_str),
            "cmd_preview": _redact_cmd(cmd_str),
            "available": Path(AVALON_BIN).exists() and AVALON_MODEL.exists(),
        })
    return out

@app.get("/api/avalon/status")
def api_avalon_status():
    return SMGR.status()

@app.get("/api/avalon/log")
def api_avalon_log():
    return {"log": "\n".join(list(SMGR.logbuf)[-200:])}

class AvalonStartBody(BaseModel):
    profile: str
    tune: dict | None = None

@app.post("/api/avalon/start")
def api_avalon_start(b: AvalonStartBody):
    return SMGR.start(b.profile, b.tune)

class AvalonTuneBody(BaseModel):
    profile: str = "flagship"
    tune: dict | None = None
    ctx: int | None = None
    batch: int | None = None
    ubatch: int | None = None
    ctk: str | None = None
    ctv: str | None = None
    draft_n: int | None = None
    ubd: int | None = None
    parallel: int | None = None
    preview: bool = False

@app.post("/api/avalon/tune")
def api_avalon_tune(b: AvalonTuneBody):
    if b.profile not in AVALON_PROFILES:
        raise HTTPException(400, "unknown profile")
    tune_dict = dict(b.tune or {})
    # The select-backed fields use 0 to mean "profile default"; draft_n/ubd are free number
    # inputs where an empty box is unset and a typed 0 means "drafting off" — so 0 is a value.
    for k in ("ctx", "batch", "ubatch", "ctk", "ctv", "draft_n", "ubd", "parallel"):
        val = getattr(b, k, None)
        if val is None or val == "":
            continue
        if k in ("ctx", "parallel") and val == 0:
            continue
        tune_dict[k] = val
    cmd_str = " ".join(SMGR.build_args(b.profile, tune_dict))
    if not b.preview:
        CFG.setdefault("avalon_tune", {})[b.profile] = tune_dict
        _save_config_secure()
    return {"ok": True, "cmd": _redact_cmd(cmd_str), "cmd_preview": _redact_cmd(cmd_str),
            "tune": tune_dict, "preview": b.preview}

@app.post("/api/avalon/stop")
def api_avalon_stop():
    return SMGR.stop()

# ---------------------------------------------------------------- avalon access
def _save_config_secure():
    CONFIG_PATH.write_text(json.dumps(CFG, indent=2))
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass

def _lan_ip():
    try:
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show", "scope", "global"],
                                      text=True, timeout=3)
        best = None
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            ip = parts[3].split("/")[0]
            if ip.startswith("192.168."):
                best = ip
                break
            if best is None and ip.startswith(("10.", "172.")):
                best = ip
        return best or "127.0.0.1"
    except Exception:
        return "127.0.0.1"

def _apply_avalon_access():
    st = SMGR.status()
    if st.get("running") and st.get("profile"):
        SMGR.stop()
        SMGR.start(st["profile"])
    return {"lan": bool(CFG.get("avalon_lan")), "api_key": CFG.get("avalon_api_key", ""),
            "ip": _lan_ip(), "port": AVALON_PORT, "status": SMGR.status()}

class AvalonAccessBody(BaseModel):
    lan: bool

@app.get("/api/avalon/access")
def api_avalon_access_get():
    return {"lan": bool(CFG.get("avalon_lan")), "api_key": CFG.get("avalon_api_key", ""),
            "ip": _lan_ip(), "port": AVALON_PORT}

@app.post("/api/avalon/access")
def api_avalon_access_set(b: AvalonAccessBody):
    if b.lan and not CFG.get("avalon_api_key"):
        CFG["avalon_api_key"] = secrets.token_urlsafe(24)
    CFG["avalon_lan"] = bool(b.lan)
    _save_config_secure()
    return _apply_avalon_access()

@app.post("/api/avalon/access/regenerate")
def api_avalon_access_regen():
    CFG["avalon_api_key"] = secrets.token_urlsafe(24)
    _save_config_secure()
    return _apply_avalon_access()

class StartReq(Request):
    pass

# ------------------------------------------------------- coding-agent registry
AGENTS_STATE_PATH = APP_DIR / "agent-registry.json"
PANEL_PROVIDER_ID = "lfm-panel"

def _load_agents_state():
    if AGENTS_STATE_PATH.exists():
        try:
            return json.loads(AGENTS_STATE_PATH.read_text())
        except Exception:
            pass
    return {"agents": {}}

def _save_agents_state(st):
    AGENTS_STATE_PATH.write_text(json.dumps(st, indent=1))

def _proc_ctx(pid):
    try:
        argv = [a.decode() for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
        if "-c" in argv:
            return int(argv[argv.index("-c") + 1])
    except Exception:
        pass
    return 8192

def _current_llm():
    """(alias, port, ctx, api_key, source) for whichever llama-server the panel manages is running."""
    m = MGR.status()
    if m.get("running") and m["port"] != AVALON_PORT:
        return m.get("alias"), m["port"], _proc_ctx(m["pid"]), "", "panel"
    a = SMGR.status()
    if a.get("running"):
        return "qwen3.8-flash-next", AVALON_PORT, _proc_ctx(a["pid"]), CFG.get("avalon_api_key", ""), "avalon"
    if m.get("running"):  # discovery adopted the systemd avalon on 8086
        return "qwen3.8-flash-next", AVALON_PORT, _proc_ctx(m["pid"]), CFG.get("avalon_api_key", ""), "avalon"
    return None

def _chat_target():
    """(port, api_key) for whichever server can answer chat completions right now, else None."""
    m = MGR.status()
    if m.get("running") and m.get("ready") and m["port"] != AVALON_PORT:
        return m["port"], ""
    a = SMGR.status()
    if a.get("running") and a.get("state") == "ready":
        return AVALON_PORT, CFG.get("avalon_api_key", "")
    if m.get("running") and m.get("ready"):  # discovery adopted the systemd avalon on 8086
        return AVALON_PORT, CFG.get("avalon_api_key", "")
    return None

def _yaml_merge(path, apply):
    path = Path(path)
    cfg = {}
    if path.exists():
        cfg = yaml.safe_load(path.read_text()) or {}
    apply(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(path)

def _json_merge(path, apply):
    path = Path(path)
    cfg = {}
    if path.exists():
        cfg = json.loads(path.read_text())
    apply(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return str(path)

def _toml_strip_section(text, headers):
    out, skip = [], False
    for ln in text.splitlines():
        if ln.startswith("[") and ln.endswith("]"):
            skip = ln.strip() in headers
        if not skip:
            out.append(ln)
    return "\n".join(out).rstrip() + "\n"

def _w_omp(alias, base, ctx):
    def apply(cfg):
        prov = {"baseUrl": base + "/v1", "api": "openai-completions", "auth": "none",
                "models": [{"id": alias, "name": alias, "api": "openai-completions",
                            "reasoning": False, "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": max(ctx, 8192), "maxTokens": min(8192, max(ctx, 8192))}]}
        cfg.setdefault("providers", {})[PANEL_PROVIDER_ID] = prov
    return _yaml_merge(HOME / ".omp/agent/models.yml", apply) + " — omp: omp --model lfm-panel/" + alias

def _w_opencode(alias, base, ctx):
    def apply(cfg):
        cfg.setdefault("provider", {})[PANEL_PROVIDER_ID] = {
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": base + "/v1"},
            "models": {alias: {"name": alias}}}
    return _json_merge(HOME / ".config/opencode/opencode.json", apply) + " — opencode: provider lfm-panel"

def _w_codex(alias, base, ctx):
    path = Path(HOME / ".codex/config.toml")
    old = path.read_text() if path.exists() else ""
    old = _toml_strip_section(old, {"[model_providers." + PANEL_PROVIDER_ID + "]", "[profiles." + PANEL_PROVIDER_ID + "]"})
    block = (f"\n[model_providers.{PANEL_PROVIDER_ID}]\nname = \"LFM Panel\"\n"
             f"base_url = \"{base}/v1\"\nwire_api = \"chat\"\n"
             f"\n[profiles.{PANEL_PROVIDER_ID}]\nmodel_provider = \"{PANEL_PROVIDER_ID}\"\nmodel = \"{alias}\"\n")
    import tomllib
    merged = old + block
    tomllib.loads(merged)  # validate before touching the user's config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged)
    return str(path) + " — codex: codex --profile lfm-panel"

def _w_aider(alias, base, ctx):
    def apply(cfg):
        cfg["openai-api-base"] = base + "/v1"
        cfg["openai-api-key"] = "panel"
        cfg["model"] = "openai/" + alias
    return _yaml_merge(HOME / ".aider.conf.yml", apply) + " — aider: aider (model openai/" + alias + ")"

def _w_zed(alias, base, ctx):
    def apply(cfg):
        lm = cfg.setdefault("language_models", {}).setdefault("openai", {})
        lm["api_url"] = base + "/v1"
        lm["available_models"] = [{"name": alias, "max_tokens": max(ctx, 8192), "tool_call": True}]
    return _json_merge(HOME / ".config/zed/settings.json", apply) + " — zed: openai provider model " + alias

def _w_continue(alias, base, ctx):
    def apply(cfg):
        models = [m for m in cfg.get("models", []) if m.get("name") != alias]
        models.append({"name": alias, "provider": "openai", "model": alias,
                       "apiBase": base + "/v1", "apiKey": "panel", "capabilities": ["tool_use"]})
        cfg["models"] = models
    return _yaml_merge(HOME / ".continue/config.yaml", apply) + " — continue: model " + alias

def _w_claude(alias, base, ctx, key):
    import httpx
    try:
        r = httpx.post(base + "/v1/messages", json={"model": alias, "max_tokens": 8,
                       "messages": [{"role": "user", "content": "hi"}]},
                       headers={"anthropic-version": "2023-06-01",
                                "x-api-key": key or "panel"}, timeout=15)
        if r.status_code == 404:
            raise HTTPException(400, "this llama-server has no Anthropic-compat /v1/messages endpoint")
    except httpx.HTTPError:
        raise HTTPException(400, "server unreachable for /v1/messages probe")
    def apply(cfg):
        cfg.setdefault("env", {}).update({
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_AUTH_TOKEN": key or "panel",
            "ANTHROPIC_MODEL": alias})
    return _json_merge(HOME / ".claude/settings.json", apply) + " — claude-code: env points at " + base

def _w_pi(alias, base, ctx, key):
    path = HOME / ".pi/agent/models.json"
    cfg = json.loads(path.read_text()) if path.exists() else {
        "$schema": "https://pi.dev/schemas/models.schema.json", "providers": {}}
    cfg.setdefault("providers", {})[PANEL_PROVIDER_ID] = {
        "baseUrl": base + "/v1", "api": "openai-completions", "apiKey": key or "none",
        "models": [{"id": alias, "name": alias, "api": "openai-completions",
                    "provider": PANEL_PROVIDER_ID, "reasoning": False, "input": ["text"],
                    "contextWindow": max(ctx, 8192), "maxTokens": min(8192, max(ctx, 8192))}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return str(path) + " — pi: provider lfm-panel"

def _w_hermes(alias, base, ctx, key):
    def apply(cfg):
        prov = cfg.setdefault("providers", {}).setdefault(PANEL_PROVIDER_ID, {})
        prov.update({"name": "LFM Panel", "base_url": base + "/v1",
                     "default_model": alias, "model": alias,
                     "models": {alias: {}}, "models_discovered": True})
    return _yaml_merge(HOME / ".hermes/config.yaml", apply) + " — hermes: pick /model custom:lfm-panel"

def _w_kimi(alias, base, ctx, key):
    path = HOME / ".kimi-code/config.toml"
    old = path.read_text() if path.exists() else ""
    old = _toml_strip_section(old, {"[providers." + PANEL_PROVIDER_ID + "]", '[models."' + alias + '"]'})
    block = (f"\n[providers.{PANEL_PROVIDER_ID}]\nname = \"LFM Panel\"\ntype = \"openai\"\n"
             f"baseUrl = \"{base}/v1\"\napiKey = \"{key or 'panel'}\"\n"
             f"\n[models.\"{alias}\"]\nprovider = \"{PANEL_PROVIDER_ID}\"\nname = \"{alias} via LFM Panel\"\n")
    import tomllib
    merged = old + block
    tomllib.loads(merged)  # validate before touching the user's config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged)
    return str(path) + " — kimi: model lfm-panel/" + alias

AGENTS = {
    "omp":         ("omp (oh-my-pi)",     lambda a, b, c, k: _w_omp(a, b, c),        "selector lfm-panel/<alias>"),
    "pi":          ("pi",                 lambda a, b, c, k: _w_pi(a, b, c, k),      "selector lfm-panel/<alias>"),
    "opencode":    ("OpenCode",           lambda a, b, c, k: _w_opencode(a, b, c),   "provider lfm-panel"),
    "codex":       ("Codex CLI",          lambda a, b, c, k: _w_codex(a, b, c),      "codex --profile lfm-panel"),
    "aider":       ("Aider",              lambda a, b, c, k: _w_aider(a, b, c),      "openai/<alias>"),
    "zed":         ("Zed",                lambda a, b, c, k: _w_zed(a, b, c),        "replaces Zed openai available_models"),
    "continue":    ("Continue",           lambda a, b, c, k: _w_continue(a, b, c),   "adds model row"),
    "claude-code": ("Claude Code",        _w_claude,                                 "needs /v1/messages; rewrites ANTHROPIC_BASE_URL"),
    "hermes":      ("Hermes Agent",       lambda a, b, c, k: _w_hermes(a, b, c, k),  "then pick /model custom:lfm-panel"),
    "kimi":        ("Kimi Code",          lambda a, b, c, k: _w_kimi(a, b, c, k),    "model lfm-panel/<alias>"),
}

@app.get("/api/agents")
def api_agents():
    return {"agents": [{"id": k, "label": v[0], "hint": v[2]} for k, v in AGENTS.items()],
            "registry": _load_agents_state(),
            "running": _current_llm()}

class AgentSyncBody(BaseModel):
    agents: list[str]

@app.post("/api/agents/sync")
def api_agents_sync(b: AgentSyncBody):
    cur = _current_llm()
    if not cur:
        raise HTTPException(409, "no running server — start a model first")
    alias, port, ctx, key, source = cur
    base = f"http://127.0.0.1:{port}"
    st = _load_agents_state()
    out = {}
    for aid in b.agents:
        if aid not in AGENTS:
            out[aid] = {"ok": False, "detail": "unknown agent"}
            continue
        try:
            detail = AGENTS[aid][1](alias, base, ctx, key)
            st.setdefault("agents", {})[aid] = {"alias": alias, "port": port, "synced_at": time.time()}
            out[aid] = {"ok": True, "detail": detail}
        except HTTPException as e:
            out[aid] = {"ok": False, "detail": str(e.detail)}
        except Exception as e:
            out[aid] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    st.update(alias=alias, port=port, source=source, synced_at=time.time())
    _save_agents_state(st)
    return {"results": out, "alias": alias, "port": port}


class StartBody(BaseModel):
    model: str
    preset: str = "auto"
    extra_args: str = ""
    ctx: int | None = None
    ngl: int | None = None
    threads: int | None = None
    fa: str = "auto"
    ctk: str = ""
    ctv: str = ""
    parallel: int | None = None
    batch: int | None = None
    ubatch: int | None = None
    ncmoe: int | None = None
    mlock: bool = False
    no_mmap: bool = False
    save_profile: bool = True

@app.post("/api/server/start")
def api_start(b: StartBody):
    return MGR.start(
        b.model, preset=b.preset, extra_args=b.extra_args, ctx=b.ctx,
        ngl=b.ngl, threads=b.threads, fa=b.fa, ctk=b.ctk, ctv=b.ctv,
        parallel=b.parallel, batch=b.batch, ubatch=b.ubatch,
        ncmoe=b.ncmoe, mlock=b.mlock, no_mmap=b.no_mmap,
        save_profile=b.save_profile,
    )

@app.post("/api/server/stop")
def api_stop():
    return MGR.stop()

# ---------------------------------------------------------------- settings api
class SettingsBody(BaseModel):
    model_dirs: list[str] | None = None
    server_binary: str | None = None
    ctx: int | None = None
    port_base: int | None = None
    idle_unload_min: int | None = None
    vram_reserve_mb: int | None = None
    hf_token: str | None = None

@app.get("/api/settings")
def api_settings_get():
    bin_path = CFG.get("server_binary", BIN)
    caps = bin_caps(bin_path)
    dirs_info = []
    for d in CFG.get("model_dirs", []):
        p = Path(d).expanduser()
        exists = p.is_dir()
        free_gb = None
        if exists:
            try:
                free_gb = round(shutil.disk_usage(str(p)).free / 2**30, 1)
            except Exception:
                pass
        dirs_info.append({"path": d, "exists": exists, "free_gb": free_gb})
    tok = CFG.get("hf_token", "")
    env_tok = bool(os.environ.get("HF_TOKEN"))
    masked = (tok[:4] + "…" + tok[-4:]) if len(tok) >= 8 else ("(from $HF_TOKEN)" if env_tok else "")
    return {
        "model_dirs": CFG.get("model_dirs", []),
        "dirs_info": dirs_info,
        "disks": dirs_info,
        "server_binary": bin_path,
        "bin_caps": caps,
        "caps": caps,
        "ctx": int(CFG.get("ctx", 8192) or 8192),
        "port_base": int(CFG.get("port_base", 8830) or 8830),
        "idle_unload_min": int(CFG.get("idle_unload_min", 0) or 0),
        "vram_reserve_mb": int(CFG.get("vram_reserve_mb", 512) or 512),
        "hf_token_set": bool(tok) or env_tok,
        "hf_token_masked": masked,
        "hf_token_preview": masked,
        "config_path": str(CONFIG_PATH),
    }

@app.post("/api/settings")
def api_settings_set(b: SettingsBody):
    if b.model_dirs is not None:
        cleaned = [d.strip() for d in b.model_dirs if d and d.strip()]
        if not cleaned:
            raise HTTPException(400, "At least one model directory is required")
        CFG["model_dirs"] = cleaned
        _model_cache.clear()
    if b.server_binary is not None and b.server_binary.strip():
        CFG["server_binary"] = str(Path(b.server_binary.strip()).expanduser())
    if b.ctx is not None:
        CFG["ctx"] = max(512, min(1048576, int(b.ctx)))
    if b.port_base is not None:
        CFG["port_base"] = max(1024, min(65000, int(b.port_base)))
    if b.idle_unload_min is not None:
        CFG["idle_unload_min"] = max(0, min(1440, int(b.idle_unload_min)))
    if b.vram_reserve_mb is not None:
        CFG["vram_reserve_mb"] = max(0, min(16384, int(b.vram_reserve_mb)))
    if b.hf_token is not None:
        val = b.hf_token.strip()
        if val == "clear":
            CFG["hf_token"] = ""
        elif val and val != "__keep__":
            CFG["hf_token"] = val
    _save_config_secure()
    return api_settings_get()

@app.post("/api/chat")
async def api_chat(req: Request):
    tgt = _chat_target()
    if not tgt:
        raise HTTPException(409, "no model ready — load one in the Models tab or start Avalon")
    MGR.touch()
    port, key = tgt
    body = await req.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "expected a JSON object body")
    body.setdefault("stream", True)
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    client = httpx.AsyncClient(timeout=None)
    async def gen():
        try:
            async with client.stream("POST", url, json=body, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    MGR.touch()
                    yield chunk
        finally:
            await client.aclose()
    return StreamingResponse(gen(), media_type="text/event-stream")

# ---------------------------------------------------------------- UI
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<title>LFM Panel · Local Inference Console</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<script>
(() => {
  try {
    const s = localStorage.getItem('lfm.theme');
    const m = document.querySelector('meta[name="color-scheme"]');
    const t = (s === 'light' || s === 'dark') ? s : (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    if (m) m.content = t;
    document.documentElement.dataset.theme = t;
  } catch (e) {}
})();
</script>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%2322d3ee'/><path d='M17.5 5l-8.5 13h5.2l-1.2 9 8.5-13h-5.2z' fill='%2305131a'/></svg>">
<style>
:root:has(> head > meta[name="color-scheme"][content="dark"]) { color-scheme: dark; }
:root:has(> head > meta[name="color-scheme"][content="light"]) { color-scheme: light; }
:root {
  --bg:#090b10; --bg-2:#0d1118; --panel:#11161f; --panel-2:#151c27; --raise:#1a2230;
  --bd:#1f2837; --bd-2:#2a3548;
  --fg:#edf2f8; --fg-2:#c0cad8; --mut:#8493a8; --faint:#59667a;
  --acc:#22d3ee; --acc-2:#f59e0b; --acc-ink:#04131a;
  --ok:#34d399; --warn:#f59e0b; --err:#f87171;
  --glow:radial-gradient(1000px 460px at 84% -10%,rgba(34,211,238,.09) 0%,rgba(245,158,11,.04) 44%,rgba(0,0,0,0) 70%);
  --grad:linear-gradient(135deg,#22d3ee,#06b6d4);
  --sh:0 2px 6px rgba(0,0,0,.28);
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --r:12px;
}
[data-theme=light] {
  --bg:#f4f6fa; --bg-2:#eaeff5; --panel:#ffffff; --panel-2:#f9fbfd; --raise:#edf2f8;
  --bd:#d9e1ec; --bd-2:#c2cedd;
  --fg:#0f1724; --fg-2:#334155; --mut:#5b6b82; --faint:#8594a8;
  --acc:#0284c7; --acc-2:#d97706; --acc-ink:#ffffff;
  --ok:#059669; --warn:#b45309; --err:#dc2626;
  --glow:radial-gradient(1000px 460px at 84% -10%,rgba(2,132,199,.08) 0%,rgba(0,0,0,0) 68%);
  --sh:0 2px 6px rgba(15,23,36,.07);
}
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;min-height:100vh;background-color:var(--bg);
  background-image:var(--glow),radial-gradient(color-mix(in srgb,var(--bd-2) 45%,transparent) 1px,transparent 1px);
  background-size:100% 100%,24px 24px;background-repeat:no-repeat,repeat;color:var(--fg);
  font:13.5px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
::selection{background:color-mix(in srgb,var(--acc) 26%,transparent);color:var(--fg)}
input,textarea{caret-color:var(--acc)}
h1,h2,h3{margin:0;letter-spacing:-.015em}
a{color:var(--acc);text-decoration:none}
a:hover{text-decoration:underline}
code,pre,.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.num{font-variant-numeric:tabular-nums}
.skip{position:absolute;left:-1px;top:-44px;background:var(--acc);color:var(--acc-ink);padding:8px 12px;border-radius:8px;z-index:99;font-weight:600}
.skip:focus{top:8px;left:8px}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px;border-radius:6px}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:var(--bd-2);border-radius:99px;border:2px solid transparent;background-clip:content-box}
::-webkit-scrollbar-track{background:transparent}

/* ---------- header ---------- */
header{position:sticky;top:0;z-index:40;display:flex;align-items:center;gap:10px;padding:8px 16px;
  background:color-mix(in srgb,var(--bg) 86%,transparent);
  backdrop-filter:blur(14px) saturate(150%);-webkit-backdrop-filter:blur(14px) saturate(150%);
  border-bottom:1px solid var(--bd)}
.brand{display:flex;align-items:center;gap:10px;flex-shrink:0}
.mark{width:30px;height:30px;border-radius:8px;background:var(--grad);display:grid;place-items:center;
  color:var(--acc-ink);box-shadow:0 1px 3px rgba(0,0,0,.35)}
.mark svg{width:16px;height:16px}
.brand .nm{font-size:14.5px;font-weight:700;line-height:1.15;letter-spacing:-.02em}
.brand .sub{display:block;font-size:11px;font-weight:500;color:var(--mut);font-family:var(--mono)}
.nav{display:flex;gap:2px;padding:3px;background:var(--panel);border:1px solid var(--bd);border-radius:10px;
  overflow-x:auto;scrollbar-width:none;min-width:0}
.nav::-webkit-scrollbar{display:none}
.nav button{display:flex;align-items:center;gap:6px;padding:6px 10px;border:0;border-radius:7px;background:transparent;
  color:var(--mut);font:inherit;font-size:12.5px;font-weight:600;cursor:pointer;white-space:nowrap;transition:background .14s,color .14s}
.nav button svg{width:14px;height:14px;opacity:.82;flex-shrink:0}
.nav button:hover{color:var(--fg);background:var(--raise)}
.nav button.on{color:var(--fg);background:var(--raise);box-shadow:inset 0 0 0 1px var(--bd-2)}
.nav button.on svg{color:var(--acc);opacity:1}
.hgap{flex:1;min-width:4px}
.tele-strip{display:flex;align-items:center;gap:6px;flex-wrap: nowrap;overflow:hidden}
.pill{display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:999px;border:1px solid var(--bd-2);
  background:var(--panel);color:var(--fg-2);font-size:12px;font-weight:600;white-space:nowrap;max-width:34vw;overflow:hidden;
  font-variant-numeric:tabular-nums}
.pill.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 36%,transparent);background:color-mix(in srgb,var(--ok) 10%,transparent)}
.pill.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 36%,transparent);background:color-mix(in srgb,var(--warn) 10%,transparent)}
.pill .dot{width:7px;height:7px;border-radius:99px;background:currentColor;flex-shrink:0}
.icon{display:grid;place-items:center;width:32px;height:32px;border-radius:8px;border:1px solid var(--bd);
  background:var(--panel);color:var(--mut);cursor:pointer;flex-shrink:0;transition:background .14s,color .14s,border-color .14s}
.icon:hover{color:var(--fg);border-color:var(--bd-2);background:var(--raise)}
.icon svg{width:15px;height:15px}
.cmdbtn{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:8px;border:1px solid var(--bd);
  background:var(--panel);color:var(--mut);font:inherit;font-size:12px;font-weight:600;cursor:pointer;transition:.14s;flex-shrink:0}
.cmdbtn:hover{color:var(--fg);border-color:var(--bd-2);background:var(--raise)}
.cmdbtn svg{width:13px;height:13px}

/* ---------- layout ---------- */
main{padding:16px 16px 44px;max-width:1320px;margin:0 auto}
section{animation:fade .18s ease-out}
@keyframes fade{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
.hide{display:none !important}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.card{background:linear-gradient(180deg,var(--panel-2),var(--panel));border:1px solid var(--bd);border-radius:var(--r);
  padding:14px 16px;margin-bottom:14px;box-shadow:var(--sh)}
.card.sub{background:var(--bg-2);box-shadow:none;padding:12px 14px;margin-bottom:10px}
.hd{display:flex;align-items:center;gap:10px;justify-content:space-between;flex-wrap:wrap;margin:-2px 0 12px}
.hd h2{font-size:14px;font-weight:650;display:flex;align-items:center;gap:8px}
.hd .meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.two{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
.two>.side{width:270px;flex:0 0 270px}
.two>.main{flex:1;min-width:300px}
.hint{color:var(--mut);font-size:12px}
.sep{height:1px;background:var(--bd);margin:12px 0}
.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px}
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-size:11.5px;font-weight:600;color:var(--fg-2);display:flex;justify-content:space-between;align-items:center}
.field label span{color:var(--mut);font-weight:500;font-family:var(--mono)}

/* ---------- controls ---------- */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:7px 12px;border-radius:8px;border:1px solid var(--bd-2);
  background:var(--raise);color:var(--fg-2);font:inherit;font-size:12.5px;font-weight:600;cursor:pointer;white-space:nowrap;
  transition:transform .1s,background .14s,border-color .14s,color .14s,opacity .14s}
.btn svg{width:14px;height:14px;flex-shrink:0}
.btn:hover{background:var(--panel-2);border-color:var(--mut);color:var(--fg)}
.btn:active{transform:translateY(1px)}
.btn.primary{background:var(--grad);border:1px solid transparent;color:var(--acc-ink);box-shadow:0 2px 6px rgba(0,0,0,.28)}
.btn.primary:hover{filter:brightness(1.07);color:var(--acc-ink)}
.btn.acc2{background:color-mix(in srgb,var(--acc-2) 15%,transparent);border-color:color-mix(in srgb,var(--acc-2) 42%,transparent);color:var(--acc-2)}
.btn.acc2:hover{background:color-mix(in srgb,var(--acc-2) 24%,transparent)}
.btn.danger{background:color-mix(in srgb,var(--err) 13%,transparent);border-color:color-mix(in srgb,var(--err) 38%,transparent);color:var(--err)}
.btn.danger:hover{background:color-mix(in srgb,var(--err) 22%,transparent);color:var(--err)}
.btn.sm{padding:5px 9px;font-size:12px;border-radius:7px}
.btn.xs{padding:3px 7px;font-size:11.5px;border-radius:6px}
.btn[disabled]{opacity:.42;cursor:not-allowed;transform:none;pointer-events:none}
select,input[type=text],input[type=password],input[type=search],input[type=number],textarea{
  background:var(--bg-2);border:1px solid var(--bd-2);color:var(--fg);border-radius:8px;padding:6px 10px;
  font:inherit;font-size:12.5px;transition:border-color .14s}
select:focus,input:focus,textarea:focus{border-color:var(--acc);outline:2px solid var(--acc);outline-offset:1px}
input[type=checkbox],input[type=radio]{accent-color:var(--acc);width:15px;height:15px;cursor:pointer}
input[type=range]{accent-color:var(--acc);width:100%;cursor:pointer}
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.search{position:relative;flex:1;min-width:190px}
.search svg{position:absolute;left:9px;top:50%;transform:translateY(-50%);width:14px;height:14px;color:var(--faint);pointer-events:none}
.search input{width:100%;padding-left:30px}
.segs{display:flex;gap:2px;padding:3px;background:var(--bg-2);border:1px solid var(--bd);border-radius:9px;flex-wrap:wrap}
.segs button{border:0;background:transparent;color:var(--mut);font:inherit;font-size:12px;font-weight:600;
  padding:4px 9px;border-radius:6px;cursor:pointer;transition:.14s;white-space:nowrap}
.segs button:hover{color:var(--fg)}
.segs button.on{background:var(--raise);color:var(--fg);box-shadow:inset 0 0 0 1px var(--bd-2)}
.switch{display:inline-flex;align-items:center;gap:7px;cursor:pointer;user-select:none}
.switch input{position:absolute;opacity:0;pointer-events:none}
.switch .track{width:36px;height:20px;border-radius:99px;background:var(--bg-2);border:1px solid var(--bd-2);position:relative;transition:.16s;flex-shrink:0}
.switch .track::after{content:"";position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:99px;background:var(--mut);transition:.16s}
.switch input:checked+.track{background:color-mix(in srgb,var(--acc) 26%,transparent);border-color:var(--acc)}
.switch input:checked+.track::after{transform:translateX(16px);background:var(--acc)}
.switch input:focus-visible+.track{outline:2px solid var(--acc);outline-offset:2px}

/* ---------- tags / stats / meters ---------- */
.tag{display:inline-flex;align-items:center;gap:4px;padding:2px 7px;border-radius:999px;border:1px solid var(--bd-2);
  background:color-mix(in srgb,var(--fg) 4%,transparent);color:var(--fg-2);font-size:11px;font-weight:600;white-space:nowrap;
  font-variant-numeric:tabular-nums}
.tag.acc{color:var(--acc);border-color:color-mix(in srgb,var(--acc) 35%,transparent);background:color-mix(in srgb,var(--acc) 10%,transparent)}
.tag.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 35%,transparent);background:color-mix(in srgb,var(--ok) 10%,transparent)}
.tag.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 35%,transparent);background:color-mix(in srgb,var(--warn) 10%,transparent)}
.tag.err{color:var(--err);border-color:color-mix(in srgb,var(--err) 35%,transparent);background:color-mix(in srgb,var(--err) 10%,transparent)}
.tag.mut{color:var(--mut)}
.kvs{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.stat{font-size:24px;font-weight:700;letter-spacing:-.03em;font-variant-numeric:tabular-nums;line-height:1.15}
.stat small{font-size:12px;font-weight:500;color:var(--mut);letter-spacing:0}
.meter{height:7px;border-radius:99px;background:var(--bg-2);border:1px solid var(--bd);overflow:hidden;margin:6px 0 2px;display:flex}
.meter i{display:block;height:100%;background:var(--grad);transition:opacity .2s}
.meter i.kv{background:var(--acc-2)}
.meter i.cmp{background:var(--ok)}
.meter.warn i{background:linear-gradient(90deg,var(--warn),#fb923c)}
.meter.err i{background:linear-gradient(90deg,var(--err),#f43f5e)}
.spark{width:100%;height:26px;display:block;margin-top:6px;opacity:.85}

/* ---------- model list ---------- */
.mlist{display:flex;flex-direction:column;gap:8px}
.mrow{display:grid;grid-template-columns:auto minmax(0,1fr) minmax(170px,225px) auto;gap:12px;align-items:center;padding:11px 13px;
  border:1px solid var(--bd);border-radius:11px;background:var(--panel);transition:border-color .14s,background .14s}
.mrow:hover{border-color:var(--bd-2);background:var(--panel-2)}
.mrow.run{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 5%,var(--panel))}
.pinbtn{background:transparent;border:0;color:var(--faint);cursor:pointer;padding:4px;border-radius:6px;display:grid;place-items:center;transition:.14s}
.pinbtn:hover{color:var(--acc-2);background:var(--raise)}
.pinbtn.on{color:var(--acc-2)}
.pinbtn svg{width:15px;height:15px}
.mrow .nm{font-weight:650;font-size:13.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:flex;align-items:center;gap:6px}
.mrow .pth{color:var(--faint);font-size:11px;font-family:var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:1px}
.mrow .tags{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.fitbox{display:flex;flex-direction:column;gap:2px;font-size:11px;font-variant-numeric:tabular-nums}
.fitbox .fit-top{display:flex;justify-content:space-between;gap:6px;color:var(--fg-2);font-weight:600}
.fitbox .fit-sub{color:var(--mut);font-size:10.5px}
.mrow .acts{display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.empty{text-align:center;color:var(--mut);padding:26px 12px;border:1px dashed var(--bd-2);border-radius:11px}
.empty b{display:block;color:var(--fg-2);font-size:13.5px;margin-bottom:4px}

/* ---------- logs & cmd preview ---------- */
.cmdbox{font-family:var(--mono);font-size:11.5px;line-height:1.5;color:var(--fg-2);background:var(--bg);
  border:1px solid var(--bd);border-radius:9px;padding:8px 11px;overflow-x:auto;white-space:pre-wrap;word-break:break-all}
.logbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.log{font-family:var(--mono);font-size:11.5px;line-height:1.55;color:var(--fg-2);background:var(--bg);
  border:1px solid var(--bd);border-radius:10px;padding:10px 12px;max-height:400px;overflow:auto;
  white-space:pre-wrap;word-break:break-word;margin:0}
.log.wrapoff{white-space:pre}
.log .e{color:var(--err)}
.log .w{color:var(--warn)}
.log .g{color:var(--faint)}

/* ---------- table ---------- */
.tbl-wrap{overflow-x:auto;border:1px solid var(--bd);border-radius:10px}
table.tbl{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
table.tbl th,table.tbl td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--bd)}
table.tbl th{background:var(--bg-2);color:var(--mut);font-weight:600;font-size:11px;white-space:nowrap}
table.tbl tr:last-child td{border-bottom:0}
table.tbl tr:hover td{background:color-mix(in srgb,var(--raise) 50%,transparent)}

/* ---------- chat ---------- */
.thread{display:flex;flex-direction:column;gap:10px;padding:2px;min-height:38vh;max-height:58vh;overflow-y:auto}
.msg{max-width:min(88%,800px);padding:10px 13px;border-radius:12px;border:1px solid var(--bd);
  background:var(--panel-2);line-height:1.6;overflow-wrap:break-word;position:relative}
.msg.u{align-self:flex-end;background:linear-gradient(180deg,color-mix(in srgb,var(--acc) 16%,var(--panel)),var(--panel));
  border-color:color-mix(in srgb,var(--acc) 34%,transparent)}
.msg.a{align-self:flex-start}
.msg .who{font-size:11px;font-weight:650;color:var(--mut);margin-bottom:4px;display:flex;align-items:center;justify-content:space-between;gap:8px}
.msg .codewrap{margin:8px 0;border:1px solid var(--bd);border-radius:9px;overflow:hidden;background:var(--bg)}
.msg .codebar{display:flex;align-items:center;justify-content:space-between;padding:4px 10px;background:var(--bg-2);
  border-bottom:1px solid var(--bd);font-family:var(--mono);font-size:11px;color:var(--mut)}
.msg pre{margin:0;padding:10px 12px;overflow:auto;font-size:12px;line-height:1.55;background:transparent;border:0}
.msg code{font-size:12px;background:color-mix(in srgb,var(--bg-2) 80%,transparent);padding:1px 5px;border-radius:5px;border:1px solid var(--bd)}
.msg pre code{background:none;border:0;padding:0}
.msg .md-h{font-weight:700;margin:8px 0 3px}
.msg ul,.msg ol{margin:6px 0;padding-left:20px}
.msg details{margin-bottom:8px;border:1px solid var(--bd);border-radius:8px;padding:6px 10px;background:var(--bg-2);color:var(--mut);font-size:12px}
.msg summary{cursor:pointer;font-weight:600;color:var(--fg-2);font-size:11.5px}
.msg details[open] summary{margin-bottom:5px}
.msg .think{white-space:pre-wrap;font-family:var(--mono);font-size:11.5px}
.msg .bar{display:flex;gap:5px;justify-content:flex-end;margin-top:6px;opacity:.75;transition:opacity .14s}
.msg:hover .bar,.msg:focus-within .bar{opacity:1}
.typing{color:var(--mut);font-size:12px;font-family:var(--mono)}
.composer{display:flex;gap:8px;align-items:flex-end;border:1px solid var(--bd-2);border-radius:11px;padding:7px 8px 7px 11px;
  background:var(--bg-2);margin-top:10px;transition:border-color .14s}
.composer:focus-within{border-color:var(--acc);outline:2px solid var(--acc);outline-offset:1px}
.composer textarea{flex:1;border:0;background:transparent;color:var(--fg);resize:none;min-height:24px;max-height:220px;
  padding:4px 0;font-size:13.5px;line-height:1.5}
.composer textarea:focus{outline:0;box-shadow:none}
.clist{display:flex;flex-direction:column;gap:4px;max-height:52vh;overflow-y:auto}
.citem{display:flex;align-items:center;gap:6px;padding:7px 8px;border-radius:8px;border:1px solid transparent;
  cursor:pointer;transition:.14s}
.citem:hover{background:var(--raise);border-color:var(--bd)}
.citem.on{background:var(--raise);border-color:var(--bd-2)}
.citem .t{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12.5px;font-weight:500}
.citem .s{font-size:11px;color:var(--faint);white-space:nowrap;font-variant-numeric:tabular-nums}

/* ---------- hf / downloads ---------- */
.frow{display:flex;align-items:center;gap:10px;padding:7px 9px;border-radius:8px;cursor:pointer;border:1px solid transparent}
.frow:hover{background:var(--raise);border-color:var(--bd)}
.frow .p{flex:1;min-width:0;font-family:var(--mono);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dl{padding:9px 0;border-bottom:1px solid var(--bd)}
.dl:last-child{border-bottom:0}

/* ---------- native dialog & command palette ---------- */
dialog{border:1px solid var(--bd-2);border-radius:14px;background:linear-gradient(180deg,var(--panel-2),var(--panel));
  color:var(--fg);padding:0;max-width:min(760px,95vw);width:100%;max-height:88vh;
  box-shadow:0 4px 10px rgba(0,0,0,.45)}
dialog:modal{overscroll-behavior:contain;display:flex;flex-direction:column}
dialog:not([open]){display:none}
dialog::backdrop{background:rgba(4,7,12,.68);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);
  overflow:hidden;overscroll-behavior:contain}
.dlg-hd{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:13px 16px;border-bottom:1px solid var(--bd)}
.dlg-hd h2{font-size:14.5px;font-weight:650}
.dlg-bd{padding:14px 16px;overflow-y:auto;min-height:0;flex:1}
.dlg-ft{display:flex;gap:8px;justify-content:flex-end;align-items:center;padding:11px 16px;border-top:1px solid var(--bd);flex-wrap:wrap;background:var(--bg-2);border-radius:0 0 14px 14px}
.pal-list{display:flex;flex-direction:column;gap:3px;max-height:52vh;overflow-y:auto;margin-top:8px}
.pal-item{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 10px;border-radius:8px;
  border:1px solid transparent;background:transparent;color:var(--fg);cursor:pointer;text-align:left;font:inherit;width:100%}
.pal-item:hover,.pal-item.sel{background:var(--raise);border-color:var(--bd-2)}
.pal-item .pi-main{display:flex;align-items:center;gap:8px;min-width:0;font-weight:600;font-size:13px}
.pal-item .pi-sub{font-size:11.5px;color:var(--mut);font-family:var(--mono);white-space:nowrap}

/* ---------- toasts ---------- */
.toasts{position:fixed;right:16px;bottom:16px;display:flex;flex-direction:column;gap:8px;z-index:95;max-width:min(430px,92vw)}
.toast{display:flex;gap:9px;align-items:flex-start;padding:10px 12px;border-radius:10px;border:1px solid var(--bd-2);
  background:var(--panel-2);box-shadow:0 6px 14px rgba(0,0,0,.4);cursor:pointer;font-size:12.5px;
  animation:tin .18s cubic-bezier(.2,.8,.2,1)}
.toast.out{animation:tout .16s forwards}
.toast .ti{width:8px;height:8px;border-radius:99px;background:var(--acc);margin-top:5px;flex-shrink:0}
.toast.ok .ti{background:var(--ok)}.toast.err{border-color:color-mix(in srgb,var(--err) 42%,transparent)}
.toast.err .ti{background:var(--err)}.toast.warn .ti{background:var(--warn)}
@keyframes tin{from{opacity:0;transform:translateY(8px) scale(.98)}to{opacity:1;transform:none}}
@keyframes tout{to{opacity:0;transform:translateY(5px)}}

/* ---------- footer ---------- */
footer{max-width:1320px;margin:0 auto;padding:0 16px 26px;color:var(--faint);font-size:11.5px;
  display:flex;gap:14px;flex-wrap:wrap;align-items:center}
kbd{font-family:var(--mono);font-size:10.5px;border:1px solid var(--bd-2);border-radius:5px;
  padding:1px 5px;background:var(--bg-2);color:var(--fg-2)}

@media (max-width:960px){
  header{flex-wrap:wrap;gap:8px}
  .brand .sub{display:none}
  .nav{order:3;width:100%}
  .two>.side{width:100%;flex:1 1 100%}
  .mrow{grid-template-columns:auto 1fr}
  .mrow .fitbox{grid-column:1 / -1}
  .mrow .acts{grid-column:1 / -1;justify-content:flex-start}
  .thread{max-height:none}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms !important;animation-iteration-count:1 !important;transition-duration:.01ms !important}
}
</style></head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header>
  <div class="brand">
    <span class="mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M13.2 2 4 14h6l-1.4 8L20 9.5h-6.2z"/></svg></span>
    <span class="nm">LFM Panel<span class="sub">local inference console</span></span>
  </div>
  <nav class="nav" id="nav" aria-label="Sections">
    <button data-t="models" class="on"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z"/><path d="M4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>Models</button>
    <button data-t="server"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><path d="M7 7.5h.01M7 16.5h.01"/></svg>Server &amp; Bench</button>
    <button data-t="avalon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/></svg>Avalon</button>
    <button data-t="vitals"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 12h4l2.5-6 4 12L16 12h5"/></svg>Vitals</button>
    <button data-t="chat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 12a8 8 0 01-8 8H7l-4 3 1.2-4.2A8 8 0 1121 12z"/></svg>Chat</button>
    <button data-t="hf"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v12m0 0l-4-4m4 4l4-4"/><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"/></svg>HF Pull</button>
    <button data-t="settings"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09a1.65 1.65 0 00-1-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09a1.65 1.65 0 001.51-1 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06a1.65 1.65 0 001.82.33h.08a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51h.08a1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82v.08a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>Settings</button>
  </nav>
  <span class="hgap"></span>
  <div class="tele-strip">
    <span class="pill" id="hdrstatus" title="Active inference server"><span class="dot"></span><span id="hdrtext">checking…</span></span>
    <span class="tag acc hide" id="hdrtps" title="Live generation throughput"></span>
    <span class="tag mut hide" id="hdrkv" title="KV cache fill"></span>
    <span class="tag mut hide" id="hdridle" title="Idle auto-unload timer"></span>
  </div>
  <button class="cmdbtn" id="cmdbtn" title="Command palette (Ctrl+K or Cmd+K)" aria-label="Open command palette">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
    <kbd>⌘K</kbd>
  </button>
  <button class="icon" id="theme" title="Toggle light / dark theme" aria-label="Toggle theme"></button>
  <button class="icon" id="logout" title="Sign out" aria-label="Sign out"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M15 12H4m0 0l3.5-3.5M4 12l3.5 3.5"/><path d="M9 4h7a2 2 0 012 2v12a2 2 0 01-2 2H9"/></svg></button>
</header>

<main id="main">
<!-- ==================== MODELS ==================== -->
<section id="t-models">
  <div class="card">
    <div class="hd">
      <h2>Model library</h2>
      <div class="meta">
        <span class="tag mut" id="libinfo">scanning…</span>
        <span class="tag mut" id="vramchip"></span>
        <button class="btn sm" id="rescan" title="Re-scan model directories">Rescan</button>
      </div>
    </div>
    <div class="toolbar">
      <div class="search">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="msearch" type="search" placeholder="Filter by name, quant, arch… (press /)" aria-label="Filter models">
      </div>
      <div class="segs" id="mfilter" role="group" aria-label="Filter models">
        <button data-f="all" class="on">All</button>
        <button data-f="pin">Pinned</button>
        <button data-f="recent">Recent</button>
        <button data-f="fit">Fits VRAM</button>
        <button data-f="moe">MoE</button>
        <button data-f="vl">Vision</button>
        <button data-f="run">Loaded</button>
      </div>
      <select id="msort" aria-label="Sort models">
        <option value="smart">Pinned &amp; Recent</option>
        <option value="name">Name A–Z</option>
        <option value="size">Largest first</option>
        <option value="small">Smallest first</option>
        <option value="speed">Fastest (bench tok/s)</option>
        <option value="arch">Architecture</option>
      </select>
    </div>
    <div id="modellist" class="mlist"></div>
    <p class="hint" style="margin:12px 0 0">VRAM fit estimates combine model weights, KV cache at default context, and compute buffer against your GPU/RAM budget. Click <b>Tune &amp; Launch</b> on any model for the interactive config optimizer.</p>
  </div>
</section>

<!-- ==================== SERVER & BENCH ==================== -->
<section id="t-server" class="hide">
  <div class="card">
    <div class="hd">
      <h2>Active llama-server</h2>
      <div class="meta">
        <span class="tag" id="srvchip">checking…</span>
        <span class="tag mut hide" id="srvidlechip"></span>
      </div>
    </div>
    <div id="srvdetail" class="kvs"></div>
    <div id="srvcmdwrap" class="hide" style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span class="hint">Active launch command</span>
        <button class="btn xs" id="srvcopycmd">Copy command</button>
      </div>
      <div class="cmdbox" id="srvcmd"></div>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center">
      <button class="btn danger" id="srvstop">Stop server</button>
      <button class="btn" id="srvtune">Tune active model…</button>
      <button class="btn" id="agentsbtn">Register in coding agents…</button>
      <span class="hint">Binds 127.0.0.1 · reachable on LAN through this panel</span>
    </div>
  </div>

  <div class="grid4" style="margin-bottom:14px">
    <div class="card" style="margin:0">
      <div class="hd" style="margin-bottom:6px"><h2>Generation speed</h2><span class="tag acc" id="t-gen-src">idle</span></div>
      <div class="stat"><span id="t-gen-tps">0</span> <small>tok/s</small></div>
      <p class="hint" style="margin:6px 0 0" id="t-gen-sub">Waiting for requests</p>
    </div>
    <div class="card" style="margin:0">
      <div class="hd" style="margin-bottom:6px"><h2>Prompt eval</h2><span class="tag mut">prefill</span></div>
      <div class="stat"><span id="t-pp-tps">0</span> <small>tok/s</small></div>
      <p class="hint" style="margin:6px 0 0" id="t-pp-sub">Prompt processing rate</p>
    </div>
    <div class="card" style="margin:0">
      <div class="hd" style="margin-bottom:6px"><h2>KV cache fill</h2><span class="tag mut" id="t-kv-pct">0%</span></div>
      <div class="stat"><span id="t-kv-used">0</span> <small>/ <span id="t-kv-max">0</span> tok</small></div>
      <div class="meter" id="t-kv-meter"><i style="width:0%"></i></div>
    </div>
    <div class="card" style="margin:0">
      <div class="hd" style="margin-bottom:6px"><h2>Parallel slots</h2><span class="tag mut" id="t-slot-badge">0 / 0</span></div>
      <div class="stat"><span id="t-slot-busy">0</span> <small>active</small></div>
      <div class="kvs" style="margin-top:6px" id="t-slot-pills"></div>
    </div>
  </div>

  <div class="card">
    <div class="hd">
      <h2>One-click benchmark &amp; history</h2>
      <div class="meta">
        <select id="benchprompt" aria-label="Benchmark prompt">
          <option value="standard">Standard reasoning &amp; math prompt</option>
          <option value="code">Code generation (Python token bucket)</option>
          <option value="short">Short latency / TTFT probe</option>
        </select>
        <select id="benchmax" aria-label="Max tokens">
          <option value="96">96 tokens</option>
          <option value="160" selected>160 tokens</option>
          <option value="256">256 tokens</option>
          <option value="384">384 tokens</option>
        </select>
        <button class="btn primary sm" id="benchrun">Run benchmark</button>
        <button class="btn xs danger" id="benchclear" title="Clear saved benchmark history">Clear</button>
      </div>
    </div>
    <div id="benchtable"></div>
  </div>

  <div class="card">
    <div class="hd"><h2>Server log</h2><span class="tag mut" id="logn"></span></div>
    <div class="logbar">
      <div class="search" style="flex:1"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="logfilter" type="search" placeholder="Filter log lines…" aria-label="Filter log"></div>
      <label class="switch"><input type="checkbox" id="logauto" checked><span class="track"></span><span class="hint">Follow</span></label>
      <label class="switch"><input type="checkbox" id="logwrap" checked><span class="track"></span><span class="hint">Wrap</span></label>
      <button class="btn sm" id="logcopy">Copy</button><button class="btn sm" id="logsave">Save</button>
    </div>
    <pre class="log" id="log"></pre>
  </div>
</section>

<!-- ==================== AVALON ==================== -->
<section id="t-avalon" class="hide">
  <div class="card">
    <div class="hd"><h2>Avalon · Qwen3.8-Flash-Next (125B MoE)</h2><span class="tag" id="avalonchip">checking…</span></div>
    <div id="avalondetail" class="kvs" style="margin-bottom:10px"></div>
    <div id="avalonprofiles"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center">
      <button class="btn primary" id="avalonstart">Start selected profile</button>
      <button class="btn danger" id="avalonstop">Stop</button>
      <button class="btn acc2" id="avalontunebtn">Config Optimizer &amp; Tuning…</button>
      <span class="hint">OpenAI-compatible · streams <code>reasoning_content</code></span>
    </div>

    <div id="avalontunebox" class="card sub hide" style="margin-top:14px">
      <div class="hd">
        <h2>Avalon Config Optimizer &amp; Parameter Overrides</h2>
        <div class="meta">
          <button class="btn xs" data-avopt="speed">Preset: Max Speed (MTP)</button>
          <button class="btn xs" data-avopt="balanced">Preset: Balanced 65k</button>
          <button class="btn xs" data-avopt="max_ctx">Preset: 131k Flagship</button>
          <button class="btn xs" data-avopt="reset">Reset overrides</button>
        </div>
      </div>
      <div class="form-grid">
        <div class="field"><label>Context window (-c) <span id="av-ctx-lbl">default</span></label>
          <select id="av-ctx"><option value="0">Profile default</option><option value="32768">32,768 (32k)</option><option value="65536">65,536 (65k)</option><option value="98304">98,304 (96k)</option><option value="131072">131,072 (131k)</option></select></div>
        <div class="field"><label>Batch / Ubatch (-b / -ub)</label>
          <div style="display:flex;gap:6px"><input id="av-batch" type="number" placeholder="batch (2048)" min="128" max="8192" step="128" style="width:50%"><input id="av-ubatch" type="number" placeholder="ubatch (512)" min="64" max="4096" step="64" style="width:50%"></div></div>
        <div class="field"><label>KV cache K / V (-ctk / -ctv)</label>
          <div style="display:flex;gap:6px">
            <select id="av-ctk" style="width:50%"><option value="">default (q8_0)</option><option value="f16">f16</option><option value="q8_0">q8_0</option><option value="q5_1">q5_1</option><option value="q4_0">q4_0</option><option value="iq4_nl">iq4_nl</option></select>
            <select id="av-ctv" style="width:50%"><option value="">default</option><option value="f16">f16</option><option value="q8_0">q8_0</option><option value="q5_1">q5_1</option><option value="q4_0">q4_0</option><option value="iq4_nl">iq4_nl</option></select>
          </div></div>
        <div class="field"><label>MTP Draft (--draft-max / -ubd)</label>
          <div style="display:flex;gap:6px"><input id="av-draft" type="number" placeholder="draft_n (3)" min="0" max="16" style="width:50%"><input id="av-ubd" type="number" placeholder="ubd (512)" min="64" max="2048" step="64" style="width:50%"></div></div>
        <div class="field"><label>Parallel slots (-np)</label>
          <select id="av-np"><option value="0">Profile default (1)</option><option value="1">1 slot</option><option value="2">2 slots</option><option value="4">4 slots</option></select></div>
      </div>
      <div style="margin-top:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
          <span class="hint">Resolved Avalon command line</span>
          <button class="btn xs" id="avcopycmd">Copy command</button>
        </div>
        <div class="cmdbox" id="avcmdpreview"></div>
      </div>
      <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:10px">
        <button class="btn sm" id="avsavetune">Save tuning</button>
        <button class="btn sm primary" id="avapplystart">Save &amp; Restart Avalon</button>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="hd"><h2>Network access</h2>
      <label class="switch"><input type="checkbox" id="avalonlan"><span class="track"></span><span class="hint">Expose to LAN</span></label></div>
    <div id="avalonaccess" class="kvs"></div>
  </div>
  <div class="card"><div class="hd"><h2>Avalon log</h2></div><pre class="log" id="avalonlog" style="max-height:280px"></pre></div>
</section>

<!-- ==================== VITALS ==================== -->
<section id="t-vitals" class="hide">
  <div class="hd" style="margin-bottom:10px"><h2>System vitals</h2>
    <div class="meta"><span class="tag mut" id="vupdatetime">ready</span><button class="btn sm" id="vrefresh">Refresh</button></div></div>
  <div id="vitals" class="grid"></div>
</section>

<!-- ==================== CHAT ==================== -->
<section id="t-chat" class="hide">
  <div class="two">
    <div class="card side">
      <div class="hd"><h2>Chats</h2><button class="btn sm primary" id="newchat">+ New</button></div>
      <div class="search" style="margin-bottom:8px">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="csearch" type="search" placeholder="Search conversations…" aria-label="Search conversations">
      </div>
      <div id="chatlist" class="clist"></div>
    </div>
    <div class="card main">
      <div class="hd">
        <h2 id="chattitle">Conversation</h2>
        <div class="meta">
          <span class="tag" id="chatmodel">no model</span>
          <button class="btn xs" id="sysbtn" title="System prompt &amp; sampling parameters">Prompt &amp; Sampling</button>
          <button class="btn xs" id="chatexportmd" title="Export chat as Markdown">Export MD</button>
          <button class="btn xs" id="chatexportjson" title="Export chat as JSON">JSON</button>
        </div>
      </div>
      <div id="sysbox" class="card sub hide">
        <div class="field" style="margin-bottom:10px">
          <label>System Prompt</label>
          <textarea id="sysprompt" rows="2" style="width:100%"></textarea>
        </div>
        <div class="form-grid">
          <div class="field"><label>Temperature <span id="sp-temp-v">0.70</span></label>
            <input id="sp-temp" type="range" min="0" max="2" step="0.05" value="0.7"></div>
          <div class="field"><label>Top P <span id="sp-topp-v">0.95</span></label>
            <input id="sp-topp" type="range" min="0.05" max="1" step="0.05" value="0.95"></div>
          <div class="field"><label>Min P <span id="sp-minp-v">0.05</span></label>
            <input id="sp-minp" type="range" min="0" max="0.5" step="0.01" value="0.05"></div>
          <div class="field"><label>Repeat Penalty <span id="sp-rep-v">1.05</span></label>
            <input id="sp-rep" type="range" min="1" max="1.5" step="0.01" value="1.05"></div>
          <div class="field"><label>Max Tokens <span id="sp-max-v">default</span></label>
            <input id="sp-max" type="number" placeholder="0 = unlimited" min="0" max="65536" step="128" value="0"></div>
        </div>
      </div>
      <div id="chat" class="thread"></div>
      <div class="composer">
        <textarea id="prompt" rows="1" placeholder="Message… Enter to send, Shift+Enter for a new line" aria-label="Message"></textarea>
        <button class="btn danger" id="stopbtn" style="display:none">Stop</button>
        <button class="btn primary" id="sendbtn">Send</button>
      </div>
      <p class="hint" style="margin:8px 2px 0" id="chatnote"></p>
    </div>
  </div>
</section>

<!-- ==================== HF PULL ==================== -->
<section id="t-hf" class="hide">
  <div class="card">
    <div class="hd"><h2>Search Hugging Face GGUF</h2><span class="tag mut" id="hfinfo"></span></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <div class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="hfq" type="search" placeholder="Search GGUF repositories (e.g. Qwen3, Llama-3.3, Gemma-3)… press Enter" aria-label="Search Hugging Face"></div>
      <button class="btn primary" id="hfsearch">Search</button>
    </div>
    <div id="hfresults" style="margin-top:10px"></div>
  </div>
  <div class="card hide" id="hffilesbox">
    <div class="hd"><h2 id="hfrepo"></h2><div class="meta"><span class="tag mut" id="hftotal"></span>
      <button class="btn sm" id="hfnone">Clear</button></div></div>
    <div id="hffiles"></div>
    <button class="btn primary" id="hfpull" style="margin-top:10px">Pull selected (resumable)</button>
  </div>
  <div class="card"><div class="hd"><h2>Downloads queue</h2><span class="tag mut" id="dlinfo"></span></div>
    <div id="downloads"><span class="hint">No downloads queued. Partial <code>.part</code> files automatically resume where they left off.</span></div></div>
</section>

<!-- ==================== SETTINGS ==================== -->
<section id="t-settings" class="hide">
  <div class="card">
    <div class="hd"><h2>Panel &amp; Runtime Settings</h2><span class="tag mut" id="set-caps-badge"></span></div>
    <div class="form-grid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">
      <div class="field" style="grid-column:1 / -1">
        <label>Model directories (one absolute path per line; first existing dir receives HF downloads) <span id="set-disks"></span></label>
        <textarea id="set-dirs" rows="3"></textarea>
      </div>
      <div class="field" style="grid-column:1 / -1">
        <label>llama-server binary path <span id="set-bin-ver"></span></label>
        <input id="set-bin" type="text" placeholder="/usr/local/bin/llama-server">
        <div class="kvs" id="set-bin-flags" style="margin-top:4px"></div>
      </div>
      <div class="field">
        <label>Default Context Window (tokens)</label>
        <input id="set-ctx" type="number" min="512" max="262144" step="512">
      </div>
      <div class="field">
        <label>Default Server Port</label>
        <input id="set-port" type="number" min="1024" max="65535">
      </div>
      <div class="field">
        <label>Auto-unload when idle (minutes, 0 = disabled)</label>
        <input id="set-idle" type="number" min="0" max="1440">
      </div>
      <div class="field">
        <label>VRAM safety reserve (MB kept free for display/OS)</label>
        <input id="set-reserve" type="number" min="0" max="16384" step="128">
      </div>
      <div class="field" style="grid-column:1 / -1">
        <label>Hugging Face Token (optional, for gated repos or higher rate limits) <span id="set-hf-status"></span></label>
        <input id="set-hftok" type="password" placeholder="hf_… (leave blank to keep existing)">
      </div>
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:14px">
      <button class="btn" id="set-reload">Reload</button>
      <button class="btn primary" id="set-save">Save settings</button>
    </div>
  </div>
</section>
</main>

<!-- ==================== MODALS (<dialog>) ==================== -->
<dialog id="launchdlg" closedby="any">
  <div class="dlg-hd">
    <h2 id="ld-title">Tune &amp; Launch Model</h2>
    <button class="icon" data-close="launchdlg" aria-label="Close"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
  </div>
  <div class="dlg-bd">
    <div class="card sub" style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px">
        <span style="font-weight:650;font-size:12.5px">Auto-Config Optimizer</span>
        <div class="segs" id="ld-opt-btns">
          <button type="button" data-opt="balanced">Balanced</button>
          <button type="button" data-opt="speed">Max Speed</button>
          <button type="button" data-opt="max_ctx">Max Context</button>
          <button type="button" data-opt="low_vram">Low VRAM</button>
        </div>
      </div>
      <div id="ld-opt-notes" class="kvs"></div>
    </div>

    <div class="card sub" style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
        <span style="font-weight:650;font-size:12.5px" id="ld-fit-label">VRAM / Memory Fit</span>
        <span class="tag" id="ld-fit-badge">calculating…</span>
      </div>
      <div class="meter" id="ld-fit-meter" style="height:9px;margin:8px 0 6px"></div>
      <div class="kvs" id="ld-fit-breakdown"></div>
    </div>

    <div class="form-grid">
      <div class="field">
        <label>Base Preset</label>
        <select id="ld-preset">
          <option value="auto">auto preset</option>
          <option value="dense-gpu">dense · full offload</option>
          <option value="moe-split">MoE · split experts</option>
          <option value="vision">vision + mmproj</option>
          <option value="igpu-full">iGPU full (780M)</option>
          <option value="igpu-vision">iGPU vision (780M)</option>
        </select>
      </div>
      <div class="field">
        <label>Context Window (-c) <span id="ld-ctx-val">16384</span></label>
        <input id="ld-ctx" type="range" min="2048" max="131072" step="2048" value="16384">
        <div class="kvs" style="margin-top:2px">
          <button type="button" class="btn xs" data-ctx="4096">4k</button>
          <button type="button" class="btn xs" data-ctx="8192">8k</button>
          <button type="button" class="btn xs" data-ctx="16384">16k</button>
          <button type="button" class="btn xs" data-ctx="32768">32k</button>
          <button type="button" class="btn xs" data-ctx="65536">65k</button>
          <button type="button" class="btn xs" data-ctx="131072">131k</button>
        </div>
      </div>
      <div class="field">
        <label>GPU Layers (-ngl) <span id="ld-ngl-val">99</span></label>
        <input id="ld-ngl" type="range" min="0" max="99" step="1" value="99">
      </div>
      <div class="field">
        <label>CPU Threads (-t, 0 = auto)</label>
        <input id="ld-threads" type="number" min="0" max="128" value="0">
      </div>
      <div class="field">
        <label>Flash Attention (-fa)</label>
        <select id="ld-fa">
          <option value="auto">auto (recommended)</option>
          <option value="on">on</option>
          <option value="off">off</option>
        </select>
      </div>
      <div class="field">
        <label>KV Cache K / V (-ctk / -ctv)</label>
        <div style="display:flex;gap:6px">
          <select id="ld-ctk" style="width:50%">
            <option value="f16">K: f16</option><option value="bf16">K: bf16</option>
            <option value="q8_0">K: q8_0</option><option value="q5_1">K: q5_1</option>
            <option value="q4_0">K: q4_0</option><option value="iq4_nl">K: iq4_nl</option>
          </select>
          <select id="ld-ctv" style="width:50%">
            <option value="f16">V: f16</option><option value="bf16">V: bf16</option>
            <option value="q8_0">V: q8_0</option><option value="q5_1">V: q5_1</option>
            <option value="q4_0">V: q4_0</option><option value="iq4_nl">V: iq4_nl</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label>Parallel Slots (--parallel)</label>
        <select id="ld-parallel">
          <option value="1">1 slot</option><option value="2">2 slots</option>
          <option value="4">4 slots</option><option value="8">8 slots</option>
        </select>
      </div>
      <div class="field">
        <label>Batch / Ubatch (-b / -ub)</label>
        <div style="display:flex;gap:6px">
          <input id="ld-batch" type="number" min="128" max="8192" step="128" value="2048" style="width:50%">
          <input id="ld-ubatch" type="number" min="64" max="4096" step="64" value="512" style="width:50%">
        </div>
      </div>
      <div class="field" id="ld-ncmoe-wrap">
        <label>MoE CPU Expert Layers (--n-cpu-moe)</label>
        <input id="ld-ncmoe" type="number" min="0" max="128" value="0">
      </div>
      <div class="field">
        <label>Memory Policy</label>
        <div style="display:flex;gap:12px;padding-top:5px">
          <label class="switch"><input type="checkbox" id="ld-mlock"><span class="track"></span><span class="hint">mlock</span></label>
          <label class="switch"><input type="checkbox" id="ld-nommap"><span class="track"></span><span class="hint">no-mmap</span></label>
        </div>
      </div>
      <div class="field" style="grid-column:1 / -1">
        <label>Extra CLI flags (appended to llama-server command)</label>
        <input id="ld-extra" type="text" placeholder="e.g. --temp 0.6 --cache-reuse 256">
      </div>
    </div>

    <div style="margin-top:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span class="hint">Live command preview</span>
        <button type="button" class="btn xs" id="ld-copycmd">Copy command</button>
      </div>
      <div class="cmdbox" id="ld-cmd"></div>
    </div>
  </div>
  <div class="dlg-ft">
    <button type="button" class="btn" data-close="launchdlg">Cancel</button>
    <button type="button" class="btn primary" id="ld-launch">Save &amp; Launch</button>
  </div>
</dialog>

<dialog id="cmdpal" closedby="any" style="max-width:min(620px,95vw)">
  <div class="dlg-bd" style="padding:12px">
    <div class="search">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
      <input id="pal-q" type="search" placeholder="Type a command, model name, or action…" aria-label="Command palette search">
    </div>
    <div id="pal-list" class="pal-list"></div>
  </div>
</dialog>

<dialog id="confirmdlg" closedby="any" style="max-width:min(460px,94vw)">
  <div class="dlg-hd">
    <h2 id="cf-title">Confirm</h2>
    <button class="icon" data-close="confirmdlg" aria-label="Close"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
  </div>
  <div class="dlg-bd"><p id="cf-msg" style="margin:0;color:var(--fg-2);line-height:1.6"></p></div>
  <div class="dlg-ft">
    <button type="button" class="btn" id="cf-no" autofocus>Cancel</button>
    <button type="button" class="btn primary" id="cf-yes">Confirm</button>
  </div>
</dialog>

<dialog id="agentsdlg" closedby="any" style="max-width:min(600px,94vw)">
  <div class="dlg-hd">
    <h2>Register into coding agents</h2>
    <button class="icon" data-close="agentsdlg" aria-label="Close"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
  </div>
  <div class="dlg-bd" id="ag-body"></div>
  <div class="dlg-ft">
    <button type="button" class="btn" data-close="agentsdlg">Close</button>
    <button type="button" class="btn primary" id="ag-apply">Apply</button>
  </div>
</dialog>

<footer>
  <span><kbd>⌘K</kbd> / <kbd>Ctrl+K</kbd> command palette</span>
  <span><kbd>1</kbd>–<kbd>7</kbd> switch tabs</span>
  <span><kbd>/</kbd> filter</span>
  <span><kbd>Esc</kbd> close</span>
  <span id="footnote"></span>
</footer>

<div class="toasts" id="toasts" aria-live="polite"></div>

<script>
/* ============================== helpers ============================== */
const $=(s,r=document)=>r.querySelector(s);
const $$=(s,r=document)=>Array.from(r.querySelectorAll(s));
const esc=v=>String(v==null?'':v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
const n0=v=>Number(v||0).toLocaleString();
const gib=b=>(Number(b||0)/1073741824).toFixed(2)+' GB';
const ago=t=>{const s=Date.now()/1000-(t||0);if(s<60)return'just now';if(s<3600)return((s/60)|0)+'m ago';
  if(s<86400)return((s/3600)|0)+'h ago';return((s/86400)|0)+'d ago'};
const fmtUp=s=>{s=s|0;const h=(s/3600)|0,m=((s%3600)/60)|0;return h?h+'h '+m+'m':m?m+'m '+(s%60)+'s':s+'s'};

async function api(url,opt){
  const r=await fetch(url,opt);
  if(!r.ok){let d=await r.text();try{const j=JSON.parse(d);d=(j&&j.detail)||d}catch(e){}
    throw new Error(String(d).slice(0,400))}
  return r.json();
}
const post=(u,b)=>api(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});

async function copy(text,label){
  try{await navigator.clipboard.writeText(text)}
  catch(e){
    const ta=document.createElement('textarea');ta.value=text;ta.style.position='fixed';ta.style.opacity='0';
    document.body.appendChild(ta);ta.select();
    let ok=false;try{ok=document.execCommand('copy')}catch(_){}
    ta.remove();if(!ok)return toast('Clipboard blocked by the browser — select and copy manually','err');
  }
  toast((label||'Copied')+' · '+String(text).slice(0,60),'ok',2400);
}

/* ============================== toasts & sound ============================== */
function toast(msg,kind,ms){
  kind=kind||'info';
  const t=document.createElement('div');t.className='toast '+kind;
  t.innerHTML='<span class="ti"></span><div>'+esc(msg)+'</div>';
  $('#toasts').appendChild(t);
  const kill=()=>{t.classList.add('out');setTimeout(()=>t.remove(),180)};
  t.addEventListener('click',kill);
  setTimeout(kill,ms||(kind==='err'?9000:4000));
}

function playReadyChime(){
  try{
    const Ctx=window.AudioContext||window.webkitAudioContext;
    if(!Ctx)return;
    const ctx=new Ctx();
    const notes=[523.25,659.25,783.99];
    notes.forEach((f,i)=>{
      const o=ctx.createOscillator(),g=ctx.createGain();
      o.type='sine';o.frequency.value=f;
      g.gain.setValueAtTime(0.0001,ctx.currentTime+i*0.09);
      g.gain.exponentialRampToValueAtTime(0.08,ctx.currentTime+i*0.09+0.02);
      g.gain.exponentialRampToValueAtTime(0.0001,ctx.currentTime+i*0.09+0.28);
      o.connect(g);g.connect(ctx.destination);
      o.start(ctx.currentTime+i*0.09);o.stop(ctx.currentTime+i*0.09+0.3);
    });
  }catch(e){}
}

/* ============================== native dialogs & light-dismiss ============================== */
$$('dialog').forEach(dlg=>{
  dlg.addEventListener('close',()=>{dlg.returnValue='';});
  if(!('closedBy' in HTMLDialogElement.prototype)){
    dlg.addEventListener('click',e=>{
      if(e.target!==dlg)return;
      const r=dlg.getBoundingClientRect();
      const inside=r.top<=e.clientY&&e.clientY<=r.top+r.height&&r.left<=e.clientX&&e.clientX<=r.left+r.width;
      if(!inside)dlg.close();
    });
  }
});
document.addEventListener('click',e=>{
  const c=e.target.closest('[data-close]');
  if(c){const d=document.getElementById(c.dataset.close);if(d&&d.open)d.close();}
});

let cfRes=null;
const cfDlg=$('#confirmdlg');
cfDlg.addEventListener('close',()=>{if(cfRes){const r=cfRes;cfRes=null;r(false);}});
$('#cf-no').addEventListener('click',()=>{if(cfRes){const r=cfRes;cfRes=null;cfDlg.close();r(false);}});
$('#cf-yes').addEventListener('click',()=>{if(cfRes){const r=cfRes;cfRes=null;cfDlg.close();r(true);}});
function confirmDlg(title,msg,okLabel,danger){
  return new Promise(res=>{
    if(cfDlg.open)cfDlg.close();
    $('#cf-title').textContent=title||'Confirm';
    $('#cf-msg').innerHTML=msg||'';
    const yes=$('#cf-yes');
    yes.textContent=okLabel||'Confirm';
    yes.className='btn '+(danger?'danger':'primary');
    cfRes=res;
    cfDlg.showModal();
    $('#cf-no').focus();
  });
}

/* ============================== theme ============================== */
const SUN='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/></svg>';
const MOON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 14a8 8 0 01-9.5-9.5A8 8 0 1020 14z"/></svg>';
const themeMeta=$('meta[name="color-scheme"]');
function applyTheme(t){
  document.documentElement.dataset.theme=t;
  if(themeMeta)themeMeta.content=t;
  $('#theme').innerHTML=t==='dark'?SUN:MOON;
  $('#theme').title=t==='dark'?'Switch to light theme':'Switch to dark theme';
}
applyTheme(document.documentElement.dataset.theme||'dark');
$('#theme').addEventListener('click',()=>{
  const next=document.documentElement.dataset.theme==='dark'?'light':'dark';
  applyTheme(next);
  const sys=matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';
  if(next===sys)localStorage.removeItem('lfm.theme');
  else localStorage.setItem('lfm.theme',next);
});
$('#logout').addEventListener('click',async()=>{
  if(!await confirmDlg('Sign out','End this panel session?','Sign out',true))return;
  try{await post('/api/logout')}catch(e){}
  location.href='/login';
});

/* ============================== tabs ============================== */
const TABS=['models','server','avalon','vitals','chat','hf','settings'];
let tab='models';
function setTab(t){
  if(TABS.indexOf(t)<0)t='models';
  tab=t;localStorage.setItem('lfm.tab',t);
  $$('.nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
  $$('main section').forEach(s=>s.classList.toggle('hide',s.id!=='t-'+t));
  if(t==='vitals')refreshVitals();
  if(t==='server'){refreshLog();pollTelemetry();loadBenchmarks();}
  if(t==='avalon'){loadAvalon();loadAvalonAccess();}
  if(t==='settings')loadSettings();
}
$$('.nav button').forEach(b=>b.addEventListener('click',()=>setTab(b.dataset.t)));

/* ============================== models & launch modal ============================== */
let MODELS=[],MMP=[],ACTIVE=null,AV=null,FILTER='all',SORT='smart',CAPS={},DEF_CTX=16384;
const STAR_ON='<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1.8"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';
const STAR_OFF='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';

async function loadModels(){
  const d=await api('/api/models');
  MODELS=d.models||[];MMP=d.mmproj||[];CAPS=d.caps||{};DEF_CTX=d.default_ctx||16384;
  renderModels();
}
function fitMeterHtml(fit){
  if(!fit)return '<div class="fitbox"><span class="hint">no estimate</span></div>';
  const tot=Math.max(1,fit.usable_mb||fit.budget_mb||1);
  const wPct=Math.min(100,Math.round(100*(fit.weights_vram_mb||0)/tot));
  const kvPct=Math.min(100-wPct,Math.round(100*(fit.kv_vram_mb||0)/tot));
  const cPct=Math.min(100-wPct-kvPct,Math.round(100*(fit.compute_mb||0)/tot));
  const tone=fit.status==='fit'?'ok':fit.status==='tight'?'warn':'err';
  const lbl=fit.status==='fit'?'Fits':fit.status==='tight'?'Tight':'Exceeds';
  return '<div class="fitbox" title="Weights: '+n0(fit.weights_vram_mb)+' MB · KV: '+n0(fit.kv_vram_mb)+' MB · Compute: '+n0(fit.compute_mb)+' MB / Budget: '+n0(fit.budget_mb)+' MB ('+esc(fit.budget_source)+')">'
    +'<div class="fit-top"><span>'+(fit.vram_used_mb/1024).toFixed(1)+' / '+((fit.usable_mb||fit.budget_mb)/1024).toFixed(1)+' GB</span>'
    +'<span class="tag '+tone+'" style="padding:0 6px;font-size:10px">'+lbl+' '+fit.pct+'%</span></div>'
    +'<div class="meter '+(tone==='ok'?'':tone)+'" style="margin:3px 0"><i style="width:'+wPct+'%"></i><i class="kv" style="width:'+kvPct+'%"></i><i class="cmp" style="width:'+cPct+'%"></i></div>'
    +'<div class="fit-sub">W '+(fit.weights_vram_mb/1024).toFixed(1)+'G · KV '+n0(fit.kv_vram_mb)+'M'+(fit.ram_offload_mb>0?' · RAM '+(fit.ram_offload_mb/1024).toFixed(1)+'G':'')+'</div></div>';
}
function modelRow(m,i){
  const running=ACTIVE&&ACTIVE.model===m.path;
  const savedP=(m.saved_profile&&m.saved_profile.preset)||'auto';
  const tags=['<span class="tag">'+esc(m.arch)+'</span>','<span class="tag acc">'+esc(m.quant)+'</span>'];
  if(m.params)tags.push('<span class="tag">'+esc(m.params)+'</span>');
  if(m.ctx_len)tags.push('<span class="tag mut">'+n0(m.ctx_len)+' ctx</span>');
  tags.push('<span class="tag mut">'+esc(m.size_h)+'</span>');
  if(m.moe)tags.push('<span class="tag warn">MoE'+(m.experts?' '+m.experts+'x':'')+'</span>');
  if(m.vl)tags.push('<span class="tag acc">VL</span>');
  if(m.imatrix)tags.push('<span class="tag mut">imatrix</span>');
  if(m.bench&&m.bench.gen_tps)tags.push('<span class="tag ok" title="Last benchmark speed">'+m.bench.gen_tps+' tok/s</span>');
  if(m.saved_profile)tags.push('<span class="tag mut" title="Custom launch profile saved">tuned</span>');
  if(running)tags.push('<span class="tag ok">loaded</span>');
  return '<div class="mrow'+(running?' run':'')+'" data-i="'+i+'" data-path="'+esc(m.path)+'" data-name="'+esc(m.name)+'" data-size="'+esc(m.size_h)+'">'
    +'<button class="pinbtn'+(m.pinned?' on':'')+'" data-act="pin" title="'+(m.pinned?'Unpin model':'Pin model to top')+'" aria-label="Pin '+esc(m.name)+'">'+(m.pinned?STAR_ON:STAR_OFF)+'</button>'
    +'<div style="min-width:0"><div class="nm" title="'+esc(m.path)+'"><span>'+esc(m.name)+'</span></div>'
    +'<div class="pth">'+esc(m.path.replace(/^\/[^/]*\//,'/…/'))+'</div>'
    +'<div class="tags">'+tags.join('')+'</div></div>'
    +fitMeterHtml(m.fit)
    +'<div class="acts"><select data-preset aria-label="Preset for '+esc(m.name)+'">'
    +['auto|auto preset','dense-gpu|dense · full offload','moe-split|MoE · split experts','vision|vision + mmproj',
      'igpu-full|iGPU full (780M)','igpu-vision|iGPU vision (780M)']
      .map(o=>{const p=o.split('|');return '<option value="'+p[0]+'"'+(p[0]===savedP?' selected':'')+'>'+p[1]+'</option>'}).join('')
    +'</select>'
    +'<button class="btn primary sm" data-act="start">'+(running?'Reload':'Load')+'</button>'
    +'<button class="btn sm acc2" data-act="tune" title="Open Launch &amp; Config Optimizer">Tune &amp; Launch…</button>'
    +'<button class="btn xs" data-act="copy" title="Copy file path">Path</button>'
    +'<button class="btn xs danger" data-act="del" title="Delete model file">Del</button></div></div>';
}
function renderModels(){
  const q=$('#msearch').value.trim().toLowerCase();
  let list=MODELS.filter(m=>{
    if(FILTER==='pin'&&!m.pinned)return false;
    if(FILTER==='recent'&&!m.last_used)return false;
    if(FILTER==='fit'&&(!m.fit||m.fit.status==='overflow'))return false;
    if(FILTER==='moe'&&!m.moe)return false;
    if(FILTER==='vl'&&!m.vl)return false;
    if(FILTER==='run'&&!(ACTIVE&&ACTIVE.model===m.path))return false;
    if(q&&!(m.name+' '+m.arch+' '+m.quant+' '+(m.params||'')).toLowerCase().includes(q))return false;
    return true;
  });
  const by={
    smart:(a,b)=>(Number(!!b.pinned)-Number(!!a.pinned))||((b.last_used||0)-(a.last_used||0))||a.name.localeCompare(b.name),
    name:(a,b)=>a.name.localeCompare(b.name),
    size:(a,b)=>b.size-a.size,
    small:(a,b)=>a.size-b.size,
    speed:(a,b)=>((b.bench&&b.bench.gen_tps)||0)-((a.bench&&a.bench.gen_tps)||0),
    arch:(a,b)=>a.arch.localeCompare(b.arch)||a.name.localeCompare(b.name)
  }[SORT]||((a,b)=>a.name.localeCompare(b.name));
  list=list.slice().sort(by);
  const bytes=MODELS.reduce((s,m)=>s+m.size,0);
  $('#libinfo').textContent=MODELS.length+' models · '+gib(bytes)+(MMP.length?' · '+MMP.length+' mmproj':'');
  if(MODELS[0]&&MODELS[0].fit){
    const f=MODELS[0].fit;
    $('#vramchip').textContent=f.budget_source+': '+((f.usable_mb||f.budget_mb)/1024).toFixed(1)+' GB usable';
  }
  if(!list.length){
    $('#modellist').innerHTML='<div class="empty"><b>'+(MODELS.length?'No model matches this filter':'No GGUF models found')+'</b>'
      +(MODELS.length?'Clear the filter or press Rescan.':'Add files under a configured model dir in Settings, or use the HF Pull tab.')+'</div>';
    return;
  }
  $('#modellist').innerHTML=list.map(m=>modelRow(m,MODELS.indexOf(m))).join('');
}
$('#msearch').addEventListener('input',renderModels);
$('#msort').addEventListener('change',e=>{SORT=e.target.value;renderModels()});
$$('#mfilter button').forEach(b=>b.addEventListener('click',()=>{
  $$('#mfilter button').forEach(x=>x.classList.toggle('on',x===b));FILTER=b.dataset.f;renderModels()}));
$('#rescan').addEventListener('click',async()=>{
  const b=$('#rescan');b.disabled=true;
  try{await loadModels();toast('Library rescanned — '+MODELS.length+' models','ok',2200)}
  catch(e){toast('Scan failed: '+e.message,'err')}
  b.disabled=false;
});
$('#modellist').addEventListener('click',async e=>{
  const b=e.target.closest('[data-act]');if(!b)return;
  const row=b.closest('.mrow');const path=row.dataset.path;
  const m=MODELS.find(x=>x.path===path);
  if(b.dataset.act==='pin'){
    try{const r=await post('/api/models/pin',{path,pinned:!(m&&m.pinned)});
      if(m)m.pinned=r.pinned;renderModels();}catch(err){toast('Pin failed: '+err.message,'err')}
    return;
  }
  if(b.dataset.act==='copy')return copy(path,'Path');
  if(b.dataset.act==='tune'){if(m)openLaunchModal(m);return;}
  if(b.dataset.act==='del'){
    if(!await confirmDlg('Delete model','<b>'+esc(row.dataset.name)+'</b> ('+esc(row.dataset.size)+')<br><code>'
      +esc(path)+'</code><br><br>This permanently removes the file from disk.','Delete',true))return;
    b.disabled=true;
    try{const d=await post('/api/models/delete',{path});
      toast('Deleted · '+d.freed_h+' freed','ok');await loadModels()}
    catch(err){toast('Delete failed: '+err.message,'err');b.disabled=false}
    return;
  }
  const sel=row.querySelector('[data-preset]');
  const saved=(m&&m.saved_profile)||{};
  const body=Object.assign({},saved,{model:path,preset:sel?sel.value:(saved.preset||'auto')});
  b.disabled=true;const old=b.textContent;b.textContent='Starting…';
  try{await post('/api/server/start',body);
    toast('Loading '+row.dataset.name+'…','ok');await pollStatus(true);await loadModels()}
  catch(err){toast('Start failed: '+err.message,'err',12000)}
  b.disabled=false;b.textContent=old;
});

/* ---------- Launch & Config Optimizer Modal ---------- */
let LD_MODEL=null,LD_TIMER=null;
function getLaunchForm(){
  return {
    model:LD_MODEL?LD_MODEL.path:'',
    preset:$('#ld-preset').value,
    ctx:Number($('#ld-ctx').value)||16384,
    ngl:Number($('#ld-ngl').value),
    threads:Number($('#ld-threads').value)||0,
    fa:$('#ld-fa').value,
    ctk:$('#ld-ctk').value,
    ctv:$('#ld-ctv').value,
    parallel:Number($('#ld-parallel').value)||1,
    batch:Number($('#ld-batch').value)||2048,
    ubatch:Number($('#ld-ubatch').value)||512,
    ncmoe:Number($('#ld-ncmoe').value)||0,
    mlock:$('#ld-mlock').checked,
    no_mmap:$('#ld-nommap').checked,
    extra_args:$('#ld-extra').value.trim(),
    save_profile:true
  };
}
function applyLaunchForm(p){
  const maxBlocks=(LD_MODEL&&LD_MODEL.blocks)||99;
  $('#ld-ngl').max=Math.max(99,maxBlocks);
  $('#ld-preset').value=p.preset||'auto';
  const ctx=p.ctx||DEF_CTX||16384;
  $('#ld-ctx').value=ctx;$('#ld-ctx-val').textContent=n0(ctx);
  const ngl=p.ngl!=null&&p.ngl>=0?p.ngl:99;
  $('#ld-ngl').value=ngl;$('#ld-ngl-val').textContent=ngl>=99?'99 (all)':String(ngl);
  $('#ld-threads').value=p.threads||0;
  $('#ld-fa').value=p.fa||'auto';
  $('#ld-ctk').value=p.ctk||'f16';
  $('#ld-ctv').value=p.ctv||'f16';
  $('#ld-parallel').value=String(p.parallel||1);
  $('#ld-batch').value=p.batch||2048;
  $('#ld-ubatch').value=p.ubatch||512;
  $('#ld-ncmoe').value=p.ncmoe||0;
  $('#ld-mlock').checked=!!p.mlock;
  $('#ld-nommap').checked=!!p.no_mmap;
  $('#ld-extra').value=p.extra_args||'';
  $('#ld-ncmoe-wrap').classList.toggle('hide',!(LD_MODEL&&LD_MODEL.moe));
}
async function updateLaunchPreview(){
  if(!LD_MODEL)return;
  const f=getLaunchForm();
  $('#ld-ctx-val').textContent=n0(f.ctx);
  $('#ld-ngl-val').textContent=f.ngl>=99?'99 (all)':String(f.ngl);
  try{
    const r=await post('/api/models/fit',f);
    const fit=r.fit||{};
    const tone=fit.status==='fit'?'ok':fit.status==='tight'?'warn':'err';
    const b=$('#ld-fit-badge');
    b.className='tag '+tone;
    b.textContent=fit.label+' · '+n0(fit.vram_used_mb)+' / '+n0(fit.usable_mb||fit.budget_mb)+' MB ('+fit.pct+'%)';
    const tot=Math.max(1,fit.usable_mb||fit.budget_mb||1);
    const wPct=Math.min(100,Math.round(100*(fit.weights_vram_mb||0)/tot));
    const kvPct=Math.min(100-wPct,Math.round(100*(fit.kv_vram_mb||0)/tot));
    const cPct=Math.min(100-wPct-kvPct,Math.round(100*(fit.compute_mb||0)/tot));
    const m=$('#ld-fit-meter');
    m.className='meter '+(tone==='ok'?'':tone);
    m.innerHTML='<i style="width:'+wPct+'%"></i><i class="kv" style="width:'+kvPct+'%"></i><i class="cmp" style="width:'+cPct+'%"></i>';
    $('#ld-fit-breakdown').innerHTML=[
      '<span class="tag">Weights VRAM: <b>'+n0(fit.weights_vram_mb)+' MB</b></span>',
      '<span class="tag warn">KV Cache ('+esc(f.ctk)+'/'+esc(f.ctv)+'): <b>'+n0(fit.kv_vram_mb)+' MB</b></span>',
      '<span class="tag ok">Compute: <b>'+n0(fit.compute_mb)+' MB</b></span>',
      fit.ram_offload_mb>0?'<span class="tag mut">CPU/RAM offload: <b>'+n0(fit.ram_offload_mb)+' MB</b></span>':''
    ].filter(Boolean).join('');
    $('#ld-cmd').textContent=r.cmd||'';
  }catch(e){}
}
function schedLaunchPreview(){
  if(LD_TIMER)clearTimeout(LD_TIMER);
  LD_TIMER=setTimeout(updateLaunchPreview,90);
}
function openLaunchModal(m){
  LD_MODEL=m;
  $('#ld-title').textContent='Tune & Launch — '+m.name;
  $('#ld-opt-notes').innerHTML='<span class="hint">Choose an optimization target above to auto-calculate flags for this '+esc(m.arch)+' ('+esc(m.size_h)+') model.</span>';
  $$('#ld-opt-btns button').forEach(b=>b.classList.remove('on'));
  const base=m.saved_profile||{preset:'auto',ctx:DEF_CTX,ngl:99,fa:'auto',ctk:'f16',ctv:'f16',parallel:1,batch:2048,ubatch:512};
  applyLaunchForm(base);
  $('#launchdlg').showModal();
  updateLaunchPreview();
}
$$('#launchdlg input, #launchdlg select').forEach(el=>el.addEventListener('input',schedLaunchPreview));
$$('#launchdlg [data-ctx]').forEach(b=>b.addEventListener('click',()=>{
  $('#ld-ctx').value=b.dataset.ctx;schedLaunchPreview();
}));
$$('#ld-opt-btns button').forEach(b=>b.addEventListener('click',async()=>{
  if(!LD_MODEL)return;
  $$('#ld-opt-btns button').forEach(x=>x.classList.toggle('on',x===b));
  try{
    const r=await post('/api/models/optimize',{path:LD_MODEL.path,target:b.dataset.opt});
    applyLaunchForm(r.recommended||{});
    $('#ld-opt-notes').innerHTML=(r.notes||[]).map(n=>'<span class="tag acc">'+esc(n)+'</span>').join('');
    $('#ld-cmd').textContent=r.cmd||'';
    updateLaunchPreview();
  }catch(e){toast('Optimize failed: '+e.message,'err')}
}));
$('#ld-copycmd').addEventListener('click',()=>copy($('#ld-cmd').textContent,'CLI command'));
$('#ld-launch').addEventListener('click',async()=>{
  if(!LD_MODEL)return;
  const btn=$('#ld-launch');btn.disabled=true;btn.textContent='Launching…';
  try{
    await post('/api/server/start',getLaunchForm());
    $('#launchdlg').close();
    toast('Launching '+LD_MODEL.name+'…','ok');
    await pollStatus(true);await loadModels();
  }catch(e){toast('Launch failed: '+e.message,'err',10000)}
  btn.disabled=false;btn.textContent='Save & Launch';
});
$('#srvtune').addEventListener('click',()=>{
  const m=ACTIVE&&MODELS.find(x=>x.path===ACTIVE.model);
  if(m)openLaunchModal(m);
  else if(MODELS[0])openLaunchModal(MODELS[0]);
  else toast('No models in library','warn');
});

/* ============================== server status, telemetry & bench ============================== */
let lastRunModel=null,wasLoading=false,TELE=null;
function chatTarget(){
  const ap=(AV&&AV.port)||8086;
  if(ACTIVE&&ACTIVE.ready&&ACTIVE.port!==ap)return{alias:ACTIVE.alias,port:ACTIVE.port,src:'panel'};
  if(AV&&AV.running&&AV.state==='ready')return{alias:AV.model||'qwen3.8-flash-next',port:ap,src:'avalon'};
  if(ACTIVE&&ACTIVE.ready&&ACTIVE.port===ap)return{alias:'qwen3.8-flash-next',port:ap,src:'avalon'};
  return null;
}
function drawHeader(){
  const p=$('#hdrstatus'),t=$('#hdrtext');
  const ct=chatTarget();
  if(ct){
    p.className='pill ok';
    t.textContent=ct.alias+' · :'+ct.port+(ct.src==='avalon'?' · avalon':'');
    document.title='● '+ct.alias+' — LFM Panel';
  }else if(ACTIVE){
    p.className='pill warn';
    t.textContent=ACTIVE.alias+' · loading…';
    document.title='◌ Loading '+ACTIVE.alias+' — LFM Panel';
  }else if(AV&&AV.running){
    p.className='pill warn';
    t.textContent='avalon · loading…';
    document.title='◌ Loading Avalon — LFM Panel';
  }else{
    p.className='pill';
    t.textContent='no server running';
    document.title='LFM Panel — Local Inference Console';
  }
  const idleChip=$('#hdridle'),srvIdle=$('#srvidlechip');
  if(ACTIVE&&ACTIVE.ready&&ACTIVE.idle_unload_min>0&&ACTIVE.idle_remaining_s!=null){
    const txt='idle unload '+fmtUp(ACTIVE.idle_remaining_s);
    idleChip.textContent=txt;idleChip.classList.remove('hide');
    srvIdle.textContent=txt;srvIdle.classList.remove('hide');
  }else{
    idleChip.classList.add('hide');srvIdle.classList.add('hide');
  }
}
function pollStatus(force){
  return api('/api/server/status').then(st=>{
    const nowLoading=!!(st.running&&!st.ready);
    if(wasLoading&&st.running&&st.ready){
      playReadyChime();
      toast('Model ready: '+st.alias+' on :'+st.port,'ok',4500);
    }
    wasLoading=nowLoading;
    ACTIVE=st.running?st:null;
    const chip=$('#srvchip');
    chip.textContent=st.running?(st.ready?'running · ready':'running · loading'):'stopped';
    chip.className='tag '+(st.running?(st.ready?'ok':'warn'):'mut');
    if(st.running){
      $('#srvdetail').innerHTML=[
        ['alias',st.alias],['pid',st.pid],['port',':'+st.port],['preset',st.preset],
        ['uptime',fmtUp(st.uptime)],st.adopted?['source','adopted']:null]
        .filter(Boolean).map(kv=>'<span class="tag">'+esc(kv[0])+' <b>'+esc(kv[1])+'</b></span>').join('')
        +'<button class="btn xs" data-copy="http://127.0.0.1:'+st.port+'/v1">Copy base URL</button>';
      if(st.cmd){$('#srvcmdwrap').classList.remove('hide');$('#srvcmd').textContent=st.cmd;}
      else $('#srvcmdwrap').classList.add('hide');
      $('#srvstop').disabled=false;$('#agentsbtn').disabled=false;
    }else{
      $('#srvdetail').innerHTML='<span class="hint">No llama-server running. Pick a model in the Models tab or press ⌘K to launch one.</span>';
      $('#srvcmdwrap').classList.add('hide');
      $('#srvstop').disabled=true;$('#agentsbtn').disabled=true;
    }
    const now=ACTIVE?ACTIVE.model:null;
    if(force||now!==lastRunModel){lastRunModel=now;renderModels()}
    drawHeader();drawChatTarget();
  }).catch(()=>{});
}
$('#srvdetail').addEventListener('click',e=>{const b=e.target.closest('[data-copy]');if(b)copy(b.dataset.copy,'Base URL')});
$('#srvcopycmd').addEventListener('click',()=>copy($('#srvcmd').textContent,'Launch command'));
$('#srvstop').addEventListener('click',async()=>{
  if(!await confirmDlg('Stop server','Stop the loaded model? Running requests will be interrupted.','Stop',true))return;
  $('#srvstop').disabled=true;
  try{await post('/api/server/stop');toast('Server stopped','ok')}catch(e){toast('Stop failed: '+e.message,'err')}
  pollStatus(true);$('#srvstop').disabled=false;
});

async function pollTelemetry(){
  try{
    const t=await api('/api/server/telemetry');TELE=t;
    const hdrTps=$('#hdrtps'),hdrKv=$('#hdrkv');
    if(t.running){
      if(t.gen_tps){hdrTps.textContent=t.gen_tps+' tok/s';hdrTps.classList.remove('hide');}
      else hdrTps.classList.add('hide');
      if(t.kv_max){hdrKv.textContent='KV '+t.kv_pct+'%';hdrKv.classList.remove('hide');}
      else hdrKv.classList.add('hide');
    }else{
      hdrTps.classList.add('hide');hdrKv.classList.add('hide');
    }
    $('#t-gen-tps').textContent=t.gen_tps!=null?t.gen_tps:'—';
    $('#t-gen-src').textContent=t.slots_busy>0?'generating':(t.running?'ready':'offline');
    $('#t-gen-src').className='tag '+(t.slots_busy>0?'ok':(t.running?'acc':'mut'));
    $('#t-gen-sub').textContent=t.tokens_predicted_total?n0(t.tokens_predicted_total)+' total generated tokens':'Waiting for requests';
    $('#t-pp-tps').textContent=t.prompt_tps!=null?t.prompt_tps:'—';
    $('#t-kv-pct').textContent=(t.kv_pct||0)+'%';
    $('#t-kv-used').textContent=n0(t.kv_used||0);
    $('#t-kv-max').textContent=n0(t.kv_max||0);
    const kvm=$('#t-kv-meter');
    kvm.className='meter '+lvlCls(t.kv_pct||0);
    kvm.innerHTML='<i style="width:'+Math.min(100,t.kv_pct||0)+'%"></i>';
    $('#t-slot-badge').textContent=(t.slots_busy||0)+' / '+(t.slots_total||0);
    $('#t-slot-busy').textContent=t.slots_busy||0;
    $('#t-slot-pills').innerHTML=(t.slots||[]).map(s=>
      '<span class="tag '+(s.busy?'ok':'mut')+'">#'+s.id+' '+(s.busy?'busy':'idle')+(s.n_past?' ('+n0(s.n_past)+' tok)':'')+'</span>'
    ).join('')||'<span class="hint">no active slots</span>';
  }catch(e){}
}

async function loadBenchmarks(){
  try{
    const list=await api('/api/benchmarks');
    if(!list.length){
      $('#benchtable').innerHTML='<div class="empty"><b>No benchmark runs recorded yet</b>Start any model (or Avalon) and click <b>Run benchmark</b> above to measure prompt eval tok/s, generation tok/s, TTFT, and VRAM usage.</div>';
      return;
    }
    const maxTps=Math.max(1,...list.map(b=>b.gen_tps||0));
    $('#benchtable').innerHTML='<div class="tbl-wrap"><table class="tbl"><thead><tr>'
      +'<th>Model</th><th>Preset</th><th>Gen speed</th><th>Prompt eval</th><th>TTFT</th><th>Tokens</th><th>VRAM</th><th>When</th>'
      +'</tr></thead><tbody>'+list.map(b=>{
        const w=Math.max(4,Math.round(100*(b.gen_tps||0)/maxTps));
        return '<tr>'
          +'<td><b>'+esc(b.model)+'</b></td>'
          +'<td><span class="tag mut">'+esc(b.preset||'—')+'</span></td>'
          +'<td style="min-width:150px"><div style="display:flex;align-items:center;gap:8px"><b style="color:var(--ok)">'+b.gen_tps+' tok/s</b>'
          +'<div class="meter" style="flex:1;margin:0;height:6px"><i style="width:'+w+'%"></i></div></div></td>'
          +'<td>'+(b.prompt_tps!=null?b.prompt_tps+' tok/s':'—')+'</td>'
          +'<td>'+n0(b.ttft_ms)+' ms</td>'
          +'<td>'+n0(b.completion_tokens)+' tok ('+b.total_s+'s)</td>'
          +'<td>'+(b.vram_mb?n0(b.vram_mb)+' MB':'—')+'</td>'
          +'<td class="hint">'+esc(ago(b.ts))+'</td></tr>';
      }).join('')+'</tbody></table></div>';
  }catch(e){}
}
$('#benchrun').addEventListener('click',async()=>{
  const b=$('#benchrun');b.disabled=true;const old=b.textContent;b.textContent='Benchmarking…';
  try{
    const r=await post('/api/benchmarks/run',{prompt_key:$('#benchprompt').value,max_tokens:Number($('#benchmax').value)||160});
    toast('Benchmark complete: '+r.gen_tps+' tok/s (TTFT '+r.ttft_ms+' ms)','ok',5000);
    await loadBenchmarks();await loadModels();
  }catch(e){toast('Benchmark failed: '+e.message,'err',9000)}
  b.disabled=false;b.textContent=old;
});
$('#benchclear').addEventListener('click',async()=>{
  if(!await confirmDlg('Clear benchmarks','Remove all saved benchmark results?','Clear',true))return;
  try{await api('/api/benchmarks',{method:'DELETE'});loadBenchmarks();loadModels();toast('Benchmark history cleared','ok')}
  catch(e){toast('Failed: '+e.message,'err')}
});

/* ============================== log viewer ============================== */
let LOGTEXT='';
function paintLog(){
  const f=$('#logfilter').value.trim().toLowerCase();
  const lines=LOGTEXT.split('\n');
  const shown=lines.filter(l=>!f||l.toLowerCase().includes(f));
  const html=shown.map(l=>{
    const e=esc(l);
    if(/error|failed|exception|abort|refused/i.test(l))return '<span class="e">'+e+'</span>';
    if(/warn|deprecat/i.test(l))return '<span class="w">'+e+'</span>';
    if(/^\s*$/.test(l))return '<span class="g">·</span>';
    return e;
  }).join('\n');
  const p=$('#log');
  if(p.innerHTML!==html){p.innerHTML=html;if($('#logauto').checked)p.scrollTop=1e9}
  $('#logn').textContent=shown.length+' / '+lines.length+' lines';
}
async function refreshLog(){
  try{const d=await api('/api/server/log');
    if(d.log!==LOGTEXT){LOGTEXT=d.log;paintLog()}}catch(e){}
}
$('#logfilter').addEventListener('input',paintLog);
$('#logauto').addEventListener('change',()=>{if($('#logauto').checked){const p=$('#log');p.scrollTop=1e9}});
$('#logwrap').addEventListener('change',e=>$('#log').classList.toggle('wrapoff',!e.target.checked));
$('#logcopy').addEventListener('click',()=>copy(LOGTEXT,'Log'));
$('#logsave').addEventListener('click',()=>{
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([LOGTEXT],{type:'text/plain'}));
  a.download='lfm-server-'+new Date().toISOString().slice(0,16).replace(/[:T]/g,'-')+'.log';
  a.click();setTimeout(()=>URL.revokeObjectURL(a.href),4000)});

/* ============================== vitals ============================== */
const VH={gpu:[],cpu:[],mem:[]};let VIT=null,VT=0;
function pushHist(){
  if(!VIT)return;
  const add=(k,v)=>{if(v==null)return;const a=VH[k];a.push(v);if(a.length>70)a.shift()};
  add('gpu',VIT.gpu?Math.round(100*(VIT.gpu.used_mb||0)/(VIT.gpu.total_mb||1)):null);
  add('cpu',VIT.cpu?VIT.cpu.percent:null);
  add('mem',VIT.mem?Math.round(100*VIT.mem.used_gb/VIT.mem.total_gb):null);
}
function spark(arr){
  if(!arr||arr.length<3)return '<div style="height:26px"></div>';
  const w=100,ht=26,mx=Math.max.apply(null,arr.concat([1]));
  const pts=arr.map((v,i)=>((i/(arr.length-1))*w).toFixed(1)+','+(ht-1.5-(v/mx)*(ht-4)).toFixed(1)).join(' ');
  return '<svg class="spark" viewBox="0 0 '+w+' '+ht+'" preserveAspectRatio="none" aria-hidden="true">'
    +'<polyline points="'+pts+'" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"/></svg>';
}
const lvlCls=p=>p>=90?'err':p>=75?'warn':'';
function card(o){
  return '<div class="card" style="margin:0;color:var(--'+(o.tone||'acc')+');'+(o.raw||'')+'">'
    +'<div class="hd" style="margin-bottom:8px"><h2 style="color:var(--fg-2);font-size:13px">'+o.title+'</h2>'
    +'<span class="tag" style="color:inherit">'+o.badge+'</span></div>'
    +'<div class="stat" style="color:var(--fg)">'+o.big+'</div>'
    +'<div class="meter '+lvlCls(o.pct||0)+'"><i style="width:'+Math.min(100,o.pct||0)+'%"></i></div>'+spark(o.hist)+'</div>';
}
function drawVitals(){
  if(!VIT){$('#vitals').innerHTML='<div class="empty"><b>No vitals yet</b>Sensors unreadable on this host.</div>';return}
  const v=VIT;let out='';
  if(v.gpu){
    const g=v.gpu,pct=Math.round(100*(g.used_mb||0)/(g.total_mb||1)),idle=g.util===0;
    const bits=[g.power_w!=null?g.power_w+' W':null,idle?null:(g.fan_rpm?g.fan_rpm+' RPM':null),
      idle?null:(g.sclk_mhz?g.sclk_mhz+' MHz':null),g.mclk_mhz?g.mclk_mhz+' MHz mem':null].filter(Boolean);
    out+=card({title:'GPU'+(g.stale?' · cached':''),tone:idle?'mut':'acc',pct:pct,
      badge:(idle?'idle':(g.temp!=null?g.temp+'°C jct':'')+(g.temp_mem!=null?' · '+g.temp_mem+'°C mem':''))+(g.stale?' · stale':'')+' · '+(g.util||0)+'% util',
      big:n0(g.used_mb)+' <small>/ '+n0(g.total_mb)+' MB VRAM</small>',hist:VH.gpu});
    if(bits.length)out+='<div class="card" style="margin:0"><div class="hd" style="margin-bottom:8px"><h2 style="color:var(--fg-2);font-size:13px">GPU clocks &amp; power</h2></div>'
      +'<div class="kvs">'+bits.map(b=>'<span class="tag">'+esc(b)+'</span>').join('')+'</div>'
      +'<p class="hint" style="margin:8px 0 0">Sensors sleep with the card; readings are held until it wakes.</p></div>';
  }
  if(v.mem){const p=Math.round(100*v.mem.used_gb/v.mem.total_gb);
    out+=card({title:'System RAM',tone:'acc',pct:p,badge:p+'%',big:v.mem.used_gb+' <small>/ '+v.mem.total_gb+' GB</small>',hist:VH.mem})}
  if(v.cpu){out+=card({title:'CPU Load',tone:'acc',pct:v.cpu.percent,badge:(v.cpu.cores||'?')+' threads',
      big:v.cpu.percent+'%',hist:VH.cpu})}
  Object.keys(v.disk||{}).forEach(k=>{const d=v.disk[k],p=Math.round(100*d.used_gb/d.total_gb);
    out+=card({title:'Disk '+k,tone:'acc',pct:p,badge:(d.free_gb!=null?d.free_gb+' GB free':p+'%'),big:d.used_gb+' <small>/ '+d.total_gb+' GB</small>'})});
  if(!out)out='<div class="empty"><b>No vitals yet</b>Sensors unreadable on this host.</div>';
  $('#vitals').innerHTML=out;
  $('#vupdatetime').textContent='updated '+Math.max(0,Math.round((Date.now()-VT)/1000))+'s ago';
}
async function refreshVitals(){
  try{VIT=await api('/api/vitals');VT=Date.now();pushHist();drawVitals()}catch(e){drawVitals()}
}
$('#vrefresh').addEventListener('click',refreshVitals);

/* ============================== chat ============================== */
let HIST=[],CHATS_CACHE=[],CID=null,BUSY=false,AC=null;
const DEF_SYS='You are a helpful local assistant. Be concise.';
$('#sysprompt').value=localStorage.getItem('lfm.sys')||DEF_SYS;
function syncSamplingLabels(){
  $('#sp-temp-v').textContent=Number($('#sp-temp').value).toFixed(2);
  $('#sp-topp-v').textContent=Number($('#sp-topp').value).toFixed(2);
  $('#sp-minp-v').textContent=Number($('#sp-minp').value).toFixed(2);
  $('#sp-rep-v').textContent=Number($('#sp-rep').value).toFixed(2);
  const mx=Number($('#sp-max').value)||0;
  $('#sp-max-v').textContent=mx>0?n0(mx)+' tok':'default';
}
['sp-temp','sp-topp','sp-minp','sp-rep','sp-max'].forEach(id=>{
  const saved=localStorage.getItem('lfm.'+id);
  if(saved!=null)$('#'+id).value=saved;
  $('#'+id).addEventListener('input',()=>{syncSamplingLabels();localStorage.setItem('lfm.'+id,$('#'+id).value)});
});
syncSamplingLabels();
$('#sysbtn').addEventListener('click',()=>{
  const box=$('#sysbox');box.classList.toggle('hide');
  if(box.classList.contains('hide'))localStorage.setItem('lfm.sys',$('#sysprompt').value||DEF_SYS);
  else $('#sysprompt').focus();
});
$('#sysprompt').addEventListener('change',()=>localStorage.setItem('lfm.sys',$('#sysprompt').value||DEF_SYS));

function drawChatTarget(){
  const ct=chatTarget(),t=$('#chatmodel');
  if(ct){t.className='tag ok';t.textContent=ct.alias+' · :'+ct.port}
  else{t.className='tag warn';t.textContent='no model loaded'}
  $('#sendbtn').disabled=BUSY||!ct;
  $('#chatnote').textContent=ct?'':'Start a model in the Models or Avalon tab to chat.';
}
function md(src){
  const blocks=[];
  let s=String(src||'').replace(/```(\w*)\n?([\s\S]*?)```/g,(m,lang,code)=>{
    const clean=String(code).replace(/\n+$/,'');
    blocks.push('<div class="codewrap"><div class="codebar"><span>'+esc(lang||'code')+'</span>'
      +'<button type="button" class="btn xs" data-copy="'+esc(clean)+'">Copy code</button></div>'
      +'<pre><code>'+esc(clean)+'</code></pre></div>');
    return '\u0000'+(blocks.length-1)+'\u0000';
  });
  s=esc(s);
  s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/(^|[\s(])\*([^*\n]+)\*/g,'$1<em>$2</em>');
  s=s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  s=s.replace(/^#{1,4}\s+(.+)$/gm,'<div class="md-h">$1</div>');
  s=s.replace(/^(?:[-*]\s+.+(?:\n|$))+/gm,b=>'<ul>'+b.trim().split('\n')
    .map(l=>'<li>'+l.replace(/^[-*]\s+/,'')+'</li>').join('')+'</ul>');
  s=s.replace(/^(?:\d+[.)]\s+.+(?:\n|$))+/gm,b=>'<ol>'+b.trim().split('\n')
    .map(l=>'<li>'+l.replace(/^\d+[.)]\s+/,'')+'</li>').join('')+'</ol>');
  s=s.replace(/\n{2,}/g,'<br><br>').replace(/\n/g,'<br>');
  return s.replace(/\u0000(\d+)\u0000/g,(m,i)=>blocks[+i]);
}
function addMsg(m,idx){
  const d=document.createElement('div');
  d.className='msg '+(m.role==='user'?'u':'a');
  if(idx!=null)d.dataset.idx=idx;
  const body=md(m.content||'');
  const actions=m.role==='user'
    ?'<div class="bar"><button class="btn xs" data-edit="'+(idx!=null?idx:'')+'">Edit &amp; Resubmit</button><button class="btn xs" data-copy="'+esc(m.content)+'">Copy</button></div>'
    :(m.content?'<div class="bar"><button class="btn xs" data-regen="'+(idx!=null?idx:'')+'">Regenerate</button><button class="btn xs" data-copy="'+esc(m.content)+'">Copy</button></div>':'');
  d.innerHTML='<div class="who"><span>'+esc(m.role==='user'?'You':'Assistant')+'</span></div>'
    +(m.reasoning?'<details><summary>Reasoning trace</summary><div class="think">'+esc(m.reasoning)+'</div></details>':'')
    +'<div class="ans">'+(body||'<span class="typing">generating response…</span>')+'</div>'
    +actions;
  $('#chat').appendChild(d);
  $('#chat').scrollTop=1e9;
  return d;
}
function redrawThread(){
  $('#chat').innerHTML='';
  if(!HIST.length){emptyThread();return;}
  HIST.forEach((m,i)=>addMsg(m,i));
}
$('#chat').addEventListener('click',async e=>{
  const cp=e.target.closest('[data-copy]');
  if(cp)return copy(cp.dataset.copy,'Copied');
  const ed=e.target.closest('[data-edit]');
  if(ed&&!BUSY){
    const idx=Number(ed.dataset.edit);
    const msg=HIST[idx];if(!msg)return;
    $('#prompt').value=msg.content;autosize();$('#prompt').focus();
    HIST=HIST.slice(0,idx);
    if(CID){try{await post('/api/chats/'+CID+'/truncate',{keep:idx})}catch(err){}}
    redrawThread();
    return;
  }
  const rg=e.target.closest('[data-regen]');
  if(rg&&!BUSY){
    let idx=Number(rg.dataset.regen);
    if(isNaN(idx)||idx<0)idx=HIST.length-1;
    let userIdx=idx-1;
    while(userIdx>=0&&HIST[userIdx].role!=='user')userIdx--;
    if(userIdx<0)return;
    const promptText=HIST[userIdx].content;
    HIST=HIST.slice(0,userIdx);
    if(CID){try{await post('/api/chats/'+CID+'/truncate',{keep:userIdx})}catch(err){}}
    redrawThread();
    $('#prompt').value=promptText;
    send();
  }
});
function renderChatList(){
  const q=($('#csearch').value||'').trim().toLowerCase();
  const cs=CHATS_CACHE.filter(c=>!q||(c.title||'').toLowerCase().includes(q));
  const delSvg='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:12px;height:12px"><path d="M18 6L6 18M6 6l12 12"/></svg>';
  const renSvg='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:12px;height:12px"><path d="M12 20h9M16.5 3.5a2.12 2.12 0 013 3L7 19l-4 1 1-4z"/></svg>';
  $('#chatlist').innerHTML=cs.length?cs.map(c=>'<div class="citem'+(c.id===CID?' on':'')+'" data-id="'+c.id+'">'
    +'<span class="t" title="'+esc(c.title)+'">'+esc(c.title||'(untitled)')+'</span>'
    +'<span class="s">'+(c.msgs||0)+' · '+esc(ago(c.updated||c.created))+'</span>'
    +'<button class="btn xs" data-ren="'+c.id+'" data-title="'+esc(c.title||'')+'" title="Rename chat" aria-label="Rename chat">'+renSvg+'</button>'
    +'<button class="btn xs danger" data-del="'+c.id+'" title="Delete chat" aria-label="Delete chat">'+delSvg+'</button></div>').join('')
    :'<span class="hint">'+(CHATS_CACHE.length?'No matching chats.':'No chats yet — start one below.')+'</span>';
}
async function loadChatList(){
  try{CHATS_CACHE=await api('/api/chats');renderChatList();}catch(e){}
}
$('#csearch').addEventListener('input',renderChatList);
$('#chatlist').addEventListener('click',async e=>{
  const ren=e.target.closest('[data-ren]');
  if(ren){
    e.stopPropagation();
    const next=prompt('Rename conversation:',ren.dataset.title||'');
    if(next&&next.trim()){
      try{await post('/api/chats/'+ren.dataset.ren+'/rename',{title:next.trim()});loadChatList();}
      catch(err){toast('Rename failed: '+err.message,'err')}
    }
    return;
  }
  const del=e.target.closest('[data-del]');
  if(del){
    e.stopPropagation();
    if(!await confirmDlg('Delete chat','This conversation will be removed from disk.','Delete',true))return;
    try{await api('/api/chats/'+del.dataset.del,{method:'DELETE'});
      if(CID===del.dataset.del)newChat();loadChatList()}catch(err){toast('Delete failed: '+err.message,'err')}
    return;
  }
  const it=e.target.closest('[data-id]');if(it)openChat(it.dataset.id);
});
async function openChat(id,quiet){
  try{
    const c=await api('/api/chats/'+id);CID=id;localStorage.setItem('lfm.cid',id);
    $('#chattitle').textContent=c.title||'Conversation';
    HIST=(c.messages||[]).filter(m=>m.role!=='system');
    redrawThread();loadChatList();return true;
  }catch(e){if(!quiet)toast('Could not open chat: '+e.message,'err');return false}
}
function emptyThread(){
  $('#chattitle').textContent='Conversation';
  $('#chat').innerHTML='<div class="empty"><b>New conversation</b>Pick a saved chat on the left, or type a message below.</div>';
}
function newChat(){CID=null;HIST=[];localStorage.removeItem('lfm.cid');emptyThread();loadChatList();$('#prompt').focus()}
$('#newchat').addEventListener('click',newChat);

$('#chatexportmd').addEventListener('click',()=>{
  if(!HIST.length)return toast('Nothing to export','warn');
  const mdOut=HIST.map(m=>'### '+(m.role==='user'?'User':'Assistant')+'\n\n'+(m.reasoning?'<think>\n'+m.reasoning+'\n</think>\n\n':'')+m.content).join('\n\n---\n\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([mdOut],{type:'text/markdown'}));
  a.download='chat-'+(CID||'export')+'.md';a.click();
});
$('#chatexportjson').addEventListener('click',()=>{
  if(!HIST.length)return toast('Nothing to export','warn');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(HIST,null,2)],{type:'application/json'}));
  a.download='chat-'+(CID||'export')+'.json';a.click();
});

const ta=$('#prompt');
function autosize(){ta.style.height='auto';ta.style.height=Math.min(220,ta.scrollHeight)+'px'}
ta.addEventListener('input',autosize);
ta.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
$('#sendbtn').addEventListener('click',send);
$('#stopbtn').addEventListener('click',()=>{if(AC)AC.abort()});

async function send(){
  const q=ta.value.trim();
  if(!q||BUSY)return;
  const ct=chatTarget();
  if(!ct)return toast('Start a model first (Models or Avalon tab)','warn');
  ta.value='';autosize();
  const th=$('#chat');if(th.querySelector('.empty'))th.innerHTML='';
  const um={role:'user',content:q};HIST.push(um);addMsg(um,HIST.length-1);
  post('/api/chats/append',{id:CID,model:ct.alias,role:'user',content:q})
    .then(r=>{CID=r.id;loadChatList()}).catch(()=>{});
  const asstIdx=HIST.length;
  const box=addMsg({role:'assistant',content:''},asstIdx);
  const ans=box.querySelector('.ans'),who=box.querySelector('.who');
  BUSY=true;$('#sendbtn').disabled=true;$('#stopbtn').style.display='';
  AC=new AbortController();
  let buff='',bufa='',t0=0,toks=0,stopped=false;
  const payload={
    messages:[{role:'system',content:$('#sysprompt').value||DEF_SYS}].concat(HIST.map(m=>({role:m.role,content:m.content}))),
    temperature:Number($('#sp-temp').value),
    top_p:Number($('#sp-topp').value),
    min_p:Number($('#sp-minp').value),
    repeat_penalty:Number($('#sp-rep').value)
  };
  const mx=Number($('#sp-max').value)||0;
  if(mx>0)payload.max_tokens=mx;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},signal:AC.signal,
      body:JSON.stringify(payload)});
    if(!r.ok){const d=await r.text();ans.innerHTML=esc(d.indexOf('no model')>=0?'No model running — load one first.':'error: '+d)}
    else{
      const rd=r.body.getReader(),dec=new TextDecoder();let sbuf='';
      for(;;){
        const it=await rd.read();if(it.done)break;
        sbuf+=dec.decode(it.value,{stream:true});
        const lines=sbuf.split('\n');sbuf=lines.pop();
        for(const ln of lines){
          if(ln.indexOf('data: ')!==0)continue;
          const d=ln.slice(6);if(d==='[DONE]')continue;
          try{
            const j=JSON.parse(d);
            if(t0===0)t0=performance.now();
            const dl=(j.choices&&j.choices[0]&&j.choices[0].delta)||{};
            if(dl.reasoning_content){buff+=dl.reasoning_content;
              let de=box.querySelector('details');
              if(!de){de=document.createElement('details');de.open=true;de.innerHTML='<summary>Reasoning trace</summary><div class="think"></div>';
                box.insertBefore(de,ans)}
              de.querySelector('.think').textContent=buff}
            if(dl.content){bufa+=dl.content;ans.innerHTML=md(bufa)}
            if(j.usage&&j.usage.completion_tokens)toks=j.usage.completion_tokens;
          }catch(e){}
        }
        $('#chat').scrollTop=1e9;
      }
    }
  }catch(e){
    if(e.name==='AbortError'){stopped=true;ans.innerHTML=md(bufa||buff||'(stopped)')}
    else ans.textContent='error: '+e.message;
  }
  const finalText=bufa||buff||'';
  const secs=Math.max(.001,(performance.now()-(t0||performance.now()))/1000);
  const tps=toks?(toks/secs).toFixed(1):Math.round((finalText.length/4)/secs);
  who.innerHTML='<span>Assistant</span><span class="tag mut">'+(stopped?'stopped · ':'')+tps+' tok/s · '+esc(ct.alias)+'</span>';
  if(finalText){
    box.insertAdjacentHTML('beforeend','<div class="bar"><button class="btn xs" data-regen="'+asstIdx+'">Regenerate</button><button class="btn xs" data-copy="'+esc(finalText)+'">Copy</button></div>');
    HIST.push({role:'assistant',content:finalText,reasoning:buff});
    post('/api/chats/append',{id:CID,model:ct.alias,role:'assistant',content:finalText,reasoning:buff})
      .then(r=>{CID=r.id;loadChatList()}).catch(()=>{});
  }
  BUSY=false;AC=null;$('#stopbtn').style.display='none';drawChatTarget();ta.focus();
}

/* ============================== agents registry ============================== */
$('#agentsbtn').addEventListener('click',async()=>{
  let d;try{d=await api('/api/agents')}catch(e){return toast('Failed: '+e.message,'err')}
  const reg=(d.registry&&d.registry.agents)||{};
  const cur=d.running?esc(d.running[0])+' @ :'+d.running[1]:'nothing running';
  const rows=d.agents.map(a=>{
    const r=reg[a.id];
    return '<label class="frow"><input type="checkbox" class="agchk" value="'+esc(a.id)+'">'
      +'<span style="flex:1;min-width:0"><b>'+esc(a.label)+'</b>'
      +(r?' <span class="tag ok">'+esc(r.alias)+'@'+r.port+' · '+esc(ago(r.synced_at))+'</span>':'')
      +'<br><span class="hint mono">'+esc(a.hint||'')+'</span></span></label>';
  }).join('');
  $('#ag-body').innerHTML='<p style="margin:0 0 10px;color:var(--fg-2)">Writes each selected agent config to point at <b>'+cur+'</b>.</p>'
    +'<div style="max-height:46vh;overflow:auto">'+rows+'</div><div id="agres" style="margin-top:10px"></div>';
  $('#agentsdlg').showModal();
});
$('#ag-apply').addEventListener('click',async()=>{
  const apply=$('#ag-apply');
  const picks=$$('.agchk',$('#agentsdlg')).filter(x=>x.checked).map(x=>x.value);
  if(!picks.length)return toast('Pick at least one agent','warn');
  apply.disabled=true;apply.textContent='Applying…';
  let r;try{r=await post('/api/agents/sync',{agents:picks})}
  catch(e){apply.disabled=false;apply.textContent='Apply';return toast('Sync failed: '+e.message,'err',12000)}
  $('#agres').innerHTML=Object.keys(r.results).map(k=>{const v=r.results[k];
    return '<div class="frow" style="cursor:default"><span class="tag '+(v.ok?'ok':'err')+'">'+(v.ok?'ok':'fail')+'</span>'
      +'<span class="p" title="'+esc(v.detail)+'">'+esc(k)+' — '+esc(v.detail)+'</span></div>'}).join('');
  apply.disabled=false;apply.textContent='Apply';
  toast('Registry updated for '+picks.length+' agent'+(picks.length>1?'s':''),'ok');
  pollStatus();
});

/* ============================== avalon & optimizer ============================== */
let ASEL=null,ALIST=[];
async function loadAvalon(){
  try{
    ALIST=await api('/api/avalon/profiles');AV=await api('/api/avalon/status');
    if(!ASEL||!ALIST.some(p=>p.key===ASEL))ASEL=(AV&&AV.running&&AV.profile)||'';
    if(!ASEL&&ALIST.length)ASEL=(ALIST.find(p=>p.available)||ALIST[0]).key;
    $('#avalonprofiles').innerHTML=ALIST.map(p=>'<label class="frow" style="align-items:flex-start;padding:10px 11px">'
      +'<input type="radio" name="sprof" value="'+esc(p.key)+'"'+(p.key===ASEL?' checked':'')+(p.available?'':' disabled')+'>'
      +'<span style="flex:1;min-width:0"><b>'+esc(p.label)+'</b>'
      +(AV.running&&AV.profile===p.key?' <span class="tag ok">running</span>':'')
      +(p.available?'':' <span class="tag err">config missing</span>')
      +'<br><span class="hint">'+esc(p.desc)+'</span><br><span class="tag ok" style="margin-top:4px">'+esc(p.speed)+'</span></span></label>').join('');
    $$('#avalonprofiles input[name=sprof]').forEach(r=>r.addEventListener('change',()=>{ASEL=r.value;loadAvalon();previewAvalonTune()}));
    const c=$('#avalonchip');
    c.textContent=AV.running?(AV.state==='ready'?'avalon · ready':'avalon · loading'):'avalon · stopped';
    c.className='tag '+(AV.running?(AV.state==='ready'?'ok':'warn'):'mut');
    $('#avalondetail').innerHTML=AV.running
      ?[['pid',AV.pid||'?'],AV.adopted?['source','adopted']:null,['up',fmtUp(AV.uptime||0)],
        ['ctx',AV.max_context?n0(AV.max_context):'…'],['port',':'+(AV.port||8086)]]
        .filter(Boolean).map(kv=>'<span class="tag">'+esc(kv[0])+' <b>'+esc(kv[1])+'</b></span>').join('')
        +(AV.model?'<span class="tag mut">'+esc(AV.model)+'</span>':'')
      :'<span class="hint">Avalon is stopped. Choose a profile below or open Config Optimizer &amp; Tuning.</span>';
    $('#avalonstart').disabled=!ASEL||!ALIST.some(p=>p.key===ASEL&&p.available);
    $('#avalonstop').disabled=!AV.running;
    const tp=ALIST.find(p=>p.key===ASEL);
    if(tp&&tp.saved&&!$('#avalontunebox').dataset.loaded){
      $('#avalontunebox').dataset.loaded='1';
      setAvalonTuneForm(tp.saved);
    }
    const selProf=ALIST.find(p=>p.key===ASEL);
    if(selProf&&selProf.cmd_preview&&!$('#avcmdpreview').textContent)$('#avcmdpreview').textContent=selProf.cmd_preview;
    const d=await api('/api/avalon/log');paintInto($('#avalonlog'),d.log);
    drawHeader();drawChatTarget();
  }catch(e){}
}
function getAvalonTuneForm(){
  // Empty number box = unset (null); the two selects use 0 = "profile default".
  const num=v=>{const s=String(v).trim();if(s==='')return null;const n=Number(s);return isNaN(n)?null:n};
  return {
    profile:ASEL||'flagship',
    ctx:Number($('#av-ctx').value)||0,
    batch:num($('#av-batch').value),
    ubatch:num($('#av-ubatch').value),
    ctk:$('#av-ctk').value,
    ctv:$('#av-ctv').value,
    draft_n:num($('#av-draft').value),
    ubd:num($('#av-ubd').value),
    parallel:Number($('#av-np').value)||0
  };
}
function setAvalonTuneForm(t){
  // Number boxes: null/undefined = unset (blank), but a saved 0 must render as "0"
  // (draft_n 0 means drafting off — blank would silently mean "profile default").
  const v=x=>(x==null||x==='')?'':String(x);
  $('#av-ctx').value=String(t.ctx||0);
  $('#av-batch').value=v(t.batch);
  $('#av-ubatch').value=v(t.ubatch);
  $('#av-ctk').value=t.ctk||'';
  $('#av-ctv').value=t.ctv||'';
  $('#av-draft').value=v(t.draft_n);
  $('#av-ubd').value=v(t.ubd);
  $('#av-np').value=String(t.parallel||0);
}
async function previewAvalonTune(){
  try{
    const r=await post('/api/avalon/tune',Object.assign(getAvalonTuneForm(),{preview:true}));
    $('#avcmdpreview').textContent=r.cmd_preview||'';
  }catch(e){}
}
$('#avalontunebtn').addEventListener('click',()=>{
  const box=$('#avalontunebox');box.classList.toggle('hide');
  if(!box.classList.contains('hide'))previewAvalonTune();
});
$$('#avalontunebox input, #avalontunebox select').forEach(el=>el.addEventListener('change',previewAvalonTune));
$$('#avalontunebox [data-avopt]').forEach(b=>b.addEventListener('click',()=>{
  const k=b.dataset.avopt;
  if(k==='speed'){ASEL='prefast';setAvalonTuneForm({ctx:65536,batch:2048,ubatch:512,ctk:'q8_0',ctv:'q8_0',draft_n:4,ubd:512,parallel:1});}
  else if(k==='balanced'){ASEL='compact';setAvalonTuneForm({ctx:65536,batch:2048,ubatch:512,ctk:'q8_0',ctv:'q8_0',draft_n:3,ubd:512,parallel:1});}
  else if(k==='max_ctx'){ASEL='flagship';setAvalonTuneForm({ctx:131072,batch:2048,ubatch:512,ctk:'q8_0',ctv:'q8_0',draft_n:0,ubd:0,parallel:1});}
  else{setAvalonTuneForm({});}
  previewAvalonTune();loadAvalon();
}));
$('#avcopycmd').addEventListener('click',()=>copy($('#avcmdpreview').textContent,'Avalon command'));
$('#avsavetune').addEventListener('click',async()=>{
  try{await post('/api/avalon/tune',getAvalonTuneForm());toast('Avalon tuning saved','ok');loadAvalon();}
  catch(e){toast('Failed: '+e.message,'err')}
});
$('#avapplystart').addEventListener('click',async()=>{
  try{await post('/api/avalon/tune',getAvalonTuneForm());$('#avalonstart').click();}
  catch(e){toast('Failed: '+e.message,'err')}
});
function paintInto(el,text){
  if(el.dataset.raw===text)return;el.dataset.raw=text;
  el.innerHTML=text.split('\n').map(l=>{
    const e=esc(l);
    if(/error|failed|exception|abort/i.test(l))return '<span class="e">'+e+'</span>';
    if(/warn/i.test(l))return '<span class="w">'+e+'</span>';
    return e}).join('\n');
  el.scrollTop=1e9;
}
$('#avalonstart').addEventListener('click',async()=>{
  $('#avalonstart').disabled=true;
  try{await post('/api/avalon/start',{profile:ASEL,tune:getAvalonTuneForm()});toast('Avalon starting — a 125B load takes a while','ok')}
  catch(e){
    const t=e.message;
    if(t.indexOf('Stop the LFM server')>=0){
      if(await confirmDlg('VRAM conflict','This profile needs the VRAM the LFM server is holding.<br><br>Stop the LFM server and start Avalon?','Stop LFM & start Avalon',true)){
        try{await post('/api/server/stop');pollStatus(true);$('#avalonstart').click()}
        catch(e2){toast('Could not stop the LFM server: '+e2.message,'err')}}
    }else toast('Start failed: '+t,'err',12000);
  }
  loadAvalon();
});
$('#avalonstop').addEventListener('click',async()=>{
  if(!await confirmDlg('Stop Avalon','Stop the Avalon server? Running requests will be interrupted.','Stop',true))return;
  $('#avalonstop').disabled=true;
  try{const r=await post('/api/avalon/stop');toast(r.stopped?'Avalon stopped':'Stop requested — port still busy, check the log',r.stopped?'ok':'warn')}
  catch(e){toast('Stop failed: '+e.message,'err')}
  loadAvalon();
});
async function loadAvalonAccess(){
  try{
    const a=await api('/api/avalon/access');
    $('#avalonlan').checked=!!a.lan;
    const port=a.port||8086;
    $('#avalonaccess').innerHTML=a.lan
      ?'<span class="tag ok">LAN · http://'+esc(a.ip)+':'+port+'/v1</span>'
        +'<button class="btn sm" data-copy="'+esc(a.api_key)+'">Copy key</button>'
        +'<span class="tag mut mono" title="API key">'+esc(a.api_key.slice(0,6))+'…'+esc(a.api_key.slice(-4))+'</span>'
        +'<button class="btn sm" id="regen">New key</button>'
        +'<span class="hint">send as <code>Authorization: Bearer &lt;key&gt;</code></span>'
      :'<span class="tag mut">local only · http://127.0.0.1:'+port+'/v1</span>'
        +'<span class="hint">toggle above to expose it on the LAN with an API key</span>';
    const rg=$('#regen');
    if(rg)rg.addEventListener('click',async()=>{
      if(!await confirmDlg('Rotate API key','Every LAN client must be updated with the new key.','Rotate',true))return;
      try{await post('/api/avalon/access/regenerate');loadAvalonAccess();toast('New key issued','ok')}
      catch(e){toast('Failed: '+e.message,'err')}});
  }catch(e){}
}
$('#avalonaccess').addEventListener('click',e=>{const b=e.target.closest('[data-copy]');if(b)copy(b.dataset.copy,'API key')});
$('#avalonlan').addEventListener('change',async e=>{
  const on=e.target.checked;e.target.disabled=true;
  try{await post('/api/avalon/access',{lan:on});
    toast(on?'Exposed on the LAN with an API key':'Back to local-only','ok')}
  catch(err){toast('Failed: '+err.message,'err');e.target.checked=!on}
  e.target.disabled=false;loadAvalonAccess();loadAvalon();
});

/* ============================== HF pull (resumable) ============================== */
let HFREPO='',DLPREV={},DLTIMER=null;
async function hfSearch(){
  const q=$('#hfq').value.trim();if(!q)return;
  $('#hfinfo').textContent='searching…';
  let res;try{res=await api('/api/hf/search?q='+encodeURIComponent(q))}
  catch(e){$('#hfinfo').textContent='';return toast('Search failed: '+e.message,'err')}
  $('#hfinfo').textContent=res.length+' repos';
  $('#hfresults').innerHTML=res.length?res.map(m=>'<div class="frow" data-repo="'+esc(m.id)+'">'
    +'<span style="flex:1;min-width:0"><b>'+esc(m.id)+'</b><br><span class="hint">downloads '
    +n0(m.downloads)+' · likes '+n0(m.likes)+'</span></span><span class="tag">Inspect files</span></div>').join('')
    :'<div class="empty"><b>No repos found</b>Try a broader search term.</div>';
}
$('#hfsearch').addEventListener('click',hfSearch);
$('#hfq').addEventListener('keydown',e=>{if(e.key==='Enter')hfSearch()});
$('#hfresults').addEventListener('click',e=>{const r=e.target.closest('[data-repo]');if(r)hfFiles(r.dataset.repo)});
async function hfFiles(repo){
  HFREPO=repo;
  let files;try{files=await api('/api/hf/files?repo='+encodeURIComponent(repo))}
  catch(e){return toast('Repo lookup failed: '+e.message,'err')}
  $('#hfrepo').textContent=repo;
  const budgetBytes=((MODELS[0]&&MODELS[0].fit&&MODELS[0].fit.budget_mb)||24000)*1048576;
  $('#hffiles').innerHTML=files.length?files.map(f=>{
    const fits=f.size&&f.size*1.12<=budgetBytes;
    return '<label class="frow"><input type="checkbox" value="'+esc(f.path)+'" data-size="'+(f.size||0)+'">'
      +'<span class="p" title="'+esc(f.path)+'">'+esc(f.path)+'</span>'
      +(f.size?'<span class="tag '+(fits?'ok':'warn')+'">'+(fits?'fits VRAM':'large')+'</span>':'')
      +'<span class="tag mut">'+esc(f.size_h)+'</span></label>';
  }).join(''):'<div class="empty"><b>No GGUF files</b>This repo has no .gguf in its root tree.</div>';
  $('#hffilesbox').classList.remove('hide');
  totsize();
}
function totsize(){
  const s=$$('#hffiles input:checked').reduce((a,i)=>a+(+i.dataset.size||0),0);
  const n=$$('#hffiles input:checked').length;
  $('#hftotal').textContent=n?n+' files · '+gib(s):'nothing selected';
}
$('#hffiles').addEventListener('change',totsize);
$('#hfnone').addEventListener('click',()=>{$$('#hffiles input').forEach(i=>i.checked=false);totsize()});
$('#hfpull').addEventListener('click',async()=>{
  const sel=$$('#hffiles input:checked').map(i=>i.value);
  if(!sel.length)return toast('Select at least one file','warn');
  try{await post('/api/hf/pull',{repo:HFREPO,files:sel});
    toast('Queued '+sel.length+' file'+(sel.length>1?'s':''),'ok');ensureDl(true);pollDownloads()}
  catch(e){toast('Pull failed: '+e.message,'err',10000)}
});
function ensureDl(on){
  if(on&&!DLTIMER)DLTIMER=setInterval(pollDownloads,1200);
  if(!on&&DLTIMER){clearInterval(DLTIMER);DLTIMER=null}
}
async function pollDownloads(){
  let ds;try{ds=await api('/api/downloads')}catch(e){return}
  if(!ds.length){$('#downloads').innerHTML='<span class="hint">No downloads queued.</span>';$('#dlinfo').textContent='';ensureDl(false);return}
  const live=ds.some(d=>d.status==='downloading'||d.status==='queued');
  $('#dlinfo').textContent=ds.length+' job'+(ds.length>1?'s':'')+(live?' · active':'');
  $('#downloads').innerHTML=ds.map(d=>{
    const pct=d.total?Math.round(100*d.done/d.total):0;
    const now=Date.now(),p=DLPREV[d.id]||{d:d.done,t:now};
    const dt=(now-p.t)/1000,spd=dt>0.4?(d.done-p.d)/dt:null;
    if(d.status==='downloading'&&dt>0.4)DLPREV[d.id]={d:d.done,t:now};
    const eta=(spd>0&&d.total&&d.done<d.total)?fmtUp((d.total-d.done)/spd):null;
    const kind=d.status==='done'?'ok':d.status==='error'?'err':(d.status==='paused'||d.status==='cancelled')?'warn':'acc';
    const canResume=d.status==='paused'||d.status==='error'||d.status==='cancelled';
    return '<div class="dl"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
      +'<span style="flex:1;min-width:0;font-family:var(--mono);font-size:12px" title="'+esc(d.dest)+'">'+esc(d.file)+'</span>'
      +'<span class="tag '+kind+'">'+esc(d.status)+'</span>'
      +(d.total?'<span class="tag mut">'+pct+'% · '+gib(d.done)+' / '+gib(d.total)+'</span>':'')
      +(d.status==='downloading'&&spd!=null?'<span class="tag mut">'+(spd/1048576).toFixed(1)+' MB/s</span>':'')
      +(d.status==='downloading'&&eta?'<span class="tag mut">ETA '+eta+'</span>':'')
      +(d.status==='downloading'?'<button class="btn xs" data-pause="'+esc(d.id)+'">Pause</button>':'')
      +(canResume?'<button class="btn xs primary" data-resume="'+esc(d.id)+'">Resume</button>':'')
      +(d.status!=='done'?'<button class="btn xs danger" data-cancel="'+esc(d.id)+'">Cancel</button>':'')
      +'</div>'+(d.total?'<div class="meter"><i style="width:'+pct+'%"></i></div>':'')
      +(d.error?'<p class="hint" style="color:var(--err);margin:5px 0 0">'+esc(d.error)+'</p>':'')+'</div>';
  }).join('');
  ensureDl(live);
}
$('#downloads').addEventListener('click',e=>{
  const p=e.target.closest('[data-pause]');
  if(p){post('/api/downloads/cancel',{id:p.dataset.pause,delete_part:false}).then(pollDownloads);return;}
  const r=e.target.closest('[data-resume]');
  if(r){post('/api/downloads/resume',{id:r.dataset.resume}).then(()=>{ensureDl(true);pollDownloads()});return;}
  const c=e.target.closest('[data-cancel]');
  if(c){post('/api/downloads/cancel',{id:c.dataset.cancel,delete_part:true}).then(pollDownloads);}
});

/* ============================== settings ============================== */
async function loadSettings(){
  try{
    const s=await api('/api/settings');
    $('#set-dirs').value=(s.model_dirs||[]).join('\n');
    $('#set-bin').value=s.server_binary||'';
    $('#set-ctx').value=s.ctx||16384;
    $('#set-port').value=s.port_base||8081;
    $('#set-idle').value=s.idle_unload_min||0;
    $('#set-reserve').value=s.vram_reserve_mb!=null?s.vram_reserve_mb:512;
    $('#set-hftok').value='';
    $('#set-hf-status').textContent=s.hf_token_set?'configured ('+s.hf_token_preview+')':'not set';
    const c=s.caps||{};
    $('#set-caps-badge').textContent=c.exists?'binary verified':'binary missing';
    $('#set-caps-badge').className='tag '+(c.exists?'ok':'err');
    $('#set-bin-ver').textContent=c.version||'';
    $('#set-bin-flags').innerHTML=['fa_value','ncmoe','metrics','fit','ctk','mlock','no_mmap','parallel']
      .map(k=>'<span class="tag '+(c[k]?'ok':'mut')+'">'+k+(c[k]?' ✓':' –')+'</span>').join('');
    $('#set-disks').innerHTML=(s.disks||[]).map(d=>d.exists?d.path+': '+d.free_gb+' GB free':d.path+': missing').join(' · ');
  }catch(e){toast('Failed to load settings: '+e.message,'err')}
}
$('#set-reload').addEventListener('click',loadSettings);
$('#set-save').addEventListener('click',async()=>{
  const btn=$('#set-save');btn.disabled=true;
  try{
    const dirs=$('#set-dirs').value.split('\n').map(x=>x.trim()).filter(Boolean);
    await post('/api/settings',{
      model_dirs:dirs,
      server_binary:$('#set-bin').value.trim(),
      ctx:Number($('#set-ctx').value)||16384,
      port_base:Number($('#set-port').value)||8081,
      idle_unload_min:Number($('#set-idle').value)||0,
      vram_reserve_mb:Number($('#set-reserve').value)||0,
      hf_token:$('#set-hftok').value.trim()
    });
    toast('Settings saved','ok');
    await loadSettings();await loadModels();
  }catch(e){toast('Save failed: '+e.message,'err')}
  btn.disabled=false;
});

/* ============================== command palette (Ctrl/Cmd+K) ============================== */
let PAL_ITEMS=[],PAL_IDX=0;
function buildPaletteItems(){
  const items=[];
  TABS.forEach((t,i)=>items.push({label:'Go to '+t.toUpperCase()+' tab',sub:'Shortcut '+(i+1),run:()=>setTab(t)}));
  items.push({label:'Run one-click benchmark on active server',sub:'Server & Bench',run:()=>{setTab('server');$('#benchrun').click()}});
  if(ACTIVE&&ACTIVE.running){
    items.push({label:'Stop active llama-server ('+ACTIVE.alias+')',sub:':'+ACTIVE.port,run:()=>$('#srvstop').click()});
  }
  items.push({label:'Toggle light / dark theme',sub:'Appearance',run:()=>$('#theme').click()});
  items.push({label:'Rescan model directories',sub:'Models',run:()=>$('#rescan').click()});
  MODELS.forEach(m=>{
    items.push({
      label:'Load model: '+m.name,
      sub:m.arch+' · '+m.quant+' · '+m.size_h+(m.fit?' · '+m.fit.label:''),
      run:async()=>{
        toast('Loading '+m.name+'…','ok');
        try{await post('/api/server/start',Object.assign({},m.saved_profile||{},{model:m.path}));pollStatus(true)}
        catch(e){toast('Start failed: '+e.message,'err')}
      }
    });
    items.push({
      label:'Tune & Optimize: '+m.name,
      sub:'Open Launch & Config Optimizer',
      run:()=>openLaunchModal(m)
    });
  });
  return items;
}
function renderPalette(){
  const q=($('#pal-q').value||'').trim().toLowerCase();
  const matched=PAL_ITEMS.filter(it=>!q||(it.label+' '+it.sub).toLowerCase().includes(q)).slice(0,24);
  if(PAL_IDX>=matched.length)PAL_IDX=0;
  $('#pal-list').innerHTML=matched.length?matched.map((it,i)=>
    '<button type="button" class="pal-item'+(i===PAL_IDX?' sel':'')+'" data-pi="'+i+'">'
    +'<span class="pi-main">'+esc(it.label)+'</span><span class="pi-sub">'+esc(it.sub)+'</span></button>'
  ).join(''):'<div class="empty">No matching commands or models</div>';
  $('#pal-list')._matched=matched;
}
function openPalette(){
  PAL_ITEMS=buildPaletteItems();PAL_IDX=0;
  $('#pal-q').value='';
  renderPalette();
  $('#cmdpal').showModal();
  $('#pal-q').focus();
}
$('#cmdbtn').addEventListener('click',openPalette);
$('#pal-q').addEventListener('input',()=>{PAL_IDX=0;renderPalette()});
$('#pal-q').addEventListener('keydown',e=>{
  const m=$('#pal-list')._matched||[];
  if(e.key==='ArrowDown'){e.preventDefault();PAL_IDX=(PAL_IDX+1)%Math.max(1,m.length);renderPalette();}
  else if(e.key==='ArrowUp'){e.preventDefault();PAL_IDX=(PAL_IDX-1+Math.max(1,m.length))%Math.max(1,m.length);renderPalette();}
  else if(e.key==='Enter'&&m[PAL_IDX]){e.preventDefault();$('#cmdpal').close();m[PAL_IDX].run();}
});
$('#pal-list').addEventListener('click',e=>{
  const b=e.target.closest('[data-pi]');if(!b)return;
  const m=($('#pal-list')._matched||[])[Number(b.dataset.pi)];
  if(m){$('#cmdpal').close();m.run();}
});

/* ============================== polling & keys ============================== */
function tick(){
  if(document.hidden)return;
  pollStatus();
  pollTelemetry();
  if(tab==='avalon')loadAvalon();
}
setInterval(tick,2500);
setInterval(()=>{if(!document.hidden&&tab==='vitals')refreshVitals()},2000);
setInterval(()=>{if(!document.hidden&&tab==='server')refreshLog()},1500);
setInterval(()=>{if(!document.hidden&&tab==='avalon')refreshAvalonLog()},1500);
async function refreshAvalonLog(){try{const d=await api('/api/avalon/log');paintInto($('#avalonlog'),d.log)}catch(e){}}
document.addEventListener('visibilitychange',()=>{if(!document.hidden){pollStatus();pollTelemetry();refreshVitals()}});
document.addEventListener('keydown',e=>{
  if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){
    e.preventDefault();
    if($('#cmdpal').open)$('#cmdpal').close();else openPalette();
    return;
  }
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if($$('dialog[open]').length)return;
  const t=e.target,typing=t&&(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.TagName==='SELECT'||t.isContentEditable);
  if(e.key==='/'&&!typing){
    e.preventDefault();
    if(tab==='hf')$('#hfq').focus();else if(tab==='server')$('#logfilter').focus();else if(tab==='chat')$('#csearch').focus();else{setTab('models');$('#msearch').focus()}
    return;
  }
  if(!typing&&TABS[Number(e.key)-1])setTab(TABS[Number(e.key)-1]);
});

/* ============================== boot ============================== */
setTab(localStorage.getItem('lfm.tab')||'models');
loadModels().then(()=>{pollStatus(true);pollTelemetry();}).catch(e=>toast('Panel load failed: '+e.message,'err'));
loadChatList();
(async()=>{const s=localStorage.getItem('lfm.cid');
  if(!s||!(await openChat(s,true))){localStorage.removeItem('lfm.cid');emptyThread()}})();
refreshVitals();loadAvalon();loadAvalonAccess();pollDownloads();loadBenchmarks();drawChatTarget();
$('#footnote').textContent='live telemetry active · polling pauses when tab is hidden';
</script>
</body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CFG.get("bind_host", "0.0.0.0"), port=PORT, log_level="info")
