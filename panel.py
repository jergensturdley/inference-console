#!/usr/bin/env python3
"""Inference Console — local model library, llama-server control, system vitals, chat.

Single-file FastAPI app. Dependencies: fastapi, uvicorn, httpx.
Config: ~/.config/inference-console/config.json (created with defaults on first run).
"""
import json, os, re, secrets, shutil, signal, socket, struct, subprocess, sys, tarfile, threading, time, uuid, zipfile, platform
from collections import deque
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

HOME = Path.home()
APP_DIR = HOME / ".local/share/inference-console"
CONFIG_PATH = HOME / ".config/inference-console/config.json"
STATE_PATH = APP_DIR / "state.json"
BIN = str(APP_DIR / "bin/llama-server")
PORT = int(os.environ.get("INFER_CONSOLE_PORT") or os.environ.get("LFM_PANEL_PORT") or "8817")
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
    "avalon_snapshots": [],      # saved one-click launch profiles (exact tune per snapshot)
    "avalon_model": "",          # pin the Avalon tab to one pack path (empty = auto: newest complete Flash-Next)
    "avalon_draft": "",          # MTP draft override (empty = first mtp-*.gguf next to the detected pack)
    # Strata (github.com/Niko1221/Strata). strata_bin empty means "find it on PATH".
    # strata_api_key is only used if you set it deliberately; Strata owns its own key.
    "strata_bin": "",
    "strata_port": 0,            # 0 = take the address the launch reported
    "strata_alias": "",          # model name to advertise to coding agents
    "strata_ctx": 8192,
    "strata_api_key": "",
    "strata_last": {},           # address/model reported by the last successful launch
    # Optional LLM provider for the Optimization Agent (empty = deterministic + local fallback)
    "agent_llm_provider": "custom",
    "agent_llm_url": "",
    "agent_llm_key": "",
    "agent_llm_model": "",
    "agent_llm_headers": "",
    "agent_llm_fallback_local": True,
    "thermal_guard": {},         # graduated GPU thermal & VRAM pressure guard settings
    # Project Maya (~/project-maya). Empty means use MAYA_DIR / MAYA_ROCM_PATH defaults.
    "maya_dir": "",
    "maya_rocm_path": "",
}

# ---------------------------------------------------------------- config
LEGACY_APP_DIR = HOME / ".local/share/lfm-panel"
LEGACY_CONFIG = HOME / ".config/lfm-panel/config.json"

def _migrate_legacy_paths():
    """One-time rename from the old lfm-panel install layout. Only moves when the new
    location is absent, so it is a no-op after the first run and never clobbers."""
    try:
        if not CONFIG_PATH.exists() and LEGACY_CONFIG.exists():
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(LEGACY_CONFIG), str(CONFIG_PATH))
        if not APP_DIR.exists() and LEGACY_APP_DIR.exists():
            shutil.move(str(LEGACY_APP_DIR), str(APP_DIR))
    except OSError:
        pass  # a failed migration must not brick the panel; it just starts with defaults

def load_config():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    cfg = json.loads(CONFIG_PATH.read_text())
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg

_migrate_legacy_paths()
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
        "draft": ("--model-draft" in help_txt) or ("--spec-type" in help_txt),
        "draft_max": "--draft-max" in help_txt,
        "ot": "--override-tensor" in help_txt,
        "tensor_split": "--tensor-split" in help_txt,
        "tbatch": "--threads-batch" in help_txt,
        "keep": bool(re.search(r"--keep\s", help_txt)),
        "jinja": "--jinja" in help_txt,
    }
    if not help_txt:   # unknown build: assume a modern llama.cpp
        caps.update(fa_value=True, ncmoe=True, metrics=True, ctk=True, mlock=True, no_mmap=True,
                    parallel=True, draft=True, draft_max=True, ot=True, tensor_split=True,
                    tbatch=True, keep=True, jinja=True)
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
# llama.cpp split files: <base>-00001-of-00033.gguf — the library groups these into one entry
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})$")
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

def find_draft(model_path: str):
    """Locate a speculative-decoding draft beside the weights: an MTP shared-expert file
    (`mtp-…`) or a classic draft GGUF (`…-draft…`). Prefers the token-overlap closest match,
    skips mmproj files and the model itself."""
    p = Path(model_path)
    want = _tokens(p.stem)
    best = None
    for sib in p.parent.glob("*.gguf"):
        if sib == p:
            continue
        sl = sib.stem.lower()
        if "mmproj" in sl or not ("mtp" in sl or "draft" in sl):
            continue
        toks = _tokens(sl.replace("mtp", "").replace("draft", ""))
        overlap = len(want & toks) / max(1, len(want | toks))
        cand = (round(overlap, 3), sib)
        if best is None or cand[0] > best[0]:
            best = cand
    return str(best[1]) if best else None

def human_size(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024

# general.file_type -> quant label. Values are the gguf LlamaFileType names with the
# MOSTLY_/ALL_ prefixes stripped; ids 4-6 and 33-35 are unassigned upstream, and 1024
# (GUESSED) is a sentinel rather than a quant, so it stays absent and renders "?".
# Kept as a literal rather than imported from gguf.constants: gguf pulls in numpy,
# requests and tqdm and is not a declared dependency, so an import-with-fallback would
# silently serve this table on any venv that lacks it. tests/test_panel_metrics.py
# asserts this table against a pinned spec fixture, and against gguf.constants when it
# happens to be importable, so drift fails loudly instead of quietly.
def _build_file_type_map():
    """Authoritative int→quant-name map (gguf-py LlamaFileType); static table = fallback."""
    try:
        from gguf.constants import LlamaFileType
        out = {}
        for m in LlamaFileType:
            if m.name == "GUESSED":
                continue
            out[int(m.value)] = m.name.replace("MOSTLY_", "").replace("ALL_F32", "F32")
        if out:
            return out
    except Exception:
        pass
    return {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",7:"Q8_0",8:"Q5_0",9:"Q5_1",10:"Q2_K",11:"Q3_K_S",12:"Q3_K_M",13:"Q3_K_L",14:"Q4_K_S",15:"Q4_K_M",16:"Q5_K_S",17:"Q5_K_M",18:"Q6_K",19:"IQ2_XXS",20:"IQ2_XS",21:"Q2_K_S",22:"IQ3_XS",23:"IQ3_XXS",24:"IQ1_S",25:"IQ4_NL",26:"IQ3_S",27:"IQ3_M",28:"IQ2_S",29:"IQ2_M",30:"IQ4_XS",31:"IQ1_M",32:"BF16",36:"TQ1_0",37:"TQ2_0",38:"MXFP4_MOE",39:"NVFP4",40:"Q1_0"}
_file_type = _build_file_type_map()

_model_cache = {}

def scan_models():
    models, accessories = [], []
    seen, _dupe_seen = set(), set()
    split_groups: dict = {}  # (dir, base-stem) -> head entry for a multi-shard pack
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
            sm = SPLIT_RE.search(f.stem)
            if sm:
                gkey = (str(f.parent), f.stem[: sm.start()])
                g = split_groups.get(gkey)
                if g is not None:
                    g["shards"].append({"path": str(f), "size": st.st_size, "name": f.name})
                    continue
            if key in _model_cache:
                head = _model_cache[key]
                if sm:
                    head["shards"] = [{"path": str(f), "size": st.st_size, "name": f.name}]
                    split_groups[gkey] = head
                models.append(head)
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
            disp_name = stem[: sm.start()] if sm else stem
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
                "path": str(f), "name": disp_name, "arch": arch,
                "quant": quant, "size": st.st_size, "size_h": human_size(st.st_size),
                "moe": moe, "vl": bool(VL_NAME_RE.search(stem)),
                "params": name_label, "ctx_len": md.get(f"{arch}.context_length", 0),
                "blocks": blocks, "embd": embd, "n_head": n_head, "n_kv": n_kv,
                "k_len": k_len, "v_len": v_len,
                "experts": expert_count, "experts_used": expert_used,
                "imatrix": bool(md.get("general.quantization_imatrix", False)),
                "mmproj": mmproj,
            }
            if sm:
                m["shards"] = [{"path": str(f), "size": st.st_size, "name": f.name}]
                split_groups[(str(f.parent), disp_name)] = m
            _model_cache[key] = m
            models.append(m)
    models.sort(key=lambda m: m["name"].lower())
    # Group totals: one entry per pack shows the sum of its shard sizes.
    for m in models:
        if m.get("shards"):
            m["size"] = sum(s["size"] for s in m["shards"])
            m["size_h"] = human_size(m["size"])
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
    draft = ""
    if target == "speed" and model_meta.get("path"):
        d = find_draft(model_meta["path"])
        if d:
            draft = d
            notes.append("Draft/MTP file found beside the weights — speculative decoding "
                         "enabled (--model-draft --draft-max 16).")
    return {
        "preset": preset, "ctx": ctx, "ngl": ngl, "threads": phys_threads,
        "fa": "on", "ctk": ctk, "ctv": ctv, "parallel": parallel,
        "batch": 2048, "ubatch": 512, "ncmoe": ncmoe, "extra_args": "",
        "draft": draft, "draft_max": 16, "jinja": "on",
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
                   draft: str = "", draft_max: int = 16, draft_min: int = 0,
                   ot: str = "", tensor_split: str = "", tbatch: int | None = None,
                   keep: int | None = None, jinja: str = "on",
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

        args = [binary, "-m", model, "--host", "127.0.0.1",
                "-c", str(eff_ctx), "--alias", mp.stem,
                "-ngl", str(eff_ngl), "-t", str(eff_t)]
        if jinja in ("on", "auto") and caps.get("jinja"):
            args += ["--jinja"]
        if pd.get("dev"):
            args += ["--device", pd["dev"]]
        # -ot: an explicit override wins over the preset default; the preset's own expert
        # split stays in play only when the user left the field blank and isn't using ncmoe.
        ot_val = [t for t in re.split(r"\s+", (ot or "").strip()) if t]
        if ot_val and caps.get("ot"):
            args += ["-ot"] + ot_val
        elif pd.get("ot") and not (ncmoe and ncmoe > 0) and caps.get("ot"):
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
        # llama.cpp takes these as repeatable/positional argv values, so "0 1" or "0,1"
        # must be split into separate args rather than passed as one string.
        ts_vals = [t for t in re.split(r"[,\s]+", tensor_split.strip()) if t] if tensor_split else []
        if ts_vals and caps.get("tensor_split"):
            args += ["--tensor-split"] + ts_vals
        if tbatch and int(tbatch) > 0 and caps.get("tbatch"):
            args += ["--threads-batch", str(int(tbatch))]
        if keep and int(keep) > 0 and caps.get("keep"):
            args += ["--keep", str(int(keep))]
        if mlock and caps.get("mlock"):
            args += ["--mlock"]
        if no_mmap and caps.get("no_mmap"):
            args += ["--no-mmap"]
        if caps.get("metrics"):
            args += ["--metrics"]
        # Speculative decoding: an MTP shared-expert or classic draft file beside the weights.
        # Only emitted when the binary actually advertises a draft flag (--model-draft or the
        # newer --spec-type), so an old build never gets a flag it rejects.
        if draft and draft.strip():
            dp = Path(draft.strip()).expanduser()
            if not dp.exists():
                raise HTTPException(400, f"draft model not found: {dp}")
            if caps.get("draft"):
                args += ["--model-draft", str(dp)]
                if caps.get("draft_max"):
                    args += ["--draft-max", str(max(1, int(draft_max))),
                             "--draft-min", str(max(0, int(draft_min)))]
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
              ncmoe=None, mlock=False, no_mmap=False, draft="", draft_max=16, draft_min=0,
              ot="", tensor_split="", tbatch=None, keep=None, jinja="on", save_profile=True):
        if preset not in ("cpu-only",) and not str(preset).startswith("igpu"):
            if MAYA.status().get("running"):
                raise HTTPException(409, "Stop Maya first — this model needs its GPU VRAM.")
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
            ncmoe=ncmoe, mlock=mlock, no_mmap=no_mmap, draft=draft, draft_max=draft_max,
            draft_min=draft_min, ot=ot, tensor_split=tensor_split, tbatch=tbatch, keep=keep,
            jinja=jinja, port=port,
        )
        if save_profile:
            profs = _read_json(LAUNCH_PROFILES_PATH, {})
            profs[model] = {
                "preset": preset, "ctx": eff_ctx, "ngl": ngl, "threads": threads,
                "fa": fa, "ctk": ctk, "ctv": ctv, "parallel": parallel,
                "batch": batch, "ubatch": ubatch, "ncmoe": ncmoe,
                "mlock": mlock, "no_mmap": no_mmap, "extra_args": extra_args,
                "draft": draft, "draft_max": draft_max, "draft_min": draft_min,
                "ot": ot, "tensor_split": tensor_split, "tbatch": tbatch, "keep": keep,
                "jinja": jinja,
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

# ---------------------------------------------------------------- avalon target detection
# The Avalon tab runs one Flash-Next pack. Which pack is resolved automatically:
# newest complete Flash-Next GGUF (split packs must have all shards) in the model
# dirs, so a fresh download enters the tab on its own. avalon_model / avalon_draft
# in config.json pin a specific pack and skip detection.
_AVALON_TTL = 60.0
_AVALON_FLASH_NEXT_RE = re.compile(r"flash[-_.]?next", re.I)
_avalon_cache: dict = {"at": 0.0, "target": None}

def _avalon_target_invalidate():
    _avalon_cache["at"] = 0.0

def _avalon_scan() -> list:
    """All complete Flash-Next packs in the model dirs, newest first.
    Complete = split packs have every -of-N shard; partial packs are invisible."""
    groups: dict = {}  # (dir, base) -> pack accumulator
    for d in CFG["model_dirs"]:
        p = Path(d).expanduser()
        if not p.is_dir():
            continue
        for f in sorted(p.rglob("*.gguf")):
            nl = f.name.lower()
            if "mmproj" in nl or nl.startswith("mtp-"):
                continue
            if not _AVALON_FLASH_NEXT_RE.search(f.name) and not _AVALON_FLASH_NEXT_RE.search(str(f.parent.name)):
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            if st.st_size < 1_000_000:
                continue
            sm = SPLIT_RE.search(f.stem)
            if sm:
                base = f.stem[: sm.start()]
                expected = int(sm.group(2))
                key = (str(f.parent), base)
                g = groups.get(key)
                if g is None:
                    g = groups[key] = {"mtime": 0.0, "head": f, "n": expected, "shards": 0, "size": 0}
                g["mtime"] = max(g["mtime"], st.st_mtime)
                g["shards"] += 1
                g["size"] += st.st_size
            else:
                groups[(str(f.parent), f.stem)] = {"mtime": st.st_mtime, "head": f, "n": 1, "shards": 1, "size": st.st_size}
    out = []
    for g in groups.values():
        if g["shards"] != g["n"]:
            continue
        head = g["head"]
        drafts = sorted(d for d in head.parent.glob("mtp-*.gguf") if d.is_file())
        try:
            mmproj = find_mmproj(str(head))
        except Exception:
            mmproj = None
        out.append({
            "path": str(head), "base": SPLIT_RE.sub("", head.stem),
            "mtime": g["mtime"], "size": g["size"], "size_h": human_size(g["size"]),
            "shards": g["shards"],
            "draft": str(drafts[0]) if drafts else None, "mmproj": mmproj,
        })
    out.sort(key=lambda x: (x["mtime"], x["size"]), reverse=True)
    return out

def _avalon_target() -> dict:
    """Returns {'model': Path|None, 'draft': Path|None, 'mmproj': str|None, 'base': str}."""
    now = time.time()
    if _avalon_cache["at"] and now - _avalon_cache["at"] < _AVALON_TTL:
        return _avalon_cache["target"]
    tgt = {"model": None, "draft": None, "mmproj": None, "base": ""}
    pinned = CFG.get("avalon_model")
    if pinned:
        tgt["model"] = Path(pinned).expanduser()
        tgt["base"] = SPLIT_RE.sub("", tgt["model"].stem)
    else:
        packs = _avalon_scan()
        if packs:
            p0 = packs[0]
            tgt["model"] = Path(p0["path"])
            tgt["base"] = p0["base"]
            tgt["draft"] = Path(p0["draft"]) if p0["draft"] else None
            tgt["mmproj"] = p0["mmproj"]
    # avalon_draft is an override, so it wins in both modes; the pinned branch used to be the
    # only one that honoured it, which silently ignored the setting whenever the tab was on auto.
    if CFG.get("avalon_draft"):
        tgt["draft"] = Path(CFG["avalon_draft"]).expanduser()
    elif tgt["draft"] is None and tgt["model"]:
        drafts = sorted(d for d in tgt["model"].parent.glob("mtp-*.gguf") if d.is_file())
        tgt["draft"] = drafts[0] if drafts else None
    if tgt["model"] and not tgt.get("mmproj"):
        try:
            tgt["mmproj"] = find_mmproj(str(tgt["model"]))
        except Exception:
            tgt["mmproj"] = None
    _avalon_cache["at"] = now
    _avalon_cache["target"] = tgt
    return tgt
AVALON_STATE_PATH = APP_DIR / "avalon-state.json"
AVALON_PORT = 8086
AVALON_PROFILES = {
    "flagship": {
        "label": "Flash-Next 262k (flagship)",
        "desc": "125B MoE IQ3KS · 262k context · MTP drafting on",
        "speed": "auto-fit: attention + KV on the eGPU, experts stream from RAM · ≈98 tok/s gen · needs the llama.cpp server stopped (VRAM)",
        "exclusive": True,
        "draft": True,
        "defaults": {"ctx": 262144, "batch": 9536, "ubatch": 9536, "ctk": "q8_0", "ctv": "q4_0", "draft_n": 2, "ubd": 256, "parallel": 1},
    },
    "prefast": {
        "label": "Fast ingestion",
        "desc": "125B MoE IQ3KS · 262k context · drafting off · prompt reading ≈2× faster",
        "speed": "needs the llama.cpp server stopped (VRAM)",
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

    def build_args(self, profile: str, tune_override: dict | None = None, target: dict | None = None) -> list[str]:
        p = AVALON_PROFILES.get(profile) or AVALON_PROFILES["flagship"]
        tgt = target or _avalon_target()
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
        # Offload placement. Default is the fork's -fit: llama.cpp counts the model's
        # dense weights, KV cache and buffers against each --device's memory and sizes
        # the GPU-side layer count itself. This 88 GiB pack (40 GiB dense + 48 GiB
        # experts + ~10 GiB KV at 262k) cannot fully offload to the 20 GB eGPU, so
        # hardcoded -ngl 99 -fit off aborted every launch; see device hints in the UI.
        device = str(defs.get("device") or "ROCm0")
        ngl = defs.get("ngl")
        fa = str(defs.get("fa") or "on")
        mlock = bool(defs.get("mlock", False))

        args = [AVALON_BIN, "-m", str(tgt["model"]), "--device", device]
        if tgt.get("mmproj"):
            args += ["--mmproj", tgt["mmproj"]]
        if ngl is None:
            args += ["-fit", "on"]
        else:
            args += ["-ngl", str(int(ngl)), "-fit", "off"]
        args += ["-fa", fa, "-ctk", ctk, "-ctv", ctv,
                 "-b", str(batch), "-ub", str(ubatch), "-c", str(ctx),
                 "--parallel", str(parallel), "--jinja", "--temp", "0.6",
                 "--metrics", "--port", str(AVALON_PORT)]
        n_cmoe = defs.get("n_cmoe")
        if n_cmoe is not None:
            args += ["--n-cpu-moe", str(int(n_cmoe))]
        threads = defs.get("threads")
        if threads is not None:
            args += ["-t", str(int(threads))]
        if mlock:
            args += ["--mlock"]
        if draft_n > 0 and tgt.get("draft") and tgt["draft"].exists():
            args += ["-md", str(tgt["draft"]), "--spec-type", "draft-mtp",
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
        tgt = _avalon_target()
        if not Path(AVALON_BIN).exists():
            raise HTTPException(503, "Avalon build not found — build ~/avalon first (cmake --build build --target llama-server).")
        if not tgt["model"] or not tgt["model"].exists():
            raise HTTPException(404, "no Flash-Next pack detected in the model dirs — finish a download first or set avalon_model in config.json.")
        if p["exclusive"] and MAYA.status().get("running"):
            raise HTTPException(409, "Stop Maya first — Avalon needs its GPU VRAM.")
        if p["exclusive"] and self._gpu_conflict():
            raise HTTPException(409, "Stop the llama.cpp server first — this profile needs its VRAM.")
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
        args = self.build_args(profile, tune, target=tgt)
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
            if "inference-console" in cmd or "panel.py" in cmd:
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

# ---------------------------------------------------------------- maya (Project Maya)
MAYA_DIR = HOME / "project-maya"
MAYA_ROCM_PATH = HOME / "maya-rocm"
MAYA_STATE_PATH = APP_DIR / "maya-state.json"
MAYA_LOG_PATH = APP_DIR / "maya-runtime.log"
_MAYA_STATUS_TTL = 2.0

_MAYA_RE_LOAD = re.compile(r"\bloaded in\s+([\d.]+)\s*s\b", re.I)
_MAYA_RE_WARMUP = re.compile(r"\bwarmed up in\s+([\d.]+)\s*s\b", re.I)
_MAYA_RE_VRAM = re.compile(r"([\d.]+)\s*GB\s+of\s+VRAM", re.I)
_MAYA_RE_CTX = re.compile(r"\bcontext\s+(\d+)\b", re.I)
_MAYA_RE_EXPERTS = re.compile(r"(\d+)\s+of\s+(\d+)\s+routed experts in VRAM\s*\(([\d.]+)%\)", re.I)
_MAYA_RE_READY = re.compile(r"\bready:\s*https?://", re.I)
_MAYA_RE_HY4_SERVE = re.compile(
    r"prompt\s+\d+\s+tokens[^()]*\(([\d.]+)\s*tok/s\)[^()]*?generated\s+in\s+[\d.]+\s*ms\s*\(([\d.]+)\s*tok/s\)",
    re.I,
)
_MAYA_RE_ANSWERING = re.compile(r"\[maya\]\s+answering:\s*(\d+)\s+of\s+max\s+\d+\s+tokens,\s*([\d.]+)\s*tok/s", re.I)
_MAYA_RE_DONE = re.compile(r"\[maya\]\s+done:\s*(\d+)\s+tokens\s+in\s+[\d.]+\s*s\s*\(([\d.]+)\s*tok/s\)", re.I)
_MAYA_RE_GLM_PREFILL = re.compile(r"glm\s+prefill:.*?([\d.]+)\s*tok/s", re.I)
_MAYA_RE_GLM_STAT = re.compile(r"glm\s+stat:.*?([\d.]+)\s*tok/s", re.I)
_MAYA_RE_QUANT = re.compile(r"(maya-xs|iq\d+_[a-z0-9]+|q\d+_[a-z0-9_]+|fp16|bf16|f16|mxfp4)", re.I)


def _tail_file_lines(path: Path, max_lines: int = 250, max_bytes: int = 131072) -> list[str]:
    """Read up to the last `max_lines` lines from `path` without loading huge logs."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return lines[-max_lines:]
    except OSError:
        return []


def _mask_secret(key: str) -> str:
    if not key:
        return ""
    return (key[:4] + "…" + key[-4:]) if len(key) >= 8 else "(short key)"


def parse_maya_log(lines: list[str]) -> dict:
    """Extract live telemetry from Project Maya engine log lines (Hy4 and GLM)."""
    out = {
        "decode_tps": None,
        "prefill_tps": None,
        "vram_gb": None,
        "experts_vram": None,
        "experts_total": None,
        "experts_vram_pct": None,
        "context": None,
        "load_s": None,
        "warmup_s": None,
        "ready_seen": False,
        "active": False,
        "last_tokens": None,
    }
    for ln in lines:
        if not ln:
            continue
        m = _MAYA_RE_VRAM.search(ln)
        if m:
            out["vram_gb"] = float(m.group(1))
        m = _MAYA_RE_CTX.search(ln)
        if m:
            out["context"] = int(m.group(1))
        m = _MAYA_RE_LOAD.search(ln)
        if m:
            out["load_s"] = float(m.group(1))
        m = _MAYA_RE_WARMUP.search(ln)
        if m:
            out["warmup_s"] = float(m.group(1))
        m = _MAYA_RE_EXPERTS.search(ln)
        if m:
            out["experts_vram"] = int(m.group(1))
            out["experts_total"] = int(m.group(2))
            out["experts_vram_pct"] = float(m.group(3))
        if _MAYA_RE_READY.search(ln):
            out["ready_seen"] = True
        m = _MAYA_RE_HY4_SERVE.search(ln)
        if m:
            out["prefill_tps"] = round(float(m.group(1)), 2)
            out["decode_tps"] = round(float(m.group(2)), 2)
            continue
        m = _MAYA_RE_ANSWERING.search(ln)
        if m:
            out["last_tokens"] = int(m.group(1))
            out["decode_tps"] = round(float(m.group(2)), 2)
            out["active"] = True
            continue
        m = _MAYA_RE_DONE.search(ln)
        if m:
            out["last_tokens"] = int(m.group(1))
            out["decode_tps"] = round(float(m.group(2)), 2)
            out["active"] = False
            continue
        m = _MAYA_RE_GLM_PREFILL.search(ln)
        if m:
            out["prefill_tps"] = round(float(m.group(1)), 2)
            continue
        m = _MAYA_RE_GLM_STAT.search(ln)
        if m:
            out["decode_tps"] = round(float(m.group(1)), 2)
    return out


class MayaMgr:
    """Manages the local Project Maya server (Hy4-Flash, GLM-5.3-Flash, etc.)."""

    def __init__(self):
        self.proc = None
        self.info = {}
        self.logbuf = deque(maxlen=400)
        self._cache = {"at": 0.0, "val": None}
        self._load_state()

    @staticmethod
    def _maya_dir() -> Path:
        raw = CFG.get("maya_dir") or ""
        return Path(raw).expanduser() if raw else MAYA_DIR

    @staticmethod
    def _rocm_path() -> str:
        raw = CFG.get("maya_rocm_path") or ""
        return str(Path(raw).expanduser()) if raw else str(MAYA_ROCM_PATH)

    def _invalidate(self):
        self._cache["at"] = 0.0

    def _load_state(self):
        if not MAYA_STATE_PATH.exists():
            return
        try:
            st = json.loads(MAYA_STATE_PATH.read_text())
            pid = st.get("pid")
            if pid and _pid_alive(pid):
                self.proc = None
                self.info = {**st, "adopted": True}
                self.log(f"adopted running maya pid={pid} config={st.get('config_id')}")
        except Exception:
            pass

    def log(self, line: str):
        self.logbuf.append(f"[{time.strftime('%H:%M:%S')}] {_redact_cmd(line)}")

    @staticmethod
    def _read_config_raw(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _extract_ctx(raw: dict) -> int:
        args = raw.get("args") or []
        if isinstance(args, list):
            for i, tok in enumerate(args):
                s = str(tok)
                if s in ("--max-context", "--ctx-size", "-c") and i + 1 < len(args):
                    try:
                        return int(args[i + 1])
                    except ValueError:
                        pass
                if s.startswith("--max-context="):
                    try:
                        return int(s.split("=", 1)[1])
                    except ValueError:
                        pass
        for k in ("max_context", "ctx", "context"):
            if raw.get(k):
                try:
                    return int(raw[k])
                except (TypeError, ValueError):
                    pass
        return 131072

    @staticmethod
    def _extract_model_size(raw: dict, base_dir: Path) -> int:
        """Compute total byte size of the model file or directory referenced in config args."""
        args = raw.get("args") or []
        candidates = []
        if isinstance(args, list):
            for i, tok in enumerate(args):
                s = str(tok)
                if s in ("-m", "--model", "--model-dir") and i + 1 < len(args):
                    candidates.append(str(args[i + 1]))
                elif s.startswith("--model="):
                    candidates.append(s.split("=", 1)[1])
                elif not s.startswith("-") and ("/" in s or s.endswith((".gguf", ".safetensors", ".bin"))):
                    candidates.append(s)
        for cand in candidates:
            p = Path(cand).expanduser()
            if not p.is_absolute():
                p = base_dir / p
            if not p.exists():
                continue
            try:
                if p.is_file():
                    return p.stat().st_size
                if p.is_dir():
                    total = 0
                    for f in p.rglob("*"):
                        if f.is_file():
                            try:
                                total += f.stat().st_size
                            except OSError:
                                pass
                    if total > 0:
                        return total
            except OSError:
                continue
        return 0

    def _resolve_log_path(self, raw: dict, cfg_path: Path) -> Path:
        base_dir = self._maya_dir()
        lp = raw.get("log")
        if lp:
            p = Path(str(lp)).expanduser()
            return p if p.is_absolute() else (base_dir / p)
        return base_dir / f"{cfg_path.stem}.log"

    def _public_config(self, cfg_path: Path, raw: dict) -> dict:
        base_dir = self._maya_dir()
        exe = str(raw.get("exe") or "")
        exe_path = Path(exe).expanduser()
        if exe and not exe_path.is_absolute():
            exe_path = base_dir / exe_path
        args = raw.get("args") or []
        args_str = " ".join(str(a) for a in args) if isinstance(args, list) else str(args)
        engine = "hy4" if "hy4" in exe.lower() or "hy4" in cfg_path.stem.lower() else "glm"
        # Never default a Maya config to 8080 (latency-speedtest owns 8080).
        port = int(raw.get("port") or 8098)
        quant_m = _MAYA_RE_QUANT.search(cfg_path.stem) or _MAYA_RE_QUANT.search(args_str)
        quant = quant_m.group(1) if quant_m else ""
        size_b = self._extract_model_size(raw, base_dir)
        key = str(raw.get("api_key") or "")
        log_path = self._resolve_log_path(raw, cfg_path)
        return {
            "id": cfg_path.stem,
            "filename": cfg_path.name,
            "path": str(cfg_path),
            "model_name": str(raw.get("model_name") or cfg_path.stem),
            "engine": engine,
            "exe": exe,
            "quant": quant,
            "ctx": self._extract_ctx(raw),
            "mtp": "--mtp" in args if isinstance(args, list) else ("--mtp" in args_str),
            "port": port,
            "host": str(raw.get("host") or "127.0.0.1"),
            "log": str(log_path),
            "sampling": raw.get("sampling") if isinstance(raw.get("sampling"), dict) else {},
            "reasoning_effort": raw.get("reasoning_effort"),
            "key_set": bool(key),
            "key_masked": _mask_secret(key),
            "dir_size_bytes": size_b,
            "dir_size_h": human_size(size_b) if size_b else "",
            "available": bool(exe) and exe_path.exists(),
            "concurrency_note": "1 request at a time" if engine == "hy4" else "",
        }

    def discover_configs(self) -> list[dict]:
        """Discover Project Maya JSON configs in MAYA_DIR, excluding *.panel-orig.json backups."""
        base_dir = self._maya_dir()
        if not base_dir.is_dir():
            return []
        out = []
        for p in sorted(base_dir.glob("*.json")):
            name = p.name
            if name.endswith(".panel-orig.json") or name.endswith(".orig.json") or name.endswith(".bak.json"):
                continue
            raw = self._read_config_raw(p)
            if not raw:
                continue
            if not (raw.get("exe") or raw.get("model_name")):
                continue
            out.append(self._public_config(p, raw))
        return out

    def _find_config(self, ref: str) -> tuple[Path, dict, dict] | None:
        """Resolve a config by full path, filename, or stem id. Returns (path, raw, public)."""
        ref_s = str(ref or "").strip()
        if not ref_s:
            return None
        base_dir = self._maya_dir()
        for pub in self.discover_configs():
            if ref_s in (pub["path"], pub["id"], pub["filename"]):
                p = Path(pub["path"])
                raw = self._read_config_raw(p)
                if raw:
                    return p, raw, pub
        # Allow an explicit path inside base_dir if valid
        cand = Path(ref_s).expanduser()
        if not cand.is_absolute():
            cand = base_dir / cand
        if cand.is_file() and not cand.name.endswith(".panel-orig.json"):
            raw = self._read_config_raw(cand)
            if raw and (raw.get("exe") or raw.get("model_name")):
                return cand, raw, self._public_config(cand, raw)
        return None

    @staticmethod
    def _port_open(port: int, timeout: float = 0.15) -> bool:
        try:
            with socket.socket() as s:
                s.settimeout(timeout)
                return s.connect_ex(("127.0.0.1", int(port))) == 0
        except Exception:
            return False

    @staticmethod
    def _probe_models(port: int, api_key: str, timeout: float = 0.8) -> tuple[bool, str | None]:
        """Key-authed GET /v1/models probe. Returns (ready, alias)."""
        if not port or not MayaMgr._port_open(port):
            return False, None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/v1/models", headers=headers, timeout=timeout)
            if r.status_code == 200:
                alias = None
                try:
                    data = r.json().get("data") or []
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        alias = data[0].get("id")
                except Exception:
                    pass
                return True, alias
            return False, None
        except Exception:
            return False, None

    @staticmethod
    def _find_proc_pid(port: int | None = None) -> int | None:
        """Best-effort PID lookup on Linux /proc for adopted Maya processes."""
        if not Path("/proc").is_dir():
            return None
        fallback = None
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except Exception:
                continue
            if "inference-console" in cmd or "panel.py" in cmd:
                continue
            if "maya.py" in cmd or "build-hip/maya" in cmd:
                pid = int(d.name)
                if port and str(port) in cmd:
                    return pid
                if fallback is None:
                    fallback = pid
        return fallback

    def _status_now(self) -> dict:
        st = self.info
        if st:
            pid = st.get("pid")
            alive = (self.proc is not None and self.proc.poll() is None) or _pid_alive(pid)
            port = int(st.get("port") or 0)
            cfg_path = Path(st["config_path"]) if st.get("config_path") else None
            raw = self._read_config_raw(cfg_path) if cfg_path and cfg_path.exists() else {}
            raw = raw or {}
            key = str(raw.get("api_key") or "")
            ready, alias = self._probe_models(port, key)
            if alive or ready:
                log_path = Path(st.get("log") or "") if st.get("log") else None
                tel = parse_maya_log(_tail_file_lines(log_path, 200)) if (log_path and log_path.exists()) else parse_maya_log([])
                return {
                    "running": True,
                    "state": "ready" if ready else "loading",
                    "pid": pid if alive else self._find_proc_pid(port),
                    "config_id": st.get("config_id"),
                    "config_path": st.get("config_path"),
                    "model_name": st.get("model_name"),
                    "alias": alias or st.get("model_name") or "maya",
                    "engine": st.get("engine") or "hy4",
                    "port": port,
                    "ctx": int(st.get("ctx") or self._extract_ctx(raw)),
                    "log": st.get("log") or "",
                    "key_set": bool(key),
                    "key_masked": _mask_secret(key),
                    "uptime": int(time.time() - st.get("started_at", time.time())),
                    "cmd": st.get("cmd", ""),
                    "adopted": bool(st.get("adopted", False)),
                    "telemetry": tel,
                }
            # Tracked process died and port is closed
            self.proc = None
            self.info = {}
            MAYA_STATE_PATH.unlink(missing_ok=True)

        # Port-probe adoption: check discovered configs for an already-running Maya server.
        for pub in self.discover_configs():
            cfg_path = Path(pub["path"])
            raw = self._read_config_raw(cfg_path) or {}
            key = str(raw.get("api_key") or "")
            ready, alias = self._probe_models(pub["port"], key)
            if ready:
                pid = self._find_proc_pid(pub["port"])
                log_path = Path(pub["log"])
                tel = parse_maya_log(_tail_file_lines(log_path, 200)) if log_path.exists() else parse_maya_log([])
                self.info = {
                    "pid": pid,
                    "config_id": pub["id"],
                    "config_path": pub["path"],
                    "model_name": pub["model_name"],
                    "engine": pub["engine"],
                    "port": pub["port"],
                    "ctx": pub["ctx"],
                    "log": pub["log"],
                    "started_at": time.time(),
                    "adopted": True,
                    "cmd": "",
                }
                return {
                    "running": True,
                    "state": "ready",
                    "pid": pid,
                    "config_id": pub["id"],
                    "config_path": pub["path"],
                    "model_name": pub["model_name"],
                    "alias": alias or pub["model_name"],
                    "engine": pub["engine"],
                    "port": pub["port"],
                    "ctx": pub["ctx"],
                    "log": pub["log"],
                    "key_set": pub["key_set"],
                    "key_masked": pub["key_masked"],
                    "uptime": 0,
                    "cmd": "",
                    "adopted": True,
                    "telemetry": tel,
                }
        return {"running": False, "state": "stopped", "telemetry": parse_maya_log([])}

    def status(self, refresh: bool = False) -> dict:
        now = time.time()
        if not refresh and self._cache["val"] is not None and (now - self._cache["at"]) < _MAYA_STATUS_TTL:
            return self._cache["val"]
        val = self._status_now()
        self._cache["val"] = val
        self._cache["at"] = now
        return val

    def active_api_key(self, config_ref: str | None = None) -> str:
        """Server-side only: return raw api_key for the active or specified Maya config."""
        ref = config_ref or self.info.get("config_path")
        if not ref:
            st = self.status()
            ref = st.get("config_path")
        if not ref:
            return ""
        found = self._find_config(ref)
        if not found:
            return ""
        return str(found[1].get("api_key") or "")

    @staticmethod
    def _gpu_conflict() -> bool:
        return AvalonManager._gpu_conflict()

    def build_start_spec(self, cfg_path: Path, pub: dict) -> tuple[list[str], dict, Path]:
        base_dir = self._maya_dir()
        venv_py = base_dir / ".venv/bin/python"
        launcher = base_dir / "maya.py"
        argv = [str(venv_py), str(launcher), "--config", str(cfg_path), "--port", str(pub["port"])]
        env = dict(os.environ)
        env["ROCM_PATH"] = self._rocm_path()
        return argv, env, base_dir

    def start(self, config_ref: str) -> dict:
        found = self._find_config(config_ref)
        if not found:
            raise HTTPException(404, f"Maya config not found: {config_ref}")
        cfg_path, raw, pub = found
        if pub["port"] == 8080:
            raise HTTPException(400, "Refusing port 8080 — reserved by latency-speedtest. Edit the config port to 8098/8099.")

        # Exclusivity: refuse a second Maya config while one serves.
        cur = self.status(refresh=True)
        if cur.get("running"):
            raise HTTPException(
                409,
                f"Maya is already running ({cur.get('model_name')} on :{cur.get('port')}) — stop it or use Switch.",
            )

        # GPU-conflict guard against other engines & VRAM.
        if MGR.status().get("running"):
            raise HTTPException(409, "Stop the llama.cpp server first — Maya needs the GPU VRAM.")
        if SMGR.status().get("running"):
            raise HTTPException(409, "Stop Avalon first — Maya needs the GPU VRAM.")
        if strata_status().get("running"):
            raise HTTPException(409, "Stop Strata first — Maya needs the GPU VRAM.")
        if self._gpu_conflict():
            raise HTTPException(409, "Another engine is holding GPU VRAM — stop it first.")

        argv, env, base_dir = self.build_start_spec(cfg_path, pub)
        if not Path(argv[0]).exists():
            raise HTTPException(503, f"Maya virtualenv python not found: {argv[0]}")
        if not Path(argv[1]).exists():
            raise HTTPException(503, f"Maya launcher not found: {argv[1]}")

        cmd_str = _redact_cmd(" ".join(argv))
        self.log(f"starting maya {pub['model_name']} ({pub['id']}) on :{pub['port']}: {cmd_str}")
        MAYA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(MAYA_LOG_PATH, "a", encoding="utf-8", errors="replace")
        try:
            self.proc = subprocess.Popen(
                argv,
                cwd=str(base_dir),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_fh.close()

        self.info = {
            "pid": self.proc.pid,
            "config_id": pub["id"],
            "config_path": str(cfg_path),
            "model_name": pub["model_name"],
            "engine": pub["engine"],
            "port": pub["port"],
            "ctx": pub["ctx"],
            "log": pub["log"],
            "started_at": time.time(),
            "adopted": False,
            "cmd": cmd_str,
        }
        self._save_state()
        self._invalidate()
        return self.status(refresh=True)

    @staticmethod
    def _kill_orphans(port: int | None = None) -> bool:
        """Terminate any untracked Maya python/engine processes on Linux /proc."""
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
            if "inference-console" in cmd or "panel.py" in cmd:
                continue
            if "maya.py" in cmd or "build-hip/maya" in cmd:
                victims.append(int(d.name))
        for v in victims:
            try:
                os.killpg(os.getpgid(v), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(v, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        for _ in range(80):
            port_free = (not port) or (not MayaMgr._port_open(port))
            if port_free and not any(_pid_alive(v) for v in victims):
                break
            time.sleep(0.1)
        for v in victims:
            if _pid_alive(v):
                try:
                    os.kill(v, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        for _ in range(30):
            if (not port) or (not MayaMgr._port_open(port)):
                break
            time.sleep(0.1)
        return bool(victims)

    def stop(self) -> dict:
        cur = self.status(refresh=True)
        st = self.info
        pid = st.get("pid") or cur.get("pid")
        port = st.get("port") or cur.get("port")
        killed = False
        if pid and _pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            for _ in range(100):
                port_free = (not port) or (not self._port_open(port))
                if not _pid_alive(pid) and port_free and not self._gpu_conflict():
                    killed = True
                    break
                time.sleep(0.1)
            if _pid_alive(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
            killed = True
        if port and self._port_open(port):
            if self._kill_orphans(port):
                killed = True
        self.proc = None
        self.info = {}
        MAYA_STATE_PATH.unlink(missing_ok=True)
        self._invalidate()
        stopped = (not port) or (not self._port_open(port))
        self.log(f"stopped maya (pid {pid})" if killed else ("maya not running" if stopped else f"stop failed: port {port} still busy"))
        return {"stopped": stopped}

    def switch(self, config_ref: str) -> dict:
        found = self._find_config(config_ref)
        if not found:
            raise HTTPException(404, f"Maya config not found: {config_ref}")
        if self.status(refresh=True).get("running"):
            res = self.stop()
            if not res.get("stopped"):
                raise HTTPException(503, "Failed to stop running Maya server before switching.")
        return self.start(str(found[0]))

    def tail_log(self, max_lines: int = 250) -> dict:
        st = self.status()
        log_path_s = st.get("log") or self.info.get("log") or ""
        lines: list[str] = []
        source_path = ""
        if log_path_s:
            lp = Path(log_path_s)
            if lp.exists():
                lines = _tail_file_lines(lp, max_lines=max_lines)
                source_path = str(lp)
        if not lines and MAYA_LOG_PATH.exists():
            lines = _tail_file_lines(MAYA_LOG_PATH, max_lines=max_lines)
            source_path = str(MAYA_LOG_PATH)
        if not lines and self.logbuf:
            lines = list(self.logbuf)[-max_lines:]
        key = self.active_api_key()
        text = "\n".join(lines)
        if key and len(key) >= 6:
            text = text.replace(key, _mask_secret(key))
        return {"log": _redact_cmd(text), "path": source_path}

    def _save_state(self):
        MAYA_STATE_PATH.write_text(json.dumps(self.info))


MAYA = MayaMgr()

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

app = FastAPI(title="Inference Console")

# ---------------------------------------------------------------- auth (sudo password)
# Password is validated live against sudo (same credential as terminal sudo).
# Every failed check counts toward PAM faillock (deny=10, unlock=120s), so the
# web limiter below must trip first: 5 failures / 5 min per IP.
SESSION_COOKIE = "ic_sid"
SESSION_TTL = 7 * 24 * 3600
_sessions: dict = {}   # token -> expiry (unix time)
_fails: dict = {}      # ip -> deque(failure unix times)

SESSIONS_PATH = APP_DIR / "sessions.json"

def _load_sessions():
    """Sessions are memory-only, so an in-place restart would sign the operator out.
    Mirror them to a 0600 file and reload on boot; expired tokens are dropped."""
    now = time.time()
    raw = _read_json(SESSIONS_PATH, {})
    if not isinstance(raw, dict):
        return {}
    out = {}
    for tok, exp in raw.items():
        try:
            if float(exp) > now:
                out[str(tok)] = float(exp)
        except (TypeError, ValueError):
            continue
    return out

def _save_sessions():
    try:
        tmp = SESSIONS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({k: v for k, v in _sessions.items() if v > time.time()}))
        os.chmod(tmp, 0o600)
        tmp.replace(SESSIONS_PATH)
    except OSError:
        pass  # a session mirror that cannot be written must not break sign-in

_sessions.update(_load_sessions())

LOGIN_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8"><title>Inference Console — Sign in</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<script>try{const t=localStorage.getItem('ic.theme');if(t==='light'||t==='dark'){document.documentElement.dataset.theme=t;document.querySelector('meta[name="color-scheme"]').content=t}}catch(_){}</script>
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
  <div><b>Inference Console</b><span class="sub">Local inference &amp; telemetry console</span></div>
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
 $('meta[name="color-scheme"]').content=localStorage.getItem('ic.theme')||'light dark';
 $('#theme').innerHTML=t==='dark'?SUN:MOON;
}
applyTheme(localStorage.getItem('ic.theme')||sysScheme());
$('#theme').onclick=()=>{
 const cur=document.documentElement.dataset.theme;
 const next=cur==='dark'?'light':'dark';
 if(next===sysScheme())localStorage.removeItem('ic.theme');
 else localStorage.setItem('ic.theme',next);
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
    _save_sessions()
    resp = JSONResponse({"ok": True})
    resp.set_cookie(SESSION_COOKIE, tok, max_age=SESSION_TTL, httponly=True,
                    samesite="lax", path="/")
    return resp

@app.post("/api/logout")
def api_logout(request: Request):
    _sessions.pop(request.cookies.get(SESSION_COOKIE, ""), None)
    _save_sessions()
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
                _avalon_target_invalidate()
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
        _avalon_target_invalidate()
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
        count = sum(1 for f in target.rglob("*") if f.is_file())
        shutil.rmtree(target)
    else:
        # A split shard deletes its whole pack: llama.cpp treats the group as one model,
        # and leaving the remaining shards behind would show a broken entry.
        size = 0
        victims = [target]
        sm = SPLIT_RE.search(target.stem)
        if sm:
            base = target.stem[: sm.start()]
            victims = sorted(
                f for f in target.parent.iterdir()
                if f.is_file() and f.suffix.lower() == ".gguf"
                and (m := SPLIT_RE.search(f.stem)) and f.stem[: m.start()] == base
            )
        for f in victims:
            size += f.stat().st_size
            f.unlink(missing_ok=True)
        count = len(victims)
    _model_cache.clear()
    MGR.log(f"deleted {target.name} ({human_size(size)} freed)")
    return {"deleted": str(target), "freed_h": human_size(size), "count": count}

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
    draft: str = ""
    draft_max: int = 16
    draft_min: int = 0
    ot: str = ""
    tensor_split: str = ""
    tbatch: int | None = None
    keep: int | None = None
    jinja: str = "on"
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
            draft=b.draft, draft_max=b.draft_max, draft_min=b.draft_min,
            ot=b.ot, tensor_split=b.tensor_split, tbatch=b.tbatch, keep=b.keep,
            jinja=b.jinja, port=int(CFG.get("port_base", 8830) or 8830),
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
            ncmoe=rec["ncmoe"], draft=rec.get("draft", ""), draft_max=rec.get("draft_max", 16),
            jinja=rec.get("jinja", "on"), port=int(CFG.get("port_base", 8830) or 8830),
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
    model_path = srv.get("model", "") if source == "panel" else str(_avalon_target().get("model") or "")
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
@app.get("/api/avalon/targets")
def api_avalon_targets():
    packs = _avalon_scan()
    pinned = CFG.get("avalon_model") or ""
    known = {p["path"] for p in packs}
    out = [{"path": "", "base": "Auto · newest complete pack", "size_h": "", "shards": 0, "draft": None, "current": not pinned}]
    for p in packs:
        out.append({"path": p["path"], "base": p["base"], "size_h": p["size_h"], "shards": p["shards"],
                    "draft": p["draft"], "current": pinned == p["path"]})
    if pinned and pinned not in known:
        bp = Path(pinned)
        out.append({"path": pinned, "base": SPLIT_RE.sub("", bp.stem) + " (pinned)", "size_h": "", "shards": 0,
                    "draft": None, "current": True})
    return {"targets": out, "has_draft": bool(_avalon_target().get("draft"))}

class AvalonTargetBody(BaseModel):
    model: str  # "" = back to auto

@app.post("/api/avalon/target")
def api_avalon_target_set(b: AvalonTargetBody):
    if b.model:
        p = Path(b.model).expanduser()
        if not p.is_file():
            raise HTTPException(404, f"model not found: {b.model}")
        CFG["avalon_model"] = str(p)
    else:
        CFG["avalon_model"] = ""
    _save_config_secure()
    _avalon_target_invalidate()
    tgt = _avalon_target()
    return {"ok": True, "model": CFG["avalon_model"], "base": tgt["base"],
            "has_draft": bool(tgt.get("draft"))}

@app.get("/api/avalon/profiles")
def api_avalon_profiles():
    out = []
    saved_all = CFG.get("avalon_tune") or {}
    tgt = _avalon_target()
    tgt_label = tgt["base"] if tgt["model"] else "no Flash-Next pack detected"
    for key, p in AVALON_PROFILES.items():
        defs = dict(p.get("defaults", {}))
        if key in saved_all and isinstance(saved_all[key], dict):
            defs.update(saved_all[key])
        cmd_str = " ".join(SMGR.build_args(key))
        out.append({
            "key": key, "label": p["label"],
            "desc": p["desc"].replace("125B MoE IQ3KS", tgt_label),
            "speed": p["speed"], "exclusive": p["exclusive"],
            "draft": p.get("draft", False) and bool(tgt.get("draft")),
            "model": tgt_label, "has_draft": bool(tgt.get("draft")),
            "defaults": defs,
            # What the user explicitly saved (empty = inherit profile defaults) — the tune
            # form restores this, not `defaults`, so unset fields stay unset.
            "saved": saved_all.get(key) if isinstance(saved_all.get(key), dict) else {},
            "cmd": _redact_cmd(cmd_str),
            "cmd_preview": _redact_cmd(cmd_str),
            "available": Path(AVALON_BIN).exists() and bool(tgt.get("model")) and tgt["model"].exists(),
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
    device: str | None = None
    ngl: int | None = None       # unset (absent) = auto-fit (-fit on); set = -ngl N -fit off
    n_cmoe: int | None = None
    fa: str | None = None
    threads: int | None = None
    mlock: bool | None = None
    preview: bool = False

@app.post("/api/avalon/tune")
def api_avalon_tune(b: AvalonTuneBody):
    if b.profile not in AVALON_PROFILES:
        raise HTTPException(400, "unknown profile")
    tune_dict = dict(b.tune or {})
    # The select-backed fields use 0 to mean "profile default"; draft_n/ubd are free number
    # inputs where an empty box is unset and a typed 0 means "drafting off" — so 0 is a value.
    for k in ("ctx", "batch", "ubatch", "ctk", "ctv", "draft_n", "ubd", "parallel",
              "device", "ngl", "n_cmoe", "fa", "threads", "mlock"):
        val = getattr(b, k, None)
        if val is None or val == "":
            continue
        # 0 is "unset" for the count fields where 0 equals the default anyway.
        if k in ("ctx", "parallel", "n_cmoe", "threads") and val == 0:
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

# ---------------------------------------------------------------- avalon snapshots
# A snapshot freezes one resolved launch (profile + exact tune) for one-click
# restarts. Launch replays the stored tune through build_args, so the command is
# byte-identical as long as the Avalon binary and model stay the same.

class AvalonSnapBody(BaseModel):
    name: str
    profile: str | None = None
    tune: dict | None = None

class AvalonSnapLaunchBody(BaseModel):
    id: str

def _snap_list() -> list:
    return CFG.setdefault("avalon_snapshots", [])

@app.get("/api/avalon/snapshots")
def api_avalon_snapshots():
    return {"snapshots": _snap_list()}

@app.post("/api/avalon/snapshots")
def api_avalon_snapshot_save(b: AvalonSnapBody):
    prof = b.profile or SMGR.status().get("profile") or "flagship"
    if prof not in AVALON_PROFILES:
        raise HTTPException(400, f"unknown avalon profile: {prof}")
    tune = b.tune if b.tune is not None else (CFG.get("avalon_tune") or {}).get(prof)
    cmd_str = " ".join(SMGR.build_args(prof, tune))
    snap = {"id": uuid.uuid4().hex[:8], "name": (b.name or "").strip()[:80] or f"{prof} snapshot",
            "profile": prof, "tune": tune, "created": int(time.time()),
            "cmd": _redact_cmd(cmd_str)}
    snaps = _snap_list()
    snaps.insert(0, snap)
    CFG["avalon_snapshots"] = snaps[:20]
    _save_config_secure()
    return snap

@app.post("/api/avalon/snapshots/launch")
def api_avalon_snapshot_launch(b: AvalonSnapLaunchBody):
    snap = next((s for s in _snap_list() if s.get("id") == b.id), None)
    if not snap:
        raise HTTPException(404, "snapshot not found")
    return SMGR.start(snap["profile"], snap.get("tune"))

@app.delete("/api/avalon/snapshots/{sid}")
def api_avalon_snapshot_delete(sid: str):
    snaps = _snap_list()
    kept = [s for s in snaps if s.get("id") != sid]
    if len(kept) == len(snaps):
        raise HTTPException(404, "snapshot not found")
    CFG["avalon_snapshots"] = kept
    _save_config_secure()
    return {"ok": True}

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
PANEL_PROVIDER_ID = "inference-console"

def _load_agents_state():
    st = _read_json(AGENTS_STATE_PATH, {})
    if not isinstance(st, dict):
        st = {}
    if not isinstance(st.get("agents"), dict):
        st["agents"] = {}
    return st

def _save_agents_state(st):
    """Atomic, and never fatal: the agent configs are already written by the time this runs,
    so a read-only state dir must not turn a successful sync into a 500."""
    try:
        _write_json(AGENTS_STATE_PATH, st)
    except OSError as e:
        print(f"agent registry not saved: {e}", flush=True)

def _proc_ctx(pid):
    try:
        argv = [a.decode() for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
        if "-c" in argv:
            return int(argv[argv.index("-c") + 1])
    except Exception:
        pass
    return 8192

def _server_facts(port, timeout: float = 1.5):
    """(model id, n_ctx) as the running server reports them over HTTP. /proc only exists on
    Linux, so the old cmdline probe told every agent `-c 8192` on macOS no matter what the
    server was really loaded with; /props answers the same question on every platform."""
    alias = nctx = None
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=timeout)
        if r.status_code == 200:
            data = (r.json() or {}).get("data") or []
            if data and isinstance(data[0], dict) and data[0].get("id"):
                alias = str(data[0]["id"])
    except Exception:
        pass
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/props", timeout=timeout)
        if r.status_code == 200:
            j = r.json() or {}
            nctx = (j.get("default_generation_settings") or {}).get("n_ctx") or j.get("n_ctx")
            nctx = int(nctx) if nctx else None
    except Exception:
        nctx = None
    if alias and ("/" in alias or "\\" in alias or alias.lower().endswith(".gguf")):
        alias = Path(alias.replace("\\", "/")).name.removesuffix(".gguf")
        alias = SPLIT_RE.sub("", alias) or None
    return alias, nctx

# ==================== STRATA (Niko1221/Strata) ====================
# Strata is a separate inference engine (github.com/Niko1221/Strata) serving the same
# Qwen3.8-Flash-Next family as the Avalon tab, but unlike Avalon it is a product: it ships
# its own desktop app, terminal monitor, model downloader, preset store and key management.
# So the panel drives it through its documented CLI (`strata launch --output json`,
# `--status`, `--stop`, `--list-presets`, `--plan`) and never parses Strata internals such
# as launch/<name>.json or the strata_control.management bridge — that bridge documents
# itself as an unaudited preview still under integration work, and its plan/execute
# preconditions belong to Strata's own Desktop host.
#
# Strata is not installed on this deployment yet. Every path therefore degrades to an
# explicit "not installed" state carrying the install command, and anything the docs do not
# define is returned as raw CLI output rather than mapped through invented field names.

STRATA_INSTALL_CMD = ("curl -fsSL --connect-timeout 15 --max-time 120 "
                      "https://github.com/Niko1221/Strata/releases/latest/download/install.sh "
                      "-o install.sh && sh install.sh")

STRATA_LOG = deque(maxlen=200)

_STRATA_TTL = 5.0
_strata_cache = {"ts": 0.0, "data": None}

def strata_log(line: str):
    STRATA_LOG.append(f"[{time.strftime('%H:%M:%S')}] {_redact_cmd(line)}")

def strata_bin() -> str | None:
    """Resolve the strata launcher. A broken symlink (an uninstalled app whose helper link
    survives) reads as absent, which is the correct answer here."""
    for cand in (CFG.get("strata_bin"), os.environ.get("STRATA_BIN"), shutil.which("strata"),
                 HOME / ".local/bin/strata", HOME / ".strata/bin/strata",
                 "/usr/local/bin/strata", "/opt/strata/bin/strata"):
        if cand and Path(str(cand)).exists():
            return str(cand)
    return None

def _strata_run(args: list[str], timeout: int = 25) -> dict:
    """Run the Strata CLI. Never raises. `json` is set only when stdout really was JSON, so
    a caller can tell a parsed result from a text one instead of assuming."""
    b = strata_bin()
    if not b:
        return {"ok": False, "rc": None, "installed": False, "json": None,
                "stdout": "", "stderr": "", "error": "strata CLI not found",
                "install_cmd": STRATA_INSTALL_CMD}
    try:
        p = subprocess.run([b] + args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": None, "installed": True, "json": None, "stdout": "",
                "stderr": "", "error": f"strata timed out after {timeout}s"}
    except OSError as e:
        return {"ok": False, "rc": None, "installed": True, "json": None, "stdout": "",
                "stderr": "", "error": f"could not run strata: {e}"}
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    parsed = None
    # Strata can append a human-readable summary line after the JSON payload, which
    # breaks json.loads on the whole stdout. Decode the first JSON value instead;
    # raw_decode stops at the end of that value and ignores the trailing text.
    dec = json.JSONDecoder()
    for idx, ch in enumerate(out):
        if ch in "{[":
            try:
                val, _ = dec.raw_decode(out[idx:])
            except Exception:
                continue
            if isinstance(val, (dict, list)):
                parsed = val
                break
    return {"ok": p.returncode == 0, "rc": p.returncode, "installed": True,
            "json": parsed if isinstance(parsed, (dict, list)) else None,
            "stdout": out[:6000], "stderr": err[:2000], "error": None}

class StrataLaunchBody(BaseModel):
    preset: str | None = None
    family: str | None = None
    model: str | None = None
    model_path: str | None = None
    context: str | None = None          # Strata takes these as words, e.g. "128k"
    kv: str | None = None               # int8 | q4_0 | k8v4 | fp16
    parallel: int | None = None
    vision: str | None = None           # on | off | gpu | cpu
    mtp: str | None = None              # on | off (draft layer)
    drafting: str | None = None         # auto | mtp | dflash2 | off
    gpu: str | None = None              # e.g. "0,1"
    tune: str | None = None             # quick | thorough | skip
    retune: bool = False
    access: str | None = None           # local | lan | cloudflare | tailscale
    port: int | None = None
    no_api_key: bool = False
    new_key: bool = False
    save_preset: str | None = None
    restart: bool = False
    set_: list[str] | None = None       # --set key=value pairs
    plan: bool = False

def strata_build_args(b: "StrataLaunchBody", plan: bool) -> list[str]:
    """Translate the form into `strata launch` flags. Only documented flags are emitted, and
    an unset field emits nothing so Strata's own defaults and presets stay in charge."""
    args = ["launch"]
    if b.preset:
        args += ["--preset", b.preset]
    if b.family:
        args += ["--family", b.family]
    if b.model:
        args += ["--model", b.model]
    if b.model_path:
        args += ["--model-path", b.model_path]
    if b.context:
        args += ["--context", b.context]
    if b.kv:
        args += ["--kv", b.kv]
    if b.parallel:
        args += ["--parallel", str(b.parallel)]
    if b.vision:
        args += ["--vision", b.vision]
    if b.mtp:
        args += ["--mtp", b.mtp]
    if b.drafting:
        args += ["--drafting", b.drafting]
    if b.gpu:
        args += ["--gpu", b.gpu]
    if b.tune:
        args += ["--tune", b.tune]
    if b.retune:
        args += ["--retune"]
    if b.access:
        args += ["--access", b.access]
    if b.port:
        args += ["--port", str(b.port)]
    if b.no_api_key:
        args += ["--no-api-key"]
    if b.new_key:
        args += ["--new-key"]
    if b.save_preset:
        args += ["--save-preset", b.save_preset]
    for kv in (b.set_ or []):
        if kv and "=" in kv:
            args += ["--set", kv]
    if b.restart:
        args += ["--restart"]
    if plan:
        args += ["--plan", "--output", "json"]
    else:
        args += ["--yes", "--no-tui", "--output", "json"]
    return args

def _strata_remember(addr: dict):
    """Persist the address a launch reported, so liveness and agent sync can find the
    server after a panel restart. The key is never stored in the clear here."""
    st = dict(CFG.get("strata_last") or {})
    for k in ("address", "port", "model", "profile", "preset"):
        if addr.get(k) not in (None, ""):
            st[k] = addr[k]
    if st:
        CFG["strata_last"] = st
        _save_config_secure()

def _parse_ctx_words(v) -> int | None:
    """'128k' / '8192' / '1m' -> token count, matching how Strata spells contexts."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([km]?)", str(v).strip().lower())
    if not m:
        return None
    n = float(m.group(1)) * (1024 if m.group(2) == "k" else 1024 * 1024 if m.group(2) == "m" else 1)
    return int(n)

def _strata_port() -> int | None:
    last = CFG.get("strata_last") or {}
    for cand in (last.get("port"), CFG.get("strata_port")):
        try:
            n = int(cand)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return None

def strata_health(port: int | None = None, timeout: float = 1.5) -> str:
    """Liveness from the HTTP surface, which is documented (OpenAI-compatible), rather than
    from a CLI status shape the docs do not define. /health first, then /v1/models."""
    port = port or _strata_port()
    if not port:
        return "unknown"
    for path in ("/health", "/v1/models"):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=timeout)
            if r.status_code in (200, 503):
                return "ready" if r.status_code == 200 else "loading"
        except Exception:
            continue
    return "down"

def strata_status(refresh: bool = False) -> dict:
    """Memoized: _current_llm() and _chat_target() run on the 2.5s header poll, and each
    would otherwise fork `strata launch --status` — the same amplification that vitals() had.
    Only the CLI result is cached; liveness is re-checked over HTTP every call, which is a
    cheap loopback GET, so a server stopped outside the panel is still noticed at once."""
    now = time.time()
    cached = _strata_cache["data"]
    if not refresh and cached is not None and now - _strata_cache["ts"] < _STRATA_TTL:
        out = dict(cached)
        out["health"] = strata_health(out.get("port"))
        out["running"] = out["running"] and out["health"] in ("ready", "loading")
        return out
    st = _strata_status_now()
    _strata_cache["ts"] = now
    _strata_cache["data"] = st
    return dict(st)

def _strata_status_now() -> dict:
    b = strata_bin()
    if not b:
        return {"installed": False, "running": False, "health": "unknown",
                "install_cmd": STRATA_INSTALL_CMD,
                "note": "Strata CLI not found. Install it, or set strata_bin in config."}
    r = _strata_run(["launch", "--status", "--output", "json"], timeout=15)
    port = _strata_port()
    out = {"installed": True, "bin": _redact_cmd(b), "running": False,
           "health": strata_health(port), "port": port,
           "parsed": isinstance(r["json"], dict), "error": r["error"],
           "raw": r["stdout"] or r["stderr"] or ""}
    j = r["json"]
    if isinstance(j, dict):
        # Only lift keys we can name. Anything else stays in `raw` for the operator to read.
        for k in ("model", "profile", "preset", "address", "version"):
            if j.get(k) not in (None, ""):
                out[k] = j[k]
        p = j.get("port")
        try:
            if p and int(p) > 0:
                out["port"] = int(p)
                out["health"] = strata_health(out["port"])
        except (TypeError, ValueError):
            pass
        runs = j.get("running")
        if isinstance(runs, list) and runs and isinstance(runs[0], dict):
            # Strata >= 0.1.62 `--status --output json` shape: {"ok": true, "running":
            # [{name, profile, state, port, url, ...}]} instead of the older flat
            # serving/address shape. Same trust level: only lift keys we can name.
            r0 = runs[0]
            if out.get("model") in (None, "") and r0.get("name"):
                out["model"] = r0["name"]
            if out.get("preset") in (None, "") and r0.get("profile"):
                out["preset"] = r0["profile"]
            if out.get("address") in (None, "") and r0.get("url"):
                out["address"] = r0["url"]
            try:
                rp = int(r0.get("port") or 0)
                if rp > 0:
                    out["port"] = rp
                    out["health"] = strata_health(rp)
            except (TypeError, ValueError):
                pass
            if r0.get("state") in ("ready", "loading"):
                out["running"] = True
        serving = j.get("serving")
        if isinstance(serving, bool):
            out["running"] = serving
        elif serving is None and out["health"] in ("ready", "loading"):
            out["running"] = True
    elif out["health"] in ("ready", "loading"):
        out["running"] = True
    return out

def strata_presets() -> list:
    r = _strata_run(["launch", "--list-presets", "--output", "json"], timeout=15)
    j = r["json"]
    if isinstance(j, list):
        return [p if isinstance(p, dict) else {"name": str(p)} for p in j]
    if isinstance(j, dict) and isinstance(j.get("presets"), list):
        return [p if isinstance(p, dict) else {"name": str(p)} for p in j["presets"]]
    # Text output: keep one name per non-empty line rather than guessing at columns.
    lines = [ln.strip() for ln in (r["stdout"] or "").splitlines() if ln.strip()]
    return [{"name": ln} for ln in lines]

@app.get("/api/strata/status")
def api_strata_status():
    return strata_status()

@app.get("/api/strata/presets")
def api_strata_presets():
    return {"presets": strata_presets()}

class StrataPresetDeleteBody(BaseModel):
    name: str

@app.post("/api/strata/presets/delete")
def api_strata_preset_delete(b: StrataPresetDeleteBody):
    if not strata_bin():
        raise HTTPException(503, f"Strata CLI not found. Install it: {STRATA_INSTALL_CMD}")
    name = b.name.strip()
    if not name:
        raise HTTPException(400, "preset name required")
    r = _strata_run(["launch", "--delete-preset", name, "--output", "json"], timeout=30)
    if not r["ok"]:
        raise HTTPException(502, _redact_cmd(r["error"] or r["stderr"] or r["stdout"] or "delete-preset failed"))
    strata_log(f"deleted preset {name}")
    return {"deleted": name, "presets": strata_presets()}

@app.get("/api/strata/key")
def api_strata_key(reveal: bool = False):
    """`strata launch --show-key` for the running launch. Masked by default; reveal=1 returns
    the raw key for the operator who is already authenticated against the panel."""
    if not strata_bin():
        raise HTTPException(503, f"Strata CLI not found. Install it: {STRATA_INSTALL_CMD}")
    r = _strata_run(["launch", "--show-key"], timeout=15)
    key = (r["stdout"] or "").strip().splitlines()[-1] if r["ok"] and r["stdout"] else ""
    if not key:
        return {"key_set": False, "masked": "", "note": _redact_cmd(r["error"] or r["stderr"] or "no key reported")}
    masked = (key[:4] + "…" + key[-4:]) if len(key) >= 8 else "(short key)"
    return {"key_set": True, "masked": masked, "key": key if reveal else ""}

@app.get("/api/strata/doctor")
def api_strata_doctor():
    """`strata doctor --live`: Strata's own environment/GPU/health report, raw text."""
    r = _strata_run(["doctor", "--live"], timeout=90)
    return {"ok": r["ok"], "raw": r["stdout"] or r["stderr"] or "", "error": r["error"]}

@app.get("/api/strata/update")
def api_strata_update():
    """`strata update --check`: report-only update check; applying it is the operator's call."""
    r = _strata_run(["update", "--check", "--output", "json"], timeout=60)
    return {"ok": r["ok"], "json": r["json"], "raw": r["stdout"] or r["stderr"] or "", "error": r["error"]}

@app.post("/api/strata/launch")
def api_strata_launch(b: StrataLaunchBody):
    if not strata_bin():
        raise HTTPException(503, f"Strata CLI not found. Install it: {STRATA_INSTALL_CMD}")
    if not b.plan and MAYA.status().get("running"):
        raise HTTPException(409, "Stop Maya first — Strata needs its GPU VRAM.")
    args = strata_build_args(b, plan=b.plan)
    # A plan is read-only, so it is worth the long timeout; a real launch downloads and
    # loads a model and must not be cut off mid-download by a short client-side wait.
    r = _strata_run(args, timeout=60 if b.plan else 900)
    cmd = _redact_cmd(" ".join(["strata"] + args))
    if b.plan:
        return {"plan": True, "cmd": cmd, "json": r["json"],
                "raw": r["stdout"] or r["stderr"] or "", "ok": r["ok"], "error": r["error"]}
    if not r["ok"]:
        raise HTTPException(502, _redact_cmd(r["error"] or r["stderr"] or r["stdout"]
                                             or "strata launch failed"))
    j = r["json"] if isinstance(r["json"], dict) else {}
    addr = {"address": j.get("address") or j.get("url"), "port": j.get("port"),
            "model": j.get("model"), "preset": j.get("preset"), "profile": j.get("profile")}
    if addr.get("address"):
        m = re.search(r":(\d{2,5})\b", str(addr["address"]))
        if m and not addr.get("port"):
            addr["port"] = int(m.group(1))
    _strata_remember(addr)
    # Remember the context so agent sync advertises the real window, not the config default.
    ctx_src = j.get("context") or b.context
    n = _parse_ctx_words(ctx_src) if ctx_src else None
    if n:
        CFG["strata_ctx"] = n
        _save_config_secure()
    strata_log(f"launch: {cmd}")
    for ln in (r["stdout"] or r["stderr"] or "").splitlines()[-12:]:
        strata_log(ln)
    return {"started": True, "cmd": cmd, "json": r["json"],
            "raw": r["stdout"] or r["stderr"] or "", "state": strata_status(refresh=True)}

@app.post("/api/strata/stop")
def api_strata_stop():
    if not strata_bin():
        raise HTTPException(503, f"Strata CLI not found. Install it: {STRATA_INSTALL_CMD}")
    r = _strata_run(["launch", "--stop", "--output", "json"], timeout=120)
    if not r["ok"]:
        raise HTTPException(502, _redact_cmd(r["error"] or r["stderr"] or "strata stop failed"))
    strata_log("stop requested")
    return {"stopped": True, "json": r["json"], "raw": r["stdout"] or r["stderr"] or "",
            "state": strata_status(refresh=True)}

@app.get("/api/strata/telemetry")
def api_strata_telemetry():
    """Strata's server exposes /metrics and its own Monitor reads it, but the counter names
    are not llama.cpp's. Parse generically and report the names actually seen, so a mapping
    can be written against real output instead of guessed llama:* names."""
    port = _strata_port()
    out = {"port": port, "ready": False, "metrics": {}, "metric_keys": []}
    if not port:
        out["note"] = "no Strata address recorded yet — launch one from this tab"
        return out
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=1.5)
        out["ready"] = r.status_code == 200
        if not out["ready"]:
            out["note"] = f"/metrics returned {r.status_code}"
            return out
        seen, vals = [], {}
        for ln in r.text.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            try:
                vals[parts[0]] = float(parts[-1])
            except ValueError:
                continue
            seen.append(parts[0])
        out["metric_keys"] = sorted(set(seen))[:120]
        out["metrics"] = {k: vals[k] for k in sorted(set(seen))[:120]}
    except Exception as e:
        out["note"] = f"could not read /metrics: {type(e).__name__}"
    return out

@app.get("/api/strata/log")
def api_strata_log():
    return {"log": "\n".join(list(STRATA_LOG)[-120:])}

# ---------------------------------------------------------------- maya api
@app.get("/api/maya/status")
def api_maya_status():
    return MAYA.status()

@app.get("/api/maya/configs")
def api_maya_configs():
    return {"configs": MAYA.discover_configs()}

class MayaStartBody(BaseModel):
    config: str
    switch: bool = False

@app.post("/api/maya/start")
def api_maya_start(b: MayaStartBody):
    return MAYA.switch(b.config) if b.switch else MAYA.start(b.config)

@app.post("/api/maya/switch")
def api_maya_switch(b: MayaStartBody):
    return MAYA.switch(b.config)

@app.post("/api/maya/stop")
def api_maya_stop():
    return MAYA.stop()

@app.get("/api/maya/log")
def api_maya_log():
    return MAYA.tail_log(max_lines=250)

@app.get("/api/maya/key")
def api_maya_key(reveal: bool = False, config: str | None = None):
    """Masked API key for the active (or specified) Maya config; reveal=1 returns the raw key."""
    key = MAYA.active_api_key(config)
    if not key:
        return {"key_set": False, "masked": "", "key": ""}
    return {"key_set": True, "masked": _mask_secret(key), "key": key if reveal else ""}

def _strata_alias() -> str:
    last = CFG.get("strata_last") or {}
    return str(last.get("model") or CFG.get("strata_alias") or "strata")

def _current_llm(prefer: str | None = None):
    """(alias, port, ctx, api_key, source) for whichever llama-server the panel manages is running.
    `prefer` pins the source: None/'auto' keeps the fallback order; a source name answers
    only that server, so the sync endpoint can say "this tab's server isn't running".
    Alias and context come from the server itself, falling back to manifest/panel state."""
    want = lambda s: prefer in (None, "auto", s)

    def from_server(fallback_alias, port, pid, fallback_ctx, key, source):
        alias, ctx = _server_facts(port)
        return (alias or fallback_alias), port, (ctx or fallback_ctx), key, source

    m = MGR.status()
    if want("panel") and m.get("running") and m["port"] != AVALON_PORT:
        return from_server(m.get("alias"), m["port"], m.get("pid"), _proc_ctx(m.get("pid")), "", "panel")
    a = SMGR.status()
    if want("avalon") and a.get("running"):
        return from_server(_avalon_target()["base"].lower() or "qwen3.8-flash-next", AVALON_PORT,
                           a.get("pid"), _proc_ctx(a.get("pid")), CFG.get("avalon_api_key", ""), "avalon")
    if want("avalon") and m.get("running"):  # discovery adopted the systemd avalon on 8086
        return from_server(_avalon_target()["base"].lower() or "qwen3.8-flash-next", AVALON_PORT,
                           m.get("pid"), _proc_ctx(m.get("pid")), CFG.get("avalon_api_key", ""), "avalon")
    s = strata_status()
    if want("strata") and s.get("running") and s.get("port"):
        # The key is only forwarded if the operator set strata_api_key deliberately. Strata
        # owns its own key (`strata launch --show-key`); auto-reading it would push a real
        # secret into every synced agent config and into panel state without being asked.
        return from_server(_strata_alias(), s["port"], None, int(CFG.get("strata_ctx") or 8192),
                           CFG.get("strata_api_key", ""), "strata")
    my = MAYA.status()
    if want("maya") and my.get("running") and my.get("port"):
        return (my.get("alias") or my.get("model_name") or "maya"), my["port"], \
            int(my.get("ctx") or 131072), MAYA.active_api_key(), "maya"
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
    s = strata_status()
    if s.get("running") and s.get("port") and s.get("health") == "ready":
        return s["port"], CFG.get("strata_api_key", "")
    my = MAYA.status()
    if my.get("running") and my.get("state") == "ready" and my.get("port"):
        return my["port"], MAYA.active_api_key()
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

def _jsonc_load(text):
    """json.loads that survives JSONC (// and /* */ comments, trailing commas) — Zed and
    opencode both ship comment-bearing settings files, and a plain json.loads turns a
    perfectly good settings file into a hard "Expecting value: line 1" failure."""
    out, inner, i, n = [], [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            out.extend(text[i:j]); inner.extend([True] * (j - i)); i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c); inner.append(False); i += 1
    chars = []
    for idx, ch in enumerate(out):
        if ch == "," and not inner[idx]:
            k = idx + 1
            while k < len(out) and out[k] in " \t\r\n":
                k += 1
            if k < len(out) and out[k] in "}]":
                continue  # trailing comma
        chars.append(ch)
    return json.loads("".join(chars))

def _json_merge(path, apply):
    path = Path(path)
    cfg = {}
    if path.exists():
        text = path.read_text()
        try:
            cfg = json.loads(text)
        except json.JSONDecodeError:
            cfg = _jsonc_load(text)  # raises with a usable message if it really is broken
        if not isinstance(cfg, dict):
            raise HTTPException(400, f"{path} is not a JSON object")
    apply(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return str(path)

def _yaml_upsert(path, parent, key, value):
    """Set <parent>.<key> in a YAML file and leave every other byte alone. Round-tripping the
    whole document through safe_load/safe_dump (the old approach) re-indented and stripped
    the comments out of files like ~/.hermes/config.yaml on every single sync."""
    path = Path(path)
    text = path.read_text() if path.exists() else ""
    try:
        root = yaml.compose(text) if text.strip() else None
    except yaml.YAMLError:
        raise HTTPException(400, f"{path} is not valid YAML — fix it first")

    def dump_block(val, col):
        s = yaml.safe_dump(val, sort_keys=False, default_flow_style=False).rstrip("\n")
        pad = " " * col
        return "\n".join(pad + ln if ln.strip() else ln for ln in s.splitlines())

    if isinstance(root, yaml.MappingNode):
        tops = {k.value: v for k, v in root.value if isinstance(k, yaml.ScalarNode)}
        pval = tops.get(parent)
        if isinstance(pval, yaml.MappingNode) and not pval.flow_style:
            pair = next(((kn, vn) for kn, vn in pval.value
                         if isinstance(kn, yaml.ScalarNode) and kn.value == key), None)
            if pair:
                knode, old = pair
                start, end = old.start_mark.index, old.end_mark.index
                line_start = text.rfind("\n", 0, start) + 1
                seg = text[start:end]
                tail_nl = "\n" if seg.endswith("\n") else ""
                if text[line_start:start].strip() == "":     # value sits on its own lines
                    text = text[:line_start] + dump_block(value, start - line_start) + tail_nl + text[end:]
                else:                                        # inline value after `key: `
                    col = knode.start_mark.column + 2
                    text = text[:start] + "\n" + dump_block(value, col) + "\n" + text[end:].lstrip("\n")
            elif pval.value:
                col = pval.value[0][0].start_mark.column
                at = pval.value[-1][1].end_mark.index
                tail = text[at:]
                if tail.startswith("\n"):
                    tail = tail[1:]
                text = text[:at].rstrip("\n") + "\n" + dump_block({key: value}, col) + "\n" + tail
            else:
                root = None  # empty/flow-style parent: fall through to the full rewrite
        elif parent not in tops:
            text = (text.rstrip("\n") + "\n" if text.strip() else "") + \
                f"{parent}:\n" + dump_block({key: value}, 2)
        else:
            root = None  # parent is a scalar/flow mapping: full rewrite below
    else:
        root = None
    if root is None:
        cfg = yaml.safe_load(text) or {}
        if not isinstance(cfg, dict):
            raise HTTPException(400, f"{path} is not a YAML mapping — fix it first")
        sub = cfg.setdefault(parent, {})
        if not isinstance(sub, dict):
            raise HTTPException(400, f"{path}: '{parent}' is not a mapping — fix it first")
        sub[key] = value
        text = yaml.safe_dump(cfg, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n")
    return str(path)

def _toml_strip_section(text, headers):
    out, skip = [], False
    for ln in text.splitlines():
        head = ln.split("#", 1)[0].strip()
        if head.startswith("[") and head.endswith("]"):
            skip = head in headers
        if not skip:
            out.append(ln)
    return "\n".join(out).rstrip() + "\n"

def _w_omp(alias, base, ctx, key=""):
    prov = {"baseUrl": base + "/v1", "api": "openai-completions",
            "auth": "apiKey" if key else "none",
            "models": [{"id": alias, "name": alias, "api": "openai-completions",
                        "reasoning": False, "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": max(ctx, 8192), "maxTokens": min(8192, max(ctx, 8192))}]}
    if key:
        prov["apiKey"] = key
    return _yaml_upsert(HOME / ".omp/agent/models.yml", "providers", PANEL_PROVIDER_ID, prov) \
        + " — omp: omp --model inference-console/" + alias

def _w_opencode(alias, base, ctx, key=""):
    def apply(cfg):
        opts = {"baseURL": base + "/v1"}
        if key:
            opts["apiKey"] = key
        cfg.setdefault("provider", {})[PANEL_PROVIDER_ID] = {
            "npm": "@ai-sdk/openai-compatible",
            "options": opts,
            "models": {alias: {"name": alias}}}
    return _json_merge(HOME / ".config/opencode/opencode.json", apply) + " — opencode: provider inference-console"

def _w_codex(alias, base, ctx, key=""):
    path = Path(HOME / ".codex/config.toml")
    old = path.read_text() if path.exists() else ""
    old = _toml_strip_section(old, {"[model_providers." + PANEL_PROVIDER_ID + "]", "[profiles." + PANEL_PROVIDER_ID + "]"})
    key_ln = f"experimental_bearer_token = \"{key}\"\n" if key else ""
    block = (f"\n[model_providers.{PANEL_PROVIDER_ID}]\nname = \"Inference Console\"\n"
             f"base_url = \"{base}/v1\"\nwire_api = \"chat\"\n{key_ln}"
             f"\n[profiles.{PANEL_PROVIDER_ID}]\nmodel_provider = \"{PANEL_PROVIDER_ID}\"\nmodel = \"{alias}\"\n")
    import tomllib
    merged = old + block
    tomllib.loads(merged)  # validate before touching the user's config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged)
    return str(path) + " — codex: codex --profile inference-console"

def _w_aider(alias, base, ctx, key=""):
    def apply(cfg):
        cfg["openai-api-base"] = base + "/v1"
        cfg["openai-api-key"] = key or "panel"
        cfg["model"] = "openai/" + alias
    return _yaml_merge(HOME / ".aider.conf.yml", apply) + " — aider: aider (model openai/" + alias + ")"

def _w_zed(alias, base, ctx):
    def apply(cfg):
        lm = cfg.setdefault("language_models", {}).setdefault("openai", {})
        lm["api_url"] = base + "/v1"
        # Merge instead of replacing: available_models is the user's own list of models
        # behind the openai provider, and wiping it unregisters everything else they had.
        keep = [m for m in (lm.get("available_models") or []) if (m or {}).get("name") != alias]
        lm["available_models"] = [{"name": alias, "max_tokens": max(ctx, 8192), "tool_call": True}] + keep
    return _json_merge(HOME / ".config/zed/settings.json", apply) + " — zed: openai provider model " + alias

def _w_continue(alias, base, ctx, key=""):
    def apply(cfg):
        models = [m for m in cfg.get("models", []) if m.get("name") != alias]
        models.append({"name": alias, "provider": "openai", "model": alias,
                       "apiBase": base + "/v1", "apiKey": key or "panel", "capabilities": ["tool_use"]})
        cfg["models"] = models
    return _yaml_merge(HOME / ".continue/config.yaml", apply) + " — continue: model " + alias

def _w_claude(alias, base, ctx, key):
    tok = key or "panel"
    try:
        r = httpx.post(base + "/v1/messages", json={"model": alias, "max_tokens": 1,
                       "messages": [{"role": "user", "content": "hi"}]},
                       headers={"anthropic-version": "2023-06-01",
                                "x-api-key": tok,
                                "Authorization": f"Bearer {tok}"}, timeout=20)
    except httpx.HTTPError:
        raise HTTPException(400, "server unreachable for /v1/messages probe")
    if r.status_code == 404:
        raise HTTPException(400, "this server has no Anthropic-compat /v1/messages endpoint")
    if r.status_code >= 400:
        # 401/403 here almost always means a mismatched key; writing the env block anyway
        # leaves Claude Code pointing at a server that will refuse every request.
        raise HTTPException(400, f"/v1/messages returned {r.status_code}: {(r.text or '')[:200]}")
    def apply(cfg):
        cfg.setdefault("env", {}).update({
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_AUTH_TOKEN": tok,
            "ANTHROPIC_MODEL": alias})
    return _json_merge(HOME / ".claude/settings.json", apply) + " — claude-code: env points at " + base

def _w_pi(alias, base, ctx, key):
    path = HOME / ".pi/agent/models.json"
    if path.exists():
        text = path.read_text()
        try:
            cfg = json.loads(text)
        except json.JSONDecodeError:
            cfg = _jsonc_load(text)
    else:
        cfg = {"$schema": "https://pi.dev/schemas/models.schema.json", "providers": {}}
    cfg.setdefault("providers", {})[PANEL_PROVIDER_ID] = {
        "baseUrl": base + "/v1", "api": "openai-completions", "apiKey": key or "none",
        "models": [{"id": alias, "name": alias, "api": "openai-completions",
                    "provider": PANEL_PROVIDER_ID, "reasoning": False, "input": ["text"],
                    "contextWindow": max(ctx, 8192), "maxTokens": min(8192, max(ctx, 8192))}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return str(path) + " — pi: provider inference-console"

def _w_hermes(alias, base, ctx, key):
    prov = {"name": "Inference Console", "base_url": base + "/v1",
            "default_model": alias, "model": alias,
            "models": {alias: {}}, "models_discovered": True}
    if key:
        prov["api_key"] = key
    return _yaml_upsert(HOME / ".hermes/config.yaml", "providers", PANEL_PROVIDER_ID, prov) \
        + " — hermes: pick /model custom:inference-console"

def _w_kimi(alias, base, ctx, key):
    path = HOME / ".kimi-code/config.toml"
    old = path.read_text() if path.exists() else ""
    old = _toml_strip_section(old, {"[providers." + PANEL_PROVIDER_ID + "]", '[models."' + alias + '"]'})
    block = (f"\n[providers.{PANEL_PROVIDER_ID}]\nname = \"Inference Console\"\ntype = \"openai\"\n"
             f"baseUrl = \"{base}/v1\"\napiKey = \"{key or 'panel'}\"\n"
             f"\n[models.\"{alias}\"]\nprovider = \"{PANEL_PROVIDER_ID}\"\nname = \"{alias} via Inference Console\"\n")
    import tomllib
    merged = old + block
    tomllib.loads(merged)  # validate before touching the user's config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged)
    return str(path) + " — kimi: model inference-console/" + alias

AGENTS = {
    "omp":         ("omp (oh-my-pi)",     lambda a, b, c, k: _w_omp(a, b, c, k),      "selector inference-console/<alias>"),
    "pi":          ("pi",                 lambda a, b, c, k: _w_pi(a, b, c, k),       "selector inference-console/<alias>"),
    "opencode":    ("OpenCode",           lambda a, b, c, k: _w_opencode(a, b, c, k), "provider inference-console"),
    "codex":       ("Codex CLI",          lambda a, b, c, k: _w_codex(a, b, c, k),    "codex --profile inference-console"),
    "aider":       ("Aider",              lambda a, b, c, k: _w_aider(a, b, c, k),    "openai/<alias>"),
    "zed":         ("Zed",                lambda a, b, c, k: _w_zed(a, b, c),         "merges into Zed openai available_models"),
    "continue":    ("Continue",           lambda a, b, c, k: _w_continue(a, b, c, k), "adds model row"),
    "claude-code": ("Claude Code",        _w_claude,                                  "needs /v1/messages; rewrites ANTHROPIC_BASE_URL"),
    "hermes":      ("Hermes Agent",       lambda a, b, c, k: _w_hermes(a, b, c, k),   "then pick /model custom:inference-console"),
    "kimi":        ("Kimi Code",          lambda a, b, c, k: _w_kimi(a, b, c, k),     "model inference-console/<alias>"),
}

@app.get("/api/agents")
def api_agents(source: str | None = None):
    m = MGR.status()
    a = SMGR.status()
    s = strata_status()
    my = MAYA.status()
    sources = {
        "panel": bool(m.get("running") and m["port"] != AVALON_PORT),
        "avalon": bool(a.get("running") or (m.get("running") and m["port"] == AVALON_PORT)),
        "strata": bool(s.get("running") and s.get("port")),
        "maya": bool(my.get("running") and my.get("port")),
    }
    targets = {}
    for src_name, up in sources.items():
        if not up:
            targets[src_name] = None
            continue
        cur = _current_llm(src_name)
        targets[src_name] = {"alias": cur[0], "port": cur[1], "ctx": cur[2], "source": cur[4]} if cur else None
    src = (source or "auto").lower()
    running = _current_llm(None if src == "auto" else src)
    return {"agents": [{"id": k, "label": v[0], "hint": v[2]} for k, v in AGENTS.items()],
            "registry": _load_agents_state(),
            "running": [running[0], running[1], running[2], running[4]] if running else None,
            "sources": sources,
            "targets": targets}

class AgentSyncBody(BaseModel):
    agents: list[str]
    source: str | None = None  # auto | panel | avalon | strata | maya

@app.post("/api/agents/sync")
def api_agents_sync(b: AgentSyncBody):
    src = (b.source or "auto").lower()
    cur = _current_llm(None if src == "auto" else src)
    if not cur:
        if src == "auto":
            raise HTTPException(409, "no running server — start a model first")
        raise HTTPException(409, f"the {src} server isn't running — start it on its tab first")
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
    if any(v["ok"] for v in out.values()):
        st.update(alias=alias, port=port, source=source, synced_at=time.time())
        _save_agents_state(st)
    return {"results": out, "alias": alias, "port": port, "source": source}


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
    draft: str = ""
    draft_max: int = 16
    draft_min: int = 0
    ot: str = ""
    tensor_split: str = ""
    tbatch: int | None = None
    keep: int | None = None
    jinja: str = "on"
    save_profile: bool = True

@app.post("/api/server/start")
def api_start(b: StartBody):
    return MGR.start(
        b.model, preset=b.preset, extra_args=b.extra_args, ctx=b.ctx,
        ngl=b.ngl, threads=b.threads, fa=b.fa, ctk=b.ctk, ctv=b.ctv,
        parallel=b.parallel, batch=b.batch, ubatch=b.ubatch,
        ncmoe=b.ncmoe, mlock=b.mlock, no_mmap=b.no_mmap,
        draft=b.draft, draft_max=b.draft_max, draft_min=b.draft_min,
        ot=b.ot, tensor_split=b.tensor_split, tbatch=b.tbatch, keep=b.keep,
        jinja=b.jinja, save_profile=b.save_profile,
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
    strata_bin: str | None = None
    strata_alias: str | None = None
    strata_api_key: str | None = None
    agent_llm_provider: str | None = None
    agent_llm_url: str | None = None
    agent_llm_key: str | None = None
    agent_llm_model: str | None = None
    agent_llm_headers: str | None = None
    agent_llm_fallback_local: bool | None = None

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
    skey = CFG.get("strata_api_key", "")
    skey_masked = (skey[:4] + "…" + skey[-4:]) if len(skey) >= 8 else ""
    akey = CFG.get("agent_llm_key", "")
    akey_masked = (akey[:4] + "…" + akey[-4:]) if len(akey) >= 8 else ""
    sbin = strata_bin()
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
        "strata_bin": CFG.get("strata_bin", ""),
        "strata_bin_resolved": sbin or "",
        "strata_installed": bool(sbin),
        "strata_alias": CFG.get("strata_alias", ""),
        "strata_api_key_set": bool(skey),
        "strata_api_key_masked": skey_masked,
        "strata_ctx": int(CFG.get("strata_ctx", 8192) or 8192),
        "agent_llm_provider": CFG.get("agent_llm_provider", "custom"),
        "agent_llm_url": CFG.get("agent_llm_url", ""),
        "agent_llm_model": CFG.get("agent_llm_model", ""),
        "agent_llm_headers": CFG.get("agent_llm_headers", ""),
        "agent_llm_key_set": bool(akey),
        "agent_llm_key_masked": akey_masked,
        "agent_llm_fallback_local": bool(CFG.get("agent_llm_fallback_local", True)),
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
    if b.strata_bin is not None:
        CFG["strata_bin"] = b.strata_bin.strip()
    if b.strata_alias is not None:
        CFG["strata_alias"] = b.strata_alias.strip()
    if b.strata_api_key is not None:
        val = b.strata_api_key.strip()
        if val == "clear":
            CFG["strata_api_key"] = ""
        elif val and val != "__keep__":
            CFG["strata_api_key"] = val
    if b.agent_llm_provider is not None:
        CFG["agent_llm_provider"] = b.agent_llm_provider.strip() or "custom"
    if b.agent_llm_url is not None:
        CFG["agent_llm_url"] = b.agent_llm_url.strip()
    if b.agent_llm_model is not None:
        CFG["agent_llm_model"] = b.agent_llm_model.strip()
    if b.agent_llm_headers is not None:
        CFG["agent_llm_headers"] = b.agent_llm_headers.strip()
    if b.agent_llm_key is not None:
        val = b.agent_llm_key.strip()
        if val == "clear":
            CFG["agent_llm_key"] = ""
        elif val and val != "__keep__":
            CFG["agent_llm_key"] = val
    if b.agent_llm_fallback_local is not None:
        CFG["agent_llm_fallback_local"] = bool(b.agent_llm_fallback_local)
    _save_config_secure()
    return api_settings_get()

# ---------------------------------------------------------------- panel process (restart)
PANEL_STARTED_AT = time.time()

def _self_argv() -> list:
    """argv that re-launches this panel exactly as it was invoked, with a resolved
    interpreter and script/shim path so execv works after a cd or a PATH change."""
    argv = list(sys.argv) or [str(Path(__file__).resolve())]
    first = Path(argv[0])
    if first.exists():
        argv[0] = str(first.resolve())
    else:
        found = shutil.which(argv[0])
        if found:
            argv[0] = found
    return [sys.executable] + argv

def _restart_self(argv: list, delay: float):
    """Replace this process image in place. Nothing else is forked, so the PID stays the
    same (systemd keeps tracking it) and the llama-server/Avalon/Strata children, which run
    in their own sessions, keep serving; the panel re-adopts them from its state files."""
    time.sleep(delay)
    print(f"panel: restarting in place: {' '.join(argv)}", flush=True)
    for f in (sys.stdout, sys.stderr):
        try:
            f.flush()
        except Exception:
            pass
    try:
        os.execv(sys.executable, argv)
    except OSError as e:
        # Do not exit: a failed restart must leave the running panel serving.
        print(f"panel: restart failed, still serving: {e}", flush=True)

class RestartBody(BaseModel):
    delay_ms: int = 600

@app.get("/api/panel/ping")
def api_panel_ping():
    argv = _self_argv()
    return {"ok": True, "pid": os.getpid(), "port": PORT,
            "started_at": PANEL_STARTED_AT, "uptime_s": round(time.time() - PANEL_STARTED_AT, 1),
            "argv": argv, "restartable": Path(argv[0]).exists()}

@app.post("/api/panel/restart")
def api_panel_restart(b: RestartBody | None = None):
    argv = _self_argv()
    if not Path(argv[0]).exists():
        raise HTTPException(400, f"cannot restart: interpreter not found: {argv[0]}")
    delay = max(0.0, min(5.0, (b.delay_ms if b else 600) / 1000))
    _save_sessions()  # keep the tab signed in across the re-exec
    threading.Thread(target=_restart_self, args=(argv, delay), daemon=True).start()
    return {"ok": True, "pid": os.getpid(), "argv": argv, "delay_s": delay}

# ---------------------------------------------------------------- engine setup (one-click installs)
# One background job at a time: download/build runs stream log lines the UI polls.
# The panel admin API can already launch arbitrary binaries, so installing official
# upstream artifacts here stays inside the same trust boundary — but every install
# still requires an explicit confirm from the UI.
SETUP_JOB = {"name": None, "state": "idle", "log": deque(maxlen=500), "started": 0.0}
_SETUP_LOCK = threading.Lock()

def _setup_log(line: str):
    SETUP_JOB["log"].append(line)

def _llama_asset_name(sysname: str, machine: str, variant: str = "auto") -> str | None:
    """Which official llama.cpp release asset fits this host + backend variant."""
    machine = (machine or "").lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if sysname.startswith("darwin"):
        return f"bin-macos-{'arm64' if arch == 'arm64' else 'x64'}.tar.gz"
    if sysname.startswith("win"):
        if variant == "cpu":
            return f"bin-win-cpu-{arch}.zip"
        return f"bin-win-vulkan-{arch}.zip"
    if sysname.startswith("linux"):
        if variant == "cpu":
            return f"bin-ubuntu-{arch}.tar.gz"
        return f"bin-ubuntu-vulkan-{arch}.tar.gz"
    return None

def _pick_llama_asset(assets: list[dict], sysname: str, machine: str,
                      variant: str = "auto") -> dict | None:
    """assets = [{name, browser_download_url}]. cuda variants match any cuda build."""
    want = _llama_asset_name(sysname, machine, variant)
    if not want:
        return None
    names = [a.get("name", "") for a in assets]
    if variant.startswith("cuda"):
        arch = "arm64" if (machine or "").lower() in ("arm64", "aarch64") else "x64"
        toks = ("cuda", arch)
        for n in names:
            if n.startswith("llama-") and all(t in n for t in toks) and (n.endswith(".tar.gz") or n.endswith(".zip")):
                return next(a for a in assets if a.get("name") == n)
        return None
    for a in assets:
        if a.get("name", "").endswith(want) and a["name"].startswith("llama-"):
            return a
    return None

def _latest_llama_release() -> tuple[str, dict] | None:
    """First recent release that actually carries binary assets (the 'latest' endpoint
    sometimes points at an assets-less nightly tag)."""
    try:
        r = httpx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8",
                      headers={"User-Agent": "inference-console"}, timeout=20)
        r.raise_for_status()
        for rel in r.json():
            assets = rel.get("assets") or []
            if assets:
                return rel.get("tag_name", ""), {a["name"]: a.get("browser_download_url", "") for a in assets}
    except Exception as e:
        _setup_log(f"llama.cpp: release lookup failed: {type(e).__name__}: {e}")
    return None

def _install_llama(variant: str = "auto"):
    rel = _latest_llama_release()
    if not rel:
        raise HTTPException(503, "could not reach the llama.cpp releases API")
    tag, urls = rel
    assets = [{"name": n, "browser_download_url": u} for n, u in urls.items()]
    asset = _pick_llama_asset(assets, sys.platform, platform.machine(), variant)
    if not asset or not asset.get("browser_download_url"):
        raise HTTPException(400, f"no llama.cpp build for {sys.platform}/{platform.machine()} variant={variant}")
    name = asset["name"]
    dest = APP_DIR / f"llama-{tag}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = dest / name
    _setup_log(f"llama.cpp: downloading {name}")
    got = 0
    last_report = 0.0
    with httpx.stream("GET", asset["browser_download_url"], timeout=httpx.Timeout(30, read=300),
                      follow_redirects=True) as r:
        r.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                f.write(chunk)
                got += len(chunk)
                if time.time() - last_report > 2:
                    last_report = time.time()
                    _setup_log(f"  … {got / 2**20:.0f} MB")
    _setup_log(f"llama.cpp: extracting into {dest}")
    if name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")
    else:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    archive.unlink(missing_ok=True)
    exe = "llama-server.exe" if sys.platform.startswith("win") else "llama-server"
    hits = [p for p in dest.rglob(exe) if p.is_file()]
    if not hits:
        raise HTTPException(500, "archive extracted but no llama-server binary found")
    binary = hits[0]
    binary.chmod(0o755)
    CFG["server_binary"] = str(binary)
    _save_config_secure()
    caps = bin_caps(str(binary))
    _setup_log(f"llama.cpp: installed {caps.get('version') or tag} at {binary}")
    _setup_log("server_binary updated in config — the llama.cpp tab now uses this build")

def _install_strata():
    _setup_log("Strata: running the official installer")
    try:
        p = subprocess.run(["sh", "-c", STRATA_INSTALL_CMD], capture_output=True, text=True,
                           timeout=900)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Strata installer timed out after 900s")
    for ln in ((p.stdout or "") + "\n" + (p.stderr or "")).splitlines():
        if ln.strip():
            _setup_log(ln)
    if p.returncode != 0:
        raise HTTPException(502, f"Strata installer exited {p.returncode}")
    b = strata_bin()
    if not b:
        raise HTTPException(502, "installer finished but no strata binary was found")
    _setup_log(f"Strata: installed at {b}")

def _build_avalon():
    src = HOME / "avalon"
    if not src.is_dir():
        raise HTTPException(400, f"no Avalon source at {src} — clone it there first")
    _setup_log("Avalon: configuring cmake build")
    steps = [["cmake", "-B", "build"], ["cmake", "--build", "build", "--config", "Release",
                                        "--target", "llama-server", "-j", str(os.cpu_count() or 4)]]
    for cmd in steps:
        p = subprocess.run(cmd, cwd=str(src), capture_output=True, text=True, timeout=3600)
        for ln in ((p.stdout or "") + "\n" + (p.stderr or "")).splitlines()[-40:]:
            if ln.strip():
                _setup_log(ln)
        if p.returncode != 0:
            raise HTTPException(502, f"{cmd[1] if len(cmd)>1 else cmd[0]} exited {p.returncode}")
    if Path(AVALON_BIN).exists():
        _setup_log(f"Avalon: built {AVALON_BIN}")
    else:
        raise HTTPException(502, "build finished but build/bin/llama-server is missing")

def _setup_job_wrap(fn):
    try:
        fn()
        SETUP_JOB["state"] = "done"
        _setup_log("done.")
    except HTTPException as e:
        SETUP_JOB["state"] = "failed"
        _setup_log(f"failed: {e.detail}")
    except Exception as e:
        SETUP_JOB["state"] = "failed"
        _setup_log(f"failed: {type(e).__name__}: {e}")

def _setup_start(name: str, fn):
    with _SETUP_LOCK:
        if SETUP_JOB["state"] == "running":
            raise HTTPException(409, f"setup job already running: {SETUP_JOB['name']}")
        SETUP_JOB.update(name=name, state="running", started=time.time())
        SETUP_JOB["log"].clear()
    threading.Thread(target=_setup_job_wrap, args=(fn,), daemon=True).start()
    return {"started": name}

class SetupInstallBody(BaseModel):
    engine: str                 # llama | strata | avalon
    variant: str = "auto"       # llama only: auto | cpu | vulkan | cuda
    confirm: bool = False

@app.get("/api/setup/status")
def api_setup_status():
    bin_path = CFG.get("server_binary", BIN)
    caps = bin_caps(bin_path)
    latest = None
    updatable = False
    if caps.get("exists"):
        rel = _latest_llama_release()
        if rel:
            latest = rel[0]
            # Release tags look like "b12430"; `--version` reports "build 12430" —
            # compare the build number, not the raw tag substring.
            build = re.sub(r"^[A-Za-z]+", "", latest or "")
            updatable = bool(build and build not in (caps.get("version") or ""))
    sbin = strata_bin()
    return {
        "llama": {"installed": bool(caps.get("exists")), "binary": bin_path,
                  "version": caps.get("version", ""), "latest": latest or "",
                  "update_available": updatable,
                  "platform": f"{sys.platform}/{platform.machine()}"},
        "strata": {"installed": bool(sbin), "binary": sbin or "",
                   "install_cmd": STRATA_INSTALL_CMD},
        "avalon": {"installed": Path(AVALON_BIN).exists(), "binary": AVALON_BIN,
                   "source": str(HOME / "avalon"), "source_exists": (HOME / "avalon").is_dir()},
        "job": {"name": SETUP_JOB["name"], "state": SETUP_JOB["state"],
                "log": list(SETUP_JOB["log"])[-60:], "started": SETUP_JOB["started"]},
    }

@app.post("/api/setup/install")
def api_setup_install(b: SetupInstallBody):
    if not b.confirm:
        raise HTTPException(400, "install requires confirm: true")
    if b.engine == "llama":
        return _setup_start(f"llama.cpp ({b.variant})", lambda: _install_llama(b.variant))
    if b.engine == "strata":
        return _setup_start("Strata", _install_strata)
    if b.engine == "avalon":
        return _setup_start("Avalon build", _build_avalon)
    raise HTTPException(400, f"unknown engine: {b.engine}")

# ---------------------------------------------------------------- optimization agent api
from agent_memory import AgentStore
from optimizer_agent import HostContext, OptimizerAgent, PROVIDER_PRESETS
from thermal_guard import ThermalGuard

AGENT_STORE = AgentStore(APP_DIR / "agent")
OPT_AGENT = OptimizerAgent(
    host=HostContext(
        cfg=CFG,
        default_bin=BIN,
        avalon_bin=AVALON_BIN,
        avalon_profiles=AVALON_PROFILES,
        launch_profiles_path=LAUNCH_PROFILES_PATH,
        vitals=vitals,
        scan_models=scan_models,
        estimate_vram=estimate_vram,
        bin_caps=bin_caps,
        find_draft=find_draft,
        save_config=_save_config_secure,
        mgr=MGR,
        smgr=SMGR,
        avalon_target=_avalon_target,
        strata_bin=strata_bin,
        strata_status=strata_status,
        strata_build_args=strata_build_args,
        strata_run=_strata_run,
        strata_launch_body_cls=StrataLaunchBody,
        strata_stop=api_strata_stop,
        strata_log_path=STRATA_LOG,
        chat_target=_chat_target,
        benchmark_run=lambda pk, mt: api_benchmark_run(BenchRunBody(prompt_key=pk, max_tokens=mt)),
        read_json=_read_json,
        write_json=_write_json,
        sync_agents_fn=lambda ags, src: api_agents_sync(AgentSyncBody(agents=ags, source=src)),
        coding_agents_list=lambda: list(AGENTS.keys()),
        hf_search_fn=api_hf_search,
        hf_files_fn=api_hf_files,
        maya_mgr=MAYA,
    ),
    store=AGENT_STORE,
)

def _save_thermal_cfg(c: dict):
    CFG["thermal_guard"] = c
    _save_config_secure()

THERMAL_GUARD = ThermalGuard(
    get_config=lambda: CFG.get("thermal_guard") or {},
    save_config=_save_thermal_cfg,
    vitals_fn=vitals,
    downshift_fn=OPT_AGENT.downshift_active_engine,
    killswitch_fn=OPT_AGENT.emergency_killswitch,
)
if (CFG.get("thermal_guard") or {}).get("enabled"):
    THERMAL_GUARD.start_loop()

class AgentAdviseBody(BaseModel):
    engine: str = "llama"
    goal: str = "balanced"
    model: str = ""
    profile: str = "flagship"
    constraints: dict | None = None
    user_prompt: str = ""
    use_llm: bool = True

class AgentApplyBody(BaseModel):
    engine: str = "llama"
    config: dict
    launch: bool = True
    save_preset_name: str = ""
    goal: str = "balanced"
    checkpoint_label: str = ""

class AgentTuneStartBody(BaseModel):
    engine: str = "llama"
    goal: str = "fastest"
    model: str = ""
    profile: str = "flagship"
    constraints: dict | None = None
    user_prompt: str = ""
    max_trials: int = 5
    max_seconds: int = 240
    save_preset_name: str = ""

class AgentTuneStopBody(BaseModel):
    rollback: bool = False

class AgentDiagnoseBody(BaseModel):
    engine: str = "llama"
    log_text: str | None = None
    last_config: dict | None = None
    use_llm: bool = True

class AgentCheckpointCreateBody(BaseModel):
    engine: str = "llama"
    label: str = ""
    reason: str = "manual"

class AgentCheckpointRestoreBody(BaseModel):
    relaunch: bool = False

class AgentPresetSaveBody(BaseModel):
    name: str
    engine: str = "llama"
    goal: str = "balanced"
    config: dict
    notes: str = ""
    model_key: str = ""

class AgentDocsBody(BaseModel):
    engine: str = "llama"
    query: str = ""

class AgentChatBody(BaseModel):
    message: str
    engine: str = "llama"
    model: str = ""

@app.get("/api/agent/status")
def api_agent_status():
    return {
        "llm": OPT_AGENT.llm_status(),
        "tune": OPT_AGENT.tune_status(),
        "checkpoints_count": len(AGENT_STORE.list_checkpoints(limit=50)),
        "presets_count": len(AGENT_STORE.list_presets()),
        "memory": AGENT_STORE.memory_summary(limit=10),
    }

@app.post("/api/agent/advise")
def api_agent_advise(b: AgentAdviseBody):
    try:
        return OPT_AGENT.advise(
            engine=b.engine, goal=b.goal, model=b.model, profile=b.profile,
            constraints=b.constraints, user_prompt=b.user_prompt, use_llm=b.use_llm,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/agent/apply")
def api_agent_apply(b: AgentApplyBody):
    try:
        return OPT_AGENT.apply_config(
            engine=b.engine, config=b.config, launch=b.launch,
            save_preset_name=b.save_preset_name, goal=b.goal,
            checkpoint_label=b.checkpoint_label,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/agent/tune/start")
def api_agent_tune_start(b: AgentTuneStartBody):
    try:
        return OPT_AGENT.start_tune(
            engine=b.engine, goal=b.goal, model=b.model, profile=b.profile,
            constraints=b.constraints, user_prompt=b.user_prompt,
            max_trials=b.max_trials, max_seconds=b.max_seconds,
            save_preset_name=b.save_preset_name,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/agent/tune/status")
def api_agent_tune_status():
    return OPT_AGENT.tune_status()

@app.post("/api/agent/tune/stop")
def api_agent_tune_stop(b: AgentTuneStopBody):
    return OPT_AGENT.stop_tune(rollback=b.rollback)

@app.post("/api/agent/diagnose")
def api_agent_diagnose(b: AgentDiagnoseBody):
    return OPT_AGENT.diagnose_failure(
        engine=b.engine, log_text=b.log_text, last_config=b.last_config, use_llm=b.use_llm,
    )

@app.post("/api/agent/heal")
def api_agent_heal(b: AgentDiagnoseBody):
    return OPT_AGENT.heal_failure(engine=b.engine, last_config=b.last_config)

@app.get("/api/agent/checkpoints")
def api_agent_checkpoints():
    return {"checkpoints": AGENT_STORE.list_checkpoints(limit=30)}

@app.post("/api/agent/checkpoints")
def api_agent_checkpoint_create(b: AgentCheckpointCreateBody):
    return OPT_AGENT.capture_checkpoint(engine=b.engine, label=b.label, reason=b.reason)

@app.post("/api/agent/checkpoints/{cp_id}/restore")
def api_agent_checkpoint_restore(cp_id: str, b: AgentCheckpointRestoreBody):
    try:
        return OPT_AGENT.restore_checkpoint(cp_id, relaunch=b.relaunch)
    except ValueError as e:
        raise HTTPException(404, str(e))

@app.delete("/api/agent/checkpoints/{cp_id}")
def api_agent_checkpoint_delete(cp_id: str):
    if not AGENT_STORE.delete_checkpoint(cp_id):
        raise HTTPException(404, "checkpoint not found")
    return {"ok": True}

@app.get("/api/agent/presets")
def api_agent_presets(engine: str | None = None):
    return {"presets": AGENT_STORE.list_presets(engine=engine)}

@app.post("/api/agent/presets")
def api_agent_preset_save(b: AgentPresetSaveBody):
    return AGENT_STORE.save_preset(
        name=b.name, engine=b.engine, goal=b.goal, config=b.config,
        notes=b.notes, model_key=b.model_key,
    )

@app.delete("/api/agent/presets/{preset_id}")
def api_agent_preset_delete(preset_id: str):
    if not AGENT_STORE.delete_preset(preset_id):
        raise HTTPException(404, "preset not found or builtin")
    return {"ok": True}

@app.get("/api/agent/memory")
def api_agent_memory():
    return AGENT_STORE.memory_summary(limit=30)

@app.delete("/api/agent/memory")
def api_agent_memory_clear():
    AGENT_STORE.clear_memory()
    return {"ok": True}

@app.post("/api/agent/docs")
def api_agent_docs(b: AgentDocsBody):
    return OPT_AGENT.docs.lookup(b.engine, b.query)

@app.post("/api/agent/chat")
def api_agent_chat(b: AgentChatBody):
    return OPT_AGENT.chat(message=b.message, engine=b.engine, model=b.model)

class AgentThermalConfigBody(BaseModel):
    enabled: bool | None = None
    auto_act: bool | None = None
    warn_temp_c: int | None = None
    throttle_temp_c: int | None = None
    kill_temp_c: int | None = None
    cool_temp_c: int | None = None
    power_throttle_pct: int | None = None
    vram_pressure_pct: int | None = None

class AgentThermalActionBody(BaseModel):
    action: str

class AgentRouteBody(BaseModel):
    task: str = "coding"
    apply: bool = False
    sync_agents: bool = False
    agents: list[str] | None = None

class AgentSpeculateBody(BaseModel):
    model: str = ""
    draft_candidates: list[str] | None = None
    apply_best: bool = True

class AgentHFRecommendBody(BaseModel):
    query: str
    goal: str = "balanced"
    max_repos: int = 4

@app.get("/api/agent/thermal")
def api_agent_thermal():
    return THERMAL_GUARD.status()

@app.post("/api/agent/thermal/config")
def api_agent_thermal_config(b: AgentThermalConfigBody):
    patch = {k: v for k, v in b.model_dump().items() if v is not None}
    return {"ok": True, "config": THERMAL_GUARD.update_config(patch), "status": THERMAL_GUARD.status()}

@app.post("/api/agent/thermal/evaluate")
def api_agent_thermal_evaluate():
    return THERMAL_GUARD.evaluate_once()

@app.post("/api/agent/thermal/action")
def api_agent_thermal_action(b: AgentThermalActionBody):
    return THERMAL_GUARD.run_manual_action(b.action)

@app.post("/api/agent/route")
def api_agent_route(b: AgentRouteBody):
    try:
        return OPT_AGENT.route_workload(
            task=b.task, apply=b.apply, sync_agents=b.sync_agents, agents=b.agents,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/agent/speculate")
def api_agent_speculate(b: AgentSpeculateBody):
    try:
        return OPT_AGENT.tune_speculation(
            model=b.model, draft_candidates=b.draft_candidates, apply_best=b.apply_best,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))

@app.post("/api/agent/hf/recommend")
def api_agent_hf_recommend(b: AgentHFRecommendBody):
    try:
        return OPT_AGENT.recommend_hf(query=b.query, goal=b.goal, max_repos=b.max_repos)
    except Exception as e:
        raise HTTPException(502, f"HF recommendation failed: {e}")

class AgentProviderBody(BaseModel):
    provider: str | None = None
    url: str | None = None
    model: str | None = None
    api_key: str | None = None
    headers: str | None = None
    fallback_local: bool | None = None

class AgentProviderModelsBody(BaseModel):
    provider: str | None = None
    url: str | None = None
    api_key: str | None = None

@app.get("/api/agent/provider")
def api_agent_provider_get():
    akey = CFG.get("agent_llm_key", "")
    akey_masked = (akey[:4] + "…" + akey[-4:]) if len(akey) >= 8 else ""
    return {
        "presets": PROVIDER_PRESETS,
        "provider": CFG.get("agent_llm_provider", "custom"),
        "url": CFG.get("agent_llm_url", ""),
        "model": CFG.get("agent_llm_model", ""),
        "headers": CFG.get("agent_llm_headers", ""),
        "key_set": bool(akey),
        "key_masked": akey_masked,
        "fallback_local": bool(CFG.get("agent_llm_fallback_local", True)),
        "status": OPT_AGENT.llm_status(),
    }

@app.post("/api/agent/provider")
def api_agent_provider_set(b: AgentProviderBody):
    if b.provider is not None:
        CFG["agent_llm_provider"] = b.provider.strip() or "custom"
    if b.url is not None:
        CFG["agent_llm_url"] = b.url.strip()
    if b.model is not None:
        CFG["agent_llm_model"] = b.model.strip()
    if b.headers is not None:
        CFG["agent_llm_headers"] = b.headers.strip()
    if b.api_key is not None:
        val = b.api_key.strip()
        if val == "clear":
            CFG["agent_llm_key"] = ""
        elif val and val != "__keep__":
            CFG["agent_llm_key"] = val
    if b.fallback_local is not None:
        CFG["agent_llm_fallback_local"] = bool(b.fallback_local)
    _save_config_secure()
    return api_agent_provider_get()

@app.post("/api/agent/provider/models")
def api_agent_provider_models(b: AgentProviderModelsBody):
    return OPT_AGENT.fetch_provider_models(provider=b.provider, url=b.url, api_key=b.api_key)

@app.post("/api/agent/provider/test")
def api_agent_provider_test():
    return OPT_AGENT.test_provider()

# ---------------------------------------------------------------- chat api
@app.post("/api/chat")
async def api_chat(req: Request):
    tgt = _chat_target()
    if not tgt:
        raise HTTPException(409, "no model ready — load one in the Models, Strata, Avalon, or Maya tab")
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
<title>Inference Console · Local Inference Manager</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<script>
(() => {
  try {
    const s = localStorage.getItem('ic.theme');
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
  --bg:#080a0f; --bg-2:#0d1118; --panel:#111620; --panel-2:#161c28; --raise:#1c2434;
  --bd:#1f2838; --bd-2:#2c384e;
  --fg:#edf2f8; --fg-2:#c4cedd; --mut:#8595ab; --faint:#5a687e;
  --acc:#22d3ee; --acc-2:#f59e0b; --acc-ink:#04131a;
  --ok:#34d399; --warn:#f59e0b; --err:#f87171;
  --glow:radial-gradient(960px 440px at 88% -8%,rgba(34,211,238,.08) 0%,rgba(245,158,11,.03) 42%,rgba(0,0,0,0) 70%);
  --grad:linear-gradient(135deg,#22d3ee,#06b6d4);
  --sh:0 1px 3px rgba(0,0,0,.32);
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --r:11px;
  --rail:224px;
}
[data-theme=light] {
  --bg:#f4f6fa; --bg-2:#eaf0f6; --panel:#ffffff; --panel-2:#f9fbfd; --raise:#edf2f8;
  --bd:#d9e1ec; --bd-2:#c0cddc;
  --fg:#0f1724; --fg-2:#334155; --mut:#5b6b82; --faint:#8594a8;
  --acc:#0284c7; --acc-2:#d97706; --acc-ink:#ffffff;
  --ok:#059669; --warn:#b45309; --err:#dc2626;
  --glow:radial-gradient(960px 440px at 88% -8%,rgba(2,132,199,.07) 0%,rgba(0,0,0,0) 68%);
  --sh:0 1px 3px rgba(15,23,36,.07);
}
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;min-height:100vh;background-color:var(--bg);
  background-image:var(--glow),radial-gradient(color-mix(in srgb,var(--bd-2) 38%,transparent) 1px,transparent 1px);
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
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:var(--bd-2);border-radius:99px;border:2px solid transparent;background-clip:content-box}
::-webkit-scrollbar-track{background:transparent}

/* ---------- workbench shell (left rail + top command bar) ---------- */
.shell{display:grid;grid-template-columns:var(--rail) minmax(0,1fr);min-height:100vh}
.rail{position:sticky;top:0;height:100vh;display:flex;flex-direction:column;background:var(--panel);
  border-right:1px solid var(--bd);padding:12px 10px;gap:10px;overflow-y:auto;z-index:45}
.rail-top{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:2px 4px 10px;border-bottom:1px solid var(--bd)}
.brand{display:flex;align-items:center;gap:9px;min-width:0}
.mark{width:28px;height:28px;border-radius:8px;background:var(--grad);display:grid;place-items:center;
  color:var(--acc-ink);box-shadow:0 1px 3px rgba(0,0,0,.35);flex-shrink:0}
.mark svg{width:15px;height:15px}
.brand .nm{font-size:14px;font-weight:700;line-height:1.15;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.brand .sub{display:block;font-size:12px;font-weight:500;color:var(--mut);font-family:var(--mono)}
.nav{display:flex;flex-direction:column;gap:12px;flex:1;min-height:0}
.nav-grp{display:flex;flex-direction:column;gap:2px}
.nav-lbl{font-size:12px;font-weight:700;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;padding:4px 9px 2px}
.nav button{display:flex;align-items:center;gap:9px;padding:7px 9px;border:1px solid transparent;border-radius:8px;background:transparent;
  color:var(--mut);font:inherit;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;text-align:left;width:100%;
  transition:background .12s,color .12s,border-color .12s}
.nav button svg{width:15px;height:15px;opacity:.8;flex-shrink:0}
.nav button .nl{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis}
.nav button .nb{font-size:11px;font-family:var(--mono);color:var(--fg-2);background:var(--bg-2);border:1px solid var(--bd);
  border-radius:999px;padding:0 6px;line-height:1.5;font-variant-numeric:tabular-nums}
.nav button .ndot{width:7px;height:7px;border-radius:99px;background:var(--faint);flex-shrink:0}
.nav button .ndot.ok{background:var(--ok)}
.nav button .ndot.warn{background:var(--warn)}
.nav button:hover{color:var(--fg);background:var(--raise)}
.nav button.on{color:var(--fg);background:var(--raise);border-color:var(--bd-2)}
.nav button.on svg{color:var(--acc);opacity:1}
.rail-foot{border-top:1px solid var(--bd);padding-top:10px;display:flex;flex-direction:column;gap:8px}
.rail-vitals{padding:8px 9px;border-radius:8px;background:var(--bg-2);border:1px solid var(--bd);cursor:pointer;transition:border-color .14s}
.rail-vitals:hover{border-color:var(--bd-2)}
.rv-row{display:flex;justify-content:space-between;align-items:center;font-size:12px;font-family:var(--mono);color:var(--fg-2);font-variant-numeric:tabular-nums}
.rv-row span:last-child{color:var(--mut)}

.work{display:flex;flex-direction:column;min-width:0;min-height:100vh}
header.topbar{position:sticky;top:0;z-index:40;display:flex;align-items:center;gap:10px;padding:9px 18px;
  background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(12px) saturate(140%);-webkit-backdrop-filter:blur(12px) saturate(140%);
  border-bottom:1px solid var(--bd)}
.mob-menu{display:none}
.view-title{font-size:14.5px;font-weight:700;letter-spacing:-.015em;white-space:nowrap}
.hgap{flex:1;min-width:4px}
.tele-strip{display:flex;align-items:center;gap:6px;flex-wrap:nowrap;overflow:hidden}
.pill{display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:999px;border:1px solid var(--bd-2);
  background:var(--panel);color:var(--fg-2);font-size:12px;font-weight:600;white-space:nowrap;max-width:32vw;overflow:hidden;
  font-variant-numeric:tabular-nums;cursor:pointer;transition:border-color .14s,background .14s}
.pill:hover{border-color:var(--mut);background:var(--raise)}
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
main{padding:18px 20px 36px;max-width:1360px;width:100%;margin:0 auto;flex:1}
section{animation:fade .16s ease-out}
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
.shardlist{display:grid;gap:2px;margin:-4px 0 6px;padding:6px 10px;border:1px solid var(--border);border-radius:8px;background:var(--surface)}
.shardlist .shrow{display:flex;justify-content:space-between;gap:10px;padding:2px 4px;border-radius:5px;cursor:copy}
.shardlist .shrow:hover{background:var(--border)}
.shardlist .shrow code{font-size:10.5px;font-family:var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.shardlist .shrow .hint{flex:none}
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
footer{max-width:1360px;width:100%;margin:0 auto;padding:0 20px 22px;color:var(--faint);font-size:11.5px;
  display:flex;gap:14px;flex-wrap:wrap;align-items:center}
kbd{font-family:var(--mono);font-size:10.5px;border:1px solid var(--bd-2);border-radius:5px;
  padding:1px 5px;background:var(--bg-2);color:var(--fg-2)}

/* ---------- responsive mobile rail ---------- */
.rail-backdrop{display:none;position:fixed;inset:0;background:rgba(4,7,12,.6);z-index:44}
@media (max-width:960px){
  .shell{grid-template-columns:1fr}
  .rail{position:fixed;left:0;top:0;bottom:0;width:250px;transform:translateX(-105%);transition:transform .2s ease;box-shadow:0 4px 12px rgba(0,0,0,.5)}
  body.nav-open .rail{transform:translateX(0)}
  body.nav-open .rail-backdrop{display:block}
  .mob-menu{display:grid}
  header.topbar{padding:8px 12px}
  main{padding:14px 12px 32px}
  .two>.side{width:100%;flex:1 1 100%}
  .mrow{grid-template-columns:auto 1fr}
  .mrow .fitbox{grid-column:1 / -1}
  .mrow .acts{grid-column:1 / -1;justify-content:flex-start}
  .thread{max-height:none}
}
/* ---------- agent chat sidebar ---------- */
.agent-drawer{position:fixed;top:0;right:0;bottom:0;width:min(410px,94vw);background:var(--panel);
  border-left:1px solid var(--bd-2);box-shadow:-8px 0 28px rgba(0,0,0,.45);z-index:85;
  display:flex;flex-direction:column;transform:translateX(100%);transition:transform .2s ease}
.agent-drawer.open{transform:translateX(0)}
.agent-drawer-hd{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px 14px;border-bottom:1px solid var(--bd)}
.agent-drawer-hd h2{font-size:14px;font-weight:650;margin:0;display:flex;align-items:center;gap:6px}
.agent-drawer-bd{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:10px}
.agent-drawer-ft{padding:10px 12px;border-top:1px solid var(--bd);display:flex;flex-direction:column;gap:8px;background:var(--bg-2)}
.agent-msg{padding:9px 11px;border-radius:10px;border:1px solid var(--bd);background:var(--bg-2);font-size:12.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
.agent-msg.usr{background:color-mix(in srgb,var(--acc) 12%,var(--panel));border-color:color-mix(in srgb,var(--acc) 35%,var(--bd));align-self:flex-end;max-width:88%}
.agent-msg.bot{align-self:flex-start;max-width:96%}
.agent-acts{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.agent-chips{display:flex;gap:5px;flex-wrap:wrap}

@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms !important;animation-iteration-count:1 !important;transition-duration:.01ms !important}
}
</style></head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="rail-backdrop" id="railbackdrop"></div>
<div class="shell">
<aside class="rail" id="rail" aria-label="Primary navigation">
  <div class="rail-top">
    <div class="brand">
      <span class="mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M13.2 2 4 14h6l-1.4 8L20 9.5h-6.2z"/></svg></span>
      <span class="nm">Inference Console<span class="sub">local model workbench</span></span>
    </div>
  </div>
  <nav class="nav" id="nav" aria-label="Sections">
    <div class="nav-grp">
      <div class="nav-lbl">Workspace</div>
      <button data-t="models" class="on"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z"/><path d="M4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg><span class="nl">Models</span><span class="nb" id="nb-models">0</span></button>
      <button data-t="agent"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2a4 4 0 014 4v1h2a2 2 0 012 2v6a5 5 0 01-5 5H9a5 5 0 01-5-5V9a2 2 0 012-2h2V6a4 4 0 014-4z"/><circle cx="9.5" cy="13" r="1"/><circle cx="14.5" cy="13" r="1"/></svg><span class="nl">Agent</span></button>
      <button data-t="chat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 12a8 8 0 01-8 8H7l-4 3 1.2-4.2A8 8 0 1121 12z"/></svg><span class="nl">Chat</span></button>
    </div>
    <div class="nav-grp">
      <div class="nav-lbl">Engines</div>
      <button data-t="server"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><path d="M7 7.5h.01M7 16.5h.01"/></svg><span class="nl">llama.cpp</span><span class="ndot" id="nd-server" title="llama.cpp status"></span></button>
      <button data-t="strata"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5"/><path d="M3 17l9 5 9-5"/></svg><span class="nl">Strata</span><span class="ndot" id="nd-strata" title="Strata status"></span></button>
      <button data-t="avalon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/></svg><span class="nl">Avalon</span><span class="ndot" id="nd-avalon" title="Avalon status"></span></button>
      <button data-t="maya"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><circle cx="12" cy="12" r="3"/></svg><span class="nl">Maya</span><span class="ndot" id="nd-maya" title="Maya status"></span></button>
    </div>
    <div class="nav-grp">
      <div class="nav-lbl">System</div>
      <button data-t="vitals"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 12h4l2.5-6 4 12L16 12h5"/></svg><span class="nl">Vitals</span></button>
      <button data-t="hf"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v12m0 0l-4-4m4 4l4-4"/><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"/></svg><span class="nl">HF Pull</span><span class="nb hide" id="nb-dl">0</span></button>
      <button data-t="settings"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09a1.65 1.65 0 00-1-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09a1.65 1.65 0 001.51-1 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06a1.65 1.65 0 001.82.33h.08a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51h.08a1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82v.08a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg><span class="nl">Settings</span></button>
    </div>
  </nav>
  <div class="rail-foot">
    <div class="rail-vitals" id="railvitals" title="Click to open System Vitals">
      <div class="rv-row"><span id="rv-mem-lbl">Memory</span><span id="rv-mem-val">…</span></div>
      <div class="meter" id="rv-mem-bar" style="margin:4px 0 6px"><i style="width:0%"></i></div>
      <div class="rv-row"><span>CPU</span><span id="rv-cpu-val">…</span></div>
    </div>
  </div>
</aside>

<div class="work">
<header class="topbar">
  <button class="icon mob-menu" id="mobmenu" aria-label="Toggle navigation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
  <span class="view-title" id="viewtitle">Models</span>
  <span class="hgap"></span>
  <div class="tele-strip">
    <button type="button" class="pill" id="hdrstatus" title="Active inference server (click to inspect)"><span class="dot"></span><span id="hdrtext">checking…</span></button>
    <span class="tag acc hide" id="hdrtps" title="Live generation throughput"></span>
    <span class="tag mut hide" id="hdrkv" title="KV cache fill"></span>
    <span class="tag mut hide" id="hdridle" title="Idle auto-unload timer"></span>
  </div>
  <button class="cmdbtn" id="agentdrawerbtn" title="Toggle Optimization Agent Chat Sidebar" aria-label="Toggle Agent Chat Sidebar">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 014 4v1h2a2 2 0 012 2v6a5 5 0 01-5 5H9a5 5 0 01-5-5V9a2 2 0 012-2h2V6a4 4 0 014-4z"/></svg>
    <span>Agent</span>
  </button>
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
      <button class="btn acc2" data-agent-quick="llama:tune">Agent Auto-Tune</button>
      <button class="btn" data-agent-quick="llama:diagnose">Diagnose failure</button>
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

<!-- ==================== STRATA ==================== -->
<section id="t-strata" class="hide">
  <div class="card">
    <div class="hd"><h2>Strata · GGUF server engine</h2><span class="tag" id="stratchip">checking…</span></div>
    <div id="stratainstall" class="hide" style="margin-bottom:10px">
      <p class="hint" style="margin:0 0 8px">Strata is a standalone inference engine (OpenAI/Anthropic-compatible server on localhost) with
        one-command model setup. Install it, or point the panel at an existing launcher in Settings.</p>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <code id="stratainstcmd" class="cmdbox" style="flex:1;min-width:300px"></code>
        <button class="btn xs" id="stratacopyinst">Copy install command</button>
      </div>
    </div>
    <div id="stratadetail" class="kvs"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center">
      <button class="btn primary" id="st-toggleform">Launch setup…</button>
      <button class="btn danger" id="stratastopbtn" disabled>Stop</button>
      <button class="btn" id="stratadoctor">Run doctor</button>
      <button class="btn" id="strataupdate">Check update</button>
      <button class="btn acc2" data-agent-quick="strata:tune">Agent Auto-Tune</button>
      <button class="btn" data-agent-quick="strata:diagnose">Diagnose</button>
      <button class="btn" id="strataagentsbtn" title="Point omp, opencode, codex, aider, zed, pi, claude-code, hermes, kimi… at the running Strata server">Register in coding agents…</button>
      <span class="tag hide" id="stratakeychip" title="API key of the running launch (strata launch --show-key)"></span>
    </div>
  </div>

  <div class="card hide" id="strataformcard" style="margin-top:14px">
    <div class="hd"><h2>Launch configuration</h2><span class="tag mut">only filled fields are passed, Strata defaults stay in charge</span></div>
    <div class="form-grid">
      <div class="field"><label>Preset</label><select id="st-preset"><option value="">(none)</option></select></div>
      <div class="field"><label>Family</label><select id="st-family"><option value="">auto</option><option value="qwen">qwen (Flash-Next)</option><option value="coder">coder</option><option value="unsloth">unsloth</option><option value="gyro">gyro</option><option value="swift">swift</option><option value="nvfp4">nvfp4</option><option value="qwen36">qwen36 (35B)</option></select></div>
      <div class="field"><label>Model / quant</label><input id="st-model" type="text" placeholder="e.g. iq3_xxs, q2_0, UD-Q4_K_XL"></div>
      <div class="field"><label>Model path (files I already have)</label><input id="st-modelpath" type="text" placeholder="/path/to/model folder or .gguf"></div>
      <div class="field"><label>Context</label><select id="st-context"><option value="">default</option><option>8k</option><option>32k</option><option>64k</option><option>128k</option><option>256k</option><option>512k</option></select></div>
      <div class="field"><label>KV format (--kv)</label><select id="st-kv"><option value="">default (int8)</option><option>int8</option><option>q4_0</option><option>k8v4</option><option>fp16</option></select></div>
      <div class="field"><label>Parallel requests</label><select id="st-parallel"><option value="">fit to card</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option>8</option></select></div>
      <div class="field"><label>Vision</label><select id="st-vision"><option value="">auto</option><option>on</option><option>off</option><option>gpu</option><option>cpu</option></select></div>
      <div class="field"><label>MTP draft layer (--mtp)</label><select id="st-mtp"><option value="">auto (on)</option><option>on</option><option>off</option></select></div>
      <div class="field"><label>Drafting mode</label><select id="st-drafting"><option value="">auto</option><option>mtp</option><option>dflash2</option><option>off</option></select></div>
      <div class="field"><label>Tuning</label><select id="st-tune"><option value="">skip (CLI default)</option><option>quick</option><option>thorough</option></select></div>
      <div class="field"><label>Access</label><select id="st-access"><option value="">local (127.0.0.1)</option><option>local</option><option>lan</option><option>cloudflare</option><option>tailscale</option></select></div>
      <div class="field"><label>Port</label><input id="st-port" type="number" min="1024" max="65535" placeholder="8080"></div>
      <div class="field"><label>GPU cards (--gpu)</label><input id="st-gpu" type="text" placeholder="e.g. 0,1 — default: all"></div>
      <div class="field"><label>Save as preset</label><input id="st-savepreset" type="text" placeholder="preset name (optional)"></div>
      <div class="field" style="grid-column:1 / -1"><label>Advanced engine settings (--set, space-separated key=value)</label><input id="st-set" type="text" placeholder="e.g. prefill=65536 STRATA_PIN_BY_NEED=1"></div>
      <div class="field" style="grid-column:1 / -1">
        <div style="display:flex;gap:16px;flex-wrap:wrap;padding-top:4px">
          <label class="switch"><input type="checkbox" id="st-restart"><span class="track"></span><span class="hint">restart if busy</span></label>
          <label class="switch"><input type="checkbox" id="st-retune"><span class="track"></span><span class="hint">re-tune</span></label>
          <label class="switch"><input type="checkbox" id="st-newkey"><span class="track"></span><span class="hint">new API key</span></label>
          <label class="switch"><input type="checkbox" id="st-nokey"><span class="track"></span><span class="hint">no API key</span></label>
        </div>
      </div>
    </div>
    <div style="margin-top:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span class="hint">Command preview</span>
        <button type="button" class="btn xs" id="st-copycmd">Copy command</button>
      </div>
      <div class="cmdbox" id="st-cmd"></div>
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:12px;flex-wrap:wrap">
      <button class="btn" id="st-plan">Preview plan (read-only)</button>
      <button class="btn primary" id="st-launch">Download &amp; Launch</button>
    </div>
    <div class="cmdbox hide" id="st-out" style="margin-top:12px;white-space:pre-wrap;max-height:340px;overflow:auto"></div>
  </div>

  <div class="card" style="margin-top:14px">
    <div class="hd"><h2>Strata presets</h2><button class="btn xs" id="st-refreshpresets">Refresh</button></div>
    <div id="st-presets" class="kvs"></div>
  </div>

  <div class="card hide" id="stratametricscard" style="margin-top:14px">
    <div class="hd"><h2>Strata live metrics</h2><span class="tag acc" id="st-metrics-chip">polling</span></div>
    <div id="st-metrics" class="kvs"></div>
  </div>

  <div class="card" style="margin-top:14px">
    <div class="hd"><h2>Strata log</h2><span class="tag mut" id="stlogn"></span></div>
    <div class="logbar">
      <div class="search" style="flex:1"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="stlogfilter" type="search" placeholder="Filter Strata log…" aria-label="Filter Strata log"></div>
      <label class="switch"><input type="checkbox" id="stlogauto" checked><span class="track"></span><span class="hint">Follow</span></label>
      <button class="btn sm" id="stlogcopy">Copy</button>
    </div>
    <pre class="log" id="stratalog" style="max-height:280px"></pre>
  </div>
</section>

<!-- ==================== AVALON ==================== -->
<section id="t-avalon" class="hide">
  <div class="card">
    <div class="hd"><h2>Avalon · Qwen3.8-Flash-Next (125B MoE)</h2><span class="tag" id="avalonchip">checking…</span></div>
    <div id="avalondetail" class="kvs" style="margin-bottom:10px"></div>
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
      <span class="hint">Avalon target</span>
      <select id="avtargetsel" style="min-width:280px;flex:1"><option>loading…</option></select>
      <span class="hint" id="avtargetdraft"></span>
    </div>
    <div id="avalonprofiles"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center">
      <button class="btn primary" id="avalonstart">Start selected profile</button>
      <button class="btn danger" id="avalonstop">Stop</button>
      <button class="btn acc2" id="avalontunebtn">Config Optimizer &amp; Tuning…</button>
      <button class="btn acc2" data-agent-quick="avalon:tune">Agent Auto-Tune</button>
      <button class="btn" data-agent-quick="avalon:diagnose">Diagnose</button>
      <button class="btn" id="avalonagentsbtn" title="Point omp, opencode, codex, aider, zed, pi, claude-code, hermes, kimi… at this Avalon server">Register in coding agents…</button>
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
        <div class="field"><label>GPU device (--device)</label>
          <select id="av-device">
            <option value="ROCm0">ROCm0 · RX 7900 XT eGPU (20 GB, recommended)</option>
            <option value="ROCm1">ROCm1 · 780M iGPU (32 GB GTT, shares system RAM)</option>
            <option value="ROCm0,ROCm1">ROCm0 + ROCm1 split</option>
          </select>
          <div class="hint">ROCm0 only. The iGPU has no FA kernels in this ROCm build, so any launch that touches it dies with “invalid device function”.</div></div>
        <div class="field"><label>GPU layers (-ngl) · empty = auto-fit</label>
          <input id="av-ngl" type="number" placeholder="auto-fit (-fit on)" min="0" max="999">
          <div class="hint">Recommended: leave empty. The fork’s <code>-fit</code> measures dense weights + KV + buffers against device memory and places what fits. A fixed <code>-ngl 99</code> only works when the whole model fits in VRAM.</div></div>
        <div class="field"><label>MoE experts in CPU RAM (--n-cpu-moe)</label>
          <input id="av-cmoe" type="number" placeholder="fork default (none forced)" min="0" max="999">
          <div class="hint">Recommended: 999 for this pack. Expert weights are 48 GiB vs 20 GB VRAM; forcing them to CPU lets <code>-fit</code> keep attention + KV on the eGPU. Empty = fork decides.</div></div>
        <div class="field"><label>Flash Attention (-fa)</label>
          <select id="av-fa"><option value="">default (on)</option><option value="on">on (recommended on ROCm0)</option><option value="off">off</option></select>
          <div class="hint">On cuts KV memory and speeds prefill on NAVI31. Only relevant if you force the iGPU, which lacks the kernels anyway.</div></div>
        <div class="field"><label>CPU threads (-t)</label>
          <input id="av-threads" type="number" placeholder="fork default (all cores)" min="1" max="256">
          <div class="hint">Threads for layers that stay on CPU. Recommended: physical cores (8 on this box), since SMT threads add contention.</div></div>
        <div class="field"><label>Lock weights in RAM (--mlock)</label>
          <select id="av-mlock"><option value="">off (recommended)</option><option value="on">on</option></select>
          <div class="hint">With 48 GiB of experts, mlock would pin them resident and OOM this 61 GB machine. Leave off; the page cache does the job.</div></div>
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
    <div class="hd"><h2>One-click launches</h2>
      <div style="display:flex;gap:6px;align-items:center">
        <input id="avsnapname" type="text" placeholder="name this launch…" style="width:220px">
        <button class="btn sm" id="avsnapsave">Save current</button>
      </div></div>
    <div class="hint" style="margin-bottom:8px">A snapshot freezes the profile and every tuning field exactly as resolved right now, and relaunches with one click using the same command line.</div>
    <div id="avsnaps"></div>
  </div>

  <div class="card">
    <div class="hd"><h2>Network access</h2>
      <label class="switch"><input type="checkbox" id="avalonlan"><span class="track"></span><span class="hint">Expose to LAN</span></label></div>
    <div id="avalonaccess" class="kvs"></div>
  </div>
  <div class="card">
    <div class="hd"><h2>Avalon log</h2><span class="tag mut" id="avlogn"></span></div>
    <div class="logbar">
      <div class="search" style="flex:1"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="avlogfilter" type="search" placeholder="Filter Avalon log…" aria-label="Filter Avalon log"></div>
      <label class="switch"><input type="checkbox" id="avlogauto" checked><span class="track"></span><span class="hint">Follow</span></label>
      <button class="btn sm" id="avlogcopy">Copy</button>
    </div>
    <pre class="log" id="avalonlog" style="max-height:280px"></pre>
  </div>
</section>

<!-- ==================== MAYA ==================== -->
<section id="t-maya" class="hide">
  <div class="card">
    <div class="hd"><h2>Project Maya</h2><span class="tag mut" id="mayachip">checking…</span></div>
    <div id="mayadetail" class="kvs" style="margin-bottom:10px"></div>

    <div class="grid g4" style="margin-bottom:12px">
      <div class="card sub" style="margin:0">
        <div class="hd" style="margin-bottom:4px"><h2>Decode speed</h2><span class="tag mut" id="m-state-tag">idle</span></div>
        <div class="stat"><span id="m-dec-tps">—</span> <small>tok/s</small></div>
        <p class="hint" style="margin:4px 0 0" id="m-dec-sub">Waiting for log telemetry</p>
      </div>
      <div class="card sub" style="margin:0">
        <div class="hd" style="margin-bottom:4px"><h2>Prefill speed</h2></div>
        <div class="stat"><span id="m-pre-tps">—</span> <small>tok/s</small></div>
        <p class="hint" style="margin:4px 0 0">Prompt processing rate</p>
      </div>
      <div class="card sub" style="margin:0">
        <div class="hd" style="margin-bottom:4px"><h2>VRAM &amp; Experts</h2></div>
        <div class="stat"><span id="m-vram">—</span> <small>GB</small></div>
        <p class="hint" style="margin:4px 0 0" id="m-exp-sub">Experts in VRAM: —</p>
      </div>
      <div class="card sub" style="margin:0">
        <div class="hd" style="margin-bottom:4px"><h2>Context &amp; Load</h2></div>
        <div class="stat"><span id="m-ctx">—</span> <small>ctx</small></div>
        <p class="hint" style="margin:4px 0 0" id="m-load-sub">Load / warmup: —</p>
      </div>
    </div>

    <div id="mayaconfigs"></div>

    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center">
      <button class="btn primary" id="mayastart">Start selected</button>
      <button class="btn acc2" id="mayaswitch" title="Stop the running Maya config and start the selected one">Switch to selected</button>
      <button class="btn danger" id="mayastop">Stop</button>
      <button class="btn acc2" data-agent-quick="maya:tune">Agent Auto-Tune</button>
      <button class="btn" data-agent-quick="maya:diagnose">Diagnose</button>
      <button class="btn" id="mayaagentsbtn" title="Point coding agents at this Maya server">Register in coding agents…</button>
      <button class="btn sm" id="mayakeybtn" title="Reveal masked API key for the active Maya config">Reveal API key</button>
      <span class="hint">OpenAI &amp; Anthropic API · exclusive GPU · <code>ROCM_PATH</code> managed automatically</span>
    </div>
  </div>

  <div class="card">
    <div class="hd">
      <h2>Maya engine log</h2>
      <div class="meta">
        <span class="hint mono" id="mayalogpath"></span>
        <button class="btn sm" id="mayalogref">Refresh</button>
      </div>
    </div>
    <pre class="log" id="mayalog" style="max-height:300px"></pre>
  </div>
</section>

<!-- ==================== AGENT / OPTIMIZER ==================== -->
<section id="t-agent" class="hide">
  <div class="card">
    <div class="hd">
      <h2>Engine Optimization Agent</h2>
      <div class="meta">
        <span class="tag mut" id="ag-llm-chip">deterministic</span>
        <span class="tag mut" id="ag-tune-chip">idle</span>
        <button class="btn sm" id="ag-open-sidebar">Open Chat Sidebar</button>
      </div>
    </div>
    <p class="hint" style="margin:0 0 12px">MoE-aware, doc-grounded optimizer across llama.cpp, Avalon, Strata, and Project Maya. Every Apply or Auto-Tune run automatically creates a safety checkpoint first so you can revert at any time.</p>
    <div class="form-grid">
      <div class="field">
        <label>Target engine</label>
        <select id="ag-engine">
          <option value="llama">llama.cpp</option>
          <option value="avalon">Avalon</option>
          <option value="strata">Strata</option>
          <option value="maya">Project Maya</option>
        </select>
      </div>
      <div class="field">
        <label>Optimization goal</label>
        <select id="ag-goal">
          <option value="fastest">Fastest (max gen tok/s)</option>
          <option value="low_latency">Lowest latency (min TTFT / prefill)</option>
          <option value="max_context">Longest stable context</option>
          <option value="balanced" selected>Balanced</option>
          <option value="low_vram">Low VRAM / cool</option>
          <option value="multi_user">Multi-user / parallel slots</option>
        </select>
      </div>
      <div class="field" id="ag-model-wrap">
        <label>Model / profile</label>
        <select id="ag-model"><option value="">(auto-select active / pinned)</option></select>
      </div>
      <div class="field">
        <label>Min context constraint (optional)</label>
        <input id="ag-minctx" type="number" min="0" step="4096" placeholder="e.g. 32768">
      </div>
      <div class="field">
        <label>Extra VRAM headroom MB (optional)</label>
        <input id="ag-headroom" type="number" min="0" step="512" placeholder="e.g. 4096">
      </div>
      <div class="field">
        <label>Max auto-tune trials</label>
        <input id="ag-trials" type="number" min="1" max="12" value="5">
      </div>
      <div class="field" style="grid-column:1 / -1">
        <label>Custom preference / constraint (natural language — works deterministically + with optional LLM)</label>
        <input id="ag-prompt" type="text" placeholder="e.g. longest context with at least 64k tokens and leave 4 GB VRAM free">
      </div>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center">
      <button class="btn primary" id="ag-advise">Dry-Run Advise</button>
      <button class="btn acc2" id="ag-tune-start">Start Active Auto-Tune</button>
      <button class="btn" id="ag-diagnose">Diagnose Engine Failure</button>
      <button class="btn" id="ag-cp-now">Create Checkpoint</button>
    </div>
  </div>

  <!-- Agent LLM Provider (Hermes-style presets + Custom OpenAI-compatible + Always-On Model Polling) -->
  <div class="card">
    <div class="hd">
      <h2>Agent LLM Provider</h2>
      <div class="meta">
        <span class="tag mut" id="ag-prov-models-chip">models: —</span>
        <button class="btn xs" id="ag-prov-poll">Poll Models</button>
        <button class="btn xs" id="ag-prov-test">Test Connection</button>
        <button class="btn primary xs" id="ag-prov-save">Save Provider</button>
      </div>
    </div>
    <p class="hint" style="margin:0 0 10px">Choose any common provider (OpenAI, Anthropic, OpenRouter, Gemini, DeepSeek, xAI, Groq, Together, Fireworks, Mistral, Moonshot/Kimi, Z.ai, Nous/Hermes, Ollama, LM Studio, vLLM) or a custom OpenAI-compatible URL. Models are polled automatically.</p>
    <div class="form-grid">
      <div class="field">
        <label>Provider</label>
        <select id="ag-prov-select"></select>
      </div>
      <div class="field">
        <label>Base URL (editable for custom providers)</label>
        <input id="ag-prov-url" type="text" placeholder="https://api.example.com/v1">
      </div>
      <div class="field">
        <label>API Key <span id="ag-prov-key-status"></span></label>
        <input id="ag-prov-key" type="password" placeholder="(leave blank to keep existing; type 'clear' to remove)">
      </div>
      <div class="field">
        <label>Model (auto-polled or type custom ID)</label>
        <div style="display:flex;gap:6px">
          <input id="ag-prov-model" type="text" list="ag-prov-models-dl" placeholder="select or type model id" style="flex:1">
          <select id="ag-prov-model-pick" aria-label="Polled models" style="max-width:48%"><option value="">— polled models —</option></select>
        </div>
        <datalist id="ag-prov-models-dl"></datalist>
      </div>
      <div class="field">
        <label>Custom Headers JSON (optional)</label>
        <input id="ag-prov-headers" type="text" placeholder='e.g. {"HTTP-Referer":"https://localhost"}'>
      </div>
      <div class="field" style="justify-content:center">
        <label class="switch"><input type="checkbox" id="ag-prov-fallback" checked><span class="track"></span><span class="hint">Fall back to active local engine when external LLM is unset</span></label>
      </div>
    </div>
    <div id="ag-prov-msg" class="hint" style="margin-top:8px"></div>
  </div>

  <div class="card hide" id="ag-result-card">
    <div class="hd">
      <h2 id="ag-result-title">Recommendation</h2>
      <div class="meta">
        <button class="btn primary sm hide" id="ag-apply-btn">Checkpoint, Apply &amp; Launch</button>
        <button class="btn sm hide" id="ag-save-preset-btn">Save as Preset</button>
      </div>
    </div>
    <div id="ag-result-moe" class="kvs" style="margin-bottom:8px"></div>
    <div id="ag-result-notes" style="margin-bottom:10px;font-size:12.5px;line-height:1.5"></div>
    <pre class="cmdbox" id="ag-result-cfg" style="white-space:pre-wrap;max-height:240px;overflow:auto"></pre>
  </div>

  <div class="card hide" id="ag-tune-card">
    <div class="hd">
      <h2>Active Auto-Tune Session</h2>
      <div class="meta">
        <span class="tag acc" id="ag-tune-state">running</span>
        <button class="btn danger sm" id="ag-tune-stop">Stop</button>
        <button class="btn danger sm" id="ag-tune-rollback">Stop &amp; Rollback</button>
      </div>
    </div>
    <div id="ag-tune-trials" style="margin-bottom:10px"></div>
    <pre class="log" id="ag-tune-log" style="max-height:200px"></pre>
  </div>

  <div class="grid">
    <div class="card">
      <div class="hd">
        <h2>Safety Checkpoints</h2>
        <button class="btn xs" id="ag-cp-refresh">Refresh</button>
      </div>
      <div id="ag-checkpoints"><span class="hint">No checkpoints yet.</span></div>
    </div>
    <div class="card">
      <div class="hd">
        <h2>Reusable Presets</h2>
        <button class="btn xs" id="ag-pre-refresh">Refresh</button>
      </div>
      <div id="ag-presets"><span class="hint">Loading presets…</span></div>
    </div>
  </div>

  <div class="card">
    <div class="hd">
      <h2>Persistent Memory &amp; VRAM Calibration</h2>
      <div class="meta">
        <button class="btn xs" id="ag-mem-refresh">Refresh</button>
        <button class="btn xs danger" id="ag-mem-clear">Clear Memory</button>
      </div>
    </div>
    <div id="ag-memory"><span class="hint">No empirical trials recorded yet.</span></div>
  </div>

  <!-- Phase 2: Workload Router & Speculative Decoding -->
  <div class="grid">
    <div class="card">
      <div class="hd"><h2>Workload Router &amp; Coding-Agent Sync</h2></div>
      <p class="hint" style="margin:0 0 10px">Pick a workload — the agent selects the best local model + engine + config (MoE-aware &amp; memory-backed) and can sync coding agents automatically.</p>
      <div class="form-grid">
        <div class="field">
          <label>Workload / Task</label>
          <select id="ag-route-task">
            <option value="coding">Coding Agent (≥32k ctx, fast)</option>
            <option value="long_doc">Long-Doc Analysis (max context)</option>
            <option value="fast_chat">Fast Chat (low latency / TTFT)</option>
            <option value="vision">Vision (mmproj)</option>
            <option value="reasoning">Deep Reasoning / Large MoE</option>
          </select>
        </div>
        <div class="field" style="justify-content:center">
          <label class="switch"><input type="checkbox" id="ag-route-sync" checked><span class="track"></span><span class="hint">Auto-sync coding agents (omp, claude-code, codex, aider…)</span></label>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
        <button class="btn" id="ag-route-preview">Preview Selection</button>
        <button class="btn primary" id="ag-route-apply">Launch &amp; Sync</button>
      </div>
      <div id="ag-route-out" style="margin-top:10px;font-size:12.5px"></div>
    </div>

    <div class="card">
      <div class="hd"><h2>Speculative Decoding Auto-Pairing</h2></div>
      <p class="hint" style="margin:0 0 10px">Benchmarks baseline (no draft) vs candidate draft/MTP models in your library and disables speculation automatically if overhead hurts net tok/s.</p>
      <div class="field">
        <label>Target model (llama.cpp)</label>
        <select id="ag-spec-model"><option value="">(auto-select active / pinned)</option></select>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
        <button class="btn acc2" id="ag-spec-run">Auto-Pair &amp; Benchmark Drafts</button>
      </div>
      <div id="ag-spec-out" style="margin-top:10px;font-size:12.5px"></div>
    </div>
  </div>

  <!-- Phase 2: Graduated Thermal & Shared-GPU Guard -->
  <div class="card">
    <div class="hd">
      <h2>Graduated Thermal &amp; Shared-GPU Guard</h2>
      <div class="meta">
        <span class="tag mut" id="tg-state-chip">normal</span>
        <button class="btn xs" id="tg-eval-now">Evaluate Now</button>
      </div>
    </div>
    <p class="hint" id="tg-hw-hint" style="margin:0 0 10px">Ladder: (1) Ramp GPU fans to 100% &amp; throttle power cap → (2) Checkpoint &amp; downshift to low-VRAM profile → (3) Emergency killswitch. Auto-restores fans &amp; power once cool.</p>
    <div class="form-grid">
      <div class="field"><label>Warn / Fan+Power (°C)</label><input id="tg-warn" type="number" value="80"></div>
      <div class="field"><label>Downshift (°C)</label><input id="tg-throttle" type="number" value="88"></div>
      <div class="field"><label>Killswitch (°C)</label><input id="tg-kill" type="number" value="95"></div>
      <div class="field"><label>Cool / Auto-Restore (°C)</label><input id="tg-cool" type="number" value="70"></div>
      <div class="field"><label>Power Throttle (%)</label><input id="tg-pwrpct" type="number" value="15" min="5" max="40"></div>
      <div class="field"><label>VRAM Pressure (%)</label><input id="tg-vrampct" type="number" value="92" min="60" max="99"></div>
    </div>
    <div style="display:flex;gap:14px;flex-wrap:wrap;margin:10px 0;align-items:center">
      <label class="switch"><input type="checkbox" id="tg-enabled"><span class="track"></span><span class="hint">Enable background watchdog</span></label>
      <label class="switch"><input type="checkbox" id="tg-autoact"><span class="track"></span><span class="hint">Auto-execute ladder actions</span></label>
      <button class="btn primary sm" id="tg-save">Save Guard Policy</button>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
      <button class="btn xs" data-tg-act="max_fans">Max Fans</button>
      <button class="btn xs" data-tg-act="throttle_power">Throttle Power</button>
      <button class="btn xs" data-tg-act="restore_hw">Restore Auto Fan/Power</button>
      <button class="btn xs" data-tg-act="downshift">Downshift Profile</button>
      <button class="btn xs danger" data-tg-act="killswitch">Emergency Killswitch</button>
    </div>
    <div id="tg-events" class="hint"></div>
  </div>

  <!-- Phase 2: Hardware-Aware HF Model / Quant Recommender -->
  <div class="card">
    <div class="hd"><h2>Hardware-Aware Hugging Face Recommender</h2></div>
    <p class="hint" style="margin:0 0 10px">Searches HF GGUF repos and scores quants against your real VRAM + RAM with MoE-aware hybrid placement (pairs matching mmproj/draft files automatically).</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input id="ag-hf-q" type="text" placeholder="e.g. Qwen3-Coder, DeepSeek, Llama-3.3, Gemma-3…" style="flex:1;min-width:220px">
      <select id="ag-hf-goal">
        <option value="balanced">Balanced</option>
        <option value="fastest">Fastest (Full GPU)</option>
        <option value="max_context">Max Context</option>
        <option value="low_vram">Low VRAM</option>
      </select>
      <button class="btn primary" id="ag-hf-go">Recommend Quants</button>
    </div>
    <div id="ag-hf-out" style="margin-top:10px"></div>
  </div>
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
          <label>System Prompt <button type="button" class="btn xs" id="sp-reset">Reset defaults</button></label>
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
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin:7px 2px 0">
        <span class="hint" id="chatnote"></span>
        <span class="hint mono" id="chatcount"></span>
      </div>
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
  <div class="card" style="margin-bottom:14px">
    <div class="hd"><h2>Engine setup</h2><span class="tag mut" id="setupjobchip">idle</span></div>
    <p class="hint" style="margin:0 0 10px">One-click installs pull official upstream artifacts (llama.cpp GitHub releases, the official Strata installer). Avalon builds from your local source tree. Each action asks for confirmation and streams its log below.</p>
    <div class="form-grid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">
      <div class="field">
        <label>llama.cpp <span id="setup-llama-chip" class="tag mut">checking…</span></label>
        <div class="kvs" id="setup-llama-detail" style="margin-bottom:6px"></div>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
          <select id="setup-llama-variant" style="width:130px">
            <option value="auto">auto backend</option>
            <option value="cpu">cpu only</option>
            <option value="vulkan">vulkan</option>
            <option value="cuda">cuda</option>
          </select>
          <button class="btn primary sm" id="setup-llama-btn">Install latest</button>
        </div>
      </div>
      <div class="field">
        <label>Strata <span id="setup-strata-chip" class="tag mut">checking…</span></label>
        <div class="kvs" id="setup-strata-detail" style="margin-bottom:6px"></div>
        <button class="btn primary sm" id="setup-strata-btn">Install (official installer)</button>
      </div>
      <div class="field">
        <label>Avalon <span id="setup-avalon-chip" class="tag mut">checking…</span></label>
        <div class="kvs" id="setup-avalon-detail" style="margin-bottom:6px"></div>
        <button class="btn sm" id="setup-avalon-btn">Build from source</button>
      </div>
    </div>
    <pre class="log hide" id="setuplog" style="max-height:220px;margin-top:10px"></pre>
  </div>
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
      <div class="field" style="grid-column:1 / -1">
        <label>Strata CLI path (optional; blank = auto-detect: $STRATA_BIN, PATH, ~/.strata/bin, /usr/local/bin, app bundle) <span id="set-strata-status"></span></label>
        <input id="set-stratabin" type="text" placeholder="/usr/local/bin/strata">
      </div>
      <div class="field">
        <label>Strata model alias (advertised to coding agents)</label>
        <input id="set-strataalias" type="text" placeholder="blank = use last launched model name">
      </div>
      <div class="field">
        <label>Strata API key override (only if you manage the key yourself) <span id="set-stratakey-status"></span></label>
        <input id="set-stratakey" type="password" placeholder="(leave blank to keep existing)">
      </div>
      <div class="field" style="grid-column:1 / -1;border-top:1px solid var(--bd);padding-top:10px;margin-top:4px">
        <label style="font-weight:650">Optimization Agent · Optional OpenAI-Compatible LLM</label>
        <span class="hint">Leave blank to run purely deterministically (or fall back to the active local model for dry-run advice).</span>
      </div>
      <div class="field">
        <label>Agent LLM Base URL</label>
        <input id="set-agenturl" type="text" placeholder="e.g. https://api.openai.com/v1 or http://127.0.0.1:11434/v1">
      </div>
      <div class="field">
        <label>Agent LLM Model</label>
        <input id="set-agentmodel" type="text" placeholder="e.g. gpt-4o-mini, claude-3-5-sonnet, qwen3-14b">
      </div>
      <div class="field">
        <label>Agent LLM API Key <span id="set-agentkey-status"></span></label>
        <input id="set-agentkey" type="password" placeholder="(leave blank to keep existing; type 'clear' to remove)">
      </div>
      <div class="field" style="justify-content:center">
        <label class="switch"><input type="checkbox" id="set-agentfallback" checked><span class="track"></span><span class="hint">Fall back to active local engine when external LLM is unset</span></label>
      </div>
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:14px">
      <button class="btn" id="set-reload">Reload</button>
      <button class="btn primary" id="set-save">Save settings</button>
    </div>
  </div>
  <div class="card">
    <div class="hd"><h2>Panel process</h2><span class="tag mut" id="panel-proc-chip">—</span></div>
    <div class="hint" style="margin-bottom:10px">Restart re-execs the panel in place. Running llama-server, Avalon, and Strata processes keep serving, and this tab reconnects on its own.</div>
    <div style="display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap">
      <span class="hint mono" id="panel-proc-detail"></span>
      <button class="btn danger" id="panelrestart">Restart panel</button>
    </div>
  </div>
</section>
</main>

<!-- ==================== AGENT CHAT SIDEBAR ==================== -->
<aside id="agentdrawer" class="agent-drawer" aria-label="Optimization Agent Chat">
  <div class="agent-drawer-hd">
    <h2>Optimization Agent</h2>
    <div style="display:flex;gap:6px;align-items:center">
      <span class="tag mut" id="ag-sb-llm">deterministic</span>
      <button class="icon" id="ag-sb-close" aria-label="Close sidebar"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
    </div>
  </div>
  <div class="agent-drawer-bd" id="ag-sb-msgs">
    <div class="agent-msg bot">Ask me to optimize any engine (llama.cpp, Avalon, Strata), explain MoE fit, diagnose a crash, or restore a checkpoint.</div>
  </div>
  <div class="agent-drawer-ft">
    <div class="agent-chips">
      <button type="button" class="btn xs" data-ag-chip="Optimize llama.cpp for fastest speed">Fastest llama.cpp</button>
      <button type="button" class="btn xs" data-ag-chip="Optimize Maya for fastest speed">Fastest Maya</button>
      <button type="button" class="btn xs" data-ag-chip="Longest context config for Avalon">Max ctx Avalon</button>
      <button type="button" class="btn xs" data-ag-chip="Diagnose last crash">Diagnose crash</button>
      <button type="button" class="btn xs" data-ag-chip="Show checkpoints">Checkpoints</button>
    </div>
    <div style="display:flex;gap:6px">
      <input id="ag-sb-in" type="text" placeholder="Ask the optimization agent…" style="flex:1">
      <button class="btn primary sm" id="ag-sb-send">Send</button>
    </div>
  </div>
</aside>

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
      <div class="field" id="ld-draft-wrap">
        <label>Speculative draft model (--model-draft)</label>
        <input id="ld-draft" type="text" list="draftlist" placeholder="MTP / draft GGUF beside the model (blank = off)">
        <datalist id="draftlist"></datalist>
      </div>
      <div class="field" id="ld-draftmax-wrap">
        <label>Draft max / min tokens</label>
        <div style="display:flex;gap:6px">
          <input id="ld-draftmax" type="number" min="1" max="64" value="16" style="width:50%">
          <input id="ld-draftmin" type="number" min="0" max="16" value="0" style="width:50%">
        </div>
      </div>
      <div class="field">
        <label>Tensor overrides (-ot, e.g. <code>*.ffn_.*_exps.=CPU</code>)</label>
        <input id="ld-ot" type="text" placeholder="blank = preset default">
      </div>
      <div class="field">
        <label>Tensor split (--tensor-split, per-device)</label>
        <input id="ld-tsplit" type="text" placeholder="e.g. 1 1 or 0,3 — blank = off">
      </div>
      <div class="field">
        <label>Batch threads (--threads-batch)</label>
        <input id="ld-tbatch" type="number" min="0" max="256" value="0">
      </div>
      <div class="field">
        <label>Keep system prompt in cache (--keep tokens)</label>
        <input id="ld-keep" type="number" min="0" max="65536" value="0">
      </div>
      <div class="field">
        <label>Chat template (--jinja)</label>
        <select id="ld-jinja">
          <option value="on">on (recommended)</option>
          <option value="off">off (legacy template)</option>
        </select>
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

<dialog id="renamedlg" closedby="any" style="max-width:min(440px,94vw)">
  <div class="dlg-hd">
    <h2>Rename conversation</h2>
    <button class="icon" data-close="renamedlg" aria-label="Close"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
  </div>
  <form id="ren-form">
    <div class="dlg-bd">
      <div class="field">
        <label for="ren-input">Conversation title</label>
        <input id="ren-input" type="text" maxlength="120" required style="width:100%">
      </div>
    </div>
    <div class="dlg-ft">
      <button type="button" class="btn" data-close="renamedlg">Cancel</button>
      <button type="submit" class="btn primary">Save title</button>
    </div>
  </form>
</dialog>

<footer>
  <span><kbd>⌘K</kbd> / <kbd>Ctrl+K</kbd> command palette</span>
  <span><kbd>1</kbd>–<kbd>8</kbd> switch tabs</span>
  <span><kbd>/</kbd> filter</span>
  <span><kbd>Esc</kbd> close</span>
  <span id="footnote"></span>
</footer>
</div></div>

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
  if(next===sys)localStorage.removeItem('ic.theme');
  else localStorage.setItem('ic.theme',next);
});
$('#logout').addEventListener('click',async()=>{
  if(!await confirmDlg('Sign out','End this panel session?','Sign out',true))return;
  try{await post('/api/logout')}catch(e){}
  location.href='/login';
});

/* ============================== tabs & workbench rail ============================== */
const TABS=['models','server','strata','avalon','maya','agent','vitals','chat','hf','settings'];
const TAB_TITLES={models:'Model Library',server:'llama.cpp Server & Bench',strata:'Strata Engine',
  avalon:'Avalon 125B MoE',maya:'Project Maya',agent:'Optimization Agent',vitals:'System Vitals',chat:'Chat',hf:'Hugging Face Pull',settings:'Settings & Setup'};
let tab='models';
function closeNav(){document.body.classList.remove('nav-open')}
$('#mobmenu').addEventListener('click',()=>document.body.classList.toggle('nav-open'));
$('#railbackdrop').addEventListener('click',closeNav);
$('#railvitals').addEventListener('click',()=>setTab('vitals'));
function setTab(t){
  if(TABS.indexOf(t)<0)t='models';
  tab=t;localStorage.setItem('ic.tab',t);
  closeNav();
  $('#viewtitle').textContent=TAB_TITLES[t]||t;
  $$('.nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
  $$('main section').forEach(s=>s.classList.toggle('hide',s.id!=='t-'+t));
  if(t==='server'){refreshLog();pollTelemetry();loadBenchmarks();}
  if(t==='strata'){loadStrata();loadStrataPresets();loadStrataKey();pollStrataTelemetry();refreshStrataLog();}
  if(t==='avalon'){loadAvalon(true);loadAvalonAccess();}
  if(t==='maya'){loadMaya();refreshMayaLog();}
  if(t==='agent'){loadAgentAll();}
  if(t==='settings'){loadSettings();loadSetup();}
  if(t==='vitals')refreshVitals();
}
$$('.nav button').forEach(b=>b.addEventListener('click',()=>setTab(b.dataset.t)));
$('#hdrstatus').addEventListener('click',()=>{
  const ct=chatTarget();
  if(ct)setTab(ct.src==='avalon'?'avalon':ct.src==='strata'?'strata':ct.src==='maya'?'maya':'server');
  else if(ACTIVE)setTab('server');
  else if(AV&&AV.running)setTab('avalon');
  else if(STRATA&&STRATA.running)setTab('strata');
  else if(MAYA&&MAYA.running)setTab('maya');
  else setTab('models');
});

/* ============================== models & launch modal ============================== */
let MODELS=[],MMP=[],ACTIVE=null,AV=null,STRATA=null,MAYA=null,FILTER='all',SORT='smart',CAPS={},DEF_CTX=16384;
const SHX=new Set(); // expanded multi-shard groups
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
  const sh=m.shards&&m.shards.length>1;
  const chev=SHX.has(m.path)
    ?'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:12px;height:12px"><path d="M6 9l6 6 6-6"/></svg>'
    :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:12px;height:12px"><path d="M9 6l6 6-6 6"/></svg>';
  if(sh)tags.push('<button class="btn xs" data-act="shards" title="Show individual shard files">'+chev+' '+m.shards.length+' shards</button>');
  return '<div class="mrow'+(running?' run':'')+'" data-i="'+i+'" data-path="'+esc(m.path)+'" data-name="'+esc(m.name)+'" data-size="'+esc(m.size_h)+'"'+(sh?' data-shards="'+m.shards.length+'"':'')+'">'
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
    +'<button class="btn xs danger" data-act="del" title="'+(sh?'Delete all shard files of this pack':'Delete model file')+'">Del</button></div></div>';
}
function shardListHtml(m){
  if(!(m.shards&&m.shards.length>1&&SHX.has(m.path)))return '';
  return '<div class="shardlist">'+m.shards.map(s=>
    '<div class="shrow" data-shardpath="'+esc(s.path)+'" title="Click to copy path"><code>'+esc(s.name)+'</code><span class="hint">'+gib(s.size)+'</span></div>'
  ).join('')+'</div>';
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
  if($('#nb-models'))$('#nb-models').textContent=String(MODELS.length);
  if(MODELS[0]&&MODELS[0].fit){
    const f=MODELS[0].fit;
    $('#vramchip').textContent=f.budget_source+': '+((f.usable_mb||f.budget_mb)/1024).toFixed(1)+' GB usable';
  }
  if(!list.length){
    $('#modellist').innerHTML='<div class="empty"><b>'+(MODELS.length?'No model matches this filter':'No GGUF models found')+'</b>'
      +(MODELS.length?'Clear the filter or press Rescan.':'Add files under a configured model dir in Settings, or use the HF Pull tab.')+'</div>';
    return;
  }
  $('#modellist').innerHTML=list.map(m=>modelRow(m,MODELS.indexOf(m))+shardListHtml(m)).join('');
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
  const shr=e.target.closest('.shrow');
  if(shr)return copy(shr.dataset.shardpath,'Shard path');
  const b=e.target.closest('[data-act]');if(!b)return;
  const row=b.closest('.mrow');const path=row.dataset.path;
  const m=MODELS.find(x=>x.path===path);
  if(b.dataset.act==='shards'){
    SHX.has(path)?SHX.delete(path):SHX.add(path);
    renderModels();return;
  }
  if(b.dataset.act==='pin'){
    try{const r=await post('/api/models/pin',{path,pinned:!(m&&m.pinned)});
      if(m)m.pinned=r.pinned;renderModels();}catch(err){toast('Pin failed: '+err.message,'err')}
    return;
  }
  if(b.dataset.act==='copy')return copy(path,'Path');
  if(b.dataset.act==='tune'){if(m)openLaunchModal(m);return;}
  if(b.dataset.act==='del'){
    const ns=row.dataset.shards;
    const what=ns?('<b>'+ns+' shard files</b> of'):('the file');
    if(!await confirmDlg('Delete model',what+' <b>'+esc(row.dataset.name)+'</b> ('+esc(row.dataset.size)+')<br><code>'
      +esc(path)+(ns?'<br><span class="hint">…and '+(ns-1)+' more shards of this pack</span>':'')+'</code><br><br>This permanently removes '+(ns?'them all':'it')+' from disk.','Delete',true))return;
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
    draft:$('#ld-draft').value.trim(),
    draft_max:Number($('#ld-draftmax').value)||16,
    draft_min:Number($('#ld-draftmin').value)||0,
    ot:$('#ld-ot').value.trim(),
    tensor_split:$('#ld-tsplit').value.trim(),
    tbatch:Number($('#ld-tbatch').value)||null,
    keep:Number($('#ld-keep').value)||null,
    jinja:$('#ld-jinja').value,
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
  $('#ld-draft').value=p.draft||'';
  $('#ld-draftmax').value=p.draft_max||16;
  $('#ld-draftmin').value=p.draft_min||0;
  $('#ld-ot').value=p.ot||'';
  $('#ld-tsplit').value=p.tensor_split||'';
  $('#ld-tbatch').value=p.tbatch||0;
  $('#ld-keep').value=p.keep||0;
  $('#ld-jinja').value=p.jinja||'on';
  $('#ld-extra').value=p.extra_args||'';
  $('#ld-ncmoe-wrap').classList.toggle('hide',!(LD_MODEL&&LD_MODEL.moe));
  $('#ld-draft-wrap').classList.toggle('hide',!CAPS.draft);
  $('#ld-draftmax-wrap').classList.toggle('hide',!CAPS.draft);
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
  const dir=m.path.replace(/\/[^/]+$/,'/');
  const cands=MODELS.filter(x=>x.path.startsWith(dir)&&x.path!==m.path
              &&/mtp|draft/i.test(x.name)&&!/mmproj/i.test(x.name));
  $('#draftlist').innerHTML=cands.map(x=>'<option value="'+esc(x.path)+'">'+esc(x.name)+'</option>').join('');
  if(!$('#ld-draft').value&&CAPS.draft&&cands.length)$('#ld-draft').value=cands[0].path;
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
  if(STRATA&&STRATA.running&&STRATA.health==='ready'&&STRATA.port)return{alias:STRATA.model||'strata',port:STRATA.port,src:'strata'};
  if(MAYA&&MAYA.running&&MAYA.state==='ready'&&MAYA.port)return{alias:MAYA.alias||MAYA.model_name||'maya',port:MAYA.port,src:'maya'};
  return null;
}
function drawHeader(){
  const p=$('#hdrstatus'),t=$('#hdrtext');
  const ct=chatTarget();
  if(ct){
    p.className='pill ok';
    t.textContent=ct.alias+' · :'+ct.port+(ct.src==='avalon'?' · avalon':ct.src==='strata'?' · strata':ct.src==='maya'?' · maya':'');
    document.title='● '+ct.alias+' — Inference Console';
  }else if(ACTIVE){
    p.className='pill warn';
    t.textContent=ACTIVE.alias+' · loading…';
    document.title='◌ Loading '+ACTIVE.alias+' — Inference Console';
  }else if(AV&&AV.running){
    p.className='pill warn';
    t.textContent='avalon · loading…';
    document.title='◌ Loading Avalon — Inference Console';
  }else if(STRATA&&STRATA.running){
    p.className='pill warn';
    t.textContent='strata · loading…';
    document.title='◌ Loading Strata — Inference Console';
  }else if(MAYA&&MAYA.running){
    p.className='pill warn';
    t.textContent=(MAYA.model_name||'maya')+' · loading…';
    document.title='◌ Loading Maya — Inference Console';
  }else{
    p.className='pill';
    t.textContent='no server running';
    document.title='Inference Console — Local Inference Manager';
  }
  const setDot=(id,st)=>{const d=$(id);if(d)d.className='ndot'+(st==='ok'?' ok':st==='warn'?' warn':'');};
  const ap=(AV&&AV.port)||8086;
  setDot('#nd-server',(ACTIVE&&ACTIVE.port!==ap)?(ACTIVE.ready?'ok':'warn'):'');
  setDot('#nd-avalon',(AV&&AV.running)?(AV.state==='ready'?'ok':'warn'):(ACTIVE&&ACTIVE.port===ap?(ACTIVE.ready?'ok':'warn'):''));
  setDot('#nd-strata',(STRATA&&STRATA.running)?(STRATA.health==='ready'?'ok':'warn'):'');
  setDot('#nd-maya',(MAYA&&MAYA.running)?(MAYA.state==='ready'?'ok':'warn'):'');
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
  a.download='inference-console-'+new Date().toISOString().slice(0,16).replace(/[:T]/g,'-')+'.log';
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
  if(v.gpu&&v.gpu.total_mb){
    const gp=Math.round(100*(v.gpu.used_mb||0)/v.gpu.total_mb);
    $('#rv-mem-lbl').textContent='VRAM';
    $('#rv-mem-val').textContent=(v.gpu.used_mb/1024).toFixed(1)+'/'+(v.gpu.total_mb/1024).toFixed(0)+'G';
    $('#rv-mem-bar').className='meter '+lvlCls(gp);
    $('#rv-mem-bar').innerHTML='<i style="width:'+Math.min(100,gp)+'%"></i>';
  }else if(v.mem&&v.mem.total_gb){
    const mp=Math.round(100*v.mem.used_gb/v.mem.total_gb);
    $('#rv-mem-lbl').textContent='RAM';
    $('#rv-mem-val').textContent=v.mem.used_gb+'/'+v.mem.total_gb+'G';
    $('#rv-mem-bar').className='meter '+lvlCls(mp);
    $('#rv-mem-bar').innerHTML='<i style="width:'+Math.min(100,mp)+'%"></i>';
  }
  if(v.cpu)$('#rv-cpu-val').textContent=v.cpu.percent+'%';
}
async function refreshVitals(){
  try{VIT=await api('/api/vitals');VT=Date.now();pushHist();drawVitals()}catch(e){drawVitals()}
}
$('#vrefresh').addEventListener('click',refreshVitals);

/* ============================== chat ============================== */
let HIST=[],CHATS_CACHE=[],CID=null,BUSY=false,AC=null;
const DEF_SYS='You are a helpful local assistant. Be concise.';
$('#sysprompt').value=localStorage.getItem('ic.sys')||DEF_SYS;
function syncSamplingLabels(){
  $('#sp-temp-v').textContent=Number($('#sp-temp').value).toFixed(2);
  $('#sp-topp-v').textContent=Number($('#sp-topp').value).toFixed(2);
  $('#sp-minp-v').textContent=Number($('#sp-minp').value).toFixed(2);
  $('#sp-rep-v').textContent=Number($('#sp-rep').value).toFixed(2);
  const mx=Number($('#sp-max').value)||0;
  $('#sp-max-v').textContent=mx>0?n0(mx)+' tok':'default';
}
['sp-temp','sp-topp','sp-minp','sp-rep','sp-max'].forEach(id=>{
  const saved=localStorage.getItem('ic.'+id);
  if(saved!=null)$('#'+id).value=saved;
  $('#'+id).addEventListener('input',()=>{syncSamplingLabels();localStorage.setItem('ic.'+id,$('#'+id).value)});
});
syncSamplingLabels();
$('#sp-reset').addEventListener('click',()=>{
  $('#sysprompt').value=DEF_SYS;localStorage.setItem('ic.sys',DEF_SYS);
  const defs={'sp-temp':'0.7','sp-topp':'0.95','sp-minp':'0.05','sp-rep':'1.05','sp-max':'0'};
  Object.keys(defs).forEach(k=>{$('#'+k).value=defs[k];localStorage.removeItem('ic.'+k);});
  syncSamplingLabels();toast('Sampling defaults restored','ok',1800);
});
$('#sysbtn').addEventListener('click',()=>{
  const box=$('#sysbox');box.classList.toggle('hide');
  if(box.classList.contains('hide'))localStorage.setItem('ic.sys',$('#sysprompt').value||DEF_SYS);
  else $('#sysprompt').focus();
});
$('#sysprompt').addEventListener('change',()=>localStorage.setItem('ic.sys',$('#sysprompt').value||DEF_SYS));

function drawChatTarget(){
  const ct=chatTarget(),t=$('#chatmodel');
  if(ct){t.className='tag ok';t.textContent=ct.alias+' · :'+ct.port}
  else{t.className='tag warn';t.textContent='no model loaded'}
  $('#sendbtn').disabled=BUSY||!ct;
  $('#chatnote').textContent=ct?'':'Start a model in the Models, Strata, Avalon, or Maya tab to chat.';
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
let REN_CID=null;
$('#ren-form').addEventListener('submit',async e=>{
  e.preventDefault();
  const next=$('#ren-input').value.trim();
  if(!next||!REN_CID)return;
  try{
    await post('/api/chats/'+REN_CID+'/rename',{title:next});
    if(CID===REN_CID)$('#chattitle').textContent=next;
    $('#renamedlg').close();loadChatList();toast('Conversation renamed','ok',2000);
  }catch(err){toast('Rename failed: '+err.message,'err')}
});
$('#chatlist').addEventListener('click',async e=>{
  const ren=e.target.closest('[data-ren]');
  if(ren){
    e.stopPropagation();
    REN_CID=ren.dataset.ren;
    $('#ren-input').value=ren.dataset.title||'';
    $('#renamedlg').showModal();
    $('#ren-input').focus();$('#ren-input').select();
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
    const c=await api('/api/chats/'+id);CID=id;localStorage.setItem('ic.cid',id);
    $('#chattitle').textContent=c.title||'Conversation';
    HIST=(c.messages||[]).filter(m=>m.role!=='system');
    redrawThread();loadChatList();return true;
  }catch(e){if(!quiet)toast('Could not open chat: '+e.message,'err');return false}
}
function emptyThread(){
  $('#chattitle').textContent='Conversation';
  $('#chat').innerHTML='<div class="empty"><b>New conversation</b>Pick a saved chat on the left, or type a message below.</div>';
}
function newChat(){CID=null;HIST=[];localStorage.removeItem('ic.cid');emptyThread();loadChatList();$('#prompt').focus()}
$('#newchat').addEventListener('click',newChat);

$('#chatexportmd').addEventListener('click',()=>{
  if(!HIST.length)return toast('Nothing to export','warn');
  const mdOut=HIST.map(m=>'### '+(m.role==='user'?'User':'Assistant')+'\n\n'+(m.reasoning?'<think>\n'+m.reasoning+'\n</think>\n\n':'')+m.content).join('\n\n---\n\n');
  const url=URL.createObjectURL(new Blob([mdOut],{type:'text/markdown'}));
  const a=document.createElement('a');a.href=url;a.download='chat-'+(CID||'export')+'.md';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),4000);
});
$('#chatexportjson').addEventListener('click',()=>{
  if(!HIST.length)return toast('Nothing to export','warn');
  const url=URL.createObjectURL(new Blob([JSON.stringify(HIST,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='chat-'+(CID||'export')+'.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),4000);
});

const ta=$('#prompt');
function autosize(){
  ta.style.height='auto';ta.style.height=Math.min(220,ta.scrollHeight)+'px';
  const len=ta.value.length;
  $('#chatcount').textContent=len?(n0(len)+' chars · ~'+n0(Math.ceil(len/4))+' tok'):'';
}
ta.addEventListener('input',autosize);
ta.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
$('#sendbtn').addEventListener('click',send);
$('#stopbtn').addEventListener('click',()=>{if(AC)AC.abort()});

async function send(){
  const q=ta.value.trim();
  if(!q||BUSY)return;
  const ct=chatTarget();
  if(!ct)return toast('Start a model first (Models, Strata, or Avalon tab)','warn');
  ta.value='';autosize();
  const th=$('#chat');if(th.querySelector('.empty'))th.innerHTML='';
  const um={role:'user',content:q};HIST.push(um);addMsg(um,HIST.length-1);
  BUSY=true;$('#sendbtn').disabled=true;$('#stopbtn').style.display='';
  try{
    const r0=await post('/api/chats/append',{id:CID,model:ct.alias,role:'user',content:q});
    if(r0&&r0.id){CID=r0.id;localStorage.setItem('ic.cid',CID);loadChatList();}
  }catch(e){}
  const asstIdx=HIST.length;
  const box=addMsg({role:'assistant',content:''},asstIdx);
  const ans=box.querySelector('.ans'),who=box.querySelector('.who');
  AC=new AbortController();
  let buff='',bufa='',t0=0,toks=0,stopped=false,hadErr=false;
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
    if(!r.ok){hadErr=true;const d=await r.text();ans.innerHTML='<span style="color:var(--err)">'+esc(d.indexOf('no model')>=0?'No model running. Load one first.':'error: '+d)+'</span>'}
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
    else{hadErr=true;ans.innerHTML='<span style="color:var(--err)">'+esc('error: '+e.message)+'</span>'}
  }
  const finalText=bufa||buff||'';
  const secs=Math.max(.001,(performance.now()-(t0||performance.now()))/1000);
  const tps=toks?(toks/secs).toFixed(1):Math.round((finalText.length/4)/secs);
  who.innerHTML='<span>Assistant</span><span class="tag mut">'+(stopped?'stopped · ':hadErr?'error · ':'')+tps+' tok/s · '+esc(ct.alias)+'</span>';
  if(finalText&&!hadErr){
    box.insertAdjacentHTML('beforeend','<div class="bar"><button class="btn xs" data-regen="'+asstIdx+'">Regenerate</button><button class="btn xs" data-copy="'+esc(finalText)+'">Copy</button></div>');
    HIST.push({role:'assistant',content:finalText,reasoning:buff});
    post('/api/chats/append',{id:CID,model:ct.alias,role:'assistant',content:finalText,reasoning:buff})
      .then(r=>{if(r&&r.id){CID=r.id;loadChatList()}}).catch(()=>{});
  }
  BUSY=false;AC=null;$('#stopbtn').style.display='none';drawChatTarget();ta.focus();
}

/* ============================== agents registry ============================== */
let AGSRC=null; // null = auto (panel first); 'panel' | 'avalon' | 'strata' | 'maya' pins the source
let AGRES='';  // last Apply's per-agent result rows, kept across post-Apply re-render
async function openAgentsDlg(source,keepResults){
  AGSRC=source||null;
  if(!keepResults)AGRES='';
  let d;try{d=await api('/api/agents')}catch(e){return toast('Failed: '+e.message,'err')}
  const reg=(d.registry&&d.registry.agents)||{};
  const srcs=d.sources||{};
  const tgts=d.targets||{};
  const selSrc=AGSRC||'auto';
  const targetFor=s=>{
    if(s==='auto')return d.running?{alias:d.running[0],port:d.running[1],source:d.running[3]||''}:null;
    return tgts[s]||null;
  };
  const fmtTarget=t=>t?(esc(t.alias)+' @ :'+t.port+(t.source?' ('+esc(t.source)+')':'')):'nothing running';
  const srcOpts=['auto','panel','avalon','strata','maya'].map(s=>{
    const t=targetFor(s);
    const up=Boolean(t||(s!=='auto'&&srcs[s]));
    return '<option value="'+s+'"'+(s===selSrc?' selected':'')+'>'+s
      +(s==='auto'?' (first running)':(up?(t?' · '+t.alias+' @ :'+t.port:' · running'):' · stopped'))+'</option>';
  }).join('');
  const curT=targetFor(selSrc);
  const srcOk=selSrc==='auto'?Boolean(d.running):Boolean(srcs[selSrc]);
  const rows=d.agents.map(a=>{
    const r=reg[a.id];
    return '<label class="frow"><input type="checkbox" class="agchk" value="'+esc(a.id)+'"'+(r?' checked':'')+'>'
      +'<span style="flex:1;min-width:0"><b>'+esc(a.label)+'</b>'
      +(r?' <span class="tag ok">'+esc(r.alias)+'@'+r.port+' · '+esc(ago(r.synced_at))+'</span>':'')
      +'<br><span class="hint mono">'+esc(a.hint||'')+'</span></span></label>';
  }).join('');
  const dlg=$('#agentsdlg');
  $('#ag-body').innerHTML='<p style="margin:0 0 8px;color:var(--fg-2)">Writes each selected agent config to point at <b id="ag-target">'+fmtTarget(curT)+'</b>.</p>'
    +'<div style="display:flex;gap:8px;align-items:center;margin:0 0 8px"><label class="hint" for="ag-src">Source</label>'
    +'<select id="ag-src" style="flex:1">'+srcOpts+'</select>'
    +'<button type="button" class="btn xs" id="ag-all">All</button>'
    +'<button type="button" class="btn xs" id="ag-none">None</button></div>'
    +'<p class="hint'+(srcOk?' hide':'')+'" id="ag-src-warn" style="margin:0 0 8px;color:var(--warn,#c80)">'
    +(srcOk?'':(selSrc==='auto'?'No server is running — start an engine first.':'The '+esc(selSrc)+' server isn’t running — start it on its tab, or pick another source.'))+'</p>'
    +'<div style="max-height:42vh;overflow:auto">'+rows+'</div>'
    +'<div id="agres" style="margin-top:10px">'+AGRES+'</div>';
  $('#ag-src').addEventListener('change',e=>{
    const v=e.target.value;
    AGSRC=v==='auto'?null:v;
    const t=targetFor(v);
    $('#ag-target').innerHTML=fmtTarget(t);
    const w=$('#ag-src-warn');
    const ok=v==='auto'?Boolean(d.running):Boolean(srcs[v]);
    if(!ok){
      w.textContent=v==='auto'?'No server is running — start an engine first.':'The '+v+' server isn’t running — start it on its tab, or pick another source.';
      w.classList.remove('hide');
    }else w.classList.add('hide');
  });
  $('#ag-all').addEventListener('click',()=>$$('.agchk',$('#agentsdlg')).forEach(x=>x.checked=true));
  $('#ag-none').addEventListener('click',()=>$$('.agchk',$('#agentsdlg')).forEach(x=>x.checked=false));
  if(!dlg.open)dlg.showModal();
  if(keepResults&&AGRES){const ar=$('#agres');if(ar)ar.scrollIntoView({block:'nearest'});}
}
$('#agentsbtn').addEventListener('click',()=>openAgentsDlg(null,false));
$('#avalonagentsbtn').addEventListener('click',()=>openAgentsDlg('avalon',false));
$('#strataagentsbtn').addEventListener('click',()=>openAgentsDlg('strata',false));
$('#mayaagentsbtn').addEventListener('click',()=>openAgentsDlg('maya',false));
$('#ag-apply').addEventListener('click',async()=>{
  const apply=$('#ag-apply');
  const picks=$$('.agchk',$('#agentsdlg')).filter(x=>x.checked).map(x=>x.value);
  if(!picks.length){
    const ar=$('#agres');
    if(ar){ar.innerHTML='<div class="frow" style="cursor:default"><span class="tag warn">select</span><span class="p">Pick at least one agent checkbox above, then click Apply.</span></div>';ar.scrollIntoView({block:'nearest'});}
    return toast('Pick at least one agent','warn');
  }
  apply.disabled=true;apply.textContent='Applying…';
  let r;try{r=await post('/api/agents/sync',{agents:picks,source:AGSRC||'auto'})}
  catch(e){
    apply.disabled=false;apply.textContent='Apply';
    const ar=$('#agres');
    if(ar){ar.innerHTML='<div class="frow" style="cursor:default"><span class="tag err">error</span><span class="p">'+esc(e.message)+'</span></div>';ar.scrollIntoView({block:'nearest'});}
    return toast('Sync failed: '+e.message,'err',12000);
  }
  const rs=Object.keys(r.results).map(k=>r.results[k]);
  const okN=rs.filter(v=>v.ok).length;
  AGRES='<div class="hint" style="margin-bottom:4px">Registered <b>'+okN+' of '+picks.length+'</b> for <b>'+esc(r.alias)+' @ :'+r.port+'</b></div>'
    +Object.keys(r.results).map(k=>{const v=r.results[k];
    return '<div class="frow" style="cursor:default"><span class="tag '+(v.ok?'ok':'err')+'">'+(v.ok?'ok':'fail')+'</span>'
      +'<span class="p" title="'+esc(v.detail)+'">'+esc(k)+' — '+esc(v.detail)+'</span></div>'}).join('');
  apply.disabled=false;apply.textContent='Apply';
  toast('Registered '+okN+' of '+picks.length+' agent'+(picks.length>1?'s':''), okN===picks.length?'ok':(okN===0?'err':'warn'), 8000);
  openAgentsDlg(AGSRC,true);  // re-render so the registry tags show the new alias/port
  pollStatus();
});

/* ============================== strata ============================== */
let STLOGTEXT='';
async function loadStrata(){
  try{
    const s=await api('/api/strata/status');STRATA=s;drawStrata(s);drawHeader();drawChatTarget();
    if(s.running)loadStrataKey();
    else $('#stratakeychip').classList.add('hide');
  }catch(e){}
}
function drawStrata(s){
  const chip=$('#stratchip');
  chip.textContent=!s.installed?'not installed':s.running?(s.health==='ready'?'serving · ready':'starting…'):'stopped';
  chip.className='tag '+(s.running?(s.health==='ready'?'ok':'warn'):(s.installed?'mut':'err'));
  $('#stratainstall').classList.toggle('hide',!!s.installed);
  if(!s.installed)$('#stratainstcmd').textContent=s.install_cmd||'';
  $('#stratadetail').innerHTML=s.installed?
    [['bin',s.bin||''],['health',s.health||''],['port',s.port?':'+s.port:'—']]
      .concat(s.model?[['model',s.model]]:[],s.preset?[['preset',s.preset]]:[],s.version?[['version',s.version]]:[])
      .map(kv=>'<span class="tag">'+esc(kv[0])+' <b>'+esc(kv[1])+'</b></span>').join('')
    +(s.raw?'<details style="margin-top:6px"><summary class="hint">raw strata launch --status output</summary><pre class="log" style="max-height:200px">'+esc(s.raw)+'</pre></details>':'')
    :'<span class="hint">'+esc(s.note||'')+'</span>';
  $('#stratastopbtn').disabled=!(s.running);
}
async function pollStrataTelemetry(){
  const card=$('#stratametricscard');
  if(!(STRATA&&STRATA.running)){card.classList.add('hide');return;}
  try{
    const d=await api('/api/strata/telemetry');
    if(!d.ready||!d.metric_keys||!d.metric_keys.length){card.classList.add('hide');return;}
    card.classList.remove('hide');
    $('#st-metrics-chip').textContent=':'+d.port+' · '+d.metric_keys.length+' metrics';
    const keys=d.metric_keys.slice(0,18);
    $('#st-metrics').innerHTML=keys.map(k=>{
      const v=d.metrics[k];
      const fmt=Number.isInteger(v)?n0(v):Number(v).toFixed(2);
      return '<span class="tag">'+esc(k)+' <b>'+esc(fmt)+'</b></span>';
    }).join('');
  }catch(e){card.classList.add('hide')}
}
function stForm(){
  const sets=$('#st-set').value.trim().split(/\s+/).filter(x=>x.includes('='));
  return {
    preset:$('#st-preset').value||null,family:$('#st-family').value||null,
    model:$('#st-model').value.trim()||null,model_path:$('#st-modelpath').value.trim()||null,
    context:$('#st-context').value||null,kv:$('#st-kv').value||null,
    parallel:Number($('#st-parallel').value)||null,vision:$('#st-vision').value||null,
    mtp:$('#st-mtp').value||null,drafting:$('#st-drafting').value||null,
    gpu:$('#st-gpu').value.trim()||null,tune:$('#st-tune').value||null,
    retune:$('#st-retune').checked,access:$('#st-access').value||null,
    port:Number($('#st-port').value)||null,no_api_key:$('#st-nokey').checked,
    new_key:$('#st-newkey').checked,save_preset:$('#st-savepreset').value.trim()||null,
    restart:$('#st-restart').checked,set_:sets.length?sets:null
  };
}
function stPreview(){
  const f=stForm();let c='strata launch';
  const map={preset:'--preset',family:'--family',model:'--model',model_path:'--model-path',
             context:'--context',kv:'--kv',vision:'--vision',mtp:'--mtp',drafting:'--drafting',
             gpu:'--gpu',tune:'--tune',access:'--access',save_preset:'--save-preset'};
  for(const k in map)if(f[k])c+=' '+map[k]+' '+f[k];
  if(f.parallel)c+=' --parallel '+f.parallel;
  if(f.port)c+=' --port '+f.port;
  (f.set_||[]).forEach(kv=>c+=' --set '+kv);
  if(f.retune)c+=' --retune';
  if(f.no_api_key)c+=' --no-api-key';
  if(f.new_key)c+=' --new-key';
  if(f.restart)c+=' --restart';
  c+=' --yes --no-tui';
  $('#st-cmd').textContent=c;
}
$$('#strataformcard input,#strataformcard select').forEach(el=>el.addEventListener('input',stPreview));
$('#st-toggleform').addEventListener('click',()=>{$('#strataformcard').classList.toggle('hide');stPreview()});
$('#st-copycmd').addEventListener('click',()=>copy($('#st-cmd').textContent,'Strata command'));
$('#stratacopyinst').addEventListener('click',()=>copy($('#stratainstcmd').textContent,'Install command'));
async function loadStrataPresets(){
  if(!$('#st-presets'))return;
  try{
    const d=await api('/api/strata/presets');
    const ps=(d.presets||[]).map(p=>typeof p==='string'?{name:p}:(p&&p.name?{name:p.name,context:p.context,model:p.model}:null)).filter(Boolean);
    const xSvg='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:11px;height:11px"><path d="M18 6L6 18M6 6l12 12"/></svg>';
    $('#st-presets').innerHTML=ps.length?ps.map(p=>'<span class="tag">'+esc(p.name)
      +(p.context?' · '+esc(p.context):'')+(p.model?' · '+esc(p.model):'')
      +' <button class="btn xs" data-stp="'+esc(p.name)+'" data-act="use">Use</button>'
      +' <button class="btn xs danger" data-stp="'+esc(p.name)+'" data-act="del" title="Delete preset" aria-label="Delete preset">'+xSvg+'</button></span>').join('')
      :'<span class="hint">No saved presets yet. Name one in the launch form to save this exact configuration.</span>';
    const sel=$('#st-preset'),cur=sel.value;
    sel.innerHTML='<option value="">(none)</option>'+ps.map(p=>'<option value="'+esc(p.name)+'">'+esc(p.name)+'</option>').join('');
    sel.value=cur;
  }catch(e){}
}
$('#st-refreshpresets').addEventListener('click',loadStrataPresets);
$('#st-presets').addEventListener('click',async e=>{
  const b=e.target.closest('[data-stp]');if(!b)return;
  if(b.dataset.act==='use'){$('#st-preset').value=b.dataset.stp;stPreview();toast('Preset '+b.dataset.stp+' selected','ok')}
  else{
    if(!await confirmDlg('Delete preset','Delete Strata preset <b>'+esc(b.dataset.stp)+'</b>?','Delete',true))return;
    try{await post('/api/strata/presets/delete',{name:b.dataset.stp});loadStrataPresets();toast('Preset deleted','ok')}
    catch(err){toast('Delete failed: '+err.message,'err',8000)}
  }
});
function stOut(txt){const o=$('#st-out');o.classList.remove('hide');o.textContent=txt}
$('#st-plan').addEventListener('click',async()=>{
  const b=$('#st-plan');b.disabled=true;b.textContent='Planning…';
  try{
    const r=await post('/api/strata/launch',Object.assign(stForm(),{plan:true}));
    stOut(r.json?JSON.stringify(r.json,null,1):(r.raw||r.error||'(empty plan)'));
  }catch(e){stOut('Plan failed: '+e.message)}
  b.disabled=false;b.textContent='Preview plan (read-only)';
});
$('#st-launch').addEventListener('click',async()=>{
  const btn=$('#st-launch');
  if(!await confirmDlg('Launch Strata','Strata may download model files (tens of GB) and load a server into memory. Continue?','Download & Launch',true))return;
  btn.disabled=true;btn.textContent='Launching…';
  try{
    const r=await post('/api/strata/launch',stForm());
    toast('Strata launch accepted — loading…','ok',4000);
    if(r.state){STRATA=r.state;drawStrata(STRATA)}
    pollStatus(true);loadStrataPresets();
  }catch(e){toast('Strata launch failed: '+e.message,'err',15000)}
  btn.disabled=false;btn.textContent='Download & Launch';
});
$('#stratastopbtn').addEventListener('click',async()=>{
  if(!await confirmDlg('Stop Strata','Unload the Strata model?','Stop',true))return;
  try{const r=await post('/api/strata/stop');STRATA=r.state;drawStrata(STRATA);pollStatus(true);toast('Strata stopped','ok')}
  catch(e){toast('Stop failed: '+e.message,'err',8000)}
});
$('#stratadoctor').addEventListener('click',async()=>{
  const b=$('#stratadoctor');b.disabled=true;b.textContent='Doctor running…';
  try{const d=await api('/api/strata/doctor');$('#strataformcard').classList.remove('hide');stOut(d.raw||d.error||'(no output)')}
  catch(e){toast('Doctor failed: '+e.message,'err')}
  b.disabled=false;b.textContent='Run doctor';
});
$('#strataupdate').addEventListener('click',async()=>{
  const b=$('#strataupdate');b.disabled=true;b.textContent='Checking…';
  try{
    const d=await api('/api/strata/update');
    $('#strataformcard').classList.remove('hide');
    stOut(d.json?JSON.stringify(d.json,null,1):(d.raw||d.error||'(no output)'));
  }catch(e){toast('Update check failed: '+e.message,'err')}
  b.disabled=false;b.textContent='Check update';
});
$('#stratakeychip').addEventListener('click',async()=>{
  const c=$('#stratakeychip');
  if(c.dataset.revealed==='1'){c.textContent='key '+(c.dataset.masked||'');c.dataset.revealed='0';return}
  try{const k=await api('/api/strata/key?reveal=1');if(k.key){c.textContent='key '+k.key;c.dataset.revealed='1'}}
  catch(e){toast('Could not read key: '+e.message,'err')}
});
async function loadStrataKey(){
  try{
    const k=await api('/api/strata/key');
    const c=$('#stratakeychip');
    if(k.key_set){
      c.dataset.masked=k.masked;
      if(c.dataset.revealed!=='1')c.textContent='key '+k.masked+' (click to reveal)';
      c.classList.remove('hide');
    }else c.classList.add('hide');
  }catch(e){}
}
function paintFilteredLog(el,text,filterId,autoId,countId){
  const f=($(filterId)&&$(filterId).value||'').trim().toLowerCase();
  const lines=(text||'').split('\n');
  const shown=lines.filter(l=>!f||l.toLowerCase().includes(f));
  const html=shown.map(l=>{
    const e=esc(l);
    if(/error|failed|exception|abort|refused/i.test(l))return '<span class="e">'+e+'</span>';
    if(/warn/i.test(l))return '<span class="w">'+e+'</span>';
    return e;
  }).join('\n');
  if(el.innerHTML!==html){el.innerHTML=html;if(!autoId||$(autoId).checked)el.scrollTop=1e9;}
  if(countId&&$(countId))$(countId).textContent=shown.length+' / '+lines.length+' lines';
}
async function refreshStrataLog(){
  try{const d=await api('/api/strata/log');STLOGTEXT=d.log||'';paintFilteredLog($('#stratalog'),STLOGTEXT,'#stlogfilter','#stlogauto','#stlogn')}catch(e){}
}
$('#stlogfilter').addEventListener('input',()=>paintFilteredLog($('#stratalog'),STLOGTEXT,'#stlogfilter','#stlogauto','#stlogn'));
$('#stlogauto').addEventListener('change',()=>{if($('#stlogauto').checked)$('#stratalog').scrollTop=1e9});
$('#stlogcopy').addEventListener('click',()=>copy(STLOGTEXT,'Strata log'));

/* ============================== avalon & optimizer ============================== */
let ASEL=null,ALIST=[],AVSNAPS=[],AVLOGTEXT='',AV_PROF_SIG='',AV_TGT_SIG='',AV_SNAP_SIG='';
async function loadAvalon(forceAux){
  try{
    ALIST=await api('/api/avalon/profiles');AV=await api('/api/avalon/status');
    if(!ASEL||!ALIST.some(p=>p.key===ASEL))ASEL=(AV&&AV.running&&AV.profile)||'';
    if(!ASEL&&ALIST.length)ASEL=(ALIST.find(p=>p.available)||ALIST[0]).key;
    const pSig=JSON.stringify({s:ASEL,r:AV.running,p:AV.profile,a:ALIST.map(x=>[x.key,x.available,x.label])});
    if(pSig!==AV_PROF_SIG){
      AV_PROF_SIG=pSig;
      $('#avalonprofiles').innerHTML=ALIST.map(p=>'<label class="frow" style="align-items:flex-start;padding:10px 11px">'
        +'<input type="radio" name="sprof" value="'+esc(p.key)+'"'+(p.key===ASEL?' checked':'')+(p.available?'':' disabled')+'>'
        +'<span style="flex:1;min-width:0"><b>'+esc(p.label)+'</b>'
        +(AV.running&&AV.profile===p.key?' <span class="tag ok">running</span>':'')
        +(p.available?'':' <span class="tag err">config missing</span>')
        +'<br><span class="hint">'+esc(p.desc)+'</span><br><span class="tag ok" style="margin-top:4px">'+esc(p.speed)+'</span></span></label>').join('');
      $$('#avalonprofiles input[name=sprof]').forEach(r=>r.addEventListener('change',()=>{ASEL=r.value;loadAvalon();previewAvalonTune()}));
    }
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
    drawHeader();drawChatTarget();
    if(forceAux||!AV_SNAP_SIG)loadAvalonSnaps();
    if(forceAux||!AV_TGT_SIG)loadAvalonTargets();
  }catch(e){}
}
function syncAvCtxLabel(){
  const v=Number($('#av-ctx').value)||0;
  $('#av-ctx-lbl').textContent=v?n0(v)+' tok':'default';
}
function getAvalonTuneForm(){
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
    parallel:Number($('#av-np').value)||0,
    device:$('#av-device').value||null,
    ngl:num($('#av-ngl').value),
    n_cmoe:num($('#av-cmoe').value),
    fa:$('#av-fa').value||null,
    threads:num($('#av-threads').value),
    mlock:$('#av-mlock').value==='on'?true:null
  };
}
function setAvalonTuneForm(t){
  const v=x=>(x==null||x==='')?'':String(x);
  $('#av-ctx').value=String(t.ctx||0);
  syncAvCtxLabel();
  $('#av-batch').value=v(t.batch);
  $('#av-ubatch').value=v(t.ubatch);
  $('#av-ctk').value=t.ctk||'';
  $('#av-ctv').value=t.ctv||'';
  $('#av-draft').value=v(t.draft_n);
  $('#av-ubd').value=v(t.ubd);
  $('#av-np').value=String(t.parallel||0);
  $('#av-device').value=t.device||'ROCm0';
  $('#av-ngl').value=v(t.ngl);
  $('#av-cmoe').value=v(t.n_cmoe);
  $('#av-fa').value=t.fa||'';
  $('#av-threads').value=v(t.threads);
  $('#av-mlock').value=t.mlock?'on':'';
}
async function previewAvalonTune(){
  syncAvCtxLabel();
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
/* ---- avalon target picker ---- */
async function loadAvalonTargets(){
  try{
    const d=await api('/api/avalon/targets');
    const sig=JSON.stringify(d);
    if(sig===AV_TGT_SIG||document.activeElement===$('#avtargetsel'))return;
    AV_TGT_SIG=sig;
    const sel=$('#avtargetsel');
    sel.innerHTML=(d.targets||[]).map(t=>'<option value="'+esc(t.path)+'"'+(t.current?' selected':'')+'>'
      +esc(t.base)+(t.size_h?' · '+esc(t.size_h):'')+(t.shards>1?' · '+t.shards+' shards':'')+(t.draft?' · MTP':'')+'</option>').join('');
    $('#avtargetdraft').textContent=d.has_draft?'MTP draft available':'no MTP draft for this pack';
  }catch(e){}
}
$('#avtargetsel').addEventListener('change',async()=>{
  const v=$('#avtargetsel').value;
  try{
    const r=await post('/api/avalon/target',{model:v});
    toast('Avalon target: '+(r.base||'auto'),'ok');
    AV_TGT_SIG='';loadAvalon(true);
  }catch(e){toast('Target change failed: '+e.message,'err');AV_TGT_SIG='';loadAvalonTargets();}
});
/* ---- one-click snapshots ---- */
async function loadAvalonSnaps(){
  try{
    const d=await api('/api/avalon/snapshots');
    AVSNAPS=d.snapshots||[];
    const sig=JSON.stringify(AVSNAPS);
    if(sig===AV_SNAP_SIG)return;
    AV_SNAP_SIG=sig;
    const el=$('#avsnaps');
    if(!AVSNAPS.length){el.innerHTML='<span class="hint">No saved launches yet. Tune the server above, then “Save current”.</span>';return;}
    const xSvg='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:12px;height:12px"><path d="M18 6L6 18M6 6l12 12"/></svg>';
    el.innerHTML=AVSNAPS.map(s=>'<div class="frow" style="padding:8px 10px;align-items:center;gap:10px">'
      +'<span style="flex:1;min-width:0"><b>'+esc(s.name)+'</b> <span class="tag mut">'+esc(s.profile)+'</span><br>'
      +'<span class="hint mono">'+esc(s.cmd||'')+'</span></span>'
      +'<button class="btn sm primary" data-snaplaunch="'+esc(s.id)+'">Launch</button>'
      +'<button class="btn xs danger" data-snapdel="'+esc(s.id)+'" title="Delete snapshot" aria-label="Delete snapshot">'+xSvg+'</button></div>').join('');
    $$('[data-snaplaunch]').forEach(b=>b.addEventListener('click',async()=>{
      b.disabled=true;
      try{await post('/api/avalon/snapshots/launch',{id:b.dataset.snaplaunch});toast('Avalon starting (125B load takes a while)','ok');loadAvalon();}
      catch(e){toast('Launch failed: '+e.message,'err',12000);b.disabled=false;}
    }));
    $$('[data-snapdel]').forEach(b=>b.addEventListener('click',async()=>{
      try{await api('/api/avalon/snapshots/'+b.dataset.snapdel,{method:'DELETE'});AV_SNAP_SIG='';loadAvalonSnaps();}
      catch(e){toast('Delete failed: '+e.message,'err')}
    }));
  }catch(e){}
}
$('#avsnapsave').addEventListener('click',async()=>{
  const name=$('#avsnapname').value.trim();
  const f=getAvalonTuneForm();
  const prof=f.profile||ASEL||'flagship';
  delete f.profile;
  try{
    const r=await post('/api/avalon/snapshots',{name:name,profile:prof,tune:f});
    $('#avsnapname').value='';
    toast('Saved launch “'+(r.name||name)+'”','ok');
    AV_SNAP_SIG='';loadAvalonSnaps();
  }catch(e){toast('Save failed: '+e.message,'err')}
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
$('#avlogfilter').addEventListener('input',()=>paintFilteredLog($('#avalonlog'),AVLOGTEXT,'#avlogfilter','#avlogauto','#avlogn'));
$('#avlogauto').addEventListener('change',()=>{if($('#avlogauto').checked)$('#avalonlog').scrollTop=1e9});
$('#avlogcopy').addEventListener('click',()=>copy(AVLOGTEXT,'Avalon log'));
$('#avalonstart').addEventListener('click',async()=>{
  $('#avalonstart').disabled=true;
  try{await post('/api/avalon/start',{profile:ASEL,tune:getAvalonTuneForm()});toast('Avalon starting — a 125B load takes a while','ok')}
  catch(e){
    const t=e.message;
    if(t.indexOf('Stop the llama.cpp server')>=0){
      if(await confirmDlg('VRAM conflict','This profile needs the VRAM the llama.cpp server is holding.<br><br>Stop the llama.cpp server and start Avalon?','Stop llama.cpp & start Avalon',true)){
        try{await post('/api/server/stop');pollStatus(true);$('#avalonstart').click()}
        catch(e2){toast('Could not stop the llama.cpp server: '+e2.message,'err')}}
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

/* ============================== maya ============================== */
let MLIST=[],MSEL='';
function drawMaya(s,list){
  const chip=$('#mayachip');
  chip.textContent=s.running?(s.state==='ready'?'maya · ready':'maya · loading'):'maya · stopped';
  chip.className='tag '+(s.running?(s.state==='ready'?'ok':'warn'):'mut');
  const tel=s.telemetry||{};
  $('#m-state-tag').textContent=tel.active?'answering':(s.running?(s.state==='ready'?'ready':'loading'):'idle');
  $('#m-state-tag').className='tag '+(tel.active?'ok':(s.running&&s.state==='ready'?'acc':'mut'));
  $('#m-dec-tps').textContent=tel.decode_tps!=null?tel.decode_tps:'—';
  $('#m-dec-sub').textContent=tel.last_tokens?n0(tel.last_tokens)+' tokens last turn':'Waiting for log telemetry';
  $('#m-pre-tps').textContent=tel.prefill_tps!=null?tel.prefill_tps:'—';
  $('#m-vram').textContent=tel.vram_gb!=null?tel.vram_gb:'—';
  $('#m-exp-sub').textContent=tel.experts_vram!=null&&tel.experts_total
    ?'Experts in VRAM: '+n0(tel.experts_vram)+' / '+n0(tel.experts_total)+' ('+tel.experts_vram_pct+'%)'
    :'Experts in VRAM: —';
  const ctxVal=tel.context||s.ctx;
  $('#m-ctx').textContent=ctxVal?n0(ctxVal):'—';
  const loadParts=[];
  if(tel.load_s!=null)loadParts.push('loaded '+tel.load_s+'s');
  if(tel.warmup_s!=null)loadParts.push('warmup '+tel.warmup_s+'s');
  $('#m-load-sub').textContent=loadParts.length?loadParts.join(' · '):'Load / warmup: —';

  if(s.running){
    const kvs=[
      ['alias',s.alias||s.model_name],
      ['engine',s.engine||'maya'],
      ['port',':'+s.port],
      s.pid?['pid',s.pid]:null,
      s.ctx?['ctx',n0(s.ctx)]:null,
      s.adopted?['source','adopted']:['up',fmtUp(s.uptime||0)],
      s.key_set?['key',s.key_masked]:null,
    ].filter(Boolean);
    $('#mayadetail').innerHTML=kvs.map(kv=>'<span class="tag">'+esc(kv[0])+' <b>'+esc(kv[1])+'</b></span>').join('')
      +(s.engine==='hy4'?'<span class="tag warn" title="Hy4 serves one request at a time">1 req/time</span>':'')
      +'<button class="btn xs" data-copy="http://127.0.0.1:'+s.port+'/v1">Copy base URL</button>';
  }else{
    $('#mayadetail').innerHTML='<span class="hint">Maya is stopped. Choose a config below to start serving.</span>';
  }

  if(list){
    if(!MSEL||!list.some(c=>c.path===MSEL))MSEL=(s.running&&s.config_path)||'';
    if(!MSEL&&list.length)MSEL=(list.find(c=>c.available)||list[0]).path;
    if(!list.length){
      $('#mayaconfigs').innerHTML='<div class="empty"><b>No Project Maya configs found</b>Expected JSON configs in <code>~/project-maya/</code> (e.g. <code>maya-hy4-flash-maya-xs.json</code>).</div>';
    }else{
      $('#mayaconfigs').innerHTML=list.map(c=>{
        const isRun=s.running&&(s.config_path===c.path||s.config_id===c.id);
        const tags=[
          '<span class="tag">'+esc(c.engine.toUpperCase())+'</span>',
          c.quant?'<span class="tag acc">'+esc(c.quant)+'</span>':'',
          '<span class="tag mut">'+n0(c.ctx)+' ctx</span>',
          '<span class="tag mut">:'+esc(c.port)+'</span>',
          c.dir_size_h?'<span class="tag mut">'+esc(c.dir_size_h)+'</span>':'',
          c.mtp?'<span class="tag ok">MTP</span>':'',
          c.reasoning_effort?'<span class="tag mut">think:'+esc(c.reasoning_effort)+'</span>':'',
          c.key_set?'<span class="tag mut mono" title="API key set (server-side)">key '+esc(c.key_masked)+'</span>':'',
          c.concurrency_note?'<span class="tag warn">'+esc(c.concurrency_note)+'</span>':'',
          isRun?'<span class="tag ok">running</span>':'',
          !c.available?'<span class="tag err">binary missing</span>':'',
        ].filter(Boolean).join(' ');
        return '<label class="frow" style="align-items:flex-start;padding:10px 11px">'
          +'<input type="radio" name="mcfg" value="'+esc(c.path)+'"'+(c.path===MSEL?' checked':'')+'>'
          +'<span style="flex:1;min-width:0"><b>'+esc(c.model_name)+'</b> <span class="hint mono">'+esc(c.filename)+'</span>'
          +'<div class="kvs" style="margin-top:4px">'+tags+'</div></span></label>';
      }).join('');
      $$('#mayaconfigs input[name=mcfg]').forEach(r=>r.addEventListener('change',()=>{MSEL=r.value;drawMaya(MAYA||s,MLIST)}));
    }
  }
  const selObj=(MLIST||[]).find(c=>c.path===MSEL);
  const selRunning=!!(s.running&&selObj&&(s.config_path===selObj.path||s.config_id===selObj.id));
  $('#mayastart').disabled=!MSEL||!!s.running;
  $('#mayaswitch').disabled=!MSEL||!s.running||selRunning;
  $('#mayastop').disabled=!s.running;
  $('#mayakeybtn').disabled=!s.key_set&&!(selObj&&selObj.key_set);
}
async function loadMaya(){
  try{
    const [cfgs,st]=await Promise.all([api('/api/maya/configs'),api('/api/maya/status')]);
    MLIST=(cfgs&&cfgs.configs)||[];
    MAYA=st;
    drawMaya(st,MLIST);
    drawHeader();drawChatTarget();
  }catch(e){}
}
async function refreshMayaLog(){
  try{
    const d=await api('/api/maya/log');
    if(d.path)$('#mayalogpath').textContent=d.path;
    paintInto($('#mayalog'),d.log||'');
  }catch(e){}
}
$('#mayadetail').addEventListener('click',e=>{const b=e.target.closest('[data-copy]');if(b)copy(b.dataset.copy,'Base URL')});
$('#mayalogref').addEventListener('click',refreshMayaLog);
$('#mayastart').addEventListener('click',async()=>{
  if(!MSEL)return;
  $('#mayastart').disabled=true;
  try{
    await post('/api/maya/start',{config:MSEL,switch:false});
    toast('Maya starting — cold start takes 30–60 s','ok');
  }catch(e){toast('Start failed: '+e.message,'err',12000)}
  loadMaya();refreshMayaLog();
});
$('#mayaswitch').addEventListener('click',async()=>{
  if(!MSEL)return;
  if(!await confirmDlg('Switch Maya config','Stop the running Maya server and start the selected config?','Switch',true))return;
  $('#mayaswitch').disabled=true;
  try{
    await post('/api/maya/switch',{config:MSEL,switch:true});
    toast('Switching Maya config — cold start takes 30–60 s','ok');
  }catch(e){toast('Switch failed: '+e.message,'err',12000)}
  loadMaya();refreshMayaLog();
});
$('#mayastop').addEventListener('click',async()=>{
  if(!await confirmDlg('Stop Maya','Stop the running Project Maya server and release VRAM?','Stop',true))return;
  $('#mayastop').disabled=true;
  try{
    const r=await post('/api/maya/stop');
    toast(r.stopped?'Maya stopped & VRAM released':'Stop requested — check log',r.stopped?'ok':'warn');
  }catch(e){toast('Stop failed: '+e.message,'err')}
  loadMaya();refreshMayaLog();
});
$('#mayakeybtn').addEventListener('click',async()=>{
  try{
    const q=MSEL?('&config='+encodeURIComponent(MSEL)):'';
    const d=await api('/api/maya/key?reveal=1'+q);
    if(!d.key_set||!d.key)return toast('No API key configured','warn');
    await copy(d.key,'Maya API key');
  }catch(e){toast('Failed: '+e.message,'err')}
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
  const checked=$$('#hffiles input:checked');
  const sel=checked.map(i=>i.value);
  const sizes={};checked.forEach(i=>{sizes[i.value]=Number(i.dataset.size)||0;});
  if(!sel.length)return toast('Select at least one file','warn');
  try{await post('/api/hf/pull',{repo:HFREPO,files:sel,sizes});
    toast('Queued '+sel.length+' file'+(sel.length>1?'s':''),'ok');ensureDl(true);pollDownloads()}
  catch(e){toast('Pull failed: '+e.message,'err',10000)}
});
function ensureDl(on){
  if(on&&!DLTIMER)DLTIMER=setInterval(pollDownloads,1200);
  if(!on&&DLTIMER){clearInterval(DLTIMER);DLTIMER=null}
}
let DL_DONE_IDS=new Set();
async function pollDownloads(){
  let ds;try{ds=await api('/api/downloads')}catch(e){return}
  const activeCount=ds.filter(d=>d.status==='downloading'||d.status==='queued').length;
  const nb=$('#nb-dl');
  if(nb){nb.textContent=String(activeCount);nb.classList.toggle('hide',activeCount===0);}
  let newlyDone=false;
  ds.forEach(d=>{if(d.status==='done'&&!DL_DONE_IDS.has(d.id)){DL_DONE_IDS.add(d.id);newlyDone=true;}});
  if(newlyDone){loadModels();toast('Download finished · model library updated','ok',3200);}
  if(!ds.length){$('#downloads').innerHTML='<span class="hint">No downloads queued.</span>';$('#dlinfo').textContent='';ensureDl(false);return}
  const live=activeCount>0;
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
/* ============================== engine setup ============================== */
let SETUP=null,SETUPTIMER=null;
async function loadSetup(){
  try{SETUP=await api('/api/setup/status');drawSetup()}catch(e){}
}
function drawSetup(){
  if(!SETUP)return;
  const L=SETUP.llama||{},S=SETUP.strata||{},A=SETUP.avalon||{},J=SETUP.job||{};
  const chip=(id,ok,txt)=>{const c=$(id);c.textContent=txt;c.className='tag '+(ok?'ok':'err')};
  chip('#setup-llama-chip',L.installed,L.installed?(L.update_available?'update available':'installed'):'not installed');
  $('#setup-llama-detail').innerHTML=[L.version?('version '+L.version):null,L.binary?('bin '+L.binary):null,
    L.update_available?('latest release '+(L.latest||'')):null,L.platform?('target '+L.platform):null]
    .filter(Boolean).map(x=>'<span class="tag">'+esc(x)+'</span>').join('');
  $('#setup-llama-btn').textContent=L.installed?(L.update_available?'Update to '+(L.latest||'latest'):'Reinstall latest'):'Install latest';
  chip('#setup-strata-chip',S.installed,S.installed?'installed':'not installed');
  $('#setup-strata-detail').innerHTML=S.binary?'<span class="tag">bin '+esc(S.binary)+'</span>':'<span class="hint">official one-line installer</span>';
  chip('#setup-avalon-chip',A.installed,A.installed?'built':(A.source_exists?'source present':'no source'));
  $('#setup-avalon-detail').innerHTML=[A.source?('src '+A.source):null,A.installed?('bin '+A.binary):null]
    .filter(Boolean).map(x=>'<span class="tag">'+esc(x)+'</span>').join('')||'<span class="hint">clone Avalon into ~/avalon first</span>';
  $('#setup-avalon-btn').disabled=!A.source_exists;
  const jc=$('#setupjobchip');
  jc.textContent=J.state==='running'?(J.name||'job')+'…':J.state;
  jc.className='tag '+(J.state==='running'?'warn':J.state==='failed'?'err':J.state==='done'?'ok':'mut');
  const box=$('#setuplog');
  if(J.log&&J.log.length){box.classList.remove('hide');paintInto(box,J.log.join('\n'))}
  const running=J.state==='running';
  $$('#t-settings button[id^="setup-"][id$="-btn"]').forEach(b=>b.disabled=running);
  if(!A.source_exists)$('#setup-avalon-btn').disabled=true;
  if(running&&!SETUPTIMER)SETUPTIMER=setInterval(()=>{loadSetup();if(tab!=='settings'){clearInterval(SETUPTIMER);SETUPTIMER=null}},1500);
  if(!running&&SETUPTIMER){clearInterval(SETUPTIMER);SETUPTIMER=null}
  if(J.state==='done'&&!SETUP._told){SETUP._told=true;toast('Engine setup finished','ok');loadSettings();loadModels();pollStatus(true)}
  if(J.state==='failed'&&!SETUP._toldf){SETUP._toldf=true;toast('Engine setup failed — see log','err',8000)}
}
async function setupInstall(engine,variant,msg){
  if(!await confirmDlg('Engine setup',msg,'Install',false))return;
  try{await post('/api/setup/install',{engine,variant,confirm:true});toast('Setup started: '+engine,'ok');loadSetup()}
  catch(e){toast('Setup failed: '+e.message,'err',8000)}
}
$('#setup-llama-btn').addEventListener('click',()=>{
  const v=$('#setup-llama-variant').value;
  setupInstall('llama',v,'Download the latest official llama.cpp release build for this machine ('+(SETUP&&SETUP.llama.platform||'')+'), unpack it under the panel data dir, and point server_binary at it?');
});
$('#setup-strata-btn').addEventListener('click',()=>{
  setupInstall('strata','auto','Run the official Strata installer script on this machine? It downloads the engine for your GPU.');
});
$('#setup-avalon-btn').addEventListener('click',()=>{
  setupInstall('avalon','auto','Configure and build llama-server from ~/avalon (cmake, full target)? This can take a while.');
});

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
    $('#set-stratabin').value=s.strata_bin||'';
    $('#set-strata-status').textContent=s.strata_installed?'found: '+(s.strata_bin_resolved||''):'not found';
    $('#set-strataalias').value=s.strata_alias||'';
    $('#set-stratakey').value='';
    $('#set-stratakey-status').textContent=s.strata_api_key_set?'overridden ('+s.strata_api_key_masked+')':'not set (Strata owns its key)';
    if($('#set-agenturl')){$('#set-agenturl').value=s.agent_llm_url||'';}
    if($('#set-agentmodel')){$('#set-agentmodel').value=s.agent_llm_model||'';}
    if($('#set-agentkey')){$('#set-agentkey').value='';}
    if($('#set-agentkey-status')){$('#set-agentkey-status').textContent=s.agent_llm_key_set?'saved ('+s.agent_llm_key_masked+')':'not set';}
    if($('#set-agentfallback')){$('#set-agentfallback').checked=s.agent_llm_fallback_local!==false;}
    const c=s.caps||{};
    $('#set-caps-badge').textContent=c.exists?'binary verified':'binary missing';
    $('#set-caps-badge').className='tag '+(c.exists?'ok':'err');
    $('#set-bin-ver').textContent=c.version||'';
    $('#set-bin-flags').innerHTML=['fa_value','ncmoe','metrics','fit','ctk','mlock','no_mmap','parallel','draft','draft_max','ot','tensor_split','tbatch','keep','jinja']
      .map(k=>'<span class="tag '+(c[k]?'ok':'mut')+'">'+k+(c[k]?' ✓':' –')+'</span>').join('');
    $('#set-disks').innerHTML=(s.disks||[]).map(d=>d.exists?d.path+': '+d.free_gb+' GB free':d.path+': missing').join(' · ');
    loadPanelProc();
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
      hf_token:$('#set-hftok').value.trim(),
      strata_bin:$('#set-stratabin').value.trim(),
      strata_alias:$('#set-strataalias').value.trim(),
      strata_api_key:$('#set-stratakey').value.trim(),
      agent_llm_url:$('#set-agenturl')?$('#set-agenturl').value.trim():undefined,
      agent_llm_model:$('#set-agentmodel')?$('#set-agentmodel').value.trim():undefined,
      agent_llm_key:$('#set-agentkey')?$('#set-agentkey').value.trim():undefined,
      agent_llm_fallback_local:$('#set-agentfallback')?$('#set-agentfallback').checked:undefined
    });
    toast('Settings saved','ok');
    await Promise.all([loadSettings(),loadModels(),loadAgentProvider(),loadAgentStatus()]);
  }catch(e){toast('Save failed: '+e.message,'err')}
  btn.disabled=false;
});

/* ============================== panel process ============================== */
const panelPing=()=>api('/api/panel/ping');
async function loadPanelProc(){
  try{
    const p=await panelPing();
    $('#panel-proc-chip').textContent='pid '+p.pid;
    $('#panel-proc-chip').className='tag ok';
    $('#panel-proc-detail').textContent='up '+fmtUp(p.uptime_s)+' · port '+p.port+(p.restartable?'':' · restart unavailable');
    $('#panelrestart').disabled=!p.restartable;
  }catch(e){
    $('#panel-proc-chip').textContent='unreachable';
    $('#panel-proc-chip').className='tag err';
  }
}
async function waitForPanelRestart(before){
  // execv keeps the PID, so watch started_at instead; every poll during the swap fails.
  for(let i=0;i<120;i++){
    await new Promise(r=>setTimeout(r,500));
    try{
      const p=await panelPing();
      if(p.ok&&p.started_at!==before){location.reload();return}
    }catch(e){}
  }
  toast('Panel never answered after restart. Check the service log.','err',15000);
}
$('#panelrestart').addEventListener('click',async()=>{
  if(!await confirmDlg('Restart panel','Re-exec the panel process now? Running models are not stopped, and this tab reconnects automatically.','Restart',true))return;
  const btn=$('#panelrestart');
  let before=null;
  try{before=(await panelPing()).started_at}catch(e){}
  btn.disabled=true;btn.textContent='Restarting…';
  try{await post('/api/panel/restart',{delay_ms:600})}
  catch(e){btn.disabled=false;btn.textContent='Restart panel';return toast('Restart failed: '+e.message,'err')}
  toast('Panel restarting…','warn',4000);
  waitForPanelRestart(before);
});

/* ============================== command palette (Ctrl/Cmd+K) ============================== */
let PAL_ITEMS=[],PAL_IDX=0;
function buildPaletteItems(){
  const items=[];
  TABS.forEach((t,i)=>items.push({label:'Go to '+t.toUpperCase()+' tab',sub:'Shortcut '+(i+1),run:()=>setTab(t)}));
  items.push({label:'Run one-click benchmark on active server',sub:'llama.cpp',run:()=>{setTab('server');$('#benchrun').click()}});
  if(ACTIVE&&ACTIVE.running){
    items.push({label:'Stop active llama-server ('+ACTIVE.alias+')',sub:':'+ACTIVE.port,run:()=>$('#srvstop').click()});
  }
  if(AV&&AV.running){
    items.push({label:'Stop Avalon ('+(AV.profile||'running')+')',sub:':'+(AV.port||8086),run:()=>$('#avalonstop').click()});
  }else if(ALIST&&ALIST.some(p=>p.available)){
    items.push({label:'Start Avalon ('+(ASEL||'flagship')+')',sub:'Avalon',run:()=>{setTab('avalon');$('#avalonstart').click()}});
  }
  (AVSNAPS||[]).forEach(s=>{
    items.push({label:'Launch Avalon snapshot: '+s.name,sub:s.profile||'snapshot',run:async()=>{
      try{await post('/api/avalon/snapshots/launch',{id:s.id});toast('Avalon starting: '+s.name,'ok');setTab('avalon');loadAvalon();}
      catch(e){toast('Launch failed: '+e.message,'err',10000)}
    }});
  });
  if(STRATA&&STRATA.running){
    items.push({label:'Stop Strata ('+(STRATA.model||'model')+')',sub:':'+(STRATA.port||''),run:()=>$('#stratastopbtn').click()});
  }
  if(MAYA&&MAYA.running){
    items.push({label:'Stop Maya ('+(MAYA.alias||MAYA.model_name||'maya')+')',sub:':'+(MAYA.port||''),run:()=>$('#mayastop').click()});
  }
  items.push({label:'Strata launch setup…',sub:'Strata',run:()=>{setTab('strata');if($('#strataformcard').classList.contains('hide'))$('#st-toggleform').click()}});
  items.push({label:'Engine setup (llama.cpp / Strata / Avalon)',sub:'Settings',run:()=>{setTab('settings');loadSetup()}});
  items.push({label:'Restart the panel process',sub:'Panel',run:()=>{setTab('settings');loadPanelProc();$('#panelrestart').click()}});
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
  const selEl=$('#pal-list .pal-item.sel');
  if(selEl)selEl.scrollIntoView({block:'nearest'});
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

/* ============================== optimization agent & sidebar ============================== */
let AG_LAST_RESULT=null,AG_PROV_PRESETS={},AG_MAYA_CFGS=[];
async function syncAgentModelOptions(){
  const sel=$('#ag-model');if(!sel)return;
  const eng=$('#ag-engine').value;
  if(eng==='llama'){
    sel.innerHTML='<option value="">(auto-select active / pinned)</option>'
      +MODELS.map(m=>'<option value="'+esc(m.path)+'">'+esc(m.name)+' ('+esc(m.quant)+(m.moe?' · MoE':'')+')</option>').join('');
  }else if(eng==='avalon'){
    sel.innerHTML='<option value="flagship">flagship</option><option value="prefast">prefast</option><option value="compact">compact</option>';
  }else if(eng==='maya'){
    try{const d=await api('/api/maya/configs');AG_MAYA_CFGS=d.configs||[];}catch(e){}
    sel.innerHTML='<option value="">(auto-select best Maya config)</option>'
      +AG_MAYA_CFGS.map(c=>'<option value="'+esc(c.id)+'">'+esc(c.model_name)+' ('+esc(c.engine)+(c.quant?' · '+esc(c.quant):'')+(c.mtp?' · MTP':'')+')</option>').join('');
  }else{
    sel.innerHTML='<option value="">(Strata default / active)</option>'
      +MODELS.map(m=>'<option value="'+esc(m.path)+'">'+esc(m.name)+'</option>').join('');
  }
}
$('#ag-engine')?.addEventListener('change',syncAgentModelOptions);

/* Agent LLM Provider (Hermes-style presets + Custom + Always-On Model Polling) */
async function loadAgentProvider(){
  try{
    const d=await api('/api/agent/provider');
    AG_PROV_PRESETS=d.presets||{};
    const sel=$('#ag-prov-select');
    if(sel&&!sel.options.length){
      sel.innerHTML=Object.entries(AG_PROV_PRESETS).map(([k,v])=>'<option value="'+esc(k)+'">'+esc(v.label)+'</option>').join('');
    }
    if(sel)sel.value=d.provider||'custom';
    if($('#ag-prov-url'))$('#ag-prov-url').value=d.url||(AG_PROV_PRESETS[d.provider]?.url||'');
    if($('#ag-prov-model'))$('#ag-prov-model').value=d.model||'';
    if($('#ag-prov-headers'))$('#ag-prov-headers').value=d.headers||'';
    if($('#ag-prov-key'))$('#ag-prov-key').value='';
    if($('#ag-prov-key-status'))$('#ag-prov-key-status').textContent=d.key_set?'saved ('+d.key_masked+')':'not set';
    if($('#ag-prov-fallback'))$('#ag-prov-fallback').checked=d.fallback_local!==false;
    pollAgentProviderModels(true);
  }catch(e){}
}
async function pollAgentProviderModels(silent){
  const chip=$('#ag-prov-models-chip'),msg=$('#ag-prov-msg');
  if(chip)chip.textContent='polling models…';
  try{
    const prov=$('#ag-prov-select')?.value||'custom';
    const url=($('#ag-prov-url')?.value||'').trim();
    const api_key=($('#ag-prov-key')?.value||'').trim();
    const r=await post('/api/agent/provider/models',{provider:prov,url,api_key});
    const list=r.models||[];
    if(chip){chip.textContent='models: '+list.length;chip.className='tag '+(list.length?'ok':'mut');}
    const dl=$('#ag-prov-models-dl'),pk=$('#ag-prov-model-pick');
    if(dl)dl.innerHTML=list.map(m=>'<option value="'+esc(m)+'"></option>').join('');
    if(pk){
      const cur=$('#ag-prov-model')?.value||'';
      pk.innerHTML='<option value="">— '+(list.length?list.length+' polled models':'no models polled')+' —</option>'
        +list.map(m=>'<option value="'+esc(m)+'"'+(m===cur?' selected':'')+'>'+esc(m)+'</option>').join('');
    }
    if(msg)msg.textContent=r.error?('Model poll note: '+r.error):(list.length?'Polled '+list.length+' model(s) from '+r.url:'');
  }catch(e){
    if(chip){chip.textContent='models: err';chip.className='tag warn';}
    if(!silent)toast('Model poll failed: '+e.message,'warn');
  }
}
$('#ag-prov-select')?.addEventListener('change',()=>{
  const p=AG_PROV_PRESETS[$('#ag-prov-select').value];
  if(p&&typeof p.url==='string')$('#ag-prov-url').value=p.url;
  pollAgentProviderModels(true);
});
$('#ag-prov-url')?.addEventListener('blur',()=>pollAgentProviderModels(true));
$('#ag-prov-key')?.addEventListener('blur',()=>pollAgentProviderModels(true));
$('#ag-prov-model-pick')?.addEventListener('change',()=>{
  const v=$('#ag-prov-model-pick').value;if(v)$('#ag-prov-model').value=v;
});
$('#ag-prov-poll')?.addEventListener('click',()=>pollAgentProviderModels(false));
$('#ag-prov-save')?.addEventListener('click',async()=>{
  const btn=$('#ag-prov-save');btn.disabled=true;
  try{
    const prov=$('#ag-prov-select').value;
    const url=$('#ag-prov-url').value.trim();
    const model=$('#ag-prov-model').value.trim();
    const fallback=$('#ag-prov-fallback').checked;
    const res=await post('/api/agent/provider',{
      provider:prov,
      url:url,
      model:model,
      api_key:$('#ag-prov-key').value.trim(),
      headers:$('#ag-prov-headers').value.trim(),
      fallback_local:fallback
    });
    if($('#set-agenturl'))$('#set-agenturl').value=res.url||'';
    if($('#set-agentmodel'))$('#set-agentmodel').value=res.model||'';
    if($('#set-agentfallback'))$('#set-agentfallback').checked=res.fallback_local!==false;
    if(res.status)renderAgentLlmChip(res.status);
    if(prov!=='local_fallback'&&!model){
      toast('Saved, but Model ID is empty — pick or enter a model to enable LLM mode','warn',8000);
    }else{
      toast('Agent LLM provider saved','ok');
    }
    await Promise.all([loadAgentProvider(),loadAgentStatus()]);
  }catch(e){toast('Save provider failed: '+e.message,'err')}
  btn.disabled=false;
});
$('#ag-prov-test')?.addEventListener('click',async()=>{
  const btn=$('#ag-prov-test');btn.disabled=true;
  try{
    const r=await post('/api/agent/provider/test',{});
    if(r.ok){toast('Provider OK ('+r.latency_ms+'ms): '+r.reply,'ok');if($('#ag-prov-msg'))$('#ag-prov-msg').textContent='Connected ('+r.latency_ms+' ms): '+r.reply;}
    else{toast('Provider test failed: '+(r.error||'check config'),'err');if($('#ag-prov-msg'))$('#ag-prov-msg').textContent=r.error||'Test failed';}
  }catch(e){toast('Test failed: '+e.message,'err')}
  btn.disabled=false;
});

function buildAgentPayload(){
  const eng=$('#ag-engine').value;
  const constraints={};
  const mc=Number($('#ag-minctx').value)||0;if(mc>0)constraints.min_ctx=mc;
  const hr=Number($('#ag-headroom').value)||0;if(hr>0)constraints.vram_headroom_mb=hr;
  return {
    engine:eng,
    goal:$('#ag-goal').value,
    model:eng==='avalon'?'':$('#ag-model').value,
    profile:eng==='avalon'?($('#ag-model').value||'flagship'):'flagship',
    constraints,
    user_prompt:$('#ag-prompt').value.trim(),
    max_trials:Number($('#ag-trials').value)||5
  };
}

function renderAgentResult(title,data){
  AG_LAST_RESULT=data;
  $('#ag-result-card').classList.remove('hide');
  $('#ag-result-title').textContent=title;
  const moe=data.moe;
  $('#ag-result-moe').innerHTML=moe
    ?('<span class="tag '+(moe.feasible?'ok':'err')+'">'+(moe.is_moe?'MoE · '+esc(moe.strategy):esc(moe.strategy))+'</span>'
      +(moe.gpu_hot_path_mb?'<span class="tag mut">GPU hot: '+moe.gpu_hot_path_mb+' MB</span>':'')
      +(moe.cpu_cold_experts_mb?'<span class="tag mut">CPU experts: '+moe.cpu_cold_experts_mb+' MB</span>':'')
      +(data.fit?'<span class="tag '+(data.fit.status==='fit'?'ok':data.fit.status==='tight'?'warn':'err')+'">'+esc(data.fit.badge)+' ('+data.fit.pct+'%)</span>':''))
    :'';
  const notes=data.notes||data.remediation_steps||[];
  const extra=data.llm_analysis||data.llm_diagnosis||'';
  $('#ag-result-notes').innerHTML=(notes.map(n=>'<div>• '+esc(n)+'</div>').join(''))
    +(extra?'<div style="margin-top:8px;padding:8px;border-left:2px solid var(--acc);background:var(--bg-2)">'+esc(extra)+'</div>':'');
  const cfg=data.config||data.fix_config;
  $('#ag-result-cfg').textContent=cfg?JSON.stringify(cfg,null,2):(data.cause||'No config changes proposed.');
  $('#ag-apply-btn').classList.toggle('hide',!cfg);
  $('#ag-save-preset-btn').classList.toggle('hide',!cfg);
}

function renderAgentLlmChip(lm){
  lm=lm||{};
  const ltxt=lm.mode==='external'?'LLM: '+lm.model:(lm.mode==='local_fallback'?'LLM: local fallback':'deterministic');
  if($('#ag-llm-chip')){$('#ag-llm-chip').textContent=ltxt;$('#ag-llm-chip').className='tag '+(lm.available?'ok':'mut');}
  if($('#ag-sb-llm')){$('#ag-sb-llm').textContent=ltxt;$('#ag-sb-llm').className='tag '+(lm.available?'ok':'mut');}
}

async function loadAgentStatus(){
  try{
    const d=await api('/api/agent/status');
    renderAgentLlmChip(d.llm);
    renderTuneStatus(d.tune);
  }catch(e){}
}

function renderTuneStatus(t){
  const running=!!(t&&t.running);
  if($('#ag-tune-chip')){$('#ag-tune-chip').textContent=running?'tuning…':(t&&t.session?t.session.state:'idle');$('#ag-tune-chip').className='tag '+(running?'acc':'mut');}
  if(!t||!t.session){$('#ag-tune-card')?.classList.add('hide');return;}
  const s=t.session;
  $('#ag-tune-card').classList.remove('hide');
  $('#ag-tune-state').textContent=s.state+' · '+s.completed_trials+'/'+s.max_trials+' trials';
  $('#ag-tune-trials').innerHTML=(s.trials||[]).map(tr=>
    '<div class="kvs" style="margin-bottom:4px">'
    +'<span class="tag '+(tr.ok?'ok':'err')+'">#'+tr.index+' '+esc(tr.label)+'</span>'
    +(tr.ok?'<span class="tag acc">'+tr.metrics.tok_s+' tok/s</span><span class="tag mut">TTFT '+tr.metrics.ttft_ms+'ms</span><span class="tag mut">VRAM '+tr.metrics.vram_mb+' MB</span><span class="tag mut">score '+tr.score+'</span>'
      :'<span class="hint">'+esc(tr.error||'failed')+'</span>')
    +'</div>'
  ).join('')||'<span class="hint">Starting first trial…</span>';
  $('#ag-tune-log').textContent=(s.log||[]).join('\n');
}

async function loadAgentCheckpoints(){
  try{
    const d=await api('/api/agent/checkpoints');
    const cps=d.checkpoints||[];
    $('#ag-checkpoints').innerHTML=cps.length?cps.map(c=>
      '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--bd)">'
      +'<div><div style="font-weight:600;font-size:12.5px">'+esc(c.label)+'</div><div class="hint">'+esc(c.id)+' · '+esc(c.engine)+' · '+new Date(c.ts*1000).toLocaleString()+'</div></div>'
      +'<div style="display:flex;gap:4px">'
      +'<button class="btn xs" data-cp-restore="'+esc(c.id)+'">Restore</button>'
      +'<button class="btn xs danger" data-cp-del="'+esc(c.id)+'">×</button>'
      +'</div></div>'
    ).join(''):'<span class="hint">No checkpoints yet. One is auto-created before any Apply or Auto-Tune.</span>';
  }catch(e){}
}

async function loadAgentPresets(){
  try{
    const d=await api('/api/agent/presets');
    const ps=d.presets||[];
    $('#ag-presets').innerHTML=ps.length?ps.map(p=>
      '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--bd)">'
      +'<div><div style="font-weight:600;font-size:12.5px">'+esc(p.name)+' <span class="tag mut">'+esc(p.engine)+'</span></div>'
      +'<div class="hint">'+esc(p.notes||p.goal)+'</div></div>'
      +'<div style="display:flex;gap:4px">'
      +'<button class="btn xs primary" data-pre-apply="'+esc(p.id)+'">Apply</button>'
      +(p.builtin?'':'<button class="btn xs danger" data-pre-del="'+esc(p.id)+'">×</button>')
      +'</div></div>'
    ).join(''):'<span class="hint">No presets.</span>';
    $('#ag-presets')._items=ps;
  }catch(e){}
}

async function loadAgentMemory(){
  try{
    const m=await api('/api/agent/memory');
    const cal=m.vram_calibration||{};
    const best=m.best||[];
    $('#ag-memory').innerHTML=
      '<div class="kvs" style="margin-bottom:8px">'
      +'<span class="tag mut">Trials: '+m.trial_count+'</span>'
      +'<span class="tag ok">Winners: '+m.best_count+'</span>'
      +'<span class="tag mut">Failure signatures: '+m.failure_count+'</span>'
      +'<span class="tag acc">VRAM calibration samples: '+(cal.total_samples||0)+'</span>'
      +'</div>'
      +(best.length?best.slice(0,8).map(b=>
        '<div class="hint" style="padding:3px 0">• <b>'+esc(b.engine)+'</b> / '+esc(b.model_key)+' ('+esc(b.goal)+'): '
        +((b.metrics&&b.metrics.tok_s)?b.metrics.tok_s+' tok/s, '+b.metrics.vram_mb+' MB VRAM':'recorded')+'</div>'
      ).join(''):'<span class="hint">Run Active Auto-Tune to record empirically verified winners & calibrate VRAM estimates.</span>');
  }catch(e){}
}

async function loadAgentAll(){
  syncAgentModelOptions();
  const sm=$('#ag-spec-model');
  if(sm)sm.innerHTML='<option value="">(auto-select active / pinned)</option>'+MODELS.map(m=>'<option value="'+esc(m.path)+'">'+esc(m.name)+'</option>').join('');
  await Promise.all([loadAgentStatus(),loadAgentProvider(),loadAgentCheckpoints(),loadAgentPresets(),loadAgentMemory(),loadThermalGuard()]);
}

/* Phase 2 UI: Thermal Guard */
async function loadThermalGuard(){
  try{
    const d=await api('/api/agent/thermal');
    const c=d.config||{};
    if($('#tg-warn'))$('#tg-warn').value=c.warn_temp_c||80;
    if($('#tg-throttle'))$('#tg-throttle').value=c.throttle_temp_c||88;
    if($('#tg-kill'))$('#tg-kill').value=c.kill_temp_c||95;
    if($('#tg-cool'))$('#tg-cool').value=c.cool_temp_c||70;
    if($('#tg-pwrpct'))$('#tg-pwrpct').value=c.power_throttle_pct||15;
    if($('#tg-vrampct'))$('#tg-vrampct').value=c.vram_pressure_pct||92;
    if($('#tg-enabled'))$('#tg-enabled').checked=!!c.enabled;
    if($('#tg-autoact'))$('#tg-autoact').checked=!!c.auto_act;
    const ch=$('#tg-state-chip');
    if(ch){
      ch.textContent=d.state+(d.hw_override_active?' · hw override':'');
      ch.className='tag '+(d.level===0?'ok':d.level===1?'warn':'err');
    }
    const hc=d.hw_caps||{};
    if($('#tg-hw-hint')&&hc.setup_hint){
      $('#tg-hw-hint').textContent='Hardware: '+hc.vendor+' (fan writable: '+hc.fan_writable+', power writable: '+hc.power_writable+'). '+hc.setup_hint;
    }
    const evs=d.events||[];
    if($('#tg-events')){
      $('#tg-events').innerHTML=evs.length?evs.slice(0,6).map(e=>'<div>• ['+new Date(e.ts*1000).toLocaleTimeString()+'] <b>'+esc(e.action)+'</b>: '+esc(e.detail)+'</div>').join(''):'No thermal events recorded.';
    }
  }catch(e){}
}
$('#tg-save')?.addEventListener('click',async()=>{
  try{
    await post('/api/agent/thermal/config',{
      enabled:$('#tg-enabled').checked,auto_act:$('#tg-autoact').checked,
      warn_temp_c:Number($('#tg-warn').value),throttle_temp_c:Number($('#tg-throttle').value),
      kill_temp_c:Number($('#tg-kill').value),cool_temp_c:Number($('#tg-cool').value),
      power_throttle_pct:Number($('#tg-pwrpct').value),vram_pressure_pct:Number($('#tg-vrampct').value)
    });
    toast('Thermal guard policy saved','ok');loadThermalGuard();
  }catch(e){toast('Save failed: '+e.message,'err')}
});
$('#tg-eval-now')?.addEventListener('click',async()=>{
  try{const r=await post('/api/agent/thermal/evaluate',{});toast('Thermal eval: '+r.state+' ('+(r.eff_temp_c||0)+'°C)','ok');loadThermalGuard();}
  catch(e){toast('Eval failed: '+e.message,'err')}
});
$$('[data-tg-act]').forEach(b=>b.addEventListener('click',async()=>{
  try{const r=await post('/api/agent/thermal/action',{action:b.dataset.tgAct});toast(r.ok?(r.detail||'Action executed'):(r.error||'Action failed'),r.ok?'ok':'warn');loadThermalGuard();}
  catch(e){toast('Action failed: '+e.message,'err')}
}));

/* Phase 2 UI: Workload Router */
async function runWorkloadRoute(apply){
  const out=$('#ag-route-out');
  try{
    const r=await post('/api/agent/route',{task:$('#ag-route-task').value,apply,sync_agents:$('#ag-route-sync').checked});
    const s=r.selected;
    out.innerHTML='<div><b>Selected:</b> '+esc(s.engine)+' · <code>'+esc(s.model_key)+'</code> (score '+s.score+')</div>'
      +'<div class="hint">'+(s.reasons||[]).map(esc).join(' · ')+'</div>'
      +(r.applied?'<div class="tag ok" style="margin-top:6px">Launched (checkpoint '+esc(r.applied.checkpoint_id)+')</div>':'')
      +(r.synced_agents?'<div class="tag acc" style="margin-top:6px">Synced coding agents</div>':'');
    if(apply){loadAgentCheckpoints();pollStatus(true);}
  }catch(e){toast('Route failed: '+e.message,'err')}
}
$('#ag-route-preview')?.addEventListener('click',()=>runWorkloadRoute(false));
$('#ag-route-apply')?.addEventListener('click',()=>runWorkloadRoute(true));

/* Phase 2 UI: Speculative Decoding Tuner */
$('#ag-spec-run')?.addEventListener('click',async()=>{
  const btn=$('#ag-spec-run');btn.disabled=true;
  try{
    const r=await post('/api/agent/speculate',{model:$('#ag-spec-model').value,apply_best:true});
    $('#ag-spec-out').innerHTML='<div><b>'+esc(r.verdict)+'</b></div>'
      +(r.trials||[]).map(t=>'<div class="hint">• '+esc(t.label)+': '+(t.ok?t.tok_s+' tok/s ('+(t.speedup_pct>=0?'+':'')+t.speedup_pct+'%)':esc(t.error))+'</div>').join('');
    loadAgentCheckpoints();loadAgentMemory();
  }catch(e){toast('Speculation tuner failed: '+e.message,'err')}
  btn.disabled=false;
});

/* Phase 2 UI: HF Recommender */
$('#ag-hf-go')?.addEventListener('click',async()=>{
  const q=($('#ag-hf-q').value||'').trim();if(!q)return;
  const btn=$('#ag-hf-go');btn.disabled=true;
  try{
    const r=await post('/api/agent/hf/recommend',{query:q,goal:$('#ag-hf-goal').value});
    const recs=r.recommendations||[];
    $('#ag-hf-out').innerHTML=recs.length?recs.map((x,i)=>
      '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--bd)">'
      +'<div><div style="font-weight:600;font-size:12.5px">'+esc(x.repo)+' · <span class="tag acc">'+esc(x.quant)+'</span> <span class="tag '+(x.is_moe?'warn':'ok')+'">'+esc(x.fit_label)+'</span></div>'
      +'<div class="hint">'+x.size_gb+' GB · GPU ~'+x.gpu_est_mb+' MB'+(x.ram_est_mb?' · RAM experts ~'+x.ram_est_mb+' MB':'')+(x.mmproj?' · +mmproj':'')+(x.draft?' · +draft':'')+'</div></div>'
      +'<button class="btn xs primary" data-hf-rec-pull="'+i+'">Queue Pull</button></div>'
    ).join(''):'<span class="hint">No fitting GGUF quants found for that query.</span>';
    $('#ag-hf-out')._recs=recs;
  }catch(e){toast('HF recommend failed: '+e.message,'err')}
  btn.disabled=false;
});
$('#ag-hf-out')?.addEventListener('click',async e=>{
  const b=e.target.closest('[data-hf-rec-pull]');if(!b)return;
  const rec=($('#ag-hf-out')._recs||[])[Number(b.dataset.hfRecPull)];if(!rec)return;
  try{await post('/api/hf/pull',{repo:rec.repo,files:rec.pull_files});toast('Queued '+rec.pull_files.length+' file(s) from '+rec.repo,'ok');pollDownloads();}
  catch(err){toast('Pull failed: '+err.message,'err')}
});

$('#ag-advise')?.addEventListener('click',async()=>{
  const btn=$('#ag-advise');btn.disabled=true;
  try{
    const d=await post('/api/agent/advise',buildAgentPayload());
    renderAgentResult('Dry-Run Recommendation ('+d.engine+' · '+d.goal+')',d);
  }catch(e){toast('Advise failed: '+e.message,'err')}
  btn.disabled=false;
});
$('#ag-tune-start')?.addEventListener('click',async()=>{
  const btn=$('#ag-tune-start');btn.disabled=true;
  try{
    const d=await post('/api/agent/tune/start',buildAgentPayload());
    renderTuneStatus(d);loadAgentCheckpoints();toast('Auto-tune started (safety checkpoint created)','ok');
  }catch(e){toast('Auto-tune failed: '+e.message,'err')}
  btn.disabled=false;
});
$('#ag-tune-stop')?.addEventListener('click',async()=>{
  await post('/api/agent/tune/stop',{rollback:false});toast('Stopping auto-tune…','ok');loadAgentStatus();
});
$('#ag-tune-rollback')?.addEventListener('click',async()=>{
  await post('/api/agent/tune/stop',{rollback:true});toast('Stopped and rolled back to checkpoint','ok');loadAgentStatus();
});
$('#ag-diagnose')?.addEventListener('click',async()=>{
  const btn=$('#ag-diagnose');btn.disabled=true;
  try{
    const d=await post('/api/agent/diagnose',{engine:$('#ag-engine').value});
    renderAgentResult('Failure Diagnosis ('+d.engine+'): '+d.title,d);
  }catch(e){toast('Diagnose failed: '+e.message,'err')}
  btn.disabled=false;
});
$('#ag-cp-now')?.addEventListener('click',async()=>{
  try{
    const cp=await post('/api/agent/checkpoints',{engine:$('#ag-engine').value,label:'Manual checkpoint ('+$('#ag-engine').value+')'});
    toast('Created checkpoint '+cp.id,'ok');loadAgentCheckpoints();
  }catch(e){toast('Checkpoint failed: '+e.message,'err')}
});
$('#ag-apply-btn')?.addEventListener('click',async()=>{
  if(!AG_LAST_RESULT)return;
  const cfg=AG_LAST_RESULT.config||AG_LAST_RESULT.fix_config;if(!cfg)return;
  const btn=$('#ag-apply-btn');btn.disabled=true;
  try{
    const r=await post('/api/agent/apply',{engine:AG_LAST_RESULT.engine,config:cfg,launch:true,goal:AG_LAST_RESULT.goal||'balanced'});
    toast('Applied & launched (checkpoint '+r.checkpoint_id+')','ok');
    loadAgentCheckpoints();pollStatus(true);
  }catch(e){toast('Apply failed: '+e.message,'err')}
  btn.disabled=false;
});
$('#ag-save-preset-btn')?.addEventListener('click',async()=>{
  if(!AG_LAST_RESULT)return;
  const cfg=AG_LAST_RESULT.config||AG_LAST_RESULT.fix_config;if(!cfg)return;
  const nm=prompt('Preset name:',(AG_LAST_RESULT.engine||'llama')+' · '+(AG_LAST_RESULT.goal||'custom'));
  if(!nm)return;
  try{
    await post('/api/agent/presets',{name:nm,engine:AG_LAST_RESULT.engine,goal:AG_LAST_RESULT.goal||'balanced',config:cfg,notes:(AG_LAST_RESULT.notes||[])[0]||''});
    toast('Preset saved','ok');loadAgentPresets();
  }catch(e){toast('Save preset failed: '+e.message,'err')}
});
$('#ag-cp-refresh')?.addEventListener('click',loadAgentCheckpoints);
$('#ag-pre-refresh')?.addEventListener('click',loadAgentPresets);
$('#ag-mem-refresh')?.addEventListener('click',loadAgentMemory);
$('#ag-mem-clear')?.addEventListener('click',async()=>{
  if(!confirm('Clear persistent agent experience memory?'))return;
  await fetch('/api/agent/memory',{method:'DELETE'});toast('Agent memory cleared','ok');loadAgentMemory();
});
$('#ag-checkpoints')?.addEventListener('click',async e=>{
  const r=e.target.closest('[data-cp-restore]');
  if(r){
    try{await post('/api/agent/checkpoints/'+r.dataset.cpRestore+'/restore',{relaunch:true});toast('Checkpoint restored','ok');pollStatus(true);}
    catch(err){toast('Restore failed: '+err.message,'err')}
    return;
  }
  const d=e.target.closest('[data-cp-del]');
  if(d){await fetch('/api/agent/checkpoints/'+d.dataset.cpDel,{method:'DELETE'});loadAgentCheckpoints();}
});
$('#ag-presets')?.addEventListener('click',async e=>{
  const a=e.target.closest('[data-pre-apply]');
  if(a){
    const p=($('#ag-presets')._items||[]).find(x=>x.id===a.dataset.preApply);
    if(!p)return;
    try{const r=await post('/api/agent/apply',{engine:p.engine,config:p.config,launch:true,goal:p.goal||'balanced'});toast('Applied preset '+p.name+' (checkpoint '+r.checkpoint_id+')','ok');loadAgentCheckpoints();pollStatus(true);}
    catch(err){toast('Apply preset failed: '+err.message,'err')}
    return;
  }
  const d=e.target.closest('[data-pre-del]');
  if(d){await fetch('/api/agent/presets/'+d.dataset.preDel,{method:'DELETE'});loadAgentPresets();}
});

/* Quick triggers on llama.cpp / Strata / Avalon tabs */
document.addEventListener('click',e=>{
  const q=e.target.closest('[data-agent-quick]');if(!q)return;
  const [eng,act]=q.dataset.agentQuick.split(':');
  setTab('agent');
  if($('#ag-engine')){$('#ag-engine').value=eng;syncAgentModelOptions();}
  if(act==='diagnose')$('#ag-diagnose')?.click();
  else $('#ag-advise')?.click();
});

/* Collapsible Agent Chat Sidebar */
function toggleAgentDrawer(force){
  const dr=$('#agentdrawer');if(!dr)return;
  dr.classList.toggle('open',typeof force==='boolean'?force:!dr.classList.contains('open'));
  if(dr.classList.contains('open')){$('#ag-sb-in')?.focus();loadAgentStatus();}
}
$('#agentdrawerbtn')?.addEventListener('click',()=>toggleAgentDrawer());
$('#ag-open-sidebar')?.addEventListener('click',()=>toggleAgentDrawer(true));
$('#ag-sb-close')?.addEventListener('click',()=>toggleAgentDrawer(false));

function appendAgentBubble(role,text,actions){
  const box=$('#ag-sb-msgs');if(!box)return;
  const div=document.createElement('div');
  div.className='agent-msg '+(role==='user'?'usr':'bot');
  div.textContent=text;
  if(actions&&actions.length){
    const acts=document.createElement('div');acts.className='agent-acts';
    actions.forEach(a=>{
      const b=document.createElement('button');b.type='button';b.className='btn xs primary';b.textContent=a.label;
      b.addEventListener('click',async()=>{
        b.disabled=true;
        try{
          if(a.type==='apply'){
            const r=await post('/api/agent/apply',{engine:a.engine,config:a.config,launch:true,goal:a.goal||'balanced'});
            toast('Applied (checkpoint '+r.checkpoint_id+')','ok');pollStatus(true);
          }else if(a.type==='start_tune'){
            await post('/api/agent/tune/start',{engine:a.engine,goal:a.goal||'fastest',model:a.model||''});
            setTab('agent');loadAgentStatus();toast('Auto-tune started','ok');
          }else if(a.type==='restore_checkpoint'){
            await post('/api/agent/checkpoints/'+a.checkpoint_id+'/restore',{relaunch:true});
            toast('Restored '+a.checkpoint_id,'ok');pollStatus(true);
          }
        }catch(err){toast('Action failed: '+err.message,'err')}
        b.disabled=false;
      });
      acts.appendChild(b);
    });
    div.appendChild(acts);
  }
  box.appendChild(div);
  box.scrollTop=box.scrollHeight;
}
async function sendAgentChat(text){
  const msg=(text||$('#ag-sb-in').value||'').trim();if(!msg)return;
  $('#ag-sb-in').value='';
  appendAgentBubble('user',msg);
  try{
    const eng=(tab==='strata'||tab==='avalon'||tab==='maya')?tab:($('#ag-engine')?.value||'llama');
    const r=await post('/api/agent/chat',{message:msg,engine:eng});
    if(r.llm)renderAgentLlmChip(r.llm);
    appendAgentBubble('bot',r.reply,r.actions||[]);
  }catch(e){appendAgentBubble('bot','Error: '+e.message);}
}
$('#ag-sb-send')?.addEventListener('click',()=>sendAgentChat());
$('#ag-sb-in')?.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();sendAgentChat();}});
$$('[data-ag-chip]').forEach(b=>b.addEventListener('click',()=>sendAgentChat(b.dataset.agChip)));

/* ============================== polling & keys ============================== */
function tick(){
  if(document.hidden)return;
  pollStatus();
  pollTelemetry();
  if(tab==='strata'){loadStrata();pollStrataTelemetry();}
  if(tab==='avalon')loadAvalon();
  if(tab==='agent')loadAgentStatus();
  if(tab==='maya')loadMaya();
}
setInterval(tick,2500);
setInterval(()=>{if(!document.hidden)refreshVitals()},5000);
setInterval(()=>{if(!document.hidden&&tab==='vitals')refreshVitals()},2000);
setInterval(()=>{if(!document.hidden&&tab==='server')refreshLog()},1500);
setInterval(()=>{if(!document.hidden&&tab==='strata')refreshStrataLog()},1500);
setInterval(()=>{if(!document.hidden&&tab==='avalon')refreshAvalonLog()},1500);
setInterval(()=>{if(!document.hidden&&tab==='maya')refreshMayaLog()},1500);
async function refreshAvalonLog(){
  try{const d=await api('/api/avalon/log');AVLOGTEXT=d.log||'';paintFilteredLog($('#avalonlog'),AVLOGTEXT,'#avlogfilter','#avlogauto','#avlogn')}catch(e){}
}
document.addEventListener('visibilitychange',()=>{if(!document.hidden){pollStatus();pollTelemetry();refreshVitals()}});
document.addEventListener('keydown',e=>{
  if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){
    e.preventDefault();
    if($('#cmdpal').open)$('#cmdpal').close();else openPalette();
    return;
  }
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if($$('dialog[open]').length)return;
  const t=e.target,typing=t&&(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.tagName==='SELECT'||t.isContentEditable);
  if(e.key==='Escape'&&document.body.classList.contains('nav-open')){closeNav();return;}
  if(e.key==='/'&&!typing){
    e.preventDefault();
    if(tab==='hf')$('#hfq').focus();else if(tab==='server')$('#logfilter').focus();else if(tab==='strata')$('#stlogfilter').focus();else if(tab==='avalon')$('#avlogfilter').focus();else if(tab==='chat')$('#csearch').focus();else{setTab('models');$('#msearch').focus()}
    return;
  }
  if(!typing&&TABS[Number(e.key)-1])setTab(TABS[Number(e.key)-1]);
});

/* ============================== boot ============================== */
setTab(localStorage.getItem('ic.tab')||'models');
loadModels().then(()=>{pollStatus(true);pollTelemetry();}).catch(e=>toast('Panel load failed: '+e.message,'err'));
loadChatList();
(async()=>{const s=localStorage.getItem('ic.cid');
  if(!s||!(await openChat(s,true))){localStorage.removeItem('ic.cid');emptyThread()}})();
refreshVitals();loadAvalon();loadAvalonAccess();loadMaya();pollDownloads();loadBenchmarks();drawChatTarget();loadStrata();loadStrataPresets();loadSetup();loadAgentStatus();
$('#footnote').textContent='live telemetry active · polling pauses when tab is hidden';
</script>
</body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CFG.get("bind_host", "0.0.0.0"), port=PORT, log_level="info")
