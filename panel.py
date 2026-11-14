#!/usr/bin/env python3
"""LFM Panel — local model library, llama-server control, system vitals, chat.

Single-file FastAPI app. Dependencies: fastapi, uvicorn, httpx.
Config: ~/.config/lfm-panel/config.json (created with defaults on first run).
"""
import asyncio, json, os, re, secrets, shutil, signal, socket, struct, subprocess, threading, time, uuid
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
    "avalon_lan": False,
    "avalon_api_key": "",
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
                if not isinstance(val, list):
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
            moe = bool(expert_count) or bool(MOE_RE.search(stem))
            name_label = md.get("general.size_label", "")
            m = {
                "path": str(f), "name": stem, "arch": arch,
                "quant": quant, "size": st.st_size, "size_h": human_size(st.st_size),
                "moe": moe, "vl": "vl" in stem.lower() or "omni" in stem.lower(),
                "params": name_label, "ctx_len": md.get(f"{arch}.context_length", 0),
                "blocks": md.get(f"{arch}.block_count", 0),
            }
            _model_cache[key] = m
            models.append(m)
    models.sort(key=lambda m: m["name"].lower())
    return {"models": models, "mmproj": accessories}

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

class ServerManager:
    def __init__(self):
        self.proc = None
        self.info = {}     # alias, model, port, preset, started_at, pid, extra
        self.logbuf = deque(maxlen=500)
        self._load_state()

    def _load_state(self):
        if STATE_PATH.exists():
            try:
                st = json.loads(STATE_PATH.read_text())
                pid, port = st.get("pid"), st.get("port")
                if pid and port and self._health(port, timeout=1.0):
                    self.proc = None
                    self.info = {**st, "adopted": True}
                    self.log("adopted running server pid=%s port=%s" % (pid, port))
                elif pid and not Path(f"/proc/{pid}").exists():
                    self.log("discarding stale state: pid %s is gone" % pid)
                    STATE_PATH.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _discover_running():
        """Find every running llama-server process via /proc; parse model, port, preset."""
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
        self.logbuf.append(f"[{time.strftime('%H:%M:%S')}] {line}")

    @staticmethod
    def _health(port, timeout=1.5):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def status(self):
        i = self.info
        if i:
            pid = i.get("pid")
            alive = (self.proc and self.proc.poll() is None) or (pid and Path(f"/proc/{pid}").exists())
            if alive:
                return {
                    "running": True, "pid": pid, "alias": i.get("alias"), "model": i.get("model"),
                    "port": i.get("port"), "preset": i.get("preset"),
                    "uptime": int(time.time() - i.get("started_at", time.time())),
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
                self._save_state()
                self.log("adopted running server pid=%s port=%s model=%s (discovered)"
                         % (cand["pid"], cand["port"], cand["alias"]))
                i = self.info
                break
        if not i:
            return {"running": False}
        return {
            "running": True, "pid": i.get("pid"), "alias": i.get("alias"), "model": i.get("model"),
            "port": i.get("port"), "preset": i.get("preset"),
            "uptime": int(time.time() - i.get("started_at", time.time())),
            "ready": self._health(i["port"], timeout=1.0), "adopted": i.get("adopted", False),
        }

    def start(self, model, preset="auto", extra_args="", ctx=None):
        if self.status()["running"]:
            self.stop()
        mp = Path(model)
        if not mp.exists():
            raise HTTPException(404, f"model not found: {model}")
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
        args_map = {
            "dense-gpu": ["-ngl", "99", "-t", "8"],
            "moe-split": ["-ngl", "99", "-ot", ".ffn_.*_exps.=CPU", "-t", "16"],
            "vision": ["-ngl", "12", "-t", "8"],
        }
        if igpu:
            args_map["igpu-full"] = ["--device", dev, "-ngl", "99", "-t", "8"]
            args_map["igpu-vision"] = ["--device", dev, "-ngl", "99", "-t", "8"]
        if preset not in args_map:
            raise HTTPException(400, f"unknown preset {preset}")
        args = [CFG["server_binary"], "-m", model, "--host", "127.0.0.1", "--jinja",
                "-c", str(ctx or CFG["ctx"]), "--alias", mp.stem] + args_map[preset]
        mm = find_mmproj(model)
        if preset in ("vision", "igpu-vision"):
            if not mm:
                raise HTTPException(400, "vision preset needs an mmproj file next to the model")
        if mm:
            args += ["--mmproj", mm]
        if extra_args:
            args += extra_args.split()
        for port in CHILD_PORTS:
            if not self._port_busy(port):
                break
        else:
            raise HTTPException(500, "no free child port")
        args += ["--port", str(port)]
        self.log("starting: " + " ".join(args))
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors="replace")
        threading.Thread(target=self._pump_logs, daemon=True).start()
        self.info = {"pid": self.proc.pid, "port": port, "alias": mp.stem, "model": model,
                     "preset": preset, "started_at": time.time(), "adopted": False}
        self._save_state()
        return self.status()

    @staticmethod
    def _port_busy(port):
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _pump_logs(self):
        proc = self.proc
        for line in proc.stdout:
            self.logbuf.append(line.rstrip())

    def stop(self):
        i = self.info
        if not i:
            return {"stopped": False}
        pid = i.get("pid")
        if pid and Path(f"/proc/{pid}").exists():
            try: os.kill(pid, signal.SIGTERM)
            except ProcessLookupError: pass
            for _ in range(50):
                if not Path(f"/proc/{pid}").exists(): break
                time.sleep(0.1)
            else:
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
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
    },
    "prefast": {
        "label": "Fast ingestion",
        "desc": "125B MoE IQ3KS · 262k context · drafting off · prompt reading ≈2× faster",
        "speed": "needs the LFM server stopped (VRAM)",
        "exclusive": True,
        "draft": False,
    },
}

class AvalonManager:
    """Runs the Avalon llama-server (Qwen3.8-Flash-Next 125B) on port 8080."""

    def __init__(self):
        self.proc = None
        self.info = {}   # profile, pid, started_at, adopted
        self.logbuf = deque(maxlen=300)
        self._load_state()

    def _load_state(self):
        if not AVALON_STATE_PATH.exists():
            return
        try:
            st = json.loads(AVALON_STATE_PATH.read_text())
            pid = st.get("pid")
            if pid and Path(f"/proc/{pid}").exists():
                self.proc = None
                self.info = {**st, "adopted": True}
                self.log(f"adopted running avalon pid={pid} profile={st.get('profile')}")
        except Exception:
            pass

    def log(self, line):
        self.logbuf.append(f"[{time.strftime('%H:%M:%S')}] {line}")

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
            alive = (self.proc and self.proc.poll() is None) or (pid and Path(f"/proc/{pid}").exists())
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
            "adopted": st.get("adopted", False),
        }

    def start(self, profile: str):
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
            raise HTTPException(503, "port 8080 is still busy — stop the running Avalon server first.")
        args = [AVALON_BIN, "-m", str(AVALON_MODEL),
                "--device", "ROCm0", "-ngl", "99", "-fit", "off",
                "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0",
                "-b", "9536", "-ub", "9536", "-c", "262144",
                "--parallel", "1", "--jinja", "--temp", "0.6",
                "--port", str(AVALON_PORT)]
        if p.get("draft") and AVALON_DRAFT.exists():
            args += ["-md", str(AVALON_DRAFT), "--spec-type", "draft-mtp",
                     "--spec-draft-n-max", "2", "-ubd", "256", "-ctkd", "q8_0", "-ctvd", "q8_0"]
        if CFG.get("avalon_lan"):
            args += ["--host", "0.0.0.0"]
            if CFG.get("avalon_api_key"):
                args += ["--api-key", CFG["avalon_api_key"]]
        self.log(f"starting profile {profile}{' [lan]' if CFG.get('avalon_lan') else ''}")
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", start_new_session=True)
        threading.Thread(target=self._pump_logs, daemon=True).start()
        self.info = {"profile": profile, "pid": self.proc.pid,
                     "started_at": time.time(), "adopted": False}
        self._save_state()
        return self.status()

    def _pump_logs(self):
        proc = self.proc
        for line in proc.stdout:
            self.logbuf.append(line.rstrip())

    def stop(self):
        st = self.info
        pid = st.get("pid")
        killed = False
        if pid and Path(f"/proc/{pid}").exists():
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try: os.kill(pid, signal.SIGTERM)
                except ProcessLookupError: pass
            for _ in range(100):
                if not (Path(f"/proc/{pid}").exists() and not self._port_free()):
                    killed = True
                    break
                time.sleep(0.1)
            if not killed:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try: os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError: pass
            killed = True
        self.log(f"stopped avalon (pid {pid})" if killed else "avalon not running")
        self.proc, self.info = None, {}
        AVALON_STATE_PATH.unlink(missing_ok=True)
        if not self._port_free():          # untracked instance still holding the port
            self._kill_orphans()
        stopped = self._port_free()
        self.log("stopped avalon" if stopped else "stop failed: port 8080 still busy")
        return {"stopped": stopped}

    @staticmethod
    def _kill_orphans():
        """Kill any avalon serve/engine processes the panel didn't spawn."""
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
            except ProcessLookupError: pass
        for _ in range(50):
            if AvalonManager._port_free():
                break
            time.sleep(0.1)
        for v in victims:
            if Path(f"/proc/{v}").exists():
                try: os.kill(v, signal.SIGKILL)
                except ProcessLookupError: pass
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

def vitals():
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
    mi = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, val = line.partition(":")
        mi[k] = int(val.split()[0])
    total, avail = mi["MemTotal"], mi["MemAvailable"]
    v["mem"] = {"used_gb": round((total - avail) / 1048576, 1), "total_gb": round(total / 1048576, 1)}
    cpu_now = Path("/proc/stat").read_text().splitlines()[0].split()[1:8]
    cur = list(map(int, cpu_now))
    if _prev_cpu:
        d = [c - p for c, p in zip(cur, _prev_cpu)]
        tot = sum(d)
        v["cpu"] = {"percent": round(100 * d[0] / tot) if tot else 0, "cores": os.cpu_count()}
    _prev_cpu = cur
    for name, path in (("home", str(HOME)), ("root", "/")):
        u = shutil.disk_usage(path)
        v.setdefault("disk", {})[name] = {"used_gb": round(u.used / 2**30, 1), "total_gb": round(u.total / 2**30, 1)}
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
<meta name="color-scheme" content="dark light">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%2363b3ff'/><path d='M17.5 5l-8.5 13h5.2l-1.2 9 8.5-13h-5.2z' fill='%2304121f'/></svg>">
<style>
:root{--bg:#0a0d12;--bg-2:#0e1319;--panel:#12171f;--panel-2:#161c25;--raise:#1b222c;--bd:#232d38;--bd-2:#2e3945;
 --fg:#e7edf5;--fg-2:#bcc7d5;--mut:#8794a6;--faint:#5d6a7c;--acc:#63b3ff;--acc-2:#4dd6c1;--acc-ink:#04121f;
 --err:#ff6f6f;--ok:#4fd48a;--grad:linear-gradient(135deg,var(--acc),var(--acc-2));
 --glow:radial-gradient(1100px 520px at 82% -12%,#16222f 0%,rgba(0,0,0,0) 62%)}
[data-theme=light]{--bg:#f5f7fa;--bg-2:#e9eef4;--panel:#fff;--panel-2:#fafbfd;--raise:#eef2f7;--bd:#dce3eb;--bd-2:#c6d0db;
 --fg:#101720;--fg-2:#39465a;--mut:#5c6b7d;--faint:#8a96a5;--acc:#1f6feb;--acc-2:#0d9488;--acc-ink:#fff;
 --err:#c0392b;--ok:#0f7a55;--glow:radial-gradient(1100px 520px at 82% -12%,#dfe8f3 0%,rgba(0,0,0,0) 62%)}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:20px;background:var(--bg) var(--glow) no-repeat;
 color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px;border-radius:6px}
.card{width:min(360px,94vw);background:linear-gradient(180deg,var(--panel-2),var(--panel));border:1px solid var(--bd);
 border-radius:16px;padding:24px;box-shadow:0 1px 2px rgba(0,0,0,.4),0 24px 60px -26px rgba(0,0,0,.8)}
.brand{display:flex;align-items:center;gap:11px;margin-bottom:20px}
.mark{width:32px;height:32px;border-radius:10px;background:var(--grad);display:grid;place-items:center;color:var(--acc-ink);
 box-shadow:0 0 0 1px rgba(255,255,255,.1),0 10px 24px -12px var(--acc)}
.mark svg{width:18px;height:18px}
.brand b{font-size:16px;letter-spacing:-.2px}
.brand small{display:block;font-size:10.5px;font-weight:550;color:var(--faint);letter-spacing:.7px;text-transform:uppercase}
label{display:block;color:var(--mut);font-size:11.5px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;margin-bottom:6px}
input{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--bd-2);background:var(--bg-2);color:var(--fg);
 font:inherit;font-size:14px;transition:border-color .16s,box-shadow .16s}
input:focus{border-color:var(--acc);box-shadow:0 0 0 3px color-mix(in srgb,var(--acc) 18%,transparent);outline:0}
button{width:100%;margin-top:14px;padding:10px;border:0;border-radius:10px;background:var(--grad);color:var(--acc-ink);
 font:inherit;font-weight:650;font-size:14px;cursor:pointer;box-shadow:0 8px 22px -12px var(--acc);transition:filter .16s,opacity .16s}
button:hover{filter:brightness(1.07)}button:active{transform:translateY(1px)}button[disabled]{opacity:.55;cursor:progress}
#msg{color:var(--err);font-size:12.5px;min-height:18px;margin-top:11px}
.hint{color:var(--faint);font-size:12px;margin:14px 0 0;text-align:center}
.theme{position:fixed;top:14px;right:14px;width:32px;height:32px;display:grid;place-items:center;border-radius:9px;
 border:1px solid var(--bd);background:var(--panel);color:var(--mut);cursor:pointer;margin:0;padding:0}
.theme:hover{color:var(--fg);background:var(--raise)}
.theme svg{width:15px;height:15px}
@media (prefers-reduced-motion:reduce){*{transition-duration:.01ms !important}}
</style></head><body>
<button class="theme" id="theme" aria-label="Toggle theme"></button>
<div class="card">
 <div class="brand"><span class="mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M13.2 2 4 14h6l-1.4 8L20 9.5h-6.2z"/></svg></span>
  <span><b>LFM Panel</b><small>local model console</small></span></div>
 <form>
  <label for="pw">Sudo password</label>
  <input id="pw" type="password" placeholder="your login password" autocomplete="current-password" autofocus>
  <button type="submit" id="go">Sign in</button>
  <div id="msg" role="alert"></div>
 </form>
 <p class="hint">Validated against sudo · 5 attempts before a 5-minute lockout</p>
</div>
<script>
const SUN='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/></svg>';
const MOON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 14a8 8 0 01-9.5-9.5A8 8 0 1020 14z"/></svg>';
function applyTheme(t){document.documentElement.dataset.theme=t;$('#theme').innerHTML=t==='dark'?SUN:MOON}
function $(s){return document.querySelector(s)}
applyTheme(localStorage.getItem('lfm.theme')||(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'));
$('#theme').onclick=()=>{const t=document.documentElement.dataset.theme==='dark'?'light':'dark';
 applyTheme(t);localStorage.setItem('lfm.theme',t)};
async function go(ev){ev.preventDefault();
 const m=$('#msg'),b=$('#go'),pw=$('#pw');m.textContent='';
 b.disabled=true;b.textContent='Signing in…';
 try{const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({password:pw.value})});
  if(r.ok){location.replace('/');return}
  m.textContent=(r.status===429)?'Too many attempts — wait 5 minutes.':'Wrong password.';
  pw.select();
 }catch(e){m.textContent='Failed: '+e}
 b.disabled=false;b.textContent='Sign in'}
document.querySelector('form').addEventListener('submit',go);
</script></body></html>"""

def _sudo_ok(pw: str) -> bool:
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

def _hf_headers():
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}

@app.get("/api/hf/search")
def api_hf_search(q: str):
    res = httpx.get("https://huggingface.co/api/models", params={"search": q, "limit": 20, "sort": "downloads", "direction": -1}, headers=_hf_headers(), timeout=15).json()
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

def _download_worker(did: str):
    d = DOWNLOADS[did]
    d["status"] = "downloading"
    dest = Path(d["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", d["url"], headers=_hf_headers(), follow_redirects=True,
                          timeout=httpx.Timeout(30, read=120)) as r:
            r.raise_for_status()
            d["total"] = int(r.headers.get("content-length", 0)) or d["total"]
            with open(tmp, "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    if d["cancel"]:
                        raise RuntimeError("cancelled")
                    fh.write(chunk)
                    d["done"] = fh.tell()
        tmp.rename(dest)
        d["status"] = "done"
        MGR.log(f"pulled {dest.name}")
    except Exception as e:
        d["status"] = "cancelled" if d["cancel"] else "error"
        if not d["cancel"]:
            d["error"] = str(e)[:300]
        tmp.unlink(missing_ok=True)

class PullBody(BaseModel):
    repo: str
    files: list[str]

@app.post("/api/hf/pull")
def api_hf_pull(b: PullBody):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", b.repo)
    started = []
    for f in b.files:
        fname = f.split("/")[-1]
        if not fname.lower().endswith(".gguf"):
            continue
        dest = Path(CFG["model_dirs"][0]).expanduser() / "hf-pulls" / safe / fname
        did = f"{b.repo}:{fname}"
        DOWNLOADS[did] = {"id": did, "repo": b.repo, "file": fname, "url": f"https://huggingface.co/{b.repo}/resolve/main/{f}",
                          "dest": str(dest), "done": 0, "total": 0, "status": "queued", "cancel": False, "error": ""}
        threading.Thread(target=_download_worker, args=(did,), daemon=True).start()
        started.append(did)
    if not started:
        raise HTTPException(400, "no .gguf files in selection")
    return {"started": started}

@app.get("/api/downloads")
def api_downloads():
    return sorted(DOWNLOADS.values(), key=lambda d: d["id"])

class CancelBody(BaseModel):
    id: str

@app.post("/api/downloads/cancel")
def api_cancel(b: CancelBody):
    d = DOWNLOADS.get(b.id)
    if not d:
        raise HTTPException(404)
    d["cancel"] = True
    return {"ok": True}

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
    return scan_models()

@app.get("/api/vitals")
def api_vitals():
    return vitals()

@app.get("/api/server/status")
def api_status():
    return MGR.status()

@app.get("/api/server/log")
def api_log():
    return {"log": "\n".join(list(MGR.logbuf)[-200:])}

# ---------------------------------------------------------------- avalon api
@app.get("/api/avalon/profiles")
def api_avalon_profiles():
    out = []
    for key, p in AVALON_PROFILES.items():
        out.append({"key": key, "label": p["label"], "desc": p["desc"],
                    "speed": p["speed"], "exclusive": p["exclusive"],
                    "available": Path(AVALON_BIN).exists() and AVALON_MODEL.exists()})
    return out

@app.get("/api/avalon/status")
def api_avalon_status():
    return SMGR.status()

@app.get("/api/avalon/log")
def api_avalon_log():
    return {"log": "\n".join(list(SMGR.logbuf)[-200:])}

class AvalonStartBody(BaseModel):
    profile: str

@app.post("/api/avalon/start")
def api_avalon_start(b: AvalonStartBody):
    return SMGR.start(b.profile)

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
            if ip.startswith("192.168.") or ip.startswith(("10.", "172.")):
                best = best or (ip if ip.startswith("192.168.") else best) or ip
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(old + block)
    import tomllib
    tomllib.loads(path.read_text())  # validate
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(old + block)
    import tomllib
    tomllib.loads(path.read_text())  # validate
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

@app.post("/api/server/start")
def api_start(b: StartBody):
    return MGR.start(b.model, b.preset, b.extra_args, b.ctx)

@app.post("/api/server/stop")
def api_stop():
    return MGR.stop()

@app.post("/api/chat")
async def api_chat(req: Request):
    tgt = _chat_target()
    if not tgt:
        raise HTTPException(409, "no model ready — load one in the Models tab or start Avalon")
    port, key = tgt
    body = await req.json()
    body.setdefault("stream", True)
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    client = httpx.AsyncClient(timeout=None)
    async def gen():
        try:
            async with client.stream("POST", url, json=body, headers=headers) as r:
                async for chunk in r.aiter_bytes():
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
<title>LFM Panel</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%2363b3ff'/><path d='M17.5 5l-8.5 13h5.2l-1.2 9 8.5-13h-5.2z' fill='%2304121f'/></svg>">
<style>
:root{
  --bg:#0a0d12;--bg-2:#0e1319;--panel:#12171f;--panel-2:#161c25;--raise:#1b222c;
  --bd:#232d38;--bd-2:#2e3945;
  --fg:#e7edf5;--fg-2:#bcc7d5;--mut:#8794a6;--faint:#5d6a7c;
  --acc:#63b3ff;--acc-2:#4dd6c1;--acc-ink:#04121f;
  --ok:#4fd48a;--warn:#f0b552;--err:#ff6f6f;
  --glow:radial-gradient(1100px 520px at 82% -12%,#16222f 0%,rgba(0,0,0,0) 62%);
  --grad:linear-gradient(135deg,var(--acc),var(--acc-2));
  --sh:0 1px 2px rgba(0,0,0,.45),0 10px 26px -14px rgba(0,0,0,.7);
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --r:14px;
}
[data-theme=light]{
  --bg:#f5f7fa;--bg-2:#e9eef4;--panel:#ffffff;--panel-2:#fafbfd;--raise:#eef2f7;
  --bd:#dce3eb;--bd-2:#c6d0db;
  --fg:#101720;--fg-2:#39465a;--mut:#5c6b7d;--faint:#8a96a5;
  --acc:#1f6feb;--acc-2:#0d9488;--acc-ink:#ffffff;
  --ok:#0f7a55;--warn:#8f6410;--err:#c0392b;
  --glow:radial-gradient(1100px 520px at 82% -12%,#dfe8f3 0%,rgba(0,0,0,0) 62%);
  --sh:0 1px 2px rgba(16,24,40,.07),0 10px 26px -16px rgba(16,24,40,.22);
}
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;min-height:100vh;background:var(--bg) var(--glow) no-repeat;color:var(--fg);
  font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
h1,h2,h3{margin:0;letter-spacing:-.1px}
a{color:var(--acc)}
code,pre,.mono{font-family:var(--mono)}
.skip{position:absolute;left:-1px;top:-40px;background:var(--acc);color:var(--acc-ink);padding:8px 12px;border-radius:8px;z-index:99}
.skip:focus{top:8px;left:8px}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px;border-radius:6px}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:var(--bd-2);border-radius:99px;border:2px solid transparent;background-clip:content-box}
::-webkit-scrollbar-track{background:transparent}

/* ---------- header ---------- */
header{position:sticky;top:0;z-index:40;display:flex;align-items:center;gap:14px;padding:9px 16px;
  background:rgba(10,13,18,.86);background:color-mix(in srgb,var(--bg) 84%,transparent);
  backdrop-filter:blur(12px) saturate(150%);-webkit-backdrop-filter:blur(12px) saturate(150%);
  border-bottom:1px solid var(--bd)}
[data-theme=light] header{background:rgba(255,255,255,.86)}
.brand{display:flex;align-items:center;gap:10px;flex-shrink:0}
.mark{width:28px;height:28px;border-radius:9px;background:var(--grad);display:grid;place-items:center;
  color:var(--acc-ink);box-shadow:0 0 0 1px rgba(255,255,255,.10),0 8px 20px -10px var(--acc)}
.mark svg{width:16px;height:16px}
.brand .nm{font-size:15px;font-weight:650;line-height:1.15}
.brand .nm small{display:block;font-size:10.5px;font-weight:550;color:var(--faint);letter-spacing:.7px;text-transform:uppercase}
.nav{display:flex;gap:2px;padding:3px;background:var(--panel);border:1px solid var(--bd);border-radius:12px;
  overflow-x:auto;scrollbar-width:none;min-width:0}
.nav::-webkit-scrollbar{display:none}
.nav button{display:flex;align-items:center;gap:7px;padding:7px 11px;border:0;border-radius:9px;background:transparent;
  color:var(--mut);font:inherit;font-size:13px;font-weight:550;cursor:pointer;white-space:nowrap;transition:background .16s,color .16s}
.nav button svg{width:15px;height:15px;opacity:.8}
.nav button:hover{color:var(--fg);background:var(--raise)}
.nav button.on{color:var(--fg);background:var(--raise);box-shadow:inset 0 0 0 1px var(--bd-2),0 2px 10px -6px rgba(0,0,0,.6)}
.nav button.on svg{color:var(--acc);opacity:1}
.hgap{flex:1;min-width:8px}
.pill{display:inline-flex;align-items:center;gap:7px;padding:5px 11px;border-radius:999px;border:1px solid var(--bd-2);
  background:var(--panel);color:var(--fg-2);font-size:12.5px;font-weight:550;white-space:nowrap;max-width:44vw;overflow:hidden}
.pill.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 35%,transparent);background:color-mix(in srgb,var(--ok) 10%,transparent)}
.pill.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 35%,transparent);background:color-mix(in srgb,var(--warn) 10%,transparent)}
.pill .dot{width:7px;height:7px;border-radius:99px;background:currentColor;flex-shrink:0}
.pill.ok .dot{animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 currentColor}50%{opacity:.65;box-shadow:0 0 0 3px transparent}}
.icon{display:grid;place-items:center;width:32px;height:32px;border-radius:9px;border:1px solid var(--bd);
  background:var(--panel);color:var(--mut);cursor:pointer;flex-shrink:0;transition:.16s}
.icon:hover{color:var(--fg);border-color:var(--bd-2);background:var(--raise)}
.icon svg{width:15px;height:15px}

/* ---------- layout ---------- */
main{padding:18px 16px 44px;max-width:1240px;margin:0 auto}
section{animation:fade .22s ease-out}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.hide{display:none}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(238px,1fr));gap:14px}
.grid.wide{grid-template-columns:1fr}
.card{background:linear-gradient(180deg,var(--panel-2),var(--panel));border:1px solid var(--bd);border-radius:var(--r);
  padding:14px 16px;margin-bottom:14px;box-shadow:var(--sh)}
.card.flat{background:var(--panel)}
.hd{display:flex;align-items:center;gap:10px;justify-content:space-between;flex-wrap:wrap;margin:-2px 0 12px}
.hd h2{font-size:14px;font-weight:650}
.hd .meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.stack{display:flex;flex-direction:column;gap:14px}
.two{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
.two>.side{width:250px;flex:0 0 250px}
.two>.main{flex:1;min-width:280px}
.hint{color:var(--faint);font-size:12px}
.sep{height:1px;background:var(--bd);margin:12px 0}

/* ---------- controls ---------- */
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 13px;border-radius:9px;border:1px solid var(--bd-2);
  background:var(--raise);color:var(--fg-2);font:inherit;font-size:13px;font-weight:600;cursor:pointer;
  transition:transform .12s,background .16s,border-color .16s,opacity .16s}
.btn:hover{background:var(--panel-2);border-color:var(--mut);color:var(--fg)}
.btn:active{transform:translateY(1px)}
.btn.primary{background:var(--grad);border:0;color:var(--acc-ink);box-shadow:0 6px 18px -10px var(--acc)}
.btn.primary:hover{filter:brightness(1.07)}
.btn.danger{background:color-mix(in srgb,var(--err) 14%,transparent);border-color:color-mix(in srgb,var(--err) 40%,transparent);color:var(--err)}
.btn.danger:hover{background:color-mix(in srgb,var(--err) 24%,transparent);color:var(--err)}
.btn.sm{padding:4px 9px;font-size:12px;border-radius:8px}
.btn.xs{padding:3px 7px;font-size:11.5px}
.btn[disabled]{opacity:.45;cursor:not-allowed;transform:none}
select,input[type=text],input[type=password],input[type=search],textarea{background:var(--bg-2);border:1px solid var(--bd-2);
  color:var(--fg);border-radius:9px;padding:7px 10px;font:inherit;font-size:13px;transition:border-color .16s,box-shadow .16s}
select:focus,input:focus,textarea:focus{border-color:var(--acc);box-shadow:0 0 0 3px color-mix(in srgb,var(--acc) 18%,transparent);outline:0}
input[type=checkbox],input[type=radio]{accent-color:var(--acc);width:15px;height:15px}
label.k{display:block;color:var(--mut);font-size:11.5px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;margin-bottom:4px}
.toolbar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:12px}
.search{position:relative;flex:1;min-width:190px}
.search svg{position:absolute;left:9px;top:50%;transform:translateY(-50%);width:14px;height:14px;color:var(--faint)}
.search input{width:100%;padding-left:30px}
.segs{display:flex;gap:2px;padding:3px;background:var(--bg-2);border:1px solid var(--bd);border-radius:10px}
.segs button{border:0;background:transparent;color:var(--mut);font:inherit;font-size:12.5px;font-weight:600;
  padding:5px 10px;border-radius:7px;cursor:pointer;transition:.16s}
.segs button:hover{color:var(--fg)}
.segs button.on{background:var(--raise);color:var(--fg);box-shadow:inset 0 0 0 1px var(--bd-2)}
.switch{display:inline-flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.switch input{position:absolute;opacity:0;pointer-events:none}
.switch .track{width:38px;height:22px;border-radius:99px;background:var(--bg-2);border:1px solid var(--bd-2);position:relative;transition:.18s;flex-shrink:0}
.switch .track::after{content:"";position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:99px;background:var(--mut);transition:.18s}
.switch input:checked+.track{background:color-mix(in srgb,var(--acc) 28%,transparent);border-color:var(--acc)}
.switch input:checked+.track::after{transform:translateX(16px);background:var(--acc)}
.switch input:focus-visible+.track{outline:2px solid var(--acc);outline-offset:2px}

/* ---------- tags / stats / meters ---------- */
.tag{display:inline-flex;align-items:center;gap:5px;padding:2px 8px;border-radius:999px;border:1px solid var(--bd-2);
  background:rgba(255,255,255,.04);color:var(--fg-2);font-size:11.5px;font-weight:600;letter-spacing:.2px;white-space:nowrap}
[data-theme=light] .tag{background:rgba(16,24,40,.04)}
.tag.acc{color:var(--acc);border-color:color-mix(in srgb,var(--acc) 34%,transparent);background:color-mix(in srgb,var(--acc) 10%,transparent)}
.tag.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 34%,transparent);background:color-mix(in srgb,var(--ok) 10%,transparent)}
.tag.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 34%,transparent);background:color-mix(in srgb,var(--warn) 10%,transparent)}
.tag.err{color:var(--err);border-color:color-mix(in srgb,var(--err) 34%,transparent);background:color-mix(in srgb,var(--err) 10%,transparent)}
.tag.mut{color:var(--mut)}
.kvs{display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.stat{font-size:27px;font-weight:700;letter-spacing:-.6px;font-variant-numeric:tabular-nums;line-height:1.1}
.stat small{font-size:12.5px;font-weight:550;color:var(--mut);letter-spacing:0}
.meter{height:7px;border-radius:99px;background:var(--bg-2);border:1px solid var(--bd);overflow:hidden;margin:8px 0 2px}
.meter i{display:block;height:100%;border-radius:99px;background:var(--grad);transition:width .55s cubic-bezier(.4,0,.2,1)}
.meter.warn i{background:linear-gradient(90deg,var(--warn),#ff9a5a)}
.meter.err i{background:linear-gradient(90deg,var(--err),#ff4d6d)}
.spark{width:100%;height:26px;display:block;margin-top:6px;opacity:.85}
.mfoot{display:flex;justify-content:space-between;gap:8px;color:var(--faint);font-size:11.5px;font-variant-numeric:tabular-nums}

/* ---------- model list ---------- */
.mlist{display:flex;flex-direction:column;gap:8px}
.mrow{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:center;padding:10px 12px;
  border:1px solid var(--bd);border-radius:12px;background:var(--panel);transition:border-color .16s,background .16s}
.mrow:hover{border-color:var(--bd-2);background:var(--panel-2)}
.mrow.run{border-color:color-mix(in srgb,var(--acc) 45%,transparent);
  box-shadow:0 0 0 1px color-mix(in srgb,var(--acc) 22%,transparent),0 0 26px -14px var(--acc)}
.mrow .nm{font-weight:620;font-size:13.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mrow .pth{color:var(--faint);font-size:11.5px;font-family:var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:2px}
.mrow .tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}
.mrow .acts{display:flex;gap:7px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.empty{text-align:center;color:var(--faint);padding:26px 10px;border:1px dashed var(--bd-2);border-radius:12px}
.empty b{display:block;color:var(--fg-2);font-size:14px;margin-bottom:4px}

/* ---------- logs ---------- */
.logbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.log{font-family:var(--mono);font-size:12px;line-height:1.6;color:var(--fg-2);background:var(--bg-2);
  border:1px solid var(--bd);border-radius:12px;padding:10px 12px;max-height:420px;overflow:auto;
  white-space:pre-wrap;word-break:break-word;margin:0}
.log.wrapoff{white-space:pre}
.log .e{color:var(--err)}
.log .w{color:var(--warn)}
.log .g{color:var(--faint)}

/* ---------- chat ---------- */
.thread{display:flex;flex-direction:column;gap:12px;padding:2px;min-height:38vh;max-height:58vh;overflow-y:auto}
.msg{max-width:min(88%,780px);padding:10px 13px;border-radius:14px;border:1px solid var(--bd);
  background:var(--panel-2);line-height:1.62;overflow-wrap:break-word;position:relative}
.msg.u{align-self:flex-end;background:linear-gradient(180deg,color-mix(in srgb,var(--acc) 20%,var(--panel)),var(--panel));
  border-color:color-mix(in srgb,var(--acc) 34%,transparent)}
.msg.a{align-self:flex-start}
.msg.err{border-color:color-mix(in srgb,var(--err) 45%,transparent);color:var(--err)}
.msg .who{font-size:10.5px;font-weight:700;letter-spacing:.7px;text-transform:uppercase;color:var(--faint);margin-bottom:5px}
.msg pre{background:var(--bg-2);border:1px solid var(--bd);border-radius:10px;padding:10px 12px;overflow:auto;
  font-size:12.5px;line-height:1.55;margin:8px 0}
.msg code{font-size:12.5px;background:color-mix(in srgb,var(--bg-2) 70%,transparent);padding:1px 5px;border-radius:5px;border:1px solid var(--bd)}
.msg pre code{background:none;border:0;padding:0}
.msg .md-h{font-weight:700;margin:8px 0 3px}
.msg ul,.msg ol{margin:6px 0;padding-left:20px}
.msg details{margin-bottom:7px;border-left:2px solid var(--bd-2);padding-left:9px;color:var(--mut);font-size:12.5px}
.msg summary{cursor:pointer;font-weight:650;color:var(--faint);font-size:11.5px;letter-spacing:.4px;text-transform:uppercase}
.msg details[open] summary{margin-bottom:5px}
.msg .think{white-space:pre-wrap}
.msg .bar{display:flex;gap:6px;justify-content:flex-end;margin-top:7px;opacity:0;transition:opacity .16s}
.msg:hover .bar,.msg:focus-within .bar{opacity:1}
.typing i{display:inline-block;width:5px;height:5px;border-radius:99px;background:var(--mut);margin-right:4px;animation:bob 1.1s infinite}
.typing i:nth-child(2){animation-delay:.15s}.typing i:nth-child(3){animation-delay:.3s}
@keyframes bob{0%,60%,100%{opacity:.3;transform:translateY(0)}30%{opacity:1;transform:translateY(-3px)}}
.composer{display:flex;gap:9px;align-items:flex-end;border:1px solid var(--bd-2);border-radius:12px;padding:7px 7px 7px 11px;
  background:var(--bg-2);margin-top:10px;transition:border-color .16s,box-shadow .16s}
.composer:focus-within{border-color:var(--acc);box-shadow:0 0 0 3px color-mix(in srgb,var(--acc) 16%,transparent)}
.composer textarea{flex:1;border:0;background:transparent;color:var(--fg);resize:none;min-height:26px;max-height:220px;
  padding:5px 0;font-size:14px;line-height:1.5}
.composer textarea:focus{outline:0;box-shadow:none}
.clist{display:flex;flex-direction:column;gap:4px;max-height:52vh;overflow-y:auto}
.citem{display:flex;align-items:center;gap:8px;padding:8px 9px;border-radius:10px;border:1px solid transparent;
  cursor:pointer;transition:.14s}
.citem:hover{background:var(--raise);border-color:var(--bd)}
.citem.on{background:var(--raise);border-color:var(--bd-2)}
.citem .t{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
.citem .s{font-size:11px;color:var(--faint);white-space:nowrap}

/* ---------- hf / downloads ---------- */
.frow{display:flex;align-items:center;gap:10px;padding:7px 9px;border-radius:9px;cursor:pointer;border:1px solid transparent}
.frow:hover{background:var(--raise);border-color:var(--bd)}
.frow .p{flex:1;min-width:0;font-family:var(--mono);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dl{padding:9px 0;border-bottom:1px solid var(--bd)}
.dl:last-child{border-bottom:0}

/* ---------- toast / dialog ---------- */
.toasts{position:fixed;right:16px;bottom:16px;display:flex;flex-direction:column;gap:8px;z-index:80;max-width:min(440px,92vw)}
.toast{display:flex;gap:9px;align-items:flex-start;padding:10px 12px;border-radius:12px;border:1px solid var(--bd-2);
  background:var(--panel-2);box-shadow:0 18px 40px -18px rgba(0,0,0,.8);cursor:pointer;font-size:13px;
  animation:tin .2s cubic-bezier(.2,.8,.2,1)}
.toast.out{animation:tout .18s forwards}
.toast .ti{width:8px;height:8px;border-radius:99px;background:var(--mut);margin-top:6px;flex-shrink:0}
.toast.ok .ti{background:var(--ok)}.toast.err{border-color:color-mix(in srgb,var(--err) 40%,transparent)}
.toast.err .ti{background:var(--err)}.toast.warn .ti{background:var(--warn)}
@keyframes tin{from{opacity:0;transform:translateY(10px) scale(.98)}to{opacity:1;transform:none}}
@keyframes tout{to{opacity:0;transform:translateY(6px)}}
.dlg{position:fixed;inset:0;z-index:90;display:grid;place-items:center;padding:16px;
  background:rgba(4,7,11,.66);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);opacity:0;transition:opacity .16s}
.dlg.in{opacity:1}
.dlg.out{opacity:0}
.dlg .box{background:linear-gradient(180deg,var(--panel-2),var(--panel));border:1px solid var(--bd-2);border-radius:16px;
  box-shadow:0 40px 90px -28px rgba(0,0,0,.9);padding:16px;max-height:88vh;display:flex;flex-direction:column;
  transform:translateY(6px) scale(.99);transition:transform .18s}
.dlg.in .box{transform:none}
.dlg-hd{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
.dlg-hd b{font-size:15px}
.dlg-bd{overflow:auto;min-height:0}
.dlg-ft{display:flex;gap:8px;justify-content:flex-end;margin-top:14px;flex-wrap:wrap}
.dlg-p{margin:0;color:var(--fg-2);line-height:1.6}
.dlg-p code{font-size:12px;word-break:break-all}

/* ---------- footer ---------- */
footer{max-width:1240px;margin:0 auto;padding:0 16px 26px;color:var(--faint);font-size:11.5px;
  display:flex;gap:14px;flex-wrap:wrap;align-items:center}
kbd{font-family:var(--mono);font-size:11px;border:1px solid var(--bd-2);border-bottom-width:2px;border-radius:5px;
  padding:1px 5px;background:var(--bg-2);color:var(--fg-2)}

@media (max-width:880px){
  header{flex-wrap:wrap;gap:9px}
  .brand .nm small{display:none}
  .nav{order:3;width:100%}
  .two>.side{width:100%;flex:1 1 100%}
  .mrow{grid-template-columns:1fr}
  .mrow .acts{justify-content:flex-start}
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
    <span class="nm">LFM Panel<small>local model console</small></span>
  </div>
  <nav class="nav" id="nav" aria-label="Sections">
    <button data-t="models" class="on"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z"/><path d="M4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>Models</button>
    <button data-t="server"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><path d="M7 7.5h.01M7 16.5h.01"/></svg>Server</button>
    <button data-t="avalon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/></svg>Avalon</button>
    <button data-t="vitals"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 12h4l2.5-6 4 12L16 12h5"/></svg>Vitals</button>
    <button data-t="chat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 12a8 8 0 01-8 8H7l-4 3 1.2-4.2A8 8 0 1121 12z"/></svg>Chat</button>
    <button data-t="hf"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v12m0 0l-4-4m4 4l4-4"/><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"/></svg>HF pull</button>
  </nav>
  <span class="hgap"></span>
  <span class="pill" id="hdrstatus" title="Active inference server"><span class="dot"></span><span id="hdrtext">checking…</span></span>
  <button class="icon" id="theme" title="Toggle light / dark theme" aria-label="Toggle theme"></button>
  <button class="icon" id="logout" title="Sign out" aria-label="Sign out"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M15 12H4m0 0l3.5-3.5M4 12l3.5 3.5"/><path d="M9 4h7a2 2 0 012 2v12a2 2 0 01-2 2H9"/></svg></button>
</header>

<main id="main">
<section id="t-models">
  <div class="card">
    <div class="hd"><h2>Model library</h2>
      <div class="meta"><span class="tag mut" id="libinfo">scanning…</span>
        <button class="btn sm" id="rescan" title="Re-scan model directories">Rescan</button></div></div>
    <div class="toolbar">
      <div class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="msearch" type="search" placeholder="Filter by name, quant, arch…  press /" aria-label="Filter models"></div>
      <div class="segs" id="mfilter" role="group" aria-label="Filter">
        <button data-f="all" class="on">All</button><button data-f="moe">MoE</button>
        <button data-f="vl">VL</button><button data-f="run">Loaded</button></div>
      <select id="msort" aria-label="Sort models">
        <option value="name">Name A–Z</option><option value="size">Largest first</option>
        <option value="small">Smallest first</option><option value="arch">Architecture</option></select>
    </div>
    <div id="modellist" class="mlist"></div>
    <p class="hint" style="margin:12px 0 0">Loading a model stops the one currently running. Presets: <b>auto</b> picks from model metadata.</p>
  </div>
</section>

<section id="t-server" class="hide">
  <div class="card">
    <div class="hd"><h2>llama-server</h2><span class="tag" id="srvchip">checking…</span></div>
    <div id="srvdetail" class="kvs"></div>
    <div class="row" style="display:flex;gap:9px;flex-wrap:wrap;margin-top:12px;align-items:center">
      <button class="btn danger" id="srvstop">Stop server</button>
      <button class="btn" id="agentsbtn">Register in coding agents…</button>
      <span class="hint">child binds 127.0.0.1 — reachable on the LAN only through this panel</span></div>
  </div>
  <div class="card">
    <div class="hd"><h2>Server log</h2><span class="tag mut" id="logn"></span></div>
    <div class="logbar">
      <div class="search" style="flex:1"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="logfilter" type="search" placeholder="Filter log lines…" aria-label="Filter log"></div>
      <label class="switch"><input type="checkbox" id="logauto" checked><span class="track"></span><span class="hint">Follow</span></label>
      <label class="switch"><input type="checkbox" id="logwrap" checked><span class="track"></span><span class="hint">Wrap</span></label>
      <button class="btn sm" id="logcopy">Copy</button><button class="btn sm" id="logsave">Save</button></div>
    <pre class="log" id="log"></pre>
  </div>
</section>

<section id="t-avalon" class="hide">
  <div class="card">
    <div class="hd"><h2>Avalon — Qwen3.8-Flash-Next (125B MoE)</h2><span class="tag" id="avalonchip">checking…</span></div>
    <div id="avalondetail" class="kvs" style="margin-bottom:10px"></div>
    <div id="avalonprofiles"></div>
    <div style="display:flex;gap:9px;flex-wrap:wrap;margin-top:12px;align-items:center">
      <button class="btn primary" id="avalonstart">Start selected profile</button>
      <button class="btn danger" id="avalonstop">Stop</button>
      <span class="hint">OpenAI-compatible · streams <code>reasoning_content</code></span></div>
  </div>
  <div class="card">
    <div class="hd"><h2>Network access</h2>
      <label class="switch"><input type="checkbox" id="avalonlan"><span class="track"></span><span class="hint">Expose to LAN</span></label></div>
    <div id="avalonaccess" class="kvs"></div>
  </div>
  <div class="card"><div class="hd"><h2>Avalon log</h2></div><pre class="log" id="avalonlog" style="max-height:280px"></pre></div>
</section>

<section id="t-vitals" class="hide">
  <div class="hd" style="margin-bottom:10px"><h2>System vitals</h2>
    <div class="meta"><span class="tag mut" id="vupdatetime">—</span><button class="btn sm" id="vrefresh">Refresh</button></div></div>
  <div id="vitals" class="grid"></div>
</section>

<section id="t-chat" class="hide">
  <div class="two">
    <div class="card side">
      <div class="hd"><h2>Chats</h2><button class="btn sm primary" id="newchat">+ New</button></div>
      <div id="chatlist" class="clist"></div>
    </div>
    <div class="card main">
      <div class="hd"><h2>Conversation</h2><div class="meta"><span class="tag" id="chatmodel">no model</span>
        <button class="btn xs" id="sysbtn" title="Edit the system prompt">System prompt</button></div></div>
      <div id="sysbox" class="hide" style="margin-bottom:10px">
        <textarea id="sysprompt" rows="3" style="width:100%"></textarea>
        <p class="hint" style="margin:5px 0 0">Saved in this browser. Applies to every new message.</p></div>
      <div id="chat" class="thread"></div>
      <div class="composer">
        <textarea id="prompt" rows="1" placeholder="Message… Enter to send, Shift+Enter for a new line" aria-label="Message"></textarea>
        <button class="btn" id="stopbtn" style="display:none">Stop</button>
        <button class="btn primary" id="sendbtn">Send</button></div>
      <p class="hint" style="margin:8px 2px 0" id="chatnote"></p>
    </div>
  </div>
</section>

<section id="t-hf" class="hide">
  <div class="card">
    <div class="hd"><h2>Search Hugging Face</h2><span class="tag mut" id="hfinfo"></span></div>
    <div style="display:flex;gap:9px;flex-wrap:wrap">
      <div class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input id="hfq" type="search" placeholder="search gguf repos… press Enter" aria-label="Search Hugging Face"></div>
      <button class="btn primary" id="hfsearch">Search</button></div>
    <div id="hfresults" style="margin-top:10px"></div>
  </div>
  <div class="card hide" id="hffilesbox">
    <div class="hd"><h2 id="hfrepo"></h2><div class="meta"><span class="tag mut" id="hftotal"></span>
      <button class="btn sm" id="hfnone">Clear</button></div></div>
    <div id="hffiles"></div>
    <button class="btn primary" id="hfpull" style="margin-top:10px">Pull selected</button>
  </div>
  <div class="card"><div class="hd"><h2>Downloads</h2><span class="tag mut" id="dlinfo"></span></div>
    <div id="downloads"><span class="hint">none</span></div></div>
</section>
</main>

<footer>
  <span><kbd>1</kbd>–<kbd>6</kbd> switch tabs</span><span><kbd>/</kbd> filter</span>
  <span><kbd>Esc</kbd> close</span><span><kbd>Enter</kbd> send · <kbd>Shift+Enter</kbd> newline</span>
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

/* ============================== toasts ============================== */
function toast(msg,kind,ms){
  kind=kind||'info';
  const t=document.createElement('div');t.className='toast '+kind;
  t.innerHTML='<span class="ti"></span><div>'+esc(msg)+'</div>';
  $('#toasts').appendChild(t);
  const kill=()=>{t.classList.add('out');setTimeout(()=>t.remove(),200)};
  t.addEventListener('click',kill);
  setTimeout(kill,ms||(kind==='err'?9000:4200));
}

/* ============================== dialogs ============================== */
let openDlg=null;
function dialog(o){
  if(openDlg)openDlg.close(null);
  const el=document.createElement('div');el.className='dlg';
  el.innerHTML='<div class="box" role="dialog" aria-modal="true" style="width:min('+(o.width||'520px')+',94vw)">'
    +'<div class="dlg-hd"><b>'+esc(o.title||'')+'</b><button class="icon" data-x aria-label="Close">✕</button></div>'
    +'<div class="dlg-bd">'+(o.html||'')+'</div><div class="dlg-ft"></div></div>';
  const inst={el:el,box:el.querySelector('.box'),body:el.querySelector('.dlg-bd'),foot:el.querySelector('.dlg-ft'),
    close(v){document.removeEventListener('keydown',onk);el.classList.add('out');
      setTimeout(()=>el.remove(),180);if(openDlg===inst)openDlg=null;if(inst.onclose)inst.onclose(v)}};
  function onk(e){if(e.key==='Escape'){e.preventDefault();inst.close(null)}}
  el.querySelector('[data-x]').addEventListener('click',()=>inst.close(null));
  el.addEventListener('mousedown',e=>{if(e.target===el)inst.close(null)});
  document.addEventListener('keydown',onk);
  document.body.appendChild(el);
  requestAnimationFrame(()=>el.classList.add('in'));
  openDlg=inst;
  return inst;
}
function mkBtn(label,kind){const b=document.createElement('button');b.className='btn '+(kind||'');b.textContent=label;return b}
function confirmDlg(title,msg,okLabel,danger){
  return new Promise(res=>{
    const d=dialog({title:title,html:'<p class="dlg-p">'+msg+'</p>',width:'500px'});
    const no=mkBtn('Cancel'),yes=mkBtn(okLabel||'Confirm',danger?'danger':'primary');
    d.foot.append(no,yes);
    no.addEventListener('click',()=>{d.close();res(false)});
    yes.addEventListener('click',()=>{d.close();res(true)});
    d.onclose=v=>res(v===true);
    yes.focus();
  });
}

/* ============================== theme ============================== */
const SUN='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/></svg>';
const MOON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 14a8 8 0 01-9.5-9.5A8 8 0 1020 14z"/></svg>';
function applyTheme(t){document.documentElement.dataset.theme=t;$('#theme').innerHTML=t==='dark'?SUN:MOON;
  $('#theme').title=t==='dark'?'Switch to light theme':'Switch to dark theme'}
applyTheme(localStorage.getItem('lfm.theme')||(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'));
$('#theme').addEventListener('click',()=>{
  const t=document.documentElement.dataset.theme==='dark'?'light':'dark';
  applyTheme(t);localStorage.setItem('lfm.theme',t)});
$('#logout').addEventListener('click',async()=>{
  if(!await confirmDlg('Sign out','End this panel session?','Sign out',true))return;
  try{await post('/api/logout')}catch(e){}
  location.href='/login'});

/* ============================== tabs ============================== */
const TABS=['models','server','avalon','vitals','chat','hf'];
let tab='models';
function setTab(t){
  if(TABS.indexOf(t)<0)t='models';
  tab=t;localStorage.setItem('lfm.tab',t);
  $$('.nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
  $$('main section').forEach(s=>s.classList.toggle('hide',s.id!=='t-'+t));
  if(t==='vitals')refreshVitals();
  if(t==='server')refreshLog();
  if(t==='avalon'){loadAvalon();loadAvalonAccess()}
}
$$('.nav button').forEach(b=>b.addEventListener('click',()=>setTab(b.dataset.t)));

/* ============================== models ============================== */
let MODELS=[],MMP=[],ACTIVE=null,AV=null,FILTER='all',SORT='name';
async function loadModels(){
  const d=await api('/api/models');MODELS=d.models||[];MMP=d.mmproj||[];renderModels();
}
function modelRow(m,i){
  const running=ACTIVE&&ACTIVE.model===m.path;
  const tags=['<span class="tag">'+esc(m.arch)+'</span>','<span class="tag acc">'+esc(m.quant)+'</span>'];
  if(m.params)tags.push('<span class="tag">'+esc(m.params)+'</span>');
  if(m.ctx_len)tags.push('<span class="tag mut">'+n0(m.ctx_len)+' ctx</span>');
  tags.push('<span class="tag mut">'+esc(m.size_h)+'</span>');
  if(m.moe)tags.push('<span class="tag warn">MoE</span>');
  if(m.vl)tags.push('<span class="tag acc">VL</span>');
  if(running)tags.push('<span class="tag ok">loaded</span>');
  return '<div class="mrow'+(running?' run':'')+'" data-i="'+i+'" data-path="'+esc(m.path)+'" data-name="'+esc(m.name)+'" data-size="'+esc(m.size_h)+'">'
    +'<div style="min-width:0"><div class="nm" title="'+esc(m.path)+'">'+esc(m.name)+'</div>'
    +'<div class="pth">'+esc(m.path.replace(/^\/[^/]*\//,'/…/'))+'</div>'
    +'<div class="tags">'+tags.join('')+'</div></div>'
    +'<div class="acts"><select data-preset aria-label="Preset for '+esc(m.name)+'">'
    +['auto|auto preset','dense-gpu|dense · full offload','moe-split|MoE · split experts','vision|vision + mmproj',
      'igpu-full|iGPU full (780M)','igpu-vision|iGPU vision (780M)']
      .map(o=>{const p=o.split('|');return '<option value="'+p[0]+'">'+p[1]+'</option>'}).join('')
    +'</select><button class="btn primary sm" data-act="start">'+(running?'Reload':'Load')+'</button>'
    +'<button class="btn sm" data-act="copy" title="Copy path">Path</button>'
    +'<button class="btn sm danger" data-act="del" title="Delete model">Delete</button></div></div>';
}
function renderModels(){
  const q=$('#msearch').value.trim().toLowerCase();
  let list=MODELS.filter(m=>{
    if(FILTER==='moe'&&!m.moe)return false;
    if(FILTER==='vl'&&!m.vl)return false;
    if(FILTER==='run'&&!(ACTIVE&&ACTIVE.model===m.path))return false;
    if(q&&!(m.name+' '+m.arch+' '+m.quant+' '+(m.params||'')).toLowerCase().includes(q))return false;
    return true;
  });
  const by={name:(a,b)=>a.name.localeCompare(b.name),size:(a,b)=>b.size-a.size,
    small:(a,b)=>a.size-b.size,arch:(a,b)=>a.arch.localeCompare(b.arch)||a.name.localeCompare(b.name)}[SORT];
  list=list.slice().sort(by);
  const bytes=MODELS.reduce((s,m)=>s+m.size,0);
  $('#libinfo').textContent=MODELS.length+' models · '+gib(bytes)+(MMP.length?' · '+MMP.length+' mmproj':'');
  if(!list.length){
    $('#modellist').innerHTML='<div class="empty"><b>'+(MODELS.length?'No model matches this filter':'No GGUF models found')+'</b>'
      +(MODELS.length?'Clear the filter or press Rescan.':'Add files under a configured model dir, or use the HF pull tab.')+'</div>';
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
  b.disabled=false});
$('#modellist').addEventListener('click',async e=>{
  const b=e.target.closest('[data-act]');if(!b)return;
  const row=b.closest('.mrow');const path=row.dataset.path;
  if(b.dataset.act==='copy')return copy(path,'Path');
  if(b.dataset.act==='del'){
    if(!await confirmDlg('Delete model','<b>'+esc(row.dataset.name)+'</b> ('+esc(row.dataset.size)+')<br><code>'
      +esc(path)+'</code><br><br>This cannot be undone.','Delete',true))return;
    b.disabled=true;
    try{const d=await post('/api/models/delete',{path:path});
      toast('Deleted · '+d.freed_h+' freed','ok');await loadModels()}
    catch(err){toast('Delete failed: '+err.message,'err');b.disabled=false}
    return;
  }
  const sel=row.querySelector('[data-preset]');
  b.disabled=true;const old=b.textContent;b.textContent='Starting…';
  try{await post('/api/server/start',{model:path,preset:sel?sel.value:'auto'});
    toast('Loading '+row.dataset.name+'…','ok');await pollStatus(true)}
  catch(err){toast('Start failed: '+err.message,'err',12000)}
  b.disabled=false;b.textContent=old;
});

/* ============================== server status ============================== */
let lastRunModel=null;
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
  if(ct){p.className='pill ok';t.textContent=ct.alias+' · :'+ct.port+(ct.src==='avalon'?' · avalon':'')}
  else if(ACTIVE){p.className='pill warn';t.textContent=ACTIVE.alias+' · loading'}
  else if(AV&&AV.running){p.className='pill warn';t.textContent='avalon · loading'}
  else{p.className='pill';t.textContent='no server running'}
}
function pollStatus(force){
  return api('/api/server/status').then(st=>{
    ACTIVE=st.running?st:null;
    const chip=$('#srvchip');
    chip.textContent=st.running?(st.ready?'running · ready':'running · loading'):'stopped';
    chip.className='tag '+(st.running?(st.ready?'ok':'warn'):'mut');
    if(st.running){
      $('#srvdetail').innerHTML=[
        ['alias',st.alias],['pid',st.pid],['port',':'+st.port],['preset',st.preset],
        ['uptime',fmtUp(st.uptime)],st.adopted?['source','adopted']:null]
        .filter(Boolean).map(kv=>'<span class="tag">'+esc(kv[0])+' <b>'+esc(kv[1])+'</b></span>').join('')
        +'<button class="btn sm" data-copy="http://127.0.0.1:'+st.port+'/v1">Copy base URL</button>';
      $('#srvstop').disabled=false;$('#agentsbtn').disabled=false;
    }else{
      $('#srvdetail').innerHTML='<span class="hint">No llama-server running. Pick a model in the Models tab and press Load.</span>';
      $('#srvstop').disabled=true;$('#agentsbtn').disabled=true;
    }
    const now=ACTIVE?ACTIVE.model:null;
    if(force||now!==lastRunModel){lastRunModel=now;renderModels()}
    drawHeader();drawChatTarget();
  }).catch(()=>{});
}
$('#srvdetail').addEventListener('click',e=>{const b=e.target.closest('[data-copy]');if(b)copy(b.dataset.copy,'Base URL')});
$('#srvstop').addEventListener('click',async()=>{
  if(!await confirmDlg('Stop server','Stop the loaded model? Running requests will be cut.','Stop',true))return;
  $('#srvstop').disabled=true;
  try{await post('/api/server/stop');toast('Server stopped','ok')}catch(e){toast('Stop failed: '+e.message,'err')}
  pollStatus(true);$('#srvstop').disabled=false;
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
      badge:(idle?'idle':g.temp+'°C jct'+(g.temp_mem!=null?' · '+g.temp_mem+'°C mem':''))+(g.stale?' · stale':'')+' · '+(g.util||0)+'% util',
      big:n0(g.used_mb)+' <small>/ '+n0(g.total_mb)+' MB VRAM</small>',hist:VH.gpu});
    if(bits.length)out+='<div class="card" style="margin:0"><div class="hd" style="margin-bottom:8px"><h2 style="color:var(--fg-2);font-size:13px">GPU clocks</h2></div>'
      +'<div class="kvs">'+bits.map(b=>'<span class="tag">'+esc(b)+'</span>').join('')+'</div>'
      +'<p class="hint" style="margin:8px 0 0">Sensors go to sleep with the card; readings are held until it wakes.</p></div>';
  }
  if(v.mem){const p=Math.round(100*v.mem.used_gb/v.mem.total_gb);
    out+=card({title:'RAM',tone:'acc',pct:p,badge:p+'%',big:v.mem.used_gb+' <small>/ '+v.mem.total_gb+' GB</small>',hist:VH.mem})}
  if(v.cpu){out+=card({title:'CPU',tone:'acc',pct:v.cpu.percent,badge:(v.cpu.cores||'?')+' threads',
      big:v.cpu.percent+'%',hist:VH.cpu})}
  Object.keys(v.disk||{}).forEach(k=>{const d=v.disk[k],p=Math.round(100*d.used_gb/d.total_gb);
    out+=card({title:'Disk '+k,tone:'acc',pct:p,badge:p+'%',big:d.used_gb+' <small>/ '+d.total_gb+' GB</small>'})});
  if(!out)out='<div class="empty"><b>No vitals yet</b>Sensors unreadable on this host.</div>';
  $('#vitals').innerHTML=out;
  $('#vupdatetime').textContent='updated '+Math.max(0,Math.round((Date.now()-VT)/1000))+'s ago';
}
async function refreshVitals(){
  try{VIT=await api('/api/vitals');VT=Date.now();pushHist();drawVitals()}catch(e){drawVitals()}
}
$('#vrefresh').addEventListener('click',refreshVitals);

/* ============================== chat ============================== */
let HIST=[],CID=null,BUSY=false,AC=null;
const DEF_SYS='You are a helpful local assistant. Be concise.';
$('#sysprompt').value=localStorage.getItem('lfm.sys')||DEF_SYS;
$('#sysbtn').addEventListener('click',()=>{
  const box=$('#sysbox');box.classList.toggle('hide');
  if(box.classList.contains('hide')){localStorage.setItem('lfm.sys',$('#sysprompt').value||DEF_SYS);toast('System prompt saved','ok',1800)}
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
    blocks.push('<pre><code>'+esc(String(code).replace(/\n+$/,''))+'</code></pre>');
    return '\u0000'+(blocks.length-1)+'\u0000'});
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
function addMsg(m){
  const d=document.createElement('div');
  d.className='msg '+(m.role==='user'?'u':'a');
  const body=md(m.content||'');
  d.innerHTML='<div class="who">'+esc(m.role==='user'?'You':'Assistant')+'</div>'
    +(m.reasoning?'<details><summary>reasoning</summary><div class="think">'+esc(m.reasoning)+'</div></details>':'')
    +'<div class="ans">'+(body||'<span class="typing"><i></i><i></i><i></i></span>')+'</div>'
    +(m.role==='assistant'&&m.content?'<div class="bar"><button class="btn xs" data-copy="'+esc(m.content)+'">Copy</button></div>':'');
  $('#chat').appendChild(d);
  $('#chat').scrollTop=1e9;
  return d;
}
$('#chat').addEventListener('click',e=>{const b=e.target.closest('[data-copy]');if(b)copy(b.dataset.copy,'Reply')});
async function loadChatList(){
  try{const cs=await api('/api/chats');
    $('#chatlist').innerHTML=cs.length?cs.map(c=>'<div class="citem'+(c.id===CID?' on':'')+'" data-id="'+c.id+'">'
      +'<span class="t" title="'+esc(c.title)+'">'+esc(c.title||'(untitled)')+'</span>'
      +'<span class="s">'+(c.msgs||0)+' · '+esc(ago(c.updated||c.created))+'</span>'
      +'<button class="btn xs danger" data-del="'+c.id+'" aria-label="Delete chat">✕</button></div>').join('')
      :'<span class="hint">No chats yet — start one below.</span>';
  }catch(e){}
}
$('#chatlist').addEventListener('click',async e=>{
  const del=e.target.closest('[data-del]');
  if(del){e.stopPropagation();
    if(!await confirmDlg('Delete chat','This chat will be removed from disk.','Delete',true))return;
    try{await api('/api/chats/'+del.dataset.del,{method:'DELETE'});
      if(CID===del.dataset.del)newChat();loadChatList()}catch(err){toast('Delete failed: '+err.message,'err')}
    return}
  const it=e.target.closest('[data-id]');if(it)openChat(it.dataset.id);
});
async function openChat(id,quiet){
  try{const c=await api('/api/chats/'+id);CID=id;localStorage.setItem('lfm.cid',id);
    HIST=(c.messages||[]).filter(m=>m.role!=='system');
    $('#chat').innerHTML='';HIST.forEach(addMsg);loadChatList();return true}
  catch(e){if(!quiet)toast('Could not open chat: '+e.message,'err');return false}
}
function emptyThread(){
  $('#chat').innerHTML='<div class="empty"><b>New conversation</b>'
    +'Pick a chat on the left, or type below to start one.</div>';
}
function newChat(){CID=null;HIST=[];localStorage.removeItem('lfm.cid');emptyThread();loadChatList();$('#prompt').focus()}
$('#newchat').addEventListener('click',newChat);

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
  const um={role:'user',content:q};HIST.push(um);addMsg(um);
  post('/api/chats/append',{id:CID,model:ct.alias,role:'user',content:q})
    .then(r=>{CID=r.id;loadChatList()}).catch(()=>{});
  const box=addMsg({role:'assistant',content:''});
  const ans=box.querySelector('.ans'),who=box.querySelector('.who');
  BUSY=true;$('#sendbtn').disabled=true;$('#stopbtn').style.display='';
  AC=new AbortController();
  let buff='',bufa='',t0=0,toks=0,stopped=false;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},signal:AC.signal,
      body:JSON.stringify({messages:[{role:'system',content:$('#sysprompt').value||DEF_SYS}]
        .concat(HIST.map(m=>({role:m.role,content:m.content})))})});
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
              if(!de){de=document.createElement('details');de.innerHTML='<summary>reasoning</summary><div class="think"></div>';
                box.insertBefore(de,ans)}
              de.querySelector('.think').textContent=buff}
            if(dl.content){bufa+=dl.content;ans.textContent=bufa}
            if(j.usage&&j.usage.completion_tokens)toks=j.usage.completion_tokens;
          }catch(e){}
        }
        $('#chat').scrollTop=1e9;
      }
    }
  }catch(e){
    if(e.name==='AbortError'){stopped=true;ans.textContent=bufa||buff||'(stopped)'}
    else ans.textContent='error: '+e.message;
  }
  const finalText=bufa||buff||'';
  const secs=Math.max(.001,(performance.now()-(t0||performance.now()))/1000);
  const tps=toks?toks/secs:Math.round((finalText.length/4)/secs);
  who.innerHTML='Assistant <span class="tag mut" style="margin-left:6px">'+(stopped?'stopped · ':'')
    +tps+' tok/s · '+esc(ct.alias)+'</span>';
  box.insertAdjacentHTML('beforeend',finalText?'<div class="bar"><button class="btn xs" data-copy="'+esc(finalText)+'">Copy</button></div>':'');
  if(finalText){
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
  const dlg=dialog({title:'Register into coding agents',width:'600px',
    html:'<p class="dlg-p" style="margin-bottom:10px">Writes each agent&rsquo;s config to point at <b>'+cur+'</b>.</p>'
      +'<div style="max-height:52vh;overflow:auto">'+rows+'</div><div id="agres" style="margin-top:10px"></div>'});
  const apply=mkBtn('Apply','primary'),close=mkBtn('Close');
  dlg.foot.append(close,apply);
  close.addEventListener('click',()=>dlg.close());
  apply.addEventListener('click',async()=>{
    const picks=$$('.agchk',dlg.el).filter(x=>x.checked).map(x=>x.value);
    if(!picks.length)return toast('Pick at least one agent','warn');
    apply.disabled=true;apply.textContent='Applying…';
    let r;try{r=await post('/api/agents/sync',{agents:picks})}
    catch(e){apply.disabled=false;apply.textContent='Apply';return toast('Sync failed: '+e.message,'err',12000)}
    $('#agres',dlg.el).innerHTML='<div class="hd" style="margin:0 0 6px"><h2 style="font-size:12px;color:var(--fg-2)">Results</h2></div>'
      +Object.keys(r.results).map(k=>{const v=r.results[k];
        return '<div class="frow" style="cursor:default"><span class="tag '+(v.ok?'ok':'err')+'">'+(v.ok?'ok':'fail')+'</span>'
          +'<span class="p" title="'+esc(v.detail)+'">'+esc(k)+' — '+esc(v.detail)+'</span></div>'}).join('');
    apply.disabled=false;apply.textContent='Apply again';
    toast('Registry updated for '+picks.length+' agent'+(picks.length>1?'s':''),'ok');
    pollStatus();
  });
});

/* ============================== avalon ============================== */
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
      +'<br><span class="hint">'+esc(p.desc)+'</span><br><span class="tag ok">'+esc(p.speed)+'</span></span></label>').join('');
    $$('#avalonprofiles input[name=sprof]').forEach(r=>r.addEventListener('change',()=>{ASEL=r.value;loadAvalon()}));
    const c=$('#avalonchip');
    c.textContent=AV.running?(AV.state==='ready'?'avalon · ready':'avalon · loading'):'avalon · stopped';
    c.className='tag '+(AV.running?(AV.state==='ready'?'ok':'warn'):'mut');
    $('#avalondetail').innerHTML=AV.running
      ?[['pid',AV.pid||'?'],AV.adopted?['source','adopted']:null,['up',fmtUp(AV.uptime||0)],
        ['ctx',AV.max_context?n0(AV.max_context):'…'],['port',':'+(AV.port||8086)]]
        .filter(Boolean).map(kv=>'<span class="tag">'+esc(kv[0])+' <b>'+esc(kv[1])+'</b></span>').join('')
        +(AV.model?'<span class="tag mut">'+esc(AV.model)+'</span>':'')
      :'<span class="hint">Avalon is stopped. Choose a profile below — it needs the LFM server&rsquo;s VRAM.</span>';
    $('#avalonstart').disabled=!ASEL||!ALIST.some(p=>p.key===ASEL&&p.available);
    $('#avalonstop').disabled=!AV.running;
    const d=await api('/api/avalon/log');paintInto($('#avalonlog'),d.log);
    drawHeader();drawChatTarget();
  }catch(e){}
}
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
  try{await post('/api/avalon/start',{profile:ASEL});toast('Avalon starting — a 125B load takes a while','ok')}
  catch(e){
    const t=e.message;
    if(t.indexOf('Stop the LFM server')>=0){
      if(await confirmDlg('VRAM conflict','This profile needs the VRAM the LFM server is holding.'
        +'<br><br>Stop the LFM server and start Avalon?','Stop LFM & start Avalon',true)){
        try{await post('/api/server/stop');pollStatus(true);$('#avalonstart').click()}
        catch(e2){toast('Could not stop the LFM server: '+e2.message,'err')}}
    }else toast('Start failed: '+t,'err',12000);
  }
  loadAvalon();
});
$('#avalonstop').addEventListener('click',async()=>{
  if(!await confirmDlg('Stop Avalon','Stop the Avalon server? Running requests will be cut.','Stop',true))return;
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
$('#avalonlan').addEventListener('change',async e=>{
  const on=e.target.checked;e.target.disabled=true;
  try{await post('/api/avalon/access',{lan:on});
    toast(on?'Exposed on the LAN with an API key':'Back to local-only','ok')}
  catch(err){toast('Failed: '+err.message,'err');e.target.checked=!on}
  e.target.disabled=false;loadAvalonAccess();loadAvalon();
});

/* ============================== HF pull ============================== */
let HFREPO='',DLPREV={},DLTIMER=null;
async function hfSearch(){
  const q=$('#hfq').value.trim();if(!q)return;
  $('#hfinfo').textContent='searching…';
  let res;try{res=await api('/api/hf/search?q='+encodeURIComponent(q))}
  catch(e){$('#hfinfo').textContent='';return toast('Search failed: '+e.message,'err')}
  $('#hfinfo').textContent=res.length+' repos';
  $('#hfresults').innerHTML=res.length?res.map(m=>'<div class="frow" data-repo="'+esc(m.id)+'">'
    +'<span style="flex:1;min-width:0"><b>'+esc(m.id)+'</b><br><span class="hint">downloads '
    +n0(m.downloads)+' · likes '+n0(m.likes)+'</span></span><span class="tag">open →</span></div>').join('')
    :'<div class="empty"><b>No repos found</b>Try a broader term.</div>';
}
$('#hfsearch').addEventListener('click',hfSearch);
$('#hfq').addEventListener('keydown',e=>{if(e.key==='Enter')hfSearch()});
$('#hfresults').addEventListener('click',e=>{const r=e.target.closest('[data-repo]');if(r)hfFiles(r.dataset.repo)});
async function hfFiles(repo){
  HFREPO=repo;
  let files;try{files=await api('/api/hf/files?repo='+encodeURIComponent(repo))}
  catch(e){return toast('Repo lookup failed: '+e.message,'err')}
  $('#hfrepo').textContent=repo;
  $('#hffiles').innerHTML=files.length?files.map(f=>'<label class="frow"><input type="checkbox" value="'+esc(f.path)+'" data-size="'
    +(f.size||0)+'"><span class="p" title="'+esc(f.path)+'">'+esc(f.path)+'</span>'
    +'<span class="tag mut">'+esc(f.size_h)+'</span></label>').join('')
    :'<div class="empty"><b>No GGUF files</b>This repo has no .gguf in its root tree.</div>';
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
  if(!ds.length){$('#downloads').innerHTML='<span class="hint">none</span>';$('#dlinfo').textContent='';ensureDl(false);return}
  const live=ds.some(d=>d.status==='downloading'||d.status==='queued');
  $('#dlinfo').textContent=ds.length+' job'+(ds.length>1?'s':'')+(live?' · active':'');
  $('#downloads').innerHTML=ds.map(d=>{
    const pct=d.total?Math.round(100*d.done/d.total):0;
    const now=Date.now(),p=DLPREV[d.id]||{d:d.done,t:now};
    const dt=(now-p.t)/1000,spd=dt>0.4?(d.done-p.d)/dt:null;
    if(d.status==='downloading'&&dt>0.4)DLPREV[d.id]={d:d.done,t:now};
    const eta=(spd>0&&d.total&&d.done<d.total)?fmtUp((d.total-d.done)/spd):null;
    const kind=d.status==='done'?'ok':d.status==='error'?'err':d.status==='cancelled'?'warn':'acc';
    return '<div class="dl"><div style="display:flex;gap:9px;align-items:center;flex-wrap:wrap">'
      +'<span style="flex:1;min-width:0;font-family:var(--mono);font-size:12.5px" title="'+esc(d.dest)+'">'+esc(d.file)+'</span>'
      +'<span class="tag '+kind+'">'+esc(d.status)+'</span>'
      +(d.status==='downloading'?'<span class="tag mut">'+pct+'% · '+gib(d.done)+' / '+gib(d.total)+'</span>'
        +(spd!=null?'<span class="tag mut">'+(spd/1048576).toFixed(1)+' MB/s</span>':'')
        +(eta?'<span class="tag mut">ETA '+eta+'</span>':''):'')
      +(d.status==='downloading'?'<button class="btn xs danger" data-cancel="'+esc(d.id)+'">Cancel</button>':'')
      +'</div>'+(d.status==='downloading'?'<div class="meter"><i style="width:'+pct+'%"></i></div>':'')
      +(d.error?'<p class="hint" style="color:var(--err);margin:5px 0 0">'+esc(d.error)+'</p>':'')+'</div>';
  }).join('');
  ensureDl(live);
}
$('#downloads').addEventListener('click',e=>{
  const b=e.target.closest('[data-cancel]');if(!b)return;
  post('/api/downloads/cancel',{id:b.dataset.cancel}).then(()=>{b.disabled=true;pollDownloads()})
    .catch(err=>toast('Cancel failed: '+err.message,'err'));
});

/* ============================== polling & keys ============================== */
function tick(){
  if(document.hidden)return;
  pollStatus();
  if(tab==='avalon')loadAvalon();
}
setInterval(tick,2500);
setInterval(()=>{if(!document.hidden&&tab==='vitals')refreshVitals()},2000);
setInterval(()=>{if(!document.hidden&&tab==='server')refreshLog()},1500);
setInterval(()=>{if(!document.hidden&&tab==='avalon')refreshAvalonLog()},1500);
async function refreshAvalonLog(){try{const d=await api('/api/avalon/log');paintInto($('#avalonlog'),d.log)}catch(e){}}
document.addEventListener('visibilitychange',()=>{if(!document.hidden){pollStatus();refreshVitals()}});
document.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const t=e.target,typing=t&&(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.isContentEditable);
  if(e.key==='Escape')return;
  if(e.key==='/'&&!typing){e.preventDefault();
    if(tab==='hf')$('#hfq').focus();else if(tab==='server')$('#logfilter').focus();else{setTab('models');$('#msearch').focus()}
    return}
  if(!typing&&TABS[Number(e.key)-1])setTab(TABS[Number(e.key)-1]);
});

/* ============================== boot ============================== */
setTab(localStorage.getItem('lfm.tab')||'models');
loadModels().then(()=>pollStatus(true)).catch(e=>toast('Panel load failed: '+e.message,'err'));
loadChatList();
(async()=>{const s=localStorage.getItem('lfm.cid');
  if(!s||!(await openChat(s,true))){localStorage.removeItem('lfm.cid');emptyThread()}})();
refreshVitals();loadAvalon();loadAvalonAccess();pollDownloads();drawChatTarget();
$('#footnote').textContent='polling pauses when the tab is hidden';
</script>
</body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
