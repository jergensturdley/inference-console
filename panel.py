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
            "pid": st.get("pid"), "profile": st.get("profile"),
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
<html><head><meta charset="utf-8"><title>LFM Panel — Sign in</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#111418;--card:#1a1f26;--fg:#dde3ea;--mut:#8b96a5;--acc:#5ab0f0;--err:#e86a6a;--bd:#2a323d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;
 display:flex;align-items:center;justify-content:center;min-height:100vh}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:28px;width:320px}
h1{font-size:16px;margin:0 0 14px}input{width:100%;padding:9px 10px;border-radius:8px;border:1px solid var(--bd);
 background:var(--bg);color:var(--fg);font-size:14px}
button{width:100%;margin-top:12px;padding:9px;border:0;border-radius:8px;background:var(--acc);color:#08131d;
 font-weight:600;font-size:14px;cursor:pointer}
#msg{color:var(--err);font-size:12.5px;min-height:18px;margin-top:10px}
</style></head><body>
<div class="card"><h1>LFM Panel</h1>
<form>
<input id="pw" type="password" placeholder="sudo password" autocomplete="current-password" autofocus>
<button type="submit">Sign in</button><div id="msg"></div></form></div>
<script>
async function go(ev){ev.preventDefault();const m=document.getElementById('msg');m.textContent='';
 try{const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({password:document.getElementById('pw').value})});
  if(r.ok){location.replace('/');}else{m.textContent=(r.status===429)?'Too many attempts — wait 5 minutes.':'Wrong password.';}
 }catch(e){m.textContent='Failed: '+e;}}
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
                        "msgs": len(c.get("messages", []))})
        except Exception:
            continue
    return sorted(out, key=lambda c: -c["created"])

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
            "ip": _lan_ip(), "status": SMGR.status()}

class AvalonAccessBody(BaseModel):
    lan: bool

@app.get("/api/avalon/access")
def api_avalon_access_get():
    return {"lan": bool(CFG.get("avalon_lan")), "api_key": CFG.get("avalon_api_key", ""), "ip": _lan_ip()}

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
    st = MGR.status()
    if not (st["running"] and st.get("ready")):
        raise HTTPException(409, "no running server")
    body = await req.json()
    body.setdefault("stream", True)
    url = f"http://127.0.0.1:{st['port']}/v1/chat/completions"
    client = httpx.AsyncClient(timeout=None)
    async def gen():
        try:
            async with client.stream("POST", url, json=body) as r:
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
<html><head><meta charset="utf-8"><title>LFM Panel</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#111418;--card:#1a1f26;--fg:#dde3ea;--mut:#8b96a5;--acc:#5ab0f0;--ok:#5ad191;--warn:#e8b45a;--err:#e86a6a;--bd:#2a323d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
header{display:flex;align-items:center;gap:20px;padding:10px 18px;border-bottom:1px solid var(--bd);position:sticky;top:0;background:var(--bg);z-index:5}
header h1{font-size:15px;margin:0;color:var(--acc)}
nav button{background:none;border:none;color:var(--mut);font:inherit;padding:6px 10px;cursor:pointer;border-radius:8px}
nav button.on{color:var(--fg);background:var(--card)}
main{padding:18px;max-width:1100px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px 16px;margin-bottom:12px}
.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.spread{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.chip{background:#232b35;border-radius:99px;padding:1px 9px;font-size:12px;color:var(--mut)}
.chip.moe{color:var(--warn)}.chip.vl{color:var(--acc)}.chip.ok{color:var(--ok)}
button.act{background:var(--acc);border:none;color:#08243c;padding:6px 14px;border-radius:8px;font-weight:600;cursor:pointer}
button.stop{background:var(--err);color:#3a0d0d}
select,input{background:#0e1216;border:1px solid var(--bd);color:var(--fg);border-radius:8px;padding:6px 8px;font:inherit}
.meter{height:9px;background:#0e1216;border-radius:99px;overflow:hidden;margin:4px 0 10px}
.meter i{display:block;height:100%;background:var(--acc);border-radius:99px}
.big{font-size:22px;font-weight:700}
pre#log{max-height:420px;overflow:auto;background:#0b0e11;padding:10px;border-radius:10px;font-size:12px;color:var(--mut)}
.mrow{display:flex;gap:8px;margin:6px 0 14px}
#chat{height:52vh;overflow-y:auto;display:flex;flex-direction:column;gap:10px;padding:4px}
.msg{max-width:82%;padding:8px 12px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.msg.u{align-self:flex-end;background:#274357}
.msg.a{align-self:flex-start;background:var(--card);border:1px solid var(--bd)}
.msg .think{color:var(--mut);font-size:12.5px;border-left:2px solid var(--bd);padding-left:8px;margin-bottom:6px;white-space:pre-wrap}
.hidden{display:none}
label.k{color:var(--mut);font-size:12px}
</style></head><body>
<header><h1>⚡ LFM Panel</h1>
<nav>
<button data-t="models" class="on">Models</button><button data-t="server">Server</button>
<button data-t="avalon">Avalon</button>
<button data-t="vitals">Vitals</button><button data-t="chat">Chat</button><button data-t="hf">HF pull</button>
</nav><span id="hdrstatus" class="chip"></span></header>
<main>
<section id="t-models"><div class="card"><div class="spread"><b>Model library</b><span class="chip" id="libinfo"></span></div><div id="modellist"></div></div></section>
<section id="t-hf" class="hidden">
 <div class="card"><b>Search Hugging Face</b>
  <div class="mrow"><input id="hfq" style="flex:1" placeholder="search gguf repos… (Enter)" onkeydown="if(event.key=='Enter')hfSearch()">
  <button class="act" onclick="hfSearch()">Search</button></div>
  <div id="hfresults"></div></div>
 <div class="card" id="hffilesbox" style="display:none"><b id="hfrepo"></b><div id="hffiles" style="margin:8px 0"></div>
  <button class="act" onclick="hfPull()">Pull selected</button></div>
 <div class="card"><b>Downloads</b><div id="downloads"></div></div>
</section>
<section id="t-server" class="hidden">
 <div class="card"><div class="spread"><b>llama-server</b><span id="srvchip" class="chip"></span></div>
  <div id="srvdetail" class="row"></div>
  <div class="row" style="margin-top:10px"><button class="act stop" onclick="stopSrv()">Stop</button>
  <span class="chip">child binds 127.0.0.1 — LAN only via this panel</span></div></div>
 <div class="card"><b>Log</b><pre id="log"></pre></div>
</section>
<section id="t-avalon" class="hidden">
 <div class="card"><div class="spread"><b>Avalon — Qwen3.8-Flash-Next (125B MoE)</b><span id="avalonchip" class="chip"></span></div>
  <div id="avalondetail" class="row" style="margin:6px 0"></div>
  <div class="row" style="margin-top:8px"><button class="act" onclick="avalonStart()">Start selected profile</button>
  <button class="act stop" onclick="avalonStop()">Stop</button>
  <span class="chip">OpenAI-compatible · streams reasoning_content</span></div>
  <div id="avalonprofiles" style="margin-top:10px"></div></div>
 <div class="card"><div class="spread"><b>Network access</b>
  <label class="row" style="gap:6px;margin:0"><input type="checkbox" id="avalonlan" onchange="avalonSetLan(this.checked)">
  <span>Expose to LAN (:8080)</span></label></div>
  <div id="avalonaccess" class="row" style="margin-top:6px"></div></div>
 <div class="card"><b>Avalon log</b><pre id="avalonlog" style="max-height:260px"></pre></div>
</section>
<section id="t-vitals" class="hidden"><div id="vitals"></div></section>
<section id="t-chat" class="hidden"><div style="display:flex;gap:12px;align-items:flex-start">
 <div class="card" style="width:240px;flex-shrink:0"><div class="spread"><b>Chats</b><button class="act" onclick="newChat()">+ New</button></div><div id="chatlist"></div></div>
 <div class="card" style="flex:1;min-width:0"><div id="chat"></div>
 <div class="mrow"><input id="prompt" style="flex:1" placeholder="message… (Enter to send)" onkeydown="if(event.key=='Enter')send()">
 <button class="act" onclick="send()">Send</button></div></div></div></section>
</main>
<script>
const $=s=>document.querySelector(s);
let MODELS=[],ACTIVE=null;
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 document.querySelectorAll('main section').forEach(s=>s.classList.add('hidden'));
 $('#t-'+b.dataset.t).classList.remove('hidden');});
function h(n){return n.toFixed?n.toFixed(1):n}
async function j(url,opt){const r=await fetch(url,opt);if(!r.ok)throw(await r.text());return r.json()}
async function loadModels(){MODELS=(await j('/api/models')).models;
 $('#libinfo').textContent=MODELS.length+' models';
 $('#modellist').innerHTML=MODELS.map(m=>`<div class="row" style="padding:8px 0;border-bottom:1px solid var(--bd)">
  <b style="min-width:280px">${m.name}</b>
  <span class="chip">${m.arch}</span><span class="chip">${m.quant}</span>
  ${m.params?`<span class="chip">${m.params}</span>`:''}<span class="chip">${m.size_h}</span>
  ${m.moe?'<span class="chip moe">MoE</span>':''}${m.vl?'<span class="chip vl">VL</span>':''}
  <span style="flex:1"></span>
  <select id="p_${btoa(m.path)}"><option value="auto">auto preset</option><option value="dense-gpu">dense-gpu</option><option value="moe-split">moe-split</option><option value="vision">vision</option><option value="igpu-full">iGPU full (780M)</option><option value="igpu-vision">iGPU vision (780M)</option></select>
  <button class="act" onclick="start('${m.path}')">Start</button>
  <button class="act stop" onclick="delModel('${m.path}','${m.name}','${m.size_h}')">✕</button></div>`).join('');}
async function delModel(path,name,size){
 if(!confirm('Delete '+name+' ('+size+')?\\n'+path+'\\n\\nThis cannot be undone.'))return;
 const r=await fetch('/api/models/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
 if(r.ok){const d=await r.json();alert('Deleted: '+d.deleted+' ('+d.freed_h+' freed)');loadModels();}
 else alert('Delete failed: '+await r.text());}
async function hfSearch(){const q=$('#hfq').value.trim();if(!q)return;
 const res=await j('/api/hf/search?q='+encodeURIComponent(q));
 $('#hfresults').innerHTML=res.map(m=>`<div class="row" style="padding:6px 0;border-bottom:1px solid var(--bd);cursor:pointer" onclick="hfFiles('${m.id}')">
  <b style="flex:1">${m.id}</b><span class="chip">↓ ${m.downloads}</span><span class="chip">★ ${m.likes}</span></div>`).join('')||'<span class="chip">no results</span>';}
async function hfFiles(repo){const files=await j('/api/hf/files?repo='+encodeURIComponent(repo));
 $('#hfrepo').textContent=repo;
 $('#hffiles').innerHTML=files.map(f=>`<label class="row" style="padding:4px 0"><input type="checkbox" value="${f.path}" data-size="${f.size}">
  <span style="flex:1;word-break:break-all">${f.path}</span><span class="chip">${f.size_h}</span></label>`).join('')||'<span class="chip">no gguf files</span>';
 $('#hffilesbox').style.display='block';}
async function hfPull(){const sel=[...document.querySelectorAll('#hffiles input:checked')].map(i=>i.value);
 if(!sel.length)return alert('select files first');
 const repo=$('#hfrepo').textContent;
 const r=await fetch('/api/hf/pull',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({repo,files:sel})});
 if(!r.ok)return alert(await r.text());
 setInterval(pollDownloads,1500);pollDownloads();}
async function pollDownloads(){const ds=await j('/api/downloads');if(!ds.length){$('#downloads').innerHTML='<span class="chip">none</span>';return}
 $('#downloads').innerHTML=ds.map(d=>{const pct=d.total?Math.round(100*d.done/d.total):0;
  return `<div style="padding:6px 0;border-bottom:1px solid var(--bd)"><div class="spread"><span>${d.file}</span>
  <span class="chip ${d.status==='done'?'ok':''}">${d.status}${d.status==='downloading'?` ${pct}% · ${h(d.done/1e9)}G`:''}</span></div>
  ${d.status==='downloading'?`<div class="meter"><i style="width:${pct}%"></i></div>`:''}
  ${d.status==='downloading'?`<button class="act stop" onclick="fetch('/api/downloads/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:'${d.id}'})})">Cancel</button>`:''}</div>`;}).join('');}
async function start(path){
 const sel=document.getElementById('p_'+btoa(path));
 try{await j('/api/server/start',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({model:path,preset:sel?sel.value:'auto'})});pollStatus(true);}
 catch(e){alert('Start failed: '+e);}}
async function stopSrv(){await j('/api/server/stop',{method:'POST'});pollStatus(true)}
function pollStatus(force){return j('/api/server/status').then(st=>{
 ACTIVE=st.running?st:null;
 $('#hdrstatus').textContent=st.running?(st.ready?'● '+st.alias:'◌ '+st.alias+' (loading)'):'○ no server';
 $('#hdrstatus').className='chip '+(st.running?(st.ready?'ok':''):'');
 $('#srvchip').textContent=st.running?(st.ready?'running · ready':'running · loading'):'stopped';
 $('#srvchip').className='chip '+(st.running&&st.ready?'ok':'');
 $('#srvdetail').innerHTML=st.running?`
  <span class="chip">alias ${st.alias}</span><span class="chip">pid ${st.pid}</span>
  <span class="chip">preset ${st.preset}</span><span class="chip">port ${st.port}</span>
  <span class="chip">uptime ${Math.floor(st.uptime/60)}m${st.uptime%60}s</span>
  <button class="act" onclick="agentsModal()">Agents…</button>`:'<span class="chip">pick a model and press Start</span>';
 if(force&&!st.running)loadModels();}).catch(()=>{})}
function closeAgModal(){document.getElementById('agmodal')?.remove();}
async function agentsModal(){
 closeAgModal();
 let d;try{d=await j('/api/agents');}catch(e){alert('Failed: '+e);return;}
 const reg=(d.registry&&d.registry.agents)||{};
 const rows=d.agents.map(a=>{
  const r=reg[a.id];
  return `<label style="display:flex;gap:8px;align-items:center;padding:4px 0">
   <input type="checkbox" class="agchk" value="${a.id}">
   <span>${a.label}</span>
   ${r?`<span class="chip ok">${r.alias}@${r.port}</span>`:''}
   ${a.hint?`<span style="opacity:.55;font-size:12px">${a.hint}</span>`:''}</label>`;}).join('');
 const el=document.createElement('div');el.id='agmodal';
 el.style.cssText='position:fixed;inset:0;background:#000a;display:flex;align-items:center;justify-content:center;z-index:99';
 el.innerHTML=`<div style="background:#14161a;color:var(--fg,#ddd);border:1px solid var(--bd,#333);padding:16px;border-radius:8px;min-width:min(420px,92vw)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
   <b>Register into coding agents</b><button class="act" onclick="closeAgModal()">✕</button></div>
  <div style="margin-bottom:6px;opacity:.7;font-size:12px">writes each agent's config to point at ${d.running?`${d.running[0]} @ :${d.running[1]}`:'—'}</div>
  ${rows}
  <div style="margin-top:12px"><button class="act" onclick="syncAgents()">Apply</button></div></div>`;
 document.body.appendChild(el);}
async function syncAgents(){
 const picks=[...document.querySelectorAll('.agchk:checked')].map(x=>x.value);
 if(!picks.length){alert('pick at least one agent');return;}
 let r;try{r=await j('/api/agents/sync',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({agents:picks})});}
 catch(e){alert('Sync failed: '+e);return;}
 alert(Object.entries(r.results).map(([k,v])=>k+': '+(v.ok?'✓ '+v.detail:'✗ '+v.detail)).join('\n'));
 closeAgModal();pollStatus(true);}
function drawVitals(v){let out='';
 if(v.gpu){const g=v.gpu,pct=Math.round(100*g.used_mb/g.total_mb),idle=g.util===0;
  const extra=[g.power_w!=null?g.power_w+' W':null,idle?null:(g.fan_rpm?g.fan_rpm+' RPM fan':null),idle?null:(g.sclk_mhz?g.sclk_mhz+' MHz':null),g.mclk_mhz?g.mclk_mhz+' MHz mem':null].filter(Boolean).join(' · ');
  out+=`<div class="card"><div class="spread"><b>GPU</b><span class="chip">${idle?'idle':g.temp+'°C junction'}${g.temp_mem!=null?' · '+g.temp_mem+'°C mem':''} · ${g.util}% util${g.stale?' · cached':''}</span></div>
  <div class="big">${g.used_mb} <small>/ ${g.total_mb} MB VRAM</small></div>
  <div class="meter"><i style="width:${pct}%"></i></div>${extra?`<div class="chip" style="margin-top:6px">${extra}</div>`:''}</div>`;}
 const m=v.mem,mp=Math.round(100*m.used_gb/m.total_gb);
 out+=`<div class="card"><b>RAM</b><div class="big">${m.used_gb} <small>/ ${m.total_gb} GB</small></div><div class="meter"><i style="width:${mp}%"></i></div></div>`;
 if(v.cpu)out+=`<div class="card"><div class="spread"><b>CPU</b><span class="chip">${v.cpu.cores} threads</span></div><div class="big">${v.cpu.percent}%</div><div class="meter"><i style="width:${v.cpu.percent}%"></i></div></div>`;
 for(const[k,d]of Object.entries(v.disk||{})){const p=Math.round(100*d.used_gb/d.total_gb);
  out+=`<div class="card"><b>Disk ${k}</b><div class="big">${d.used_gb} <small>/ ${d.total_gb} GB</small></div><div class="meter"><i style="width:${p}%"></i></div></div>`;}
 $('#vitals').innerHTML=out;}
let HIST=[],CID=null,SYSPROMPT='You are a helpful local assistant. Be concise.';
const escq=s=>esc(s).replace(/"/g,'&quot;');
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function addMsg(m){const d=document.createElement('div');d.className='msg '+(m.role==='user'?'u':'a');
 d.innerHTML=(m.reasoning?'<span class="think">'+esc(m.reasoning)+'</span>':'')+'<span class="ans">'+esc(m.content||'')+'</span>';
 $('#chat').appendChild(d);$('#chat').scrollTop=1e9;return d}
async function loadChatList(){try{const cs=await j('/api/chats');
 $('#chatlist').innerHTML='<div class="row" style="padding:2px 0 8px"><button class="act" style="width:100%" onclick="newChat()">+ New chat</button></div>'
  +(cs.map(c=>`<div class="row" style="padding:5px 0;border-bottom:1px solid var(--bd);cursor:pointer" onclick="openChat('${c.id}')">
   <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escq(c.title)}">${esc(c.title)}</span>
   <span class="chip">${c.msgs}</span>
   <button class="act stop" onclick="event.stopPropagation();delChat('${c.id}')">✕</button></div>`).join('')||'<span class="chip">no chats yet</span>');}catch(e){}}
async function openChat(id){const c=await j('/api/chats/'+id);CID=id;HIST=c.messages.filter(m=>m.role!=='system');
 $('#chat').innerHTML='';HIST.forEach(addMsg);}
function newChat(){CID=null;HIST=[];$('#chat').innerHTML='';}
async function delChat(id){if(!confirm('Delete this chat?'))return;
 await fetch('/api/chats/'+id,{method:'DELETE'});if(CID===id)newChat();loadChatList();}
async function send(){const t=$('#prompt');const q=t.value.trim();if(!q)return;t.value='';
 const um={role:'user',content:q};HIST.push(um);addMsg(um);
 const model=ACTIVE?ACTIVE.alias:'';
 j('/api/chats/append',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({id:CID,model:model,role:'user',content:q})}).then(r=>{CID=r.id;loadChatList()});
 const box=addMsg({role:'assistant',content:''});
 const think=box.querySelector('.think'),ans=box.querySelector('.ans');
 let buff='',bufa='';
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({messages:[{role:'system',content:SYSPROMPT},...HIST.map(m=>({role:m.role,content:m.content}))]})});
  if(!r.ok){const d=await r.text();ans.textContent=d.includes('no running server')?'No model running — start one in the Models tab first.':'error: '+d;return}
  const rd=r.body.getReader(),dec=new TextDecoder();let sbuf='';
  for(;;){const{done,value}=await rd.read();if(done)break;sbuf+=dec.decode(value,{stream:true});
   const lines=sbuf.split('\n');sbuf=lines.pop();
   for(const ln of lines){if(!ln.startsWith('data: '))continue;const d=ln.slice(6);
    if(d==='[DONE]')continue;
    try{const j2=JSON.parse(d),dl=j2.choices?.[0]?.delta||{};
     if(dl.reasoning_content){buff+=dl.reasoning_content;think.textContent=buff}
     if(dl.content){bufa+=dl.content;ans.textContent=bufa}
     $('#chat').scrollTop=1e9;}catch(e){}}}
  HIST.push({role:'assistant',content:bufa||buff,reasoning:buff});
  j('/api/chats/append',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({id:CID,model:model,role:'assistant',content:bufa||buff,reasoning:buff})}).then(()=>loadChatList());
 }catch(e){ans.textContent='error: '+e}}
async function refreshLog(){try{const{log}=(await j('/api/server/log'));const p=$('#log');
if(p.textContent!==log){p.textContent=log;p.scrollTop=1e9;}}catch(e){}}
let AVALON_SEL=null,AVALON_LIST=[];
function fmtUp(s){s|=0;const m=(s/60)|0;return m>=60?((m/60)|0)+'h'+(m%60)+'m':m+'m'+(s%60)+'s'}
async function loadAvalon(){try{AVALON_LIST=await j('/api/avalon/profiles');const st=await j('/api/avalon/status');
 if(AVALON_SEL===null)AVALON_SEL=(st.running&&st.profile)||'balanced';
 $('#avalonprofiles').innerHTML=AVALON_LIST.map(p=>`
  <label class="row" style="padding:9px 10px;border-bottom:1px solid var(--bd);cursor:pointer;border-radius:8px;${p.key===AVALON_SEL?'background:#1c2630':''}">
   <input type="radio" name="sprof" value="${p.key}" ${p.key===AVALON_SEL?'checked':''} ${p.available?'':'disabled'} onchange="AVALON_SEL='${p.key}';loadAvalon()">
   <span style="flex:1"><b>${p.label}</b>${st.running&&st.profile===p.key?' <span class="chip ok">running</span>':''}<br>
   <span class="chip">${p.desc}</span> <span class="chip ok">${p.speed}</span>${p.available?'':' <span class="chip" style="color:var(--err)">config missing</span>'}</span>
  </label>`).join('');
 const c=$('#avalonchip');
 c.textContent=st.running?(st.state==='ready'?'● avalon · '+(st.profile||''):'◌ avalon loading…'):'○ avalon stopped';
 c.className='chip '+(st.running&&st.state==='ready'?'ok':'');
 $('#avalondetail').innerHTML=st.running?`<span class="chip">pid ${st.pid||'?'}${st.adopted?' · adopted':''}</span><span class="chip">up ${fmtUp(st.uptime||0)}</span><span class="chip">ctx ${st.max_context||'…'}</span><span class="chip">${st.model||''}</span>`:'';
 const{log}=await j('/api/avalon/log');const p2=$('#avalonlog');
 if(p2.textContent!==log){p2.textContent=log;p2.scrollTop=1e9;}}catch(e){}}
async function avalonStart(){try{
  await j('/api/avalon/start',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({profile:AVALON_SEL})});}
 catch(e){const t=String(e);
  if(t.includes('Stop the LFM server')&&confirm('This profile needs the VRAM the LFM server is using.\\nStop the LFM server and start Avalon?')){
   await j('/api/server/stop',{method:'POST'});return avalonStart();}
  alert('Start failed: '+t);}
 loadAvalon().catch(()=>{})}
async function avalonStop(){await j('/api/avalon/stop',{method:'POST'});loadAvalon().catch(()=>{})}
async function loadAvalonAccess(){try{const a=await j('/api/avalon/access');
 $('#avalonlan').checked=a.lan;
 $('#avalonaccess').innerHTML=a.lan
  ?`<span class="chip ok">LAN · http://${a.ip}:8080/v1</span>
    <span class="chip">key: <code>${esc(a.api_key)}</code></span>
    <button class="act" style="padding:2px 8px" onclick="avalonRegen()">New key</button>
    <span class="chip">send as: Authorization: Bearer &lt;key&gt;</span>`
  :`<span class="chip">local only · http://127.0.0.1:8080/v1</span>`;}catch(e){}}
async function avalonSetLan(on){try{await j('/api/avalon/access',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lan:on})});}catch(e){alert('Failed: '+e)}loadAvalonAccess();loadAvalon().catch(()=>{})}
async function avalonRegen(){if(!confirm('Generate a new API key? Every LAN client must update its key.'))return;
 await j('/api/avalon/access/regenerate',{method:'POST'});loadAvalonAccess();}
loadModels();pollStatus();loadChatList();loadAvalon();loadAvalonAccess();
setInterval(()=>{pollStatus();loadAvalon();},2500);
setInterval(()=>{if(!document.getElementById('t-vitals').classList.contains('hidden'))j('/api/vitals').then(drawVitals)},2000);
setInterval(()=>{if(!document.getElementById('t-server').classList.contains('hidden'))refreshLog()},1500);
document.getElementById('t-vitals').classList.contains('hidden')||j('/api/vitals').then(drawVitals);
</script></body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
