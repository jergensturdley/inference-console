#!/usr/bin/env python3
"""Metric-path tests for panel.py (GGUF parse -> VRAM fit, server status, telemetry, bench).

panel.py targets a Linux host, so these tests stub the host-specific probes (nvidia-smi,
/proc, llama-server /health) and point HOME at a temp dir. Runnable either way:

    python3 tests/test_panel_metrics.py
    python3 -m pytest tests/test_panel_metrics.py
"""
import json
import os
import struct
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOME_TMP = tempfile.mkdtemp(prefix="lfm-panel-home-")
os.environ["HOME"] = HOME_TMP          # must be set before panel is imported
sys.path.insert(0, str(REPO))

import httpx                                          # noqa: E402
import panel                                          # noqa: E402

MODELS_DIR = Path(HOME_TMP) / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
panel.CFG["model_dirs"] = [str(MODELS_DIR)]

VIT_24G = {"gpu": {"used_mb": 1000, "total_mb": 24576, "util": 5, "temp": 50,
                  "name": "Test GPU"},
           "mem": {"total_gb": 32.0, "used_gb": 10.0},
           "cpu": {"percent": 10, "cores": 16}}


def gguf(path: Path, arch="qwen3moe", kv=None):
    """Write a minimal GGUF v3 header. `kv` entries are (key, type, payload)."""
    def s(v):
        b = v.encode()
        return struct.pack("<Q", len(b)) + b

    def u32(v):
        return struct.pack("<I", v)

    def arr(vals):                      # array payload = element type + count + values
        return struct.pack("<I", 4) + struct.pack("<Q", len(vals)) + b"".join(u32(v) for v in vals)

    pairs = [
        s("general.architecture") + struct.pack("<I", 8) + s(arch),
        s("general.file_type") + struct.pack("<I", 4) + u32(10),
        s(f"{arch}.embedding_length") + struct.pack("<I", 4) + u32(2048),
        s(f"{arch}.attention.head_count") + struct.pack("<I", 4) + u32(32),
        s(f"{arch}.block_count") + struct.pack("<I", 4) + u32(48),
        s(f"{arch}.context_length") + struct.pack("<I", 4) + u32(40960),
    ]
    # head_count_kv / key_length / value_length are ARRAY-typed in real GGUF files
    pairs.append(s(f"{arch}.attention.head_count_kv") + struct.pack("<I", 9) + arr([4]))
    pairs.append(s(f"{arch}.attention.key_length") + struct.pack("<I", 9) + arr([128]))
    pairs.append(s(f"{arch}.attention.value_length") + struct.pack("<I", 9) + arr([128]))
    if arch.endswith("moe"):
        pairs.append(s(f"{arch}.expert_count") + struct.pack("<I", 4) + u32(128))
        pairs.append(s(f"{arch}.expert_used_count") + struct.pack("<I", 4) + u32(8))
    if kv:
        for k, t, p in kv:
            pairs.append(s(k) + struct.pack("<I", t) + p)
    body = b"".join(pairs)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(pairs))
    # >1 MB so scan_models() doesn't treat it as an LFS pointer
    path.write_bytes(header + body + b"\0" * (1_100_000 - len(header) - len(body)))
    return path


# ---------------------------------------------------------------- GGUF parsing
def test_gguf_array_kv_collapsed():
    p = gguf(MODELS_DIR / "synth-a.gguf")
    md = panel.gguf_metadata(p)
    arch = md["general.architecture"]
    assert md[f"{arch}.attention.head_count"] == 32
    assert md[f"{arch}.attention.head_count_kv"] == 4, "GQA kv heads must survive array typing"
    assert md[f"{arch}.attention.key_length"] == 128
    assert md[f"{arch}.attention.value_length"] == 128


def test_scan_models_maps_gqa_dims():
    gguf(MODELS_DIR / "synth-b.gguf")
    m = next(m for m in panel.scan_models()["models"] if m["name"] == "synth-b")
    assert (m["n_kv"], m["k_len"], m["v_len"]) == (4, 128, 128)
    assert m["blocks"] == 48 and m["moe"] is True


# ---------------------------------------------------------------- VRAM fit
def test_estimate_vram_kv_uses_gqa_heads():
    calls = {"vit": 0, "status": 0}

    def fake_vitals(refresh=False):
        calls["vit"] += 1
        return json.loads(json.dumps(VIT_24G))

    class St:
        def status(self):
            calls["status"] += 1
            return {"running": True}

    ov = (panel.vitals, panel.MGR, panel.SMGR)
    panel.vitals, panel.MGR, panel.SMGR = fake_vitals, St(), St()
    try:
        meta = {"size": 20 * 1024**3, "blocks": 48, "n_kv": 4, "k_len": 128, "v_len": 128,
                "moe": True, "vl": False, "mmproj": None}
        fit = panel.estimate_vram(meta, ctx=8192, ngl=99, ctk="f16", ctv="f16",
                                  parallel=1, preset="dense-gpu")
        # 48 blocks * 4 kv heads * (128*2 + 128*2) B/tok * 8192 tok
        assert fit["kv_vram_mb"] == 768, f"KV term was {fit['kv_vram_mb']} MiB, expected 768"
        assert fit["weights_vram_mb"] == 20480
        assert fit["device"] == "Test GPU"
        # Regression guard: before array KVs were parsed, n_kv fell back to n_head (32) and
        # k_len to embd/n_head (64), giving 4608 MiB of KV and a false "Exceeds memory".
        assert fit["status"] == "tight", f"GQA model mis-rated as {fit['status']} ({fit['badge']})"
        inflated = panel.estimate_vram({**meta, "n_kv": 32, "k_len": 64}, ctx=8192, ngl=99,
                                       ctk="f16", ctv="f16", parallel=1, preset="dense-gpu")
        assert inflated["kv_vram_mb"] == 4608 and inflated["status"] == "overflow"
        # A snapshot passed in must suppress per-call probing entirely.
        calls["vit"] = calls["status"] = 0
        panel.estimate_vram(meta, ctx=8192, ngl=99, ctk="q8_0", ctv="q4_0", parallel=1,
                            preset="dense-gpu", vit=VIT_24G, srv_running=True)
        assert calls["vit"] == 0 and calls["status"] == 0
    finally:
        panel.vitals, panel.MGR, panel.SMGR = ov


def test_api_models_probes_hardware_once():
    n = 25
    models = [{"path": f"/m/{i}.gguf", "name": f"m{i}", "arch": "qwen3moe", "quant": "Q4_K_M",
               "size": 20 * 1024**3, "size_h": "20 GB", "moe": True, "vl": False, "params": "",
               "ctx_len": 40960, "blocks": 48, "embd": 2048, "n_head": 32, "n_kv": 4,
               "k_len": 128, "v_len": 128, "experts": 128, "experts_used": 8, "imatrix": False,
               "mmproj": None} for i in range(n)]
    calls = {"vit": 0, "status": 0}

    def fake_vitals(refresh=False):
        calls["vit"] += 1
        return json.loads(json.dumps(VIT_24G))

    class St:
        def status(self):
            calls["status"] += 1
            return {"running": True}

    ov = (panel.scan_models, panel.vitals, panel.MGR, panel.SMGR, panel.bin_caps)
    panel.scan_models = lambda: {"models": models, "mmproj": []}
    panel.vitals = fake_vitals
    panel.MGR, panel.SMGR = St(), St()
    panel.bin_caps = lambda b: {"exists": True, "version": "test"}
    try:
        data = panel.api_models()
        assert len(data["models"]) == n
        assert calls["vit"] == 1, f"vitals() called {calls['vit']}x for one page load, expected 1"
        assert calls["status"] == 1, f"status() called {calls['status']}x, expected 1"
        assert all(m["fit"]["kv_vram_mb"] == 768 for m in data["models"])
    finally:
        panel.scan_models, panel.vitals, panel.MGR, panel.SMGR, panel.bin_caps = ov


def test_optimizer_does_not_refork_per_candidate():
    calls = {"vit": 0}

    def fake_vitals(refresh=False):
        calls["vit"] += 1
        return {"gpu": {"used_mb": 20000, "total_mb": 24576, "name": "Test GPU"},
                "mem": {"total_gb": 64.0, "used_gb": 10.0}}

    class St:
        def status(self):
            return {"running": False}

    ov = (panel.vitals, panel.MGR, panel.SMGR)
    panel.vitals, panel.MGR, panel.SMGR = fake_vitals, St(), St()
    try:
        meta = {"size": 40 * 1024**3, "blocks": 64, "n_kv": 8, "k_len": 128, "v_len": 128,
                "moe": False, "vl": False, "mmproj": None, "ctx_len": 131072}
        rec = panel.recommend_model_config(meta, "balanced")
        assert calls["vit"] == 1, f"optimizer forked nvidia-smi {calls['vit']} times, expected 1"
        assert rec["fit"]["status"] in ("fit", "tight", "overflow")
        assert 0 <= rec["ngl"] <= 64
    finally:
        panel.vitals, panel.MGR, panel.SMGR = ov


# ---------------------------------------------------------------- launch flags
def test_threads_zero_means_preset_default():
    mp = MODELS_DIR / "launch-me.gguf"
    gguf(mp, arch="qwen2")
    meta = {"path": str(mp), "name": mp.stem, "moe": False, "vl": False, "blocks": 48}
    ov = (panel.scan_models, panel.bin_caps, panel.igpu_device)
    panel.scan_models = lambda: {"models": [meta], "mmproj": []}
    panel.bin_caps = lambda b: {"exists": True, "metrics": True, "fa_value": True,
                                "ctk": True, "mlock": True, "no_mmap": True,
                                "parallel": True, "ncmoe": True}
    panel.igpu_device = lambda refresh=False: (None, 0)
    try:
        args, _, _ = panel.MGR.build_args(str(mp), preset="dense-gpu", ctx=8192, threads=0)
        line = " ".join(args)
        assert "-t 0" not in line, f"-t 0 leaked into the launch line: {line}"
        assert "-t 8" in line, f"preset default threads missing: {line}"
        args, _, _ = panel.MGR.build_args(str(mp), preset="dense-gpu", ctx=8192, threads=12)
        assert "-t 12" in " ".join(args)
        args, _, _ = panel.MGR.build_args(str(mp), preset="moe-split", ctx=8192, threads=None)
        assert "-t 16" in " ".join(args)
    finally:
        panel.scan_models, panel.bin_caps, panel.igpu_device = ov


# ---------------------------------------------------------------- server status
class Resp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code, self._json, self.text = status_code, json_data, text

    def json(self):
        return self._json


def test_status_exposes_idle_countdown_fields():
    real_get = httpx.get
    httpx.get = lambda url, **kw: Resp(200, {"status": "ok"})
    ov = (panel.CFG["idle_unload_min"], panel.MGR.info, panel.MGR.last_active)
    try:
        panel.CFG["idle_unload_min"] = 10
        panel.MGR.info = {"pid": os.getpid(), "port": 8830, "alias": "t", "model": "/m.gguf",
                          "preset": "dense-gpu", "ctx": 8192, "started_at": __import__("time").time()}
        panel.MGR.last_active = __import__("time").time() - 60
        st = panel.MGR.status()
        assert st["running"] is True
        assert st["idle_unload_min"] == 10
        assert 530 < st["idle_remaining_s"] <= 600, st
        assert st["idle_unload_in"] == st["idle_remaining_s"], "legacy key must stay in sync"
    finally:
        httpx.get = real_get
        panel.CFG["idle_unload_min"], panel.MGR.info, panel.MGR.last_active = ov


def test_stop_handles_foreign_pid():
    import signal as _sig
    real_kill = os.kill
    boom = {"n": 0}

    def kill(pid, sig):
        boom["n"] += 1
        if sig == _sig.SIGTERM:
            raise PermissionError()
    os.kill = kill
    ov = panel.MGR.info
    panel.MGR.info = {"pid": 424242, "alias": "foreign", "port": 8830}
    try:
        raised = None
        try:
            panel.MGR.stop()
        except panel.HTTPException as e:
            raised = e
        assert raised is not None and raised.status_code == 403, "foreign pid must 403, not 500"
        assert panel.MGR.info == {}, "panel must stop tracking a process it cannot signal"
    finally:
        os.kill = real_kill
        panel.MGR.info = ov


def test_avalon_stop_survives_foreign_pid():
    """Avalon's killpg fallback must not 500 on a process it cannot signal."""
    real = (os.kill, os.killpg, os.getpgid)
    os.getpgid = lambda pid: (_ for _ in ()).throw(PermissionError())
    def kill(pid, sig):
        raise PermissionError()
    os.kill = kill
    os.killpg = lambda pgid, sig: (_ for _ in ()).throw(PermissionError())
    ov = panel.SMGR.info
    panel.SMGR.info = {"pid": 424242, "profile": "flagship", "started_at": __import__("time").time()}
    try:
        out = panel.SMGR.stop()          # must return, not raise
        assert isinstance(out, dict)
        assert panel.SMGR.info == {}
    finally:
        os.kill, os.killpg, os.getpgid = real
        panel.SMGR.info = ov


# ---------------------------------------------------------------- telemetry
SLOTS = [{"id": 0, "is_processing": True, "n_ctx": 8192, "n_past": 2048},
         {"id": 1, "is_processing": False, "n_ctx": 8192, "n_past": 0}]
METRICS = """# HELP llama_generated_tokens_total
llama:tokens_predicted_total 12345
llama:tokens_predicted_seconds_total 1520.5
llama:tokens_prompted_total 6789
llama:tokens_prompted_seconds_total 301.2
llama:kv_cache_usage_ratio 0.25
llama:requests_processing 1
llama:requests_deferred 2
"""


def test_telemetry_parses_slots_and_metrics():
    real_get = httpx.get

    def fake_get(url, **kw):
        if url.endswith("/slots"):
            return Resp(200, json.loads(json.dumps(SLOTS)))
        if url.endswith("/metrics"):
            return Resp(200, None, METRICS)
        return Resp(200, {"status": "ok"})
    httpx.get = fake_get
    try:
        t = panel.MGR.telemetry(port=8830)
        assert t["ready"] is True and t["running"] is True
        assert t["gen_tps"] == 1520.5 and t["prompt_tps"] == 301.2
        assert t["tokens_predicted_total"] == 12345, "counter must reach the UI's total-tokens field"
        assert t["tokens_prompted_total"] == 6789
        assert t["kv_used"] == 2048 and t["kv_max"] == 16384
        assert t["kv_pct"] == 25.0
        assert t["slots_busy"] == 1 and t["slots_total"] == 2
        assert t["requests_processing"] == 1 and t["requests_deferred"] == 2
    finally:
        httpx.get = real_get


def test_telemetry_handles_server_without_metrics():
    real_get = httpx.get

    def fake_get(url, **kw):
        if url.endswith("/slots"):
            return Resp(200, json.loads(json.dumps(SLOTS)))
        if url.endswith("/metrics"):
            return Resp(404, None, "")
        return Resp(200, {"status": "ok"})
    httpx.get = fake_get
    try:
        t = panel.MGR.telemetry(port=8830)
        assert t["ready"] and t["tokens_predicted_total"] == 0
        assert t["kv_used"] == 2048
    finally:
        httpx.get = real_get


# ---------------------------------------------------------------- benchmarks
class FakeStream:
    """Mimics `with httpx.stream(...) as r` for a llama.cpp SSE completion."""
    def __init__(self, timings, n_chunks=8, piece="hello world "):
        self.timings, self.n, self.piece = timings, n_chunks, piece

    def __call__(self, method, url, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield "data: " + json.dumps({"choices": [{"delta": {"content": self.piece}}]})
        for _ in range(self.n - 1):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": self.piece}}]})
        yield "data: " + json.dumps({"choices": [{"delta": {}}], "timings": self.timings})
        yield "data: [DONE]"


def test_benchmark_run_records_metrics():
    mp = MODELS_DIR / "bench-me.gguf"
    gguf(mp, arch="qwen2")
    timings = {"prompt_n": 96, "prompt_ms": 480.0, "prompt_per_second": 200.0,
               "predicted_n": 64, "predicted_ms": 1000.0, "predicted_per_second": 64.0}
    real_stream, real_get = httpx.stream, httpx.get
    httpx.stream = FakeStream(timings)
    httpx.get = lambda url, **kw: Resp(200, {"status": "ok"})
    ov = (panel._chat_target, panel._current_llm, panel.MGR.info, panel.vitals,
          panel.BENCHMARKS_PATH)
    panel._chat_target = lambda: (8830, "")
    panel._current_llm = lambda: ("bench-model", None, None, None, "panel")
    panel.MGR.info = {"pid": os.getpid(), "port": 8830, "alias": "bench-model",
                      "model": str(mp), "preset": "dense-gpu",
                      "started_at": __import__("time").time()}
    panel.vitals = lambda refresh=False: json.loads(json.dumps(VIT_24G))
    panel.BENCHMARKS_PATH = Path(HOME_TMP) / "benchmarks-test.json"
    try:
        entry = panel.api_benchmark_run(panel.BenchRunBody(prompt_key="code", max_tokens=64))
        assert entry["source"] == "panel" and entry["alias"] == "bench-model"
        assert entry["model_path"] == str(mp)
        assert entry["prompt_tokens"] == 96 and entry["completion_tokens"] == 64
        assert entry["gen_tps"] == 64.0 and entry["prompt_tps"] == 200.0
        assert entry["tok_s"] == entry["gen_tps"] and entry["prompt_s"] == entry["prompt_tps"]
        assert entry["ttft_ms"] >= 0 and entry["total_s"] >= 0
        assert entry["vram_mb"] == 1000

        saved = json.loads(panel.BENCHMARKS_PATH.read_text())
        assert saved[0]["id"] == entry["id"], "bench result must persist for the history table"

        # and it must resurface on the model card through /api/models
        models = [{"path": str(mp), "name": mp.stem, "arch": "qwen2", "quant": "Q4_K_M",
                   "size": mp.stat().st_size, "size_h": "1 MB", "moe": False, "vl": False,
                   "params": "", "ctx_len": 4096, "blocks": 48, "embd": 2048, "n_head": 32,
                   "n_kv": 4, "k_len": 128, "v_len": 128, "experts": 0, "experts_used": 0,
                   "imatrix": False, "mmproj": None}]
        real_scan = panel.scan_models
        panel.scan_models = lambda: {"models": models, "mmproj": []}
        panel.vitals = lambda refresh=False: json.loads(json.dumps(VIT_24G))
        try:
            data = panel.api_models()
            b = data["models"][0]["bench"]
            assert b and b["gen_tps"] == 64.0 and b["prompt_tps"] == 200.0
        finally:
            panel.scan_models = real_scan
    finally:
        httpx.stream, httpx.get = real_stream, real_get
        (panel._chat_target, panel._current_llm, panel.MGR.info, panel.vitals,
         panel.BENCHMARKS_PATH) = ov


def test_benchmark_requires_ready_server():
    ov = panel._chat_target
    panel._chat_target = lambda: None
    try:
        raised = None
        try:
            panel.api_benchmark_run(panel.BenchRunBody())
        except panel.HTTPException as e:
            raised = e
        assert raised is not None and raised.status_code == 409
    finally:
        panel._chat_target = ov


# ---------------------------------------------------------------- chat persistence
def test_truncate_accepts_keep_and_drops_turns():
    import time as _t
    cid = "a" * 32
    panel.CHATS_DIR.mkdir(parents=True, exist_ok=True)
    (panel.CHATS_DIR / f"{cid}.json").write_text(json.dumps({
        "id": cid, "title": "t", "created": 0, "updated": 0, "model": "m",
        "messages": [{"role": "user", "content": "one"},
                     {"role": "assistant", "content": "two"},
                     {"role": "user", "content": "three"},
                     {"role": "assistant", "content": "four"}]}))
    out = panel.api_chat_truncate(cid, panel.TruncateChatBody(keep=1))
    assert out["msgs"] == 1
    msgs = json.loads((panel.CHATS_DIR / f"{cid}.json").read_text())["messages"]
    assert [m["content"] for m in msgs] == ["one"], "edit/resend must drop the discarded turns"
    panel.api_chat_truncate(cid, panel.TruncateChatBody(keep=1, content="edited"))
    msgs = json.loads((panel.CHATS_DIR / f"{cid}.json").read_text())["messages"]
    assert msgs[0]["content"] == "edited" and msgs[0]["ts"] >= _t.time() - 5


# ---------------------------------------------------------------- avalon tuning
def test_avalon_tune_preview_does_not_persist():
    draft = Path(HOME_TMP) / "draft.gguf"
    draft.write_bytes(b"x" * 16)
    ov = (panel.AVALON_DRAFT, panel.CFG.get("avalon_tune"), panel.SMGR.info)
    panel.AVALON_DRAFT = draft
    panel.SMGR.info = {}
    try:
        panel.CFG["avalon_tune"] = {}
        r = panel.api_avalon_tune(panel.AvalonTuneBody(profile="flagship", draft_n=0, preview=True))
        assert panel.CFG["avalon_tune"] == {}, "preview must not write the saved tune"
        assert r["tune"]["draft_n"] == 0, "0 is a value for draft_n (drafting off)"
        assert "--spec-draft-n-max" not in r["cmd"], f"draft_n=0 must disable drafting: {r['cmd']}"

        panel.api_avalon_tune(panel.AvalonTuneBody(profile="flagship", draft_n=4, ubd=384))
        assert panel.CFG["avalon_tune"]["flagship"] == {"draft_n": 4, "ubd": 384}
        cmd = " ".join(panel.SMGR.build_args("flagship"))
        assert "--spec-draft-n-max 4" in cmd and "-ubd 384" in cmd

        # select-backed fields use 0 = "profile default" and must not be stored
        panel.CFG["avalon_tune"] = {}
        panel.api_avalon_tune(panel.AvalonTuneBody(profile="flagship", ctx=0, parallel=0))
        assert panel.CFG["avalon_tune"]["flagship"] == {}
        args = " ".join(panel.SMGR.build_args("flagship"))
        assert f"-c {panel.AVALON_PROFILES['flagship']['defaults']['ctx']}" in args
    finally:
        panel.AVALON_DRAFT, panel.SMGR.info = ov[0], ov[2]
        panel.CFG["avalon_tune"] = ov[1]


def test_profiles_expose_saved_tune():
    ov = panel.CFG.get("avalon_tune")
    panel.CFG["avalon_tune"] = {"flagship": {"draft_n": 0}}
    try:
        out = panel.api_avalon_profiles()
        flagship = next(p for p in out if p["key"] == "flagship")
        assert flagship["saved"] == {"draft_n": 0}, "tune form restores `saved`, not `defaults`"
        assert flagship["defaults"]["draft_n"] == 0
        prefast = next(p for p in out if p["key"] == "prefast")
        assert prefast["saved"] == {}
    finally:
        panel.CFG["avalon_tune"] = ov


# ---------------------------------------------------------------- host vitals
def test_vitals_reads_real_host_metrics():
    """Runs unstubbed: exercises the real macOS fallbacks (no /proc, no nvidia-smi)."""
    v = panel._vitals_now()
    assert "mem" in v and v["mem"]["total_gb"] > 0
    assert v["mem"]["used_gb"] <= v["mem"]["total_gb"]
    assert "cpu" in v and v["cpu"]["cores"] > 0 and 0 <= v["cpu"]["percent"] <= 100
    assert "disk" in v and v["disk"]["home"]["free_gb"] >= 0


def test_vitals_memoized_across_calls():
    ov = panel._VITALS_TTL
    real = panel._vitals_now
    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return json.loads(json.dumps(VIT_24G))
    panel._vitals_now = counting
    panel._vitals_cache["data"] = None
    try:
        for _ in range(20):
            panel.vitals()
        assert calls["n"] == 1, f"20 vitals() calls forked {calls['n']} samples, expected 1"
        panel.vitals(refresh=True)
        assert calls["n"] == 2, "refresh=True must bypass the cache"
    finally:
        panel._vitals_now, panel._VITALS_TTL = real, ov
        panel._vitals_cache["data"] = None


def test_sudo_rejects_empty_password():
    assert panel._sudo_ok("") is False


def test_agent_toml_invalid_leaves_file_untouched():
    """A TOML error must fail before the write, not after leaving the config mangled."""
    real_home = panel.HOME
    h = Path(tempfile.mkdtemp(prefix="lfm-agents-"))
    panel.HOME = h
    try:
        p = h / ".codex" / "config.toml"
        p.parent.mkdir(parents=True)
        broken = "model = o3\n[this is not valid toml\n"
        p.write_text(broken)
        raised = None
        try:
            panel._w_codex("model-a", "http://127.0.0.1:8830", 8192)
        except Exception as e:
            raised = e
        assert raised is not None, "invalid TOML must raise, not silently write"
        assert p.read_text() == broken, "user config must be untouched when the merge is invalid"

        # a valid config still gains the provider block
        p.write_text('model = "o3"\n')
        panel._w_codex("model-a", "http://127.0.0.1:8830", 8192)
        after = p.read_text()
        assert "model_providers.lfm-panel" in after and 'model = "o3"' in after
        import tomllib
        tomllib.loads(after)
    finally:
        panel.HOME = real_home


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and isinstance(f, types.FunctionType)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
