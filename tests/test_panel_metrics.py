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
import threading
import time
import unittest
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOME_TMP = tempfile.mkdtemp(prefix="inference-console-home-")
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


def gguf(path: Path, arch="qwen3moe", kv=None, file_type=10):
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
        s("general.file_type") + struct.pack("<I", 4) + u32(file_type),
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


# ------------------------------------------------------- general.file_type labels
# Pinned from gguf-py 0.19.0 LlamaFileType, MOSTLY_/ALL_ stripped. Unconditional so the
# check runs in any venv; the optional gguf cross-check below catches this fixture drifting
# from the spec. Ids 4-6 and 33-35 are unassigned upstream; 1024 (GUESSED) is a sentinel.
FILE_TYPE_SPEC = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}


def test_file_type_table_matches_pinned_spec():
    assert panel._file_type == FILE_TYPE_SPEC, (
        "mismatch: " + ", ".join(
            f"id {i}: panel={panel._file_type.get(i)!r} spec={v!r}"
            for i, v in sorted(FILE_TYPE_SPEC.items()) if panel._file_type.get(i) != v)
        + " | extra: " + repr({i: v for i, v in panel._file_type.items()
                               if i not in FILE_TYPE_SPEC}))


def test_file_type_table_has_no_duplicate_labels():
    """The pre-fix table listed IQ4_XS twice (ids 26 and 30), which is how the off-by-one
    cascade past 26 went unnoticed: a duplicated label looks like a valid row."""
    labels = list(panel._file_type.values())
    dupes = {x for x in labels if labels.count(x) > 1}
    assert not dupes, f"duplicate quant labels: {sorted(dupes)}"
    assert "?" not in labels and "" not in labels


def test_file_type_table_covers_every_assigned_id():
    missing = sorted(set(FILE_TYPE_SPEC) - set(panel._file_type))
    assert not missing, f"ids with no label would render '?' in the library: {missing}"
    # the ids the old table got wrong, pinned individually so a regression names itself
    for i, want in ((26, "IQ3_S"), (27, "IQ3_M"), (28, "IQ2_S"), (29, "IQ2_M"),
                    (31, "IQ1_M"), (32, "BF16"), (38, "MXFP4_MOE"), (39, "NVFP4")):
        assert panel._file_type.get(i) == want, f"id {i}: {panel._file_type.get(i)} != {want}"


def test_file_type_label_reaches_a_model_with_no_filename_token():
    """The table is only consulted when the filename carries no quant token, which is the
    path that was showing '?' for NVFP4 and a wrong label for BF16."""
    nv = MODELS_DIR / "plain-weights-nvfp4-a.gguf"     # no quant token in the stem
    gguf(nv, arch="llama", file_type=39)
    bf = MODELS_DIR / "plain-weights-bf16-b.gguf"
    gguf(bf, arch="llama", file_type=32)
    try:
        models = {m["name"]: m for m in panel.scan_models()["models"]}
        assert models[nv.stem]["quant"] == "NVFP4", models[nv.stem]["quant"]
        assert models[bf.stem]["quant"] == "BF16", models[bf.stem]["quant"]
        # a filename token still wins over the table, unchanged by this fix. The quant
        # regexes are case-sensitive uppercase, so this needs the conventional spelling.
        tok = MODELS_DIR / "token-wins-Q4_K_M.gguf"
        gguf(tok, arch="llama", file_type=39)
        assert next(m for m in panel.scan_models()["models"]
                    if m["name"] == tok.stem)["quant"] == "Q4_K_M"
    finally:
        panel._model_cache.clear()


def test_file_type_table_matches_gguf_constants_when_importable():
    """Optional cross-check so the pinned fixture itself cannot silently drift from spec."""
    try:
        from gguf.constants import LlamaFileType
    except ImportError:
        return          # gguf is not a panel dependency; the fixture test above still ran
    import re as _re
    spec = {}
    for m in LlamaFileType:
        if m.value == 1024:            # GUESSED sentinel, not a quant label
            continue
        spec[m.value] = _re.sub(r"^(MOSTLY_|ALL_)", "", m.name)
    assert panel._file_type == spec, (
        f"panel table disagrees with gguf-py {spec!r}: "
        + repr({i: (panel._file_type.get(i), spec[i]) for i in set(spec) | set(panel._file_type)
                if panel._file_type.get(i) != spec.get(i)}))


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


class _FakeProc(type(Path())):
    """Path subclass mapping a few absolute paths onto in-memory content, so the
    Linux-only /proc branches in panel.py can be exercised on macOS. Anything not mapped
    delegates to the real filesystem."""
    files = {}      # str path -> bytes
    dirs = set()    # str path -> treated as an existing directory
    children = {}   # str dir path -> [child paths]

    def read_text(self, *a, **k):
        p = str(self)
        if p in _FakeProc.files:
            data = _FakeProc.files[p]
            return data.decode() if isinstance(data, bytes) else data
        return super().read_text(*a, **k)

    def read_bytes(self):
        p = str(self)
        return _FakeProc.files[p] if p in _FakeProc.files else super().read_bytes()

    def exists(self):
        return str(self) in _FakeProc.files or str(self) in _FakeProc.dirs or super().exists()

    def is_dir(self):
        return str(self) in _FakeProc.dirs

    def iterdir(self):
        p = str(self)
        if p in _FakeProc.children:
            return iter(_FakeProc(c) for c in _FakeProc.children[p])
        return super().iterdir()


_MEMINFO = b"MemTotal:       32768000 kB\nMemAvailable:   16384000 kB\n"


def _stat(user, idle):
    # cpu  user nice system idle iowait irq softirq
    return f"cpu  {user} 200 300 {idle} 100 10 20\n".encode()


def test_vitals_first_cpu_sample_reports_a_value_instead_of_omitting_the_key():
    """Before the fix the first sample after start had no /proc/stat delta window, so the
    cpu key was absent and the CPU card did not render on the first poll."""
    real_path, real_prev = panel.Path, panel._prev_cpu
    _FakeProc.files = {"/proc/meminfo": _MEMINFO, "/proc/stat": _stat(1000, 5000)}
    panel.Path = _FakeProc
    panel._prev_cpu = None
    try:
        v = panel._vitals_now()
        assert "cpu" in v, "first sample must still emit a cpu entry"
        assert v["cpu"]["percent"] == 0, v["cpu"]
        assert v["cpu"]["cores"] > 0
        # 32768000 kB / 1048576 = 31.25 -> 31.2 (banker's rounding); proves the /proc
        # branch ran rather than the macOS fallback (which reports 8.0 / 16.0)
        assert v["mem"] == {"used_gb": 15.6, "total_gb": 31.2}, v["mem"]
    finally:
        panel.Path, panel._prev_cpu = real_path, real_prev
        _FakeProc.files = {}


def test_vitals_second_cpu_sample_reports_the_real_delta():
    """The first sample must seed _prev_cpu so the next one reports real load, not 0."""
    real_path, real_prev = panel.Path, panel._prev_cpu
    _FakeProc.files = {"/proc/meminfo": _MEMINFO, "/proc/stat": _stat(1000, 5000)}
    panel.Path = _FakeProc
    panel._prev_cpu = None
    try:
        assert panel._vitals_now()["cpu"]["percent"] == 0
        # user +500, idle +100 => 500/600 of the sampled time
        _FakeProc.files["/proc/stat"] = _stat(1500, 5100)
        second = panel._vitals_now()
        assert second["cpu"]["percent"] == 83, second["cpu"]
    finally:
        panel.Path, panel._prev_cpu = real_path, real_prev
        _FakeProc.files = {}


def test_kill_orphans_survives_foreign_owned_pids():
    """A foreign-owned avalon pid raises EPERM, not ESRCH. Before the fix that propagated
    through stop() to the UI as a 500."""
    real_path = panel.Path
    real_kill = os.kill
    real_free = panel.AvalonManager.__dict__["_port_free"]
    pids = [424241, 424242]
    _FakeProc.dirs = {"/proc"}
    _FakeProc.children = {"/proc": [f"/proc/{p}" for p in pids]}
    _FakeProc.files = {f"/proc/{p}/cmdline": b"/home/u/avalon/build/bin/llama-server\0-m\0x.gguf\0"
                       for p in pids}
    panel.Path = _FakeProc
    os.kill = lambda pid, sig: (_ for _ in ()).throw(PermissionError())
    panel.AvalonManager._port_free = staticmethod(lambda: True)
    try:
        assert panel.AvalonManager._kill_orphans() is True, "foreign victims were discovered"
    finally:
        panel.Path, os.kill = real_path, real_kill
        panel.AvalonManager._port_free = real_free
        _FakeProc.dirs = _FakeProc.children = _FakeProc.files = {}


def test_stop_reports_honest_failure_when_a_foreign_orphan_keeps_the_port():
    """Swallowing EPERM must not become a fake success: the port check still decides."""
    real_free = panel.AvalonManager.__dict__["_port_free"]
    real_orphans = panel.AvalonManager.__dict__["_kill_orphans"]
    real_info = panel.SMGR.info
    panel.AvalonManager._port_free = staticmethod(lambda: False)
    panel.AvalonManager._kill_orphans = staticmethod(lambda: True)
    panel.SMGR.info = {}
    n_log = len(panel.SMGR.logbuf)
    try:
        assert panel.SMGR.stop() == {"stopped": False}
        assert "still busy" in list(panel.SMGR.logbuf)[-1], list(panel.SMGR.logbuf)[-1]
    finally:
        panel.AvalonManager._port_free = real_free
        panel.AvalonManager._kill_orphans = real_orphans
        panel.SMGR.info = real_info
        while len(panel.SMGR.logbuf) > n_log:
            panel.SMGR.logbuf.pop()


def test_sudo_rejects_empty_password():
    assert panel._sudo_ok("") is False


# ------------------------------------------------- fit percentage / badge agreement
def test_fit_pct_and_badge_share_a_denominator():
    """pct used to divide by budget_mb while the badge divided by usable_mb, so a
    "Tight fit" badge could sit next to a smaller percentage than it implies."""
    ov = (panel.vitals, panel.MGR, panel.SMGR, panel.CFG["vram_reserve_mb"])

    def fake_vitals(refresh=False):
        return {"gpu": {"used_mb": 0, "total_mb": 24576, "name": "T"},
                "mem": {"total_gb": 32.0, "used_gb": 1.0}}

    class St:
        def status(self):
            return {"running": True}

    panel.vitals, panel.MGR, panel.SMGR = fake_vitals, St(), St()
    panel.CFG["vram_reserve_mb"] = 512
    try:
        saw_denominators_differ = False
        for size_mb in (500, 2000, 6000, 10000, 15000, 20000, 25000, 32000, 48000):
            meta = {"size": size_mb * 1048576, "blocks": 8, "n_kv": 4, "k_len": 128,
                    "v_len": 128, "moe": False, "vl": False, "mmproj": None}
            f = panel.estimate_vram(meta, ctx=1024, ngl=99, ctk="f16", ctv="f16",
                                    parallel=1, preset="dense-gpu")
            assert f["budget_mb"] > f["usable_mb"], "reserve must separate the two"
            assert f["pct"] == min(999, round(100 * f["total_vram_mb"] / f["usable_mb"])), \
                f"pct must be measured against usable_mb at {size_mb} MB"
            # The badge thresholds are 0.85 and 1.0 of usable, so pct must land on the
            # same side of those lines as the badge it is displayed next to.
            if f["status"] == "fit":
                assert f["pct"] <= 85, f"fit badge with pct {f['pct']}"
            elif f["status"] == "tight":
                assert 85 < f["pct"] <= 100, f"tight badge with pct {f['pct']}"
            else:
                assert f["pct"] > 100, f"overflow badge with pct {f['pct']}"
            if round(100 * f["total_vram_mb"] / f["budget_mb"]) != f["pct"]:
                saw_denominators_differ = True
        assert saw_denominators_differ, "test is vacuous unless the old denominator differs"
    finally:
        panel.vitals, panel.MGR, panel.SMGR, panel.CFG["vram_reserve_mb"] = ov


# ------------------------------------------------- vision-language detection
def test_vl_name_detection():
    yes = ["qwen2.5-vl-7b-instruct", "internvl3-8b", "Qwen3VL-8B", "llava-v1.6-mistral",
           "pixtral-12b-2409", "qwen2.5-omni-7b", "minicpm-v-2_6", "moe-vl-mixtral",
           "phi3.5-vision-instruct", "molmo-7b-d"]
    no = ["omnidoc-bench-parser", "meta-llama-3.1-8b-instruct", "deepseek-r1-q4_k_m",
          "revaluation-notes-13b", "volume-knob-8b", "qwen3-30b-a3b", "nova-bench-v2"]
    for name in yes:
        assert panel.VL_NAME_RE.search(name), f"{name} should be vision-language"
    for name in no:
        assert not panel.VL_NAME_RE.search(name), f"{name} should not be vision-language"


def test_loose_mmproj_sidecar_does_not_flag_every_model_as_vl():
    """find_mmproj() matches on name-token overlap, so a sibling that merely shares a
    prefix "finds" a sidecar. Using that as a vl signal tagged every model VL and flipped
    the `auto` preset to vision; the name marker is the signal instead."""
    base = MODELS_DIR / "plain-dense-weights.gguf"
    sibling = MODELS_DIR / "plain-dense-other.gguf"   # shares 2 of 5 tokens with the sidecar
    for p in (base, sibling):
        if not p.exists():
            gguf(p, arch="llama")
    sidecar = MODELS_DIR / "plain-dense-weights-mmproj-f16.gguf"
    sidecar.write_bytes(b"GGUF" + b"\0" * 1_100_000)
    try:
        assert panel.find_mmproj(str(base)) == str(sidecar)
        assert panel.find_mmproj(str(sibling)) == str(sidecar), \
            "overlap is loose by design: a prefix-sharing sibling also matches"
        models = {m["name"]: m for m in panel.scan_models()["models"]}
        for name in (base.stem, sibling.stem):
            m = models[name]
            assert m["vl"] is False, f"{name} must not be tagged VL from a loose sidecar"
            assert m["mmproj"], "the sidecar is still reported for explicit vision launches"
    finally:
        sidecar.unlink(missing_ok=True)
        panel._model_cache.clear()


def test_model_cache_prunes_entries_for_gone_files():
    panel.scan_models()
    stale = ("/nonexistent/dir/gone.gguf", 1, 2)
    panel._model_cache[stale] = {"path": stale[0]}
    panel.scan_models()
    assert stale not in panel._model_cache, "cache must drop files that no longer exist"


# ------------------------------------------------- metrics counter self-diagnosis
def test_telemetry_reports_counter_names_the_build_emits():
    """A build whose counter names the panel doesn't know must be diagnosable from the
    API rather than by scraping /metrics on the host."""
    body = (METRICS + "llama:tokens_generated_total 999\n"
            "llama:some_brand_new_counter 3\n")
    real_get = httpx.get

    def fake_get(url, **kw):
        if url.endswith("/slots"):
            return Resp(200, json.loads(json.dumps(SLOTS)))
        if url.endswith("/metrics"):
            return Resp(200, None, body)
        return Resp(200, {"status": "ok"})
    httpx.get = fake_get
    port = 8899           # fresh port: the name cache keeps the first scrape per port
    panel._METRIC_KEYS.pop(port, None)
    try:
        t = panel.MGR.telemetry(port=port)
        assert "llama:tokens_generated_total" in t["metric_keys"], t["metric_keys"]
        assert "llama:some_brand_new_counter" in t["metric_keys"]
        assert "llama:tokens_predicted_total" in t["metric_keys"]
        assert t["tokens_predicted_total"] == 12345
        assert panel._METRIC_KEYS[port] == sorted(set(body.split() and [
            "llama:tokens_predicted_total", "llama:tokens_predicted_seconds_total",
            "llama:tokens_prompted_total", "llama:tokens_prompted_seconds_total",
            "llama:kv_cache_usage_ratio", "llama:requests_processing",
            "llama:requests_deferred", "llama:tokens_generated_total",
            "llama:some_brand_new_counter"]))
    finally:
        httpx.get = real_get
        panel._METRIC_KEYS.pop(port, None)


def test_counter_family_matches_spelling_variants():
    for name in ("llama:tokens_predicted_total", "llama_tokens_predicted_total",
                 "tokens_predicted_total", "llama:tokens_prompted_total"):
        m = panel._METRIC_TOK_TOTAL_RE.match(name)
        assert m, f"{name} must match the counter family"
        assert m.group(1) == ("predicted" if "predicted" in name else "prompted")
    assert not panel._METRIC_TOK_TOTAL_RE.match("llama:tokens_predicted_seconds_total")


# ------------------------------------------------- download job state
def test_downloads_list_is_json_safe_and_lock_guarded():
    """_thread must not leak into the API, and listing while a pull registers a job must
    not raise 'dictionary changed size during iteration'."""
    did = "repo/concurrent.gguf"
    stop = threading.Event()
    holder = threading.Thread(target=stop.wait, daemon=True)
    holder.start()
    panel.DOWNLOADS[did] = {"id": did, "repo": "repo", "file": "concurrent.gguf",
                            "path": "concurrent.gguf", "url": "http://x",
                            "dest": "/tmp/concurrent.gguf", "done": 0, "total": 1,
                            "status": "downloading", "cancel": False,
                            "delete_part": False, "error": "", "_thread": holder}
    errors = []

    def churn():
        try:
            for i in range(300):
                extra = f"repo/churn{i}.gguf"
                panel.DOWNLOADS[extra] = {"id": extra, "status": "queued", "cancel": False,
                                          "dest": f"/tmp/c{i}.gguf"}
                if i % 3 == 0:
                    panel.DOWNLOADS.pop(extra, None)
        except Exception as e:                       # pragma: no cover - the point is it can't
            errors.append(e)

    t = threading.Thread(target=churn)
    t.start()
    try:
        for _ in range(200):
            listed = panel.api_downloads()
            json.dumps(listed)
            assert all("_thread" not in d for d in listed), "Thread must not reach the API"
    finally:
        stop.set()
        t.join(5)
        for k in [k for k in panel.DOWNLOADS if k.startswith("repo/churn")]:
            panel.DOWNLOADS.pop(k, None)
        panel.DOWNLOADS.pop(did, None)
    assert not errors, f"concurrent mutation escaped the lock: {errors!r}"


class _ChunkStream:
    """httpx.stream stand-in that yields chunks until the job is cancelled."""

    def __init__(self, job, chunk=b"x" * 512, rounds=400):
        self.job, self.chunk, self.rounds = job, chunk, rounds

    def __call__(self, method, url, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def status_code(self):
        return 200

    @property
    def headers(self):
        return {"content-length": "200000"}

    def raise_for_status(self):
        return None

    def iter_bytes(self, n):
        for _ in range(self.rounds):
            if self.job["cancel"]:
                raise RuntimeError("paused")
            time.sleep(0.004)
            yield self.chunk


def test_cancel_while_worker_is_live_defers_the_delete():
    d = MODELS_DIR / "dl"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / "cancel-me.gguf"
    part = Path(str(dest) + ".part")
    part.write_bytes(b"")
    job = {"id": "r/cancel-me.gguf", "repo": "r", "file": "cancel-me.gguf",
           "path": "cancel-me.gguf", "url": "http://x", "dest": str(dest),
           "done": 0, "total": 200000, "status": "queued", "cancel": False,
           "delete_part": False, "error": ""}
    real_stream = httpx.stream
    httpx.stream = _ChunkStream(job)
    panel.DOWNLOADS[job["id"]] = job
    try:
        th = threading.Thread(target=panel._download_worker, args=(job["id"],), daemon=True)
        job["_thread"] = th
        th.start()
        for _ in range(200):
            if job["status"] == "downloading":
                break
            time.sleep(0.01)
        r = panel.api_cancel(panel.CancelBody(id=job["id"], delete_part=True))
        assert r["deferred_to_worker"] is True, r
        assert job["id"] in panel.DOWNLOADS, "a live worker must not be yanked mid-write"
        assert part.exists(), "the .part the worker holds open must survive for now"
        th.join(10)
        assert job["id"] not in panel.DOWNLOADS, "worker must drop the entry on cancel+delete"
        assert not part.exists(), "worker must unlink the .part it owns"
        assert job["status"] == "paused", job["status"]
        assert not dest.exists(), "a cancelled download must never be published"
    finally:
        httpx.stream = real_stream
        panel.DOWNLOADS.pop(job["id"], None)
        part.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)


def test_resume_clears_cancel_on_a_live_job():
    """The old code returned early on status=="downloading" without clearing cancel,
    leaving the job stuck showing "downloading" forever."""
    did = "r/resume-me.gguf"
    stop = threading.Event()
    holder = threading.Thread(target=stop.wait, daemon=True)
    holder.start()
    panel.DOWNLOADS[did] = {"id": did, "repo": "r", "file": "resume-me.gguf",
                            "path": "resume-me.gguf", "url": "http://x",
                            "dest": "/tmp/resume-me.gguf", "done": 1, "total": 2,
                            "status": "downloading", "cancel": True,
                            "delete_part": True, "error": "boom", "_thread": holder}
    try:
        r = panel.api_resume(panel.CancelBody(id=did))
        assert r["resumed_live"] is True, r
        job = panel.DOWNLOADS[did]
        assert job["cancel"] is False, "cancel must clear or the worker keeps aborting"
        assert job["delete_part"] is False and job["error"] == ""
        assert job["status"] == "downloading", "a live job must not be re-queued"
    finally:
        stop.set()
        holder.join(5)
        panel.DOWNLOADS.pop(did, None)


def test_pull_reuses_a_live_job_for_the_same_destination():
    did = "r:same.gguf"          # api_hf_pull keys jobs as "{repo}:{filename}"
    stop = threading.Event()
    holder = threading.Thread(target=stop.wait, daemon=True)
    holder.start()
    panel.DOWNLOADS[did] = {"id": did, "repo": "r", "file": "same.gguf",
                            "path": "same.gguf", "url": "http://x",
                            "dest": "/tmp/same.gguf", "done": 0, "total": 1,
                            "status": "downloading", "cancel": False,
                            "delete_part": False, "error": "", "_thread": holder}
    real_dirs = panel.CFG["model_dirs"]
    panel.CFG["model_dirs"] = [str(MODELS_DIR)]
    try:
        r = panel.api_hf_pull(panel.PullBody(repo="r", files=["same.gguf"],
                                             sizes={"same.gguf": 10}))
        assert r["started"] == [did]
        assert panel.DOWNLOADS[did]["_thread"] is holder, \
            "a second pull must attach to the running job, not race a second writer"
    finally:
        panel.CFG["model_dirs"] = real_dirs
        stop.set()
        holder.join(5)
        panel.DOWNLOADS.pop(did, None)


# ------------------------------------------------- launch secret redaction
def test_launch_secrets_are_redacted_in_every_exposed_surface():
    assert panel._redact_cmd("run --api-key SECRETTOKEN -c 8192") == \
        "run --api-key *** -c 8192"
    assert panel._redact_cmd("run --api-key=SECRETTOKEN") == "run --api-key=***"
    ov_key, ov_lan = panel.CFG.get("avalon_api_key"), panel.CFG.get("avalon_lan")
    # build_args only appends --api-key when Avalon is LAN-exposed, which is exactly the
    # case where the key is visible in `ps` and worth keeping out of the panel's surfaces.
    panel.CFG["avalon_api_key"] = "SECRETTOKEN"
    panel.CFG["avalon_lan"] = True
    n_log = len(panel.SMGR.logbuf)
    try:
        args = panel.SMGR.build_args("flagship")
        assert "SECRETTOKEN" in args, "the real argv must still carry the key"
        cmd = " ".join(args)
        assert "SECRETTOKEN" in cmd
        for p in panel.api_avalon_profiles():
            assert "SECRETTOKEN" not in p["cmd"], p["cmd"]
            assert "SECRETTOKEN" not in p["cmd_preview"], p["cmd_preview"]
        t = panel.api_avalon_tune(panel.AvalonTuneBody(profile="flagship", preview=True))
        assert "SECRETTOKEN" not in t["cmd"] and "SECRETTOKEN" not in t["cmd_preview"]
        panel.SMGR.log("starting: " + cmd)
        assert "SECRETTOKEN" not in list(panel.SMGR.logbuf)[-1], "log line leaked the key"
        stored = panel._redact_cmd(cmd)
        assert "SECRETTOKEN" not in stored
    finally:
        panel.CFG["avalon_api_key"], panel.CFG["avalon_lan"] = ov_key, ov_lan
        while len(panel.SMGR.logbuf) > n_log:
            panel.SMGR.logbuf.pop()


def test_agent_toml_invalid_leaves_file_untouched():
    """A TOML error must fail before the write, not after leaving the config mangled."""
    real_home = panel.HOME
    h = Path(tempfile.mkdtemp(prefix="inference-console-agents-"))
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
        assert "model_providers.inference-console" in after and 'model = "o3"' in after
        import tomllib
        tomllib.loads(after)
    finally:
        panel.HOME = real_home


# ---------------------------------------------------------------- llama.cpp extra knobs
_FULL_CAPS = {"exists": True, "metrics": True, "fa_value": True, "ctk": True, "mlock": True,
              "no_mmap": True, "parallel": True, "ncmoe": True, "draft": True, "draft_max": True,
              "ot": True, "tensor_split": True, "tbatch": True, "keep": True, "jinja": True}


def test_new_llama_flags_emitted_when_supported():
    mp = MODELS_DIR / "specme.gguf"
    gguf(mp, arch="qwen2")
    draft = MODELS_DIR / "mtp-specme.gguf"
    gguf(draft, arch="qwen2")
    meta = {"path": str(mp), "name": mp.stem, "moe": False, "vl": False, "blocks": 48}
    ov = (panel.scan_models, panel.bin_caps, panel.igpu_device)
    panel.scan_models = lambda: {"models": [meta], "mmproj": []}
    panel.bin_caps = lambda b: dict(_FULL_CAPS)
    panel.igpu_device = lambda refresh=False: (None, 0)
    try:
        args, _, _ = panel.MGR.build_args(
            str(mp), preset="dense-gpu", ctx=8192, threads=8,
            draft=str(draft), draft_max=24, ot="exps=CPU", tensor_split="1 2",
            tbatch=6, keep=128)
        line = " ".join(args)
        assert f"--model-draft {draft}" in line
        assert "--draft-max 24" in line and "--draft-min 0" in line
        assert "-ot exps=CPU" in line
        assert "--tensor-split 1 2" in line          # separate argv values, not one string
        assert "--threads-batch 6" in line and "--keep 128" in line
        assert "--jinja" in line
        args, _, _ = panel.MGR.build_args(str(mp), preset="dense-gpu", ctx=8192, threads=8,
                                          jinja="off")
        assert "--jinja" not in " ".join(args)
    finally:
        panel.scan_models, panel.bin_caps, panel.igpu_device = ov


def test_new_llama_flags_suppressed_on_old_builds():
    mp = MODELS_DIR / "oldbuild.gguf"
    gguf(mp, arch="qwen2")
    draft = MODELS_DIR / "mtp-oldbuild.gguf"
    gguf(draft, arch="qwen2")
    meta = {"path": str(mp), "name": mp.stem, "moe": False, "vl": False, "blocks": 48}
    ov = (panel.scan_models, panel.bin_caps, panel.igpu_device)
    panel.scan_models = lambda: {"models": [meta], "mmproj": []}
    # A build without the newer flags must never receive them.
    panel.bin_caps = lambda b: {"exists": True, "metrics": True, "fa_value": True,
                                "ctk": True, "mlock": True, "no_mmap": True, "parallel": True}
    panel.igpu_device = lambda refresh=False: (None, 0)
    try:
        args, _, _ = panel.MGR.build_args(
            str(mp), preset="dense-gpu", ctx=8192, threads=8,
            draft=str(draft), tensor_split="1 1", tbatch=4, keep=64)
        line = " ".join(args)
        for flag in ("--model-draft", "--draft-max", "--tensor-split",
                     "--threads-batch", "--keep", "--jinja"):
            assert flag not in line, f"{flag} leaked on a build without it: {line}"
    finally:
        panel.scan_models, panel.bin_caps, panel.igpu_device = ov


def test_draft_path_validated_before_launch():
    mp = MODELS_DIR / "validatedraft.gguf"
    gguf(mp, arch="qwen2")
    meta = {"path": str(mp), "name": mp.stem, "moe": False, "vl": False, "blocks": 48}
    ov = (panel.scan_models, panel.bin_caps, panel.igpu_device)
    panel.scan_models = lambda: {"models": [meta], "mmproj": []}
    panel.bin_caps = lambda b: dict(_FULL_CAPS)
    panel.igpu_device = lambda refresh=False: (None, 0)
    try:
        try:
            panel.MGR.build_args(str(mp), preset="dense-gpu", ctx=8192,
                                 draft=str(MODELS_DIR / "nope-draft.gguf"))
            assert False, "missing draft file must raise, not launch without spec-decode"
        except panel.HTTPException as e:
            assert e.status_code == 400
    finally:
        panel.scan_models, panel.bin_caps, panel.igpu_device = ov


def test_find_draft_finds_mtp_sibling_skips_mmproj():
    d = MODELS_DIR / "draftdir"
    d.mkdir(exist_ok=True)
    model = d / "Qwen3-Foo-IQ3.gguf"
    gguf(model, arch="qwen3moe")
    mtp = d / "mtp-Qwen3-Foo-shared-IQ4.gguf"
    gguf(mtp, arch="qwen3moe")
    (d / "Qwen3-Foo-mmproj-f16.gguf").write_bytes(b"GGUF")
    assert panel.find_draft(str(model)) == str(mtp)
    plain = d / "Other-Bar.gguf"
    gguf(plain, arch="qwen2")
    assert panel.find_draft(str(plain)) == str(mtp)  # falls back to the only draft in the dir
    solo = MODELS_DIR / "draftdir" / "lonely.gguf"
    gguf(solo, arch="qwen2")
    assert panel.find_draft(str(solo)) == str(mtp)


def test_optimizer_speed_target_enables_found_draft():
    d = MODELS_DIR / "optdir"
    d.mkdir(exist_ok=True)
    mp = d / "Qwen3-Spec.gguf"
    gguf(mp, arch="qwen2")
    (d / "mtp-Qwen3-Spec.gguf").write_bytes(b"GGUF")
    meta = {"path": str(mp), "name": mp.stem, "size": 4 * 1024**3, "blocks": 48,
            "n_kv": 8, "k_len": 128, "v_len": 128, "moe": False, "vl": False,
            "mmproj": None, "ctx_len": 32768}
    class St:
        @staticmethod
        def status():
            return {"running": False}
    ov = (panel.vitals, panel.MGR, panel.SMGR)
    panel.vitals = lambda refresh=False: VIT_24G
    panel.MGR, panel.SMGR = St, St
    try:
        rec = panel.recommend_model_config(meta, "speed")
        assert rec["draft"] == str(d / "mtp-Qwen3-Spec.gguf")
        assert rec["draft_max"] == 16
        assert any("speculative" in n for n in rec["notes"])
        rec = panel.recommend_model_config(meta, "balanced")
        assert rec["draft"] == ""
    finally:
        panel.vitals, panel.MGR, panel.SMGR = ov


# ---------------------------------------------------------------- strata
def test_strata_build_args_documented_flags_only():
    b = panel.StrataLaunchBody(
        preset="mywork", family="qwen", model="iq3_xxs", model_path="/m/here",
        context="128k", kv="q4_0", parallel=4, vision="gpu", mtp="on", drafting="mtp",
        gpu="0,1", tune="quick", retune=True, access="lan", port=8123,
        no_api_key=False, new_key=True, save_preset="mywork", restart=True,
        set_=["prefill=65536", "STRATA_PIN_BY_NEED=1", "badpair"])
    args = panel.strata_build_args(b, plan=False)
    line = " ".join(args)
    for want in ("--preset mywork", "--family qwen", "--model iq3_xxs",
                 "--model-path /m/here", "--context 128k", "--kv q4_0",
                 "--parallel 4", "--vision gpu", "--mtp on", "--drafting mtp",
                 "--gpu 0,1", "--tune quick", "--retune", "--access lan", "--port 8123",
                 "--new-key", "--save-preset mywork", "--restart",
                 "--set prefill=65536", "--set STRATA_PIN_BY_NEED=1"):
        assert want in line, f"missing {want!r} in {line}"
    assert "badpair" not in line                      # malformed --set pairs are dropped
    assert "--no-tui" in args and "--yes" in args and "--output" in args
    # An empty form emits nothing but the safe non-interactive launch tail.
    empty = panel.strata_build_args(panel.StrataLaunchBody(), plan=False)
    assert empty == ["launch", "--yes", "--no-tui", "--output", "json"]
    plan = panel.strata_build_args(panel.StrataLaunchBody(preset="x"), plan=True)
    assert plan == ["launch", "--preset", "x", "--plan", "--output", "json"]
    assert "--yes" not in plan


def test_parse_ctx_words():
    assert panel._parse_ctx_words("128k") == 131072
    assert panel._parse_ctx_words("8192") == 8192
    assert panel._parse_ctx_words("1m") == 1048576
    assert panel._parse_ctx_words(" 64K ") == 65536
    assert panel._parse_ctx_words("huge") is None


def test_strata_status_without_cli_is_not_installed():
    ov = panel.CFG.get("strata_bin")
    ov_env = os.environ.get("STRATA_BIN")
    try:
        panel.CFG["strata_bin"] = str(Path(HOME_TMP) / "definitely-missing-strata")
        os.environ["STRATA_BIN"] = panel.CFG["strata_bin"]
        st = panel.strata_status()
        assert st["installed"] is False
        assert "install.sh" in st["install_cmd"]
        assert st["running"] is False
    finally:
        panel.CFG["strata_bin"] = ov
        if ov_env is None:
            os.environ.pop("STRATA_BIN", None)
        else:
            os.environ["STRATA_BIN"] = ov_env


def test_strata_key_endpoint_masks_by_default():
    ov_bin, ov_run = panel.CFG.get("strata_bin"), panel._strata_run
    stub = Path(HOME_TMP) / "strata-stub"
    try:
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
        panel.CFG["strata_bin"] = str(stub)
        panel._strata_run = lambda args, timeout=20: {
            "ok": True, "json": None, "stdout": "sk-strata-abcdefghijklmnop\n",
            "stderr": "", "error": None}
        k = panel.api_strata_key(reveal=False)
        assert k["key_set"] and k["key"] == ""
        assert "abcdefghijklmnop" not in k["masked"]
        k = panel.api_strata_key(reveal=True)
        assert k["key"] == "sk-strata-abcdefghijklmnop"
    finally:
        panel.CFG["strata_bin"] = ov_bin
        panel._strata_run = ov_run
        stub.unlink(missing_ok=True)


# ---------------------------------------------------------------- engine setup
_ASSETS = [
    {"name": "llama-b9999-bin-macos-arm64.tar.gz", "browser_download_url": "u1"},
    {"name": "llama-b9999-bin-macos-x64.tar.gz", "browser_download_url": "u2"},
    {"name": "llama-b9999-bin-ubuntu-x64.tar.gz", "browser_download_url": "u3"},
    {"name": "llama-b9999-bin-ubuntu-vulkan-x64.tar.gz", "browser_download_url": "u4"},
    {"name": "llama-b9999-bin-ubuntu-cuda-12.8-x64.tar.gz", "browser_download_url": "u5"},
    {"name": "llama-b9999-bin-win-cpu-x64.zip", "browser_download_url": "u6"},
    {"name": "llama-b9999-bin-win-vulkan-x64.zip", "browser_download_url": "u7"},
    {"name": "cudart-llama-b9999-bin-win-cuda-12.4-x64.zip", "browser_download_url": "u8"},
    {"name": "llama-b9999-ui.tar.gz", "browser_download_url": "u9"},
]


def test_llama_asset_picker_matches_platform_and_variant():
    def pick(*a, **kw):
        got = panel._pick_llama_asset(*a, **kw)
        assert got is not None, a
        return got
    a = pick(_ASSETS, "darwin", "arm64")
    assert a["name"].endswith("bin-macos-arm64.tar.gz")
    a = pick(_ASSETS, "darwin", "x86_64")
    assert a["name"].endswith("bin-macos-x64.tar.gz")
    a = pick(_ASSETS, "linux", "x86_64")
    assert a["name"].endswith("bin-ubuntu-vulkan-x64.tar.gz")
    a = pick(_ASSETS, "linux", "x86_64", variant="cpu")
    assert a["name"].endswith("bin-ubuntu-x64.tar.gz")
    a = pick(_ASSETS, "linux", "x86_64", variant="cuda")
    assert "cuda" in a["name"] and a["name"].startswith("llama-")   # never the cudart pack
    a = pick(_ASSETS, "win32", "AMD64")
    assert a["name"].endswith("bin-win-vulkan-x64.zip")
    assert panel._pick_llama_asset(_ASSETS, "plan9", "mips") is None
    assert panel._pick_llama_asset([], "darwin", "arm64") is None


def test_setup_install_requires_confirmation():
    try:
        panel.api_setup_install(panel.SetupInstallBody(engine="llama", confirm=False))
        assert False, "install without confirm must be refused"
    except panel.HTTPException as e:
        assert e.status_code == 400
    try:
        panel.api_setup_install(panel.SetupInstallBody(engine="bogus", confirm=True))
        assert False
    except panel.HTTPException as e:
        assert e.status_code == 400


def test_setup_job_is_single_flight():
    ov = dict(panel.SETUP_JOB)
    try:
        panel.SETUP_JOB.update(name="other", state="running")
        try:
            panel._setup_start("mine", lambda: None)
            assert False, "second concurrent job must be refused"
        except panel.HTTPException as e:
            assert e.status_code == 409
    finally:
        panel.SETUP_JOB.clear()
        panel.SETUP_JOB.update(ov)


def test_setup_job_records_failure_state():
    ov = dict(panel.SETUP_JOB)
    try:
        panel.SETUP_JOB.update(name=None, state="idle")
        panel.SETUP_JOB["log"].clear()
        def boom():
            raise panel.HTTPException(502, "installer exited 1")
        panel._setup_job_wrap(boom)
        assert panel.SETUP_JOB["state"] == "failed"
        assert "installer exited 1" in list(panel.SETUP_JOB["log"])[-1]
    finally:
        panel.SETUP_JOB.clear()
        panel.SETUP_JOB.update(ov)


def test_setup_status_shape_without_binaries():
    ov_bin, ov_sbin = panel.CFG.get("server_binary"), panel.CFG.get("strata_bin")
    ov_env = os.environ.get("STRATA_BIN")
    try:
        missing = str(Path(HOME_TMP) / "no-llama-server")
        panel.CFG["server_binary"] = missing
        panel.CFG["strata_bin"] = missing
        os.environ["STRATA_BIN"] = missing
        st = panel.api_setup_status()
        assert st["llama"]["installed"] is False
        assert st["strata"]["installed"] is False
        assert "avalon" in st and "job" in st
    finally:
        panel.CFG["server_binary"] = ov_bin
        panel.CFG["strata_bin"] = ov_sbin
        if ov_env is None:
            os.environ.pop("STRATA_BIN", None)
        else:
            os.environ["STRATA_BIN"] = ov_env


def test_legacy_path_migration_moves_once_and_never_clobbers():
    import shutil as _sh
    from pathlib import Path as _P
    root = _P(HOME_TMP) / "migr"
    ov = (panel.HOME, panel.APP_DIR, panel.CONFIG_PATH,
          panel.LEGACY_APP_DIR, panel.LEGACY_CONFIG)
    try:
        _sh.rmtree(root, ignore_errors=True)
        (root / ".config" / "lfm-panel").mkdir(parents=True)
        (root / ".config" / "lfm-panel" / "config.json").write_text('{"ctx": 12345}')
        (root / ".local" / "share" / "lfm-panel").mkdir(parents=True)
        (root / ".local" / "share" / "lfm-panel" / "state.json").write_text('{"a": 1}')
        panel.APP_DIR = root / ".local" / "share" / "inference-console"
        panel.CONFIG_PATH = root / ".config" / "inference-console" / "config.json"
        panel.LEGACY_APP_DIR = root / ".local" / "share" / "lfm-panel"
        panel.LEGACY_CONFIG = root / ".config" / "lfm-panel" / "config.json"
        panel._migrate_legacy_paths()
        assert json.loads(panel.CONFIG_PATH.read_text())["ctx"] == 12345
        assert json.loads((panel.APP_DIR / "state.json").read_text()) == {"a": 1}
        assert not panel.LEGACY_CONFIG.exists()      # moved, not copied
        # Second run is a no-op even when new files change: never clobber.
        panel.CONFIG_PATH.write_text('{"ctx": 999}')
        (root / ".config" / "lfm-panel").mkdir(parents=True, exist_ok=True)  # dir survives; only the file moved
        (root / ".config" / "lfm-panel" / "config.json").write_text('{"ctx": 1}')
        panel._migrate_legacy_paths()
        assert json.loads(panel.CONFIG_PATH.read_text())["ctx"] == 999
    finally:
        (panel.HOME, panel.APP_DIR, panel.CONFIG_PATH,
         panel.LEGACY_APP_DIR, panel.LEGACY_CONFIG) = ov
        _sh.rmtree(root, ignore_errors=True)


def test_inline_js_parses():
    """Catch JS syntax errors in the served HTML (a bad paren blanks the whole UI)."""
    import re
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("node not installed")
    blocks = re.findall(r"<script>(.*?)</script>", panel.HTML, re.S)
    assert blocks, "no inline <script> blocks found"
    for i, src in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            r = subprocess.run([node, "--check", path],
                               capture_output=True, text=True)
            assert r.returncode == 0, f"script block {i}: {r.stderr[:500]}"
        finally:
            os.unlink(path)


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
