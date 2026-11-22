#!/usr/bin/env python3
"""Stubbed unit tests for Phase 1 of the Optimization Agent (agent_memory.py, optimizer_agent.py, panel.py API)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOME_TMP = tempfile.mkdtemp(prefix="inference-console-agent-test-")
os.environ["HOME"] = HOME_TMP
sys.path.insert(0, str(REPO))

from agent_memory import AgentStore  # noqa: E402
from optimizer_agent import (  # noqa: E402
    HostContext,
    OptimizerAgent,
    analyze_moe_mechanics,
    score_trial,
)
import panel  # noqa: E402


def _make_stub_agent(tmp_path: Path, *, models=None, caps=None, vit=None, bench_fn=None, launch_fn=None):
    store = AgentStore(tmp_path / "agent")
    cfg = {
        "server_binary": "/nonexistent/llama-server",
        "avalon_tune": {},
        "strata_last": {},
        "strata_ctx": 8192,
        "vram_reserve_mb": 512,
        "ctx": 8192,
        "agent_llm_url": "",
        "agent_llm_key": "",
        "agent_llm_model": "",
        "agent_llm_fallback_local": False,
    }
    lp_path = tmp_path / "launch_profiles.json"
    lp_path.write_text("{}")

    default_models = models if models is not None else [
        {
            "name": "Qwen3-80B-MoE-Q4_K_M",
            "path": str(tmp_path / "qwen3-moe.gguf"),
            "size": 48 * 1024**3,  # 48 GiB total > 20 GB VRAM!
            "arch": "qwen3moe",
            "quant": "Q4_K_M",
            "blocks": 48,
            "n_kv": 4,
            "k_len": 128,
            "v_len": 128,
            "ctx_len": 131072,
            "moe": True,
            "experts": 128,
            "experts_used": 8,
            "vl": False,
            "pinned": True,
        },
        {
            "name": "Llama-3-8B-Q4_K_M",
            "path": str(tmp_path / "llama3-8b.gguf"),
            "size": 5 * 1024**3,
            "arch": "llama",
            "quant": "Q4_K_M",
            "blocks": 32,
            "n_kv": 8,
            "k_len": 128,
            "v_len": 128,
            "ctx_len": 32768,
            "moe": False,
            "vl": False,
            "pinned": False,
        },
    ]

    default_vit = vit or {
        "gpu": {"name": "RX 7900 XT", "total_mb": 20480, "used_mb": 500, "temp": 48, "util": 5},
        "mem": {"total_gb": 64.0, "used_gb": 10.0},
        "cpu": {"cores": 16, "percent": 5},
    }

    default_caps = caps if caps is not None else {
        "exists": True,
        "flash_attn": True,
        "kv_quant": True,
        "n_cpu_moe": True,
        "draft": True,
    }

    mgr_calls = []

    class FakeMgr:
        logbuf = []

        def status(self):
            return {"running": False, "model": "", "preset": "auto", "ctx": 8192}

        def start(self, model_path, **kwargs):
            mgr_calls.append(("llama", model_path, kwargs))
            if launch_fn:
                return launch_fn("llama", model_path, kwargs)
            return {"ok": True, "port": 8830}

        def stop(self):
            return {"ok": True}

    class FakeAvalonMgr:
        logbuf = []

        def status(self):
            return {"running": False, "profile": "flagship"}

        def build_args(self, profile, tune=None):
            return ["/bin/avalon", "-fit", "on", f"--profile={profile}"]

        def start(self, profile, tune=None):
            mgr_calls.append(("avalon", profile, tune))
            if launch_fn:
                return launch_fn("avalon", profile, tune)
            return {"ok": True, "port": 8820}

        def stop(self):
            return {"ok": True}

    def fake_strata_run(args, timeout=20):
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"plan": "ok", "args": args}),
            stderr="",
        )

    def default_bench(pk, mt):
        if bench_fn:
            return bench_fn(pk, mt, mgr_calls[-1] if mgr_calls else None)
        return {"tok_s": 42.5, "prompt_s": 510.0, "ttft_ms": 180, "vram_mb": 14200}

    host = HostContext(
        cfg=cfg,
        default_bin="/nonexistent/llama-server",
        avalon_bin="/nonexistent/avalon",
        avalon_profiles={"flagship": {"label": "Flagship", "defaults": {}}},
        launch_profiles_path=lp_path,
        vitals=lambda refresh=False: default_vit,
        scan_models=lambda: {"models": default_models},
        estimate_vram=lambda m, **kw: panel.estimate_vram(m, **{**kw, "vit": kw.get("vit") or default_vit, "srv_running": False}),
        bin_caps=lambda b=None: default_caps,
        find_draft=lambda p: "",
        save_config=lambda: None,
        mgr=FakeMgr(),
        smgr=FakeAvalonMgr(),
        avalon_target=lambda: {"base": "Flash-Next-125B", "model": tmp_path / "fn.gguf", "draft": tmp_path / "mtp.gguf"},
        strata_bin=lambda: "/usr/local/bin/strata",
        strata_status=lambda refresh=False: {"running": False},
        strata_build_args=panel.strata_build_args,
        strata_run=fake_strata_run,
        strata_launch_body_cls=panel.StrataLaunchBody,
        strata_stop=lambda: {"ok": True},
        strata_log_path=tmp_path / "strata.log",
        chat_target=lambda: None,
        benchmark_run=default_bench,
        read_json=panel._read_json,
        write_json=panel._write_json,
    )
    return OptimizerAgent(host, store), store, mgr_calls, host


# ---------------------------------------------------------------- MoE mechanics
def test_moe_not_written_off_when_total_exceeds_vram():
    """48 GiB MoE model on 20 GB VRAM + 54 GB free RAM MUST be feasible via expert CPU offload."""
    meta = {"size": 48 * 1024**3, "blocks": 48, "moe": True, "experts": 128, "experts_used": 8}
    res = analyze_moe_mechanics(meta, usable_vram_mb=19500, avail_ram_mb=55000, kv_vram_mb=1200)
    assert res["is_moe"] is True
    assert res["feasible"] is True
    assert res["strategy"] in ("moe_partial_cpu_experts", "moe_full_cpu_experts")
    assert res["recommended_ncmoe"] > 0
    assert res["gpu_hot_path_mb"] <= 19500


def test_moe_full_gpu_when_fits():
    meta = {"size": 12 * 1024**3, "blocks": 32, "moe": True, "experts": 64, "experts_used": 4}
    res = analyze_moe_mechanics(meta, usable_vram_mb=20000, avail_ram_mb=32000, kv_vram_mb=1000)
    assert res["feasible"] is True
    assert res["strategy"] == "moe_full_gpu"
    assert res["recommended_ncmoe"] == 0


# ---------------------------------------------------------------- Dry-Run Advisor (3 engines)
def test_advise_all_three_engines_and_prompt_constraints(tmp_path):
    agent, _, _, _ = _make_stub_agent(tmp_path)

    # 1. llama.cpp with MoE model + natural-language constraint
    r_llama = agent.advise(
        engine="llama",
        goal="balanced",
        model="Qwen3-80B-MoE-Q4_K_M",
        user_prompt="longest context with at least 64k tokens and leave 2 GB VRAM free",
        use_llm=False,
    )
    assert r_llama["engine"] == "llama"
    assert r_llama["goal"] == "max_context"
    assert r_llama["config"]["ctx"] >= 65536
    assert r_llama["moe"]["is_moe"] is True
    assert r_llama["moe"]["feasible"] is True
    assert r_llama["config"]["ctk"] == "q8_0" and r_llama["config"]["ctv"] == "q4_0"

    # 2. Avalon
    r_av = agent.advise(engine="avalon", goal="fastest", profile="flagship", use_llm=False)
    assert r_av["engine"] == "avalon"
    assert r_av["config"]["draft_n"] > 0  # draft was detected
    assert "cmd_preview" in r_av

    # 3. Strata (includes --plan --output json dry-run!)
    r_st = agent.advise(engine="strata", goal="max_context", use_llm=False)
    assert r_st["engine"] == "strata"
    assert r_st["config"]["kv"] == "k8v4"
    assert r_st["plan_preview"]["plan"] == "ok"


def test_advise_respects_binary_capabilities(tmp_path):
    agent, _, _, _ = _make_stub_agent(
        tmp_path,
        caps={"exists": True, "flash_attn": False, "kv_quant": False, "n_cpu_moe": False, "draft": False},
    )
    r = agent.advise(engine="llama", goal="max_context", model="Llama-3-8B-Q4_K_M", use_llm=False)
    assert r["config"]["ctk"] == "f16" and r["config"]["ctv"] == "f16"
    assert r["config"]["fa"] == "auto"


# ---------------------------------------------------------------- Checkpoints & Rollback
def test_apply_creates_checkpoint_and_restore_reverts(tmp_path):
    agent, store, _, host = _make_stub_agent(tmp_path)
    host.cfg["avalon_tune"] = {"flagship": {"ctx": 16384}}

    res = agent.apply_config(
        engine="avalon",
        config={"profile": "flagship", "ctx": 131072, "ctk": "q8_0", "ctv": "q4_0"},
        launch=False,
        goal="max_context",
    )
    cp_id = res["checkpoint_id"]
    assert host.cfg["avalon_tune"]["flagship"]["ctx"] == 131072
    assert store.get_checkpoint(cp_id) is not None

    # Now restore checkpoint -> original ctx=16384 must be back!
    restored = agent.restore_checkpoint(cp_id, relaunch=False)
    assert restored["ok"] is True
    assert host.cfg["avalon_tune"]["flagship"]["ctx"] == 16384


# ---------------------------------------------------------------- Presets
def test_presets_crud_and_builtins(tmp_path):
    _, store, _, _ = _make_stub_agent(tmp_path)
    builtins = store.list_presets()
    assert any(p["engine"] == "llama" for p in builtins)
    assert any(p["engine"] == "avalon" for p in builtins)
    assert any(p["engine"] == "strata" for p in builtins)

    saved = store.save_preset(
        name="My Coding Rig",
        engine="llama",
        goal="fastest",
        config={"ctx": 32768, "ctk": "q8_0", "ctv": "q8_0"},
        notes="Verified fast",
    )
    assert any(p["id"] == saved["id"] for p in store.list_presets(engine="llama"))
    assert store.delete_preset(saved["id"]) is True
    assert store.delete_preset("builtin-llama-fastest") is False


# ---------------------------------------------------------------- Active Auto-Tune & VRAM Calibration
def test_active_autotune_selects_winner_and_calibrates_vram(tmp_path):
    def fake_bench(pk, mt, last_call):
        # Make batch=4096 measurably faster than baseline!
        kwargs = last_call[2] if last_call else {}
        if kwargs.get("batch") == 4096:
            return {"tok_s": 88.0, "prompt_s": 900.0, "ttft_ms": 95, "vram_mb": 6400}
        return {"tok_s": 62.0, "prompt_s": 600.0, "ttft_ms": 140, "vram_mb": 6200}

    agent, store, _, _ = _make_stub_agent(tmp_path, bench_fn=fake_bench)
    st = agent.start_tune(
        engine="llama",
        goal="fastest",
        model="Llama-3-8B-Q4_K_M",
        max_trials=3,
        save_preset_name="Auto-Tuned Llama3",
        sync_run=True,
    )
    sess = st["session"]
    assert sess["state"] == "done"
    assert sess["checkpoint_id"].startswith("cp-")
    assert sess["best"]["label"] == "larger-batch-4096"
    assert sess["best"]["metrics"]["tok_s"] == 88.0

    # Persistent memory & preset recorded
    best_mem = store.get_best("llama", "Llama-3-8B-Q4_K_M", "fastest")
    assert best_mem is not None and best_mem["metrics"]["tok_s"] == 88.0
    assert any(p["name"] == "Auto-Tuned Llama3" for p in store.list_presets("llama"))

    # Real-world VRAM calibration recorded samples!
    cal = store.get_vram_calibration("llama", moe=False, engine="llama")
    assert cal["applied"] is True and cal["samples"] >= 2


# ---------------------------------------------------------------- Crash & OOM Self-Healing
def test_diagnose_moe_oom_switches_to_moe_split(tmp_path):
    agent, _, _, _ = _make_stub_agent(tmp_path)
    oom_log = "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate 38000 MB\ncudaMalloc failed: out of memory"
    diag = agent.diagnose_failure(
        engine="llama",
        log_text=oom_log,
        last_config={"model": "Qwen3-80B-MoE-Q4_K_M", "preset": "dense-gpu", "ctx": 16384, "ngl": 99, "ctk": "f16", "ctv": "f16"},
        use_llm=False,
    )
    assert diag["pattern_id"] == "vram_oom"
    assert diag["fix_config"]["preset"] == "moe-split"
    assert diag["fix_config"]["ctk"] == "q8_0" and diag["fix_config"]["ctv"] == "q4_0"


def test_diagnose_draft_mismatch(tmp_path):
    agent, _, _, _ = _make_stub_agent(tmp_path)
    diag = agent.diagnose_failure(
        engine="llama",
        log_text="error: draft model vocab does not match target model",
        last_config={"model": "Llama-3-8B-Q4_K_M", "draft": "/tmp/bad-draft.gguf", "draft_max": 16},
        use_llm=False,
    )
    assert diag["pattern_id"] == "draft_mismatch"
    assert diag["fix_config"]["draft"] == ""


# ---------------------------------------------------------------- Docs & Sidebar Chat
def test_docs_oracle_and_deterministic_sidebar_chat(tmp_path):
    agent, _, _, _ = _make_stub_agent(tmp_path)
    doc = agent.docs.lookup("llama", "moe")
    assert "moe" in doc["guides"]

    resp = agent.chat("Optimize llama.cpp for fastest speed", engine="llama")
    assert "LLAMA recommendation" in resp["reply"]
    assert any(a["type"] == "apply" for a in resp["actions"])
    assert any(a["type"] == "start_tune" for a in resp["actions"])


# ---------------------------------------------------------------- Panel API smoke test
def test_panel_agent_endpoints_and_settings_masking():
    st = panel.api_agent_status()
    assert "llm" in st and "tune" in st and "memory" in st

    panel.api_settings_set(panel.SettingsBody(
        agent_llm_url="https://example.test/v1",
        agent_llm_model="gpt-4o-mini",
        agent_llm_key="sk-test-secret-123456",
        agent_llm_fallback_local=True,
    ))
    got = panel.api_settings_get()
    assert got["agent_llm_url"] == "https://example.test/v1"
    assert got["agent_llm_model"] == "gpt-4o-mini"
    assert got["agent_llm_key_set"] is True
    assert "123456" not in got["agent_llm_key_masked"] or got["agent_llm_key_masked"].startswith("sk-t…")
    assert got["agent_llm_key_masked"] != "sk-test-secret-123456"


# ---------------------------------------------------------------- Phase 2: Thermal Guard
def test_thermal_guard_graduated_ladder_and_auto_restore(tmp_path):
    from thermal_guard import GPUHardwareController, ThermalGuard

    # Create fake writable amdgpu hwmon sysfs directory
    hwmon = tmp_path / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("amdgpu\n")
    (hwmon / "fan1_input").write_text("1200\n")
    (hwmon / "pwm1_enable").write_text("2\n")
    (hwmon / "pwm1").write_text("100\n")
    (hwmon / "power1_cap").write_text("300000000\n")       # 300 W
    (hwmon / "power1_cap_min").write_text("150000000\n")   # 150 W

    guard_cfg = {"enabled": True, "auto_act": True, "warn_temp_c": 80, "throttle_temp_c": 88, "kill_temp_c": 95, "cool_temp_c": 70, "power_throttle_pct": 15}
    calls = []

    hw = GPUHardwareController(hwmon_root=tmp_path / "hwmon")
    guard = ThermalGuard(
        get_config=lambda: guard_cfg,
        save_config=lambda c: guard_cfg.update(c),
        vitals_fn=lambda refresh=False: {"gpu": {"temp": 50, "used_mb": 2000, "total_mb": 20480}},
        downshift_fn=lambda: (calls.append("downshift") or {"ok": True}),
        killswitch_fn=lambda: (calls.append("killswitch") or {"ok": True}),
        hw=hw,
    )

    # Level 1: 82C -> max fans + 15% power throttle (300W -> 255W)
    r1 = guard.evaluate_once({"gpu": {"temp": 82, "used_mb": 8000, "total_mb": 20480}})
    assert r1["level"] == 1 and r1["state"] == "cool_hw"
    assert (hwmon / "pwm1_enable").read_text().strip() == "1"
    assert (hwmon / "pwm1").read_text().strip() == "255"
    assert int((hwmon / "power1_cap").read_text().strip()) == 255_000_000

    # Level 2: 89C -> downshift profile
    r2 = guard.evaluate_once({"gpu": {"temp": 89, "used_mb": 8000, "total_mb": 20480}})
    assert r2["level"] == 2 and "downshift" in calls

    # Level 3: 96C -> killswitch
    r3 = guard.evaluate_once({"gpu": {"temp": 96, "used_mb": 8000, "total_mb": 20480}})
    assert r3["level"] == 3 and "killswitch" in calls

    # Hysteresis cool-down: 65C <= 70C -> auto-restores pwm1_enable=2 & power1_cap=300W!
    r4 = guard.evaluate_once({"gpu": {"temp": 65, "used_mb": 1000, "total_mb": 20480}})
    assert r4["level"] == 0 and r4["state"] == "normal"
    assert (hwmon / "pwm1_enable").read_text().strip() == "2"
    assert int((hwmon / "power1_cap").read_text().strip()) == 300_000_000


# ---------------------------------------------------------------- Phase 2: Workload Router & Sync
def test_workload_router_and_coding_agent_sync(tmp_path):
    agent, store, _, host = _make_stub_agent(tmp_path)
    synced = []
    host.coding_agents_list = lambda: ["claude-code", "codex"]
    host.sync_agents_fn = lambda ags, src: (synced.append((ags, src)) or {"ok": True, "agents": ags})

    r = agent.route_workload(task="coding", apply=True, sync_agents=True)
    assert r["selected"]["engine"] == "llama"
    assert r["selected"]["model_key"] == "Qwen3-80B-MoE-Q4_K_M"
    assert r["applied"]["checkpoint_id"].startswith("cp-")
    assert synced == [(["claude-code", "codex"], "panel")]
    assert store.get_task_best("coding") is not None

    # fast_chat should prefer the smaller full-GPU model
    r_fast = agent.route_workload(task="fast_chat", apply=False)
    assert r_fast["selected"]["model_key"] == "Llama-3-8B-Q4_K_M"


# ---------------------------------------------------------------- Phase 2: Speculative Decoding Tuner
def test_speculative_tuner_enables_faster_and_disables_slower(tmp_path):
    draft_path = str(tmp_path / "mtp-draft.gguf")

    # Case A: draft gives a clear speedup (60 -> 88 tok/s)
    def bench_faster(pk, mt, last_call):
        kw = last_call[2] if last_call else {}
        if kw.get("draft") == draft_path and kw.get("draft_max") == 16:
            return {"tok_s": 88.0, "prompt_s": 500.0, "ttft_ms": 110, "vram_mb": 6500}
        return {"tok_s": 60.0, "prompt_s": 500.0, "ttft_ms": 120, "vram_mb": 6200}

    agent_a, _, _, _ = _make_stub_agent(tmp_path, bench_fn=bench_faster)
    ra = agent_a.tune_speculation(model="Llama-3-8B-Q4_K_M", draft_candidates=[draft_path], apply_best=True)
    assert ra["speculation_enabled"] is True
    assert ra["winner"]["draft"] == draft_path and ra["winner"]["draft_max"] == 16

    # Case B: draft hurts speed (60 -> 42 tok/s) -> must disable speculation!
    def bench_slower(pk, mt, last_call):
        kw = last_call[2] if last_call else {}
        if kw.get("draft"):
            return {"tok_s": 42.0, "prompt_s": 400.0, "ttft_ms": 150, "vram_mb": 6500}
        return {"tok_s": 60.0, "prompt_s": 500.0, "ttft_ms": 120, "vram_mb": 6200}

    agent_b, _, _, _ = _make_stub_agent(tmp_path, bench_fn=bench_slower)
    rb = agent_b.tune_speculation(model="Llama-3-8B-Q4_K_M", draft_candidates=[draft_path], apply_best=True)
    assert rb["speculation_enabled"] is False
    assert rb["winner"]["draft"] == ""


# ---------------------------------------------------------------- Phase 2: HF Recommender (MoE-aware)
def test_hf_recommender_moe_aware_fit_and_companions(tmp_path):
    agent, _, _, host = _make_stub_agent(tmp_path)
    host.hf_search_fn = lambda q: [{"id": "unsloth/Qwen3-80B-A3B-MoE-GGUF", "downloads": 50000}]
    host.hf_files_fn = lambda repo: [
        {"path": "Qwen3-80B-A3B-Q4_K_M-00001-of-00002.gguf", "size": 24 * 1024**3},
        {"path": "Qwen3-80B-A3B-Q4_K_M-00002-of-00002.gguf", "size": 24 * 1024**3},
        {"path": "mmproj-F16.gguf", "size": 600 * 1024**2},
        {"path": "mtp-draft-Q4_0.gguf", "size": 900 * 1024**2},
    ]
    res = agent.recommend_hf(query="Qwen3-80B-A3B", goal="balanced")
    recs = res["recommendations"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["is_moe"] is True
    assert rec["fit_label"] == "MoE Hybrid (GPU+RAM)"
    assert rec["quant"] == "Q4_K_M"
    assert "mmproj-F16.gguf" in rec["pull_files"]
    assert "mtp-draft-Q4_0.gguf" in rec["pull_files"]
    assert len(rec["pull_files"]) == 4  # both shards + mmproj + draft


def test_panel_phase2_thermal_endpoint():
    st = panel.api_agent_thermal()
    assert "config" in st and "hw_caps" in st and "state" in st
    upd = panel.api_agent_thermal_config(panel.AgentThermalConfigBody(warn_temp_c=82, auto_act=False))
    assert upd["ok"] is True and upd["config"]["warn_temp_c"] == 82


# ---------------------------------------------------------------- Maya Engine + Custom Providers & Model Polling
def test_maya_advise_apply_checkpoint_restore_and_diagnose(tmp_path):
    maya_dir = tmp_path / "project-maya"
    maya_dir.mkdir(parents=True)
    cfg_file = maya_dir / "qwen-maya.json"
    orig_cfg = {
        "exe": "/bin/true",
        "model_name": "Qwen3-Maya-MoE",
        "port": 8098,
        "ctx": 32768,
        "args": ["--ctx", "32768", "--mtp"],
        "reasoning_effort": "medium",
    }
    cfg_file.write_text(json.dumps(orig_cfg, indent=2))

    class FakeMayaMgr:
        def discover_configs(self):
            raw = json.loads(cfg_file.read_text())
            return [{
                "id": "qwen-maya",
                "path": str(cfg_file),
                "model_name": raw["model_name"],
                "engine": "glm",
                "quant": "maya-xs",
                "ctx": raw["ctx"],
                "mtp": "--mtp" in raw.get("args", []),
                "port": raw["port"],
                "reasoning_effort": raw.get("reasoning_effort"),
                "dir_size_bytes": 40 * 1024**3,
                "available": True,
            }]

        def _find_config(self, ref):
            raw = json.loads(cfg_file.read_text())
            pub = self.discover_configs()[0]
            return cfg_file, raw, pub

        def status(self):
            return {"running": False, "config_id": "qwen-maya"}

        def start(self, cid):
            return {"ok": True, "running": True, "config_id": cid}

        def switch(self, cid):
            return {"ok": True, "running": True, "config_id": cid}

        def stop(self):
            return {"ok": True}

        def tail_log(self, max_lines=150):
            return {"log": "HIP out of memory while allocating KV cache"}

    agent, _, _, host = _make_stub_agent(tmp_path)
    host.maya_mgr = FakeMayaMgr()

    # 1. Advise Maya for max_context (should disable --mtp and boost ctx >= 131072)
    rec = agent.advise(engine="maya", goal="max_context", use_llm=False)
    assert rec["engine"] == "maya"
    assert rec["config"]["mtp"] is False
    assert rec["config"]["ctx"] >= 131072
    assert rec["moe"]["is_moe"] is True

    # 2. Apply creates checkpoint & patches Maya JSON config
    applied = agent.apply_config(engine="maya", config=rec["config"], launch=False, goal="max_context")
    patched = json.loads(cfg_file.read_text())
    assert patched["ctx"] >= 131072
    assert "--mtp" not in patched["args"]

    # 3. Restore checkpoint reverts Maya JSON config cleanly back to original!
    agent.restore_checkpoint(applied["checkpoint_id"], relaunch=False)
    reverted = json.loads(cfg_file.read_text())
    assert reverted["ctx"] == 32768
    assert "--mtp" in reverted["args"]

    # 4. Diagnose Maya OOM from Maya log tail
    diag = agent.diagnose_failure(engine="maya", last_config={"ctx": 65536, "mtp": True}, use_llm=False)
    assert diag["pattern_id"] == "vram_oom"
    assert diag["fix_config"]["mtp"] is False
    assert diag["fix_config"]["ctx"] == 32768

    # 5. Sidebar chat recognizes Maya
    chat_r = agent.chat("Optimize Maya for fastest speed")
    assert "MAYA recommendation" in chat_r["reply"]
    assert any(a["engine"] == "maya" for a in chat_r["actions"])


def test_hermes_provider_presets_custom_provider_and_model_polling(tmp_path, monkeypatch):
    from optimizer_agent import PROVIDER_PRESETS

    for expected in ("custom", "openai", "anthropic", "openrouter", "google", "deepseek", "xai", "groq", "together", "fireworks", "mistral", "moonshot", "zai", "nous", "ollama", "lmstudio", "vllm"):
        assert expected in PROVIDER_PRESETS

    # Test Panel provider endpoints (custom provider + masked key)
    panel.api_agent_provider_set(panel.AgentProviderBody(
        provider="custom",
        url="https://my-custom-llm.internal/v1",
        model="my-custom-model",
        api_key="sk-custom-99887766",
        headers='{"X-Custom": "1"}',
        fallback_local=True,
    ))
    got = panel.api_agent_provider_get()
    assert got["provider"] == "custom"
    assert got["url"] == "https://my-custom-llm.internal/v1"
    assert got["model"] == "my-custom-model"
    assert got["key_set"] is True and got["key_masked"] != "sk-custom-99887766"

    # Stub httpx.get to verify model polling for OpenAI-compatible & Anthropic providers
    seen_calls = []

    class FakeResp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    def fake_get(url, headers=None, timeout=None):
        seen_calls.append((url, headers or {}))
        return FakeResp({"data": [{"id": "model-b"}, {"id": "model-a"}]})

    import optimizer_agent
    monkeypatch.setattr(optimizer_agent.httpx, "get", fake_get)

    agent, _, _, _ = _make_stub_agent(tmp_path)
    polled = agent.fetch_provider_models(provider="custom", url="https://my-custom-llm.internal/v1", api_key="sk-abc")
    assert polled["ok"] is True
    assert polled["models"] == ["model-a", "model-b"]
    assert seen_calls[-1][0] == "https://my-custom-llm.internal/v1/models"
    assert seen_calls[-1][1].get("Authorization") == "Bearer sk-abc"

    # Anthropic uses x-api-key + anthropic-version header
    polled_anth = agent.fetch_provider_models(provider="anthropic", api_key="sk-ant-123")
    assert polled_anth["ok"] is True
    assert seen_calls[-1][1].get("x-api-key") == "sk-ant-123"
    assert "anthropic-version" in seen_calls[-1][1]


def test_agent_chat_surfaces_provider_errors_and_handles_reasoning_models(tmp_path, monkeypatch):
    """Configured provider failures must report the error rather than claiming deterministic mode."""
    import optimizer_agent

    agent, _, _, host = _make_stub_agent(tmp_path)
    host.cfg["agent_llm_provider"] = "openai"
    host.cfg["agent_llm_url"] = "https://api.openai.com/v1"
    host.cfg["agent_llm_model"] = "o3-mini"
    host.cfg["agent_llm_key"] = "sk-test"

    class FakePostResp:
        def __init__(self, status_code, body, reason="OK"):
            self.status_code = status_code
            self._body = body
            self.text = json.dumps(body) if isinstance(body, dict) else str(body)
            self.reason_phrase = reason

        def json(self):
            return self._body

    # 1. HTTP 401 should surface the real error, NOT "Deterministic mode active"
    monkeypatch.setattr(
        optimizer_agent.httpx,
        "post",
        lambda *a, **kw: FakePostResp(401, {"error": {"message": "Invalid API key"}}, "Unauthorized"),
    )
    res_err = agent.chat("Hello, what GPU do I have?")
    assert "Deterministic mode active" not in res_err["reply"]
    assert "HTTP 401" in res_err["reply"] and "Invalid API key" in res_err["reply"]
    assert res_err["llm"]["mode"] == "external"

    # 2. HTTP 400 demanding max_completion_tokens retries cleanly and reads reasoning_content fallback
    calls = []

    def fake_reasoning_post(url, headers=None, json=None, timeout=None):
        calls.append(dict(json or {}))
        if "max_tokens" in (json or {}):
            return FakePostResp(400, {"error": {"message": "Use max_completion_tokens instead of max_tokens"}}, "Bad Request")
        return FakePostResp(200, {"choices": [{"message": {"content": "", "reasoning_content": "You have a 24GB GPU."}}]})

    monkeypatch.setattr(optimizer_agent.httpx, "post", fake_reasoning_post)
    res_ok = agent.chat("Hello, what GPU do I have?")
    assert len(calls) == 2
    assert "max_completion_tokens" in calls[1]
    assert "24GB GPU" in res_ok["reply"]


def test_strata_timeout_preserves_output_and_advise_inherits_last_preset(tmp_path, monkeypatch):
    """Strata timeouts must keep partial stdout/stderr and auto-tune must log the command + use >=300s timeout."""
    import subprocess

    # 1. panel._strata_run preserves partial stderr/stdout on TimeoutExpired
    monkeypatch.setattr(panel, "strata_bin", lambda: "/usr/local/bin/strata")

    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired(
            cmd=a[0],
            timeout=kw.get("timeout", 10),
            output=b"Loading shards...\n",
            stderr=b"Downloading qwen3.8-flash-next 14%\n",
        )

    monkeypatch.setattr(panel.subprocess, "run", fake_run)
    r = panel._strata_run(["launch", "--context", "64k"], timeout=120)
    assert r["ok"] is False
    assert "14%" in r["error"]
    assert "Downloading qwen3.8-flash-next 14%" in r["stderr"]
    assert "Loading shards" in r["stdout"]

    # 2. _advise_strata inherits preset from strata_last & handles dict strata_run results cleanly
    agent, _, _, host = _make_stub_agent(tmp_path)
    host.cfg["strata_last"] = {"preset": "my-local-preset", "port": 8080}
    host.strata_bin = lambda: "/usr/local/bin/strata"
    seen = []

    def fake_strata_run(args, timeout=25):
        seen.append((list(args), timeout))
        if "--plan" in args:
            return {"ok": True, "rc": 0, "json": {"vram_gb": 18.2}, "stdout": '{"vram_gb": 18.2}', "stderr": "", "error": None}
        return {"ok": False, "rc": None, "json": None, "stdout": "", "stderr": "waiting on GPU 0", "error": "strata timed out after 300s"}

    host.strata_run = fake_strata_run
    rec = agent.advise(engine="strata", goal="balanced", use_llm=False)
    assert rec["config"].get("preset") == "my-local-preset"
    assert rec["plan_preview"] == {"vram_gb": 18.2}

    ok, err = agent._launch_for_trial("strata", rec["config"])
    assert ok is False
    assert seen[-1][1] == 300
    assert "waiting on GPU 0" in err and "timed out after 300s" in err




