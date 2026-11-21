#!/usr/bin/env python3
"""Stubbed unit & API tests for the Project Maya engine tab in panel.py.

Runnable via:
    uv run --with pytest --with fastapi --with uvicorn --with httpx --with pyyaml pytest tests/test_panel_maya.py
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOME_TMP = tempfile.mkdtemp(prefix="inference-console-maya-")
os.environ["HOME"] = HOME_TMP  # must be set before importing panel
sys.path.insert(0, str(REPO))

from fastapi import HTTPException  # noqa: E402
import panel  # noqa: E402


HY4_LOG_SAMPLE = """\
INFO maya-hy4: Hy4-flash, 42 layers, 512 experts (16 a token), context 131072, 19.3 GB of VRAM over 1 GPUs, loaded in 4.5 s
INFO CPU lane: 410 of 20992 routed experts in VRAM (2%), the rest on 8 CPU threads from the model's file
INFO maya-hy4: warmed up in 12.4 s
ready: http://0.0.0.0:8098
[maya] answering: 233 of max 256 tokens, 17.2 tok/s, 15 s
maya hy4 serve: prompt 47 tokens = 0 reused + 47 read in 1543 ms (30.5 tok/s), 256 generated in 14448 ms (17.72 tok/s)
[maya] done: 256 tokens in 15 s (18.6 tok/s)
"""

GLM_LOG_SAMPLE = """\
INFO maya-glm: GLM-5.3-Flash, context 131072, 18.8 GB of VRAM over 1 GPUs, loaded in 6.1 s
INFO maya-glm: warmed up in 9.0 s
ready: http://0.0.0.0:8099
glm prefill: 128 tokens in 2400 ms (53.3 tok/s)
glm stat: 200 tokens in 10.2 s (19.6 tok/s)
"""


def _make_maya_env(tmp_path: Path):
    maya_dir = tmp_path / "project-maya"
    (maya_dir / ".venv/bin").mkdir(parents=True)
    py_bin = maya_dir / ".venv/bin/python"
    py_bin.write_text("#!/bin/sh\n")
    (maya_dir / "maya.py").write_text("# stub launcher\n")
    (maya_dir / "build-hip").mkdir(parents=True)
    (maya_dir / "build-hip/maya-hy4").write_bytes(b"ELF")
    (maya_dir / "build-hip/maya").write_bytes(b"ELF")

    model_dir = maya_dir / "models/hy4-flash"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"x" * 4096)

    hy4_log = maya_dir / "maya-hy4-flash-maya-xs.log"
    hy4_log.write_text(HY4_LOG_SAMPLE)

    hy4_cfg = {
        "exe": "build-hip/maya-hy4",
        "args": ["-m", "models/hy4-flash", "--max-context", "131072"],
        "port": 8098,
        "host": "0.0.0.0",
        "api_key": "sk-maya-super-secret-hy4-key",
        "log": "maya-hy4-flash-maya-xs.log",
        "model_name": "hy4-flash",
        "sampling": {"temperature": 0.9, "top_p": 1.0},
        "reasoning_effort": "high",
    }
    glm_cfg = {
        "exe": "build-hip/maya",
        "args": ["-m", "models/glm-5.3", "--max-context", "131072", "--mtp"],
        "port": 8099,
        "host": "0.0.0.0",
        "api_key": "sk-maya-super-secret-glm-key",
        "log": "maya-glm.log",
        "model_name": "glm-5.3-flash",
    }
    backup_cfg = dict(hy4_cfg, port=8098, model_name="should-be-ignored")

    hy4_path = maya_dir / "maya-hy4-flash-maya-xs.json"
    glm_path = maya_dir / "maya-maya-xs-iq1_s-hip.json"
    bak_path = maya_dir / "maya-hy4-flash-maya-xs.panel-orig.json"
    noise_path = maya_dir / "random.json"

    hy4_path.write_text(json.dumps(hy4_cfg))
    glm_path.write_text(json.dumps(glm_cfg))
    bak_path.write_text(json.dumps(backup_cfg))
    noise_path.write_text(json.dumps({"unrelated": True}))

    return maya_dir, hy4_path, glm_path, hy4_log


@pytest.fixture
def maya_env(tmp_path, monkeypatch):
    maya_dir, hy4_path, glm_path, hy4_log = _make_maya_env(tmp_path)
    rocm_dir = tmp_path / "maya-rocm"
    monkeypatch.setattr(panel, "MAYA_DIR", maya_dir)
    monkeypatch.setattr(panel, "MAYA_ROCM_PATH", rocm_dir)
    monkeypatch.setattr(panel, "MAYA_STATE_PATH", tmp_path / "maya-state.json")
    monkeypatch.setattr(panel, "MAYA_LOG_PATH", tmp_path / "maya-runtime.log")
    monkeypatch.setitem(panel.CFG, "maya_dir", "")
    monkeypatch.setitem(panel.CFG, "maya_rocm_path", "")
    panel.MAYA.proc = None
    panel.MAYA.info = {}
    panel.MAYA._invalidate()
    # Default all external engine/host probes to inactive
    monkeypatch.setattr(panel.MGR, "status", lambda: {"running": False})
    monkeypatch.setattr(panel.SMGR, "status", lambda: {"running": False})
    monkeypatch.setattr(panel, "strata_status", lambda refresh=False: {"running": False})
    monkeypatch.setattr(panel.AvalonManager, "_gpu_conflict", staticmethod(lambda: False))
    monkeypatch.setattr(panel.MayaMgr, "_port_open", staticmethod(lambda port, timeout=0.15: False))
    monkeypatch.setattr(panel.MayaMgr, "_find_proc_pid", staticmethod(lambda port=None: None))
    return types.SimpleNamespace(
        maya_dir=maya_dir,
        rocm_dir=rocm_dir,
        hy4_path=hy4_path,
        glm_path=glm_path,
        hy4_log=hy4_log,
    )


# ---------------------------------------------------------------- config discovery
def test_discover_configs_excludes_backups_and_hides_api_key(maya_env):
    cfgs = panel.MAYA.discover_configs()
    ids = [c["id"] for c in cfgs]
    assert ids == ["maya-hy4-flash-maya-xs", "maya-maya-xs-iq1_s-hip"]

    hy4 = cfgs[0]
    assert hy4["model_name"] == "hy4-flash"
    assert hy4["engine"] == "hy4"
    assert hy4["port"] == 8098
    assert hy4["ctx"] == 131072
    assert hy4["quant"] == "maya-xs"
    assert hy4["dir_size_bytes"] == 4096
    assert hy4["available"] is True
    assert hy4["key_set"] is True
    assert "super-secret" not in json.dumps(cfgs)
    assert "api_key" not in hy4

    glm = cfgs[1]
    assert glm["model_name"] == "glm-5.3-flash"
    assert glm["engine"] == "glm"
    assert glm["port"] == 8099
    assert glm["mtp"] is True


# ---------------------------------------------------------------- log telemetry parsing
def test_parse_maya_log_hy4_lines():
    tel = panel.parse_maya_log(HY4_LOG_SAMPLE.splitlines())
    assert tel["vram_gb"] == 19.3
    assert tel["context"] == 131072
    assert tel["load_s"] == 4.5
    assert tel["warmup_s"] == 12.4
    assert tel["experts_vram"] == 410
    assert tel["experts_total"] == 20992
    assert tel["experts_vram_pct"] == 2.0
    assert tel["prefill_tps"] == 30.5
    assert tel["decode_tps"] == 18.6  # latest done line wins
    assert tel["ready_seen"] is True
    assert tel["active"] is False
    assert tel["last_tokens"] == 256


def test_parse_maya_log_glm_lines():
    tel = panel.parse_maya_log(GLM_LOG_SAMPLE.splitlines())
    assert tel["vram_gb"] == 18.8
    assert tel["context"] == 131072
    assert tel["load_s"] == 6.1
    assert tel["warmup_s"] == 9.0
    assert tel["prefill_tps"] == 53.3
    assert tel["decode_tps"] == 19.6
    assert tel["ready_seen"] is True


def test_parse_maya_log_answering_sets_active():
    tel = panel.parse_maya_log([
        "[maya] answering: 233 of max 256 tokens, 17.2 tok/s, 15 s",
    ])
    assert tel["active"] is True
    assert tel["decode_tps"] == 17.2
    assert tel["last_tokens"] == 233


# ---------------------------------------------------------------- adoption, _current_llm, agent sync
def test_adoption_and_current_llm_maya(maya_env, monkeypatch):
    def fake_probe(port, api_key, timeout=0.8):
        if port == 8098 and api_key == "sk-maya-super-secret-hy4-key":
            return True, "hy4-flash"
        return False, None

    monkeypatch.setattr(panel.MayaMgr, "_probe_models", staticmethod(fake_probe))
    panel.MAYA._invalidate()

    st = panel.MAYA.status(refresh=True)
    assert st["running"] is True
    assert st["state"] == "ready"
    assert st["adopted"] is True
    assert st["port"] == 8098
    assert st["alias"] == "hy4-flash"
    assert st["telemetry"]["decode_tps"] == 18.6
    assert "super-secret" not in json.dumps(st)

    cur = panel._current_llm(prefer="maya")
    assert cur == ("hy4-flash", 8098, 131072, "sk-maya-super-secret-hy4-key", "maya")

    ct = panel._chat_target()
    assert ct == (8098, "sk-maya-super-secret-hy4-key")

    agents_info = panel.api_agents()
    assert agents_info["sources"]["maya"] is True

    # Sync one agent (aider) with source="maya"
    res = panel.api_agents_sync(panel.AgentSyncBody(agents=["aider"], source="maya"))
    assert res["alias"] == "hy4-flash"
    assert res["port"] == 8098
    assert res["results"]["aider"]["ok"] is True
    reg = panel._load_agents_state()
    assert reg["source"] == "maya"
    assert reg["port"] == 8098
    assert reg["alias"] == "hy4-flash"


# ---------------------------------------------------------------- start argv / env & switch
def test_start_argv_env_and_exclusivity(maya_env, monkeypatch):
    captured = {}

    class DummyProc:
        pid = 42424
        def poll(self):
            return None

    def fake_popen(argv, cwd=None, env=None, stdout=None, stderr=None, start_new_session=False):
        captured["argv"] = argv
        captured["cwd"] = cwd
        captured["env"] = env
        captured["start_new_session"] = start_new_session
        return DummyProc()

    monkeypatch.setattr(panel.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(panel, "_pid_alive", lambda pid: pid == 42424)
    monkeypatch.setattr(panel.MayaMgr, "_probe_models", staticmethod(lambda p, k, timeout=0.8: (False, None)))

    st = panel.MAYA.start(str(maya_env.hy4_path))
    assert st["running"] is True
    assert st["state"] == "loading"
    assert st["pid"] == 42424
    assert captured["argv"] == [
        str(maya_env.maya_dir / ".venv/bin/python"),
        str(maya_env.maya_dir / "maya.py"),
        "--config", str(maya_env.hy4_path),
        "--port", "8098",
    ]
    assert captured["env"]["ROCM_PATH"] == str(maya_env.rocm_dir)
    assert captured["cwd"] == str(maya_env.maya_dir)
    assert captured["start_new_session"] is True

    # Refuses a second Maya config while one serves
    with pytest.raises(HTTPException) as exc:
        panel.MAYA.start(str(maya_env.glm_path))
    assert exc.value.status_code == 409

    # switch() stops then starts the second config
    monkeypatch.setattr(panel.MAYA, "stop", lambda: (setattr(panel.MAYA, "proc", None), setattr(panel.MAYA, "info", {}), panel.MAYA._invalidate(), {"stopped": True})[-1])
    st2 = panel.MAYA.switch(str(maya_env.glm_path))
    assert st2["running"] is True
    assert st2["port"] == 8099
    assert st2["model_name"] == "glm-5.3-flash"


# ---------------------------------------------------------------- GPU-conflict guard both ways
def test_gpu_conflict_guard_blocks_maya_when_other_engines_run(maya_env, monkeypatch):
    # 1. llama.cpp running
    monkeypatch.setattr(panel.MGR, "status", lambda: {"running": True, "port": 8830})
    with pytest.raises(HTTPException) as e1:
        panel.MAYA.start(str(maya_env.hy4_path))
    assert e1.value.status_code == 409

    # 2. Avalon running
    monkeypatch.setattr(panel.MGR, "status", lambda: {"running": False})
    monkeypatch.setattr(panel.SMGR, "status", lambda: {"running": True, "port": 8086})
    with pytest.raises(HTTPException) as e2:
        panel.MAYA.start(str(maya_env.hy4_path))
    assert e2.value.status_code == 409

    # 3. Strata running
    monkeypatch.setattr(panel.SMGR, "status", lambda: {"running": False})
    monkeypatch.setattr(panel, "strata_status", lambda refresh=False: {"running": True, "port": 9000})
    with pytest.raises(HTTPException) as e3:
        panel.MAYA.start(str(maya_env.hy4_path))
    assert e3.value.status_code == 409

    # 4. Raw VRAM conflict
    monkeypatch.setattr(panel, "strata_status", lambda refresh=False: {"running": False})
    monkeypatch.setattr(panel.AvalonManager, "_gpu_conflict", staticmethod(lambda: True))
    with pytest.raises(HTTPException) as e4:
        panel.MAYA.start(str(maya_env.hy4_path))
    assert e4.value.status_code == 409


def test_gpu_conflict_guard_blocks_other_engines_when_maya_runs(maya_env, monkeypatch):
    monkeypatch.setattr(panel.MAYA, "status", lambda refresh=False: {"running": True, "state": "ready", "port": 8098})

    # llama.cpp refuses
    with pytest.raises(HTTPException) as e_llama:
        panel.MGR.start("/nonexistent.gguf", preset="dense-gpu")
    assert e_llama.value.status_code == 409

    # Avalon refuses
    monkeypatch.setattr(panel, "_avalon_target", lambda: {"model": maya_env.hy4_path, "base": "test", "draft": None, "mmproj": None})
    monkeypatch.setattr(panel, "AVALON_BIN", str(maya_env.maya_dir / "build-hip/maya"))
    with pytest.raises(HTTPException) as e_av:
        panel.SMGR.start("flagship")
    assert e_av.value.status_code == 409

    # Strata refuses (non-plan)
    monkeypatch.setattr(panel, "strata_bin", lambda: "/usr/bin/true")
    with pytest.raises(HTTPException) as e_st:
        panel.api_strata_launch(panel.StrataLaunchBody(plan=False))
    assert e_st.value.status_code == 409


# ---------------------------------------------------------------- key masking, log tail, UI markers
def test_key_endpoint_and_log_redaction(maya_env, monkeypatch):
    maya_env.hy4_log.write_text(HY4_LOG_SAMPLE + "\nleaked sk-maya-super-secret-hy4-key here\n")
    monkeypatch.setattr(
        panel.MayaMgr, "_probe_models",
        staticmethod(lambda p, k, timeout=0.8: (True, "hy4-flash") if p == 8098 else (False, None)),
    )
    panel.MAYA._invalidate()

    masked = panel.api_maya_key(reveal=False)
    assert masked["key_set"] is True
    assert masked["key"] == ""
    assert "super-secret" not in masked["masked"]

    revealed = panel.api_maya_key(reveal=True)
    assert revealed["key"] == "sk-maya-super-secret-hy4-key"

    log_out = panel.api_maya_log()
    assert "super-secret" not in log_out["log"]
    assert "18.6 tok/s" in log_out["log"]


def test_ui_contains_maya_tab_markers():
    assert 'data-t="maya"' in panel.HTML
    assert 'id="nd-maya"' in panel.HTML
    assert 'id="t-maya"' in panel.HTML
    assert "'maya'" in panel.HTML
    assert "/api/maya/status" in panel.HTML
    assert "/api/maya/configs" in panel.HTML
    assert "/api/maya/start" in panel.HTML
    assert "/api/maya/stop" in panel.HTML
    assert "/api/maya/log" in panel.HTML
