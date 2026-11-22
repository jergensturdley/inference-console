#!/usr/bin/env python3
"""Persistent memory, checkpoints, presets, and real-world VRAM calibration for the Optimization Agent.

All state is stored as atomic JSON files under `<app_dir>/agent/` so the agent stays
persistently aware of what worked best for each (model, engine, goal, task) across restarts,
and can always roll back to a pre-optimization checkpoint if a config is a dud.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


BUILTIN_PRESETS: list[dict[str, Any]] = [
    {
        "id": "builtin-llama-fastest",
        "name": "llama.cpp · Max Speed",
        "engine": "llama",
        "goal": "fastest",
        "builtin": True,
        "notes": "Full GPU offload where possible, Q8_0 KV cache, Flash Attention, speculative decoding if draft exists.",
        "config": {"ctk": "q8_0", "ctv": "q8_0", "fa": "on", "parallel": 1, "batch": 2048, "ubatch": 512},
    },
    {
        "id": "builtin-llama-maxctx",
        "name": "llama.cpp · Longest Context",
        "engine": "llama",
        "goal": "max_context",
        "builtin": True,
        "notes": "Q8_0/Q4_0 KV quantization + Flash Attention to maximize usable context window in VRAM.",
        "config": {"ctk": "q8_0", "ctv": "q4_0", "fa": "on", "parallel": 1, "batch": 2048, "ubatch": 512},
    },
    {
        "id": "builtin-llama-moe",
        "name": "llama.cpp · MoE Hybrid Split",
        "engine": "llama",
        "goal": "balanced",
        "builtin": True,
        "notes": "Attention, router, and KV cache on GPU; routed FFN experts offloaded to system RAM.",
        "config": {"preset": "moe-split", "ctk": "q8_0", "ctv": "q8_0", "fa": "on", "parallel": 1},
    },
    {
        "id": "builtin-avalon-fastest",
        "name": "Avalon · Fast MoE + Draft",
        "engine": "avalon",
        "goal": "fastest",
        "builtin": True,
        "notes": "Fit-aware GPU/RAM split (-fit on) with Q8_0 KV and Flash Attention.",
        "config": {"profile": "flagship", "ctk": "q8_0", "ctv": "q8_0", "fa": "on", "batch": 2048, "ubatch": 512},
    },
    {
        "id": "builtin-avalon-maxctx",
        "name": "Avalon · Long Context",
        "engine": "avalon",
        "goal": "max_context",
        "builtin": True,
        "notes": "Q8_0/Q4_0 KV cache for deep context while keeping attention & KV on GPU.",
        "config": {"profile": "flagship", "ctx": 65536, "ctk": "q8_0", "ctv": "q4_0", "fa": "on", "parallel": 1},
    },
    {
        "id": "builtin-strata-fastest",
        "name": "Strata · Max Speed",
        "engine": "strata",
        "goal": "fastest",
        "builtin": True,
        "notes": "Int8 KV cache with automatic MTP/dflash2 drafting and single slot.",
        "config": {"kv": "int8", "parallel": 1, "drafting": "auto", "mtp": "on"},
    },
    {
        "id": "builtin-strata-maxctx",
        "name": "Strata · Longest Context",
        "engine": "strata",
        "goal": "max_context",
        "builtin": True,
        "notes": "k8v4 KV cache format to fit 128k+ context comfortably.",
        "config": {"context": "128k", "kv": "k8v4", "parallel": 1},
    },
]


class AgentStore:
    """Thread-safe persistent store for checkpoints, presets, experience memory, and VRAM calibration."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = self.base_dir / "checkpoints.json"
        self.presets_path = self.base_dir / "presets.json"
        self.memory_path = self.base_dir / "memory.json"
        self.calibration_path = self.base_dir / "vram_calibration.json"
        self._lock = threading.RLock()

    # ------------------------------------------------------------ checkpoints
    def create_checkpoint(
        self,
        label: str,
        engine: str,
        snapshot: dict[str, Any],
        reason: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            cps = _read_json(self.checkpoints_path, [])
            cp = {
                "id": f"cp-{uuid.uuid4().hex[:10]}",
                "ts": time.time(),
                "label": label or f"Pre-agent checkpoint ({engine})",
                "engine": engine,
                "reason": reason,
                "snapshot": snapshot,
            }
            cps.insert(0, cp)
            cps = cps[:50]
            _write_json(self.checkpoints_path, cps)
            return cp

    def list_checkpoints(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            cps = _read_json(self.checkpoints_path, [])
            return cps[:limit]

    def get_checkpoint(self, cp_id: str) -> dict[str, Any] | None:
        with self._lock:
            for cp in _read_json(self.checkpoints_path, []):
                if cp.get("id") == cp_id:
                    return cp
            return None

    def delete_checkpoint(self, cp_id: str) -> bool:
        with self._lock:
            cps = _read_json(self.checkpoints_path, [])
            new_cps = [c for c in cps if c.get("id") != cp_id]
            if len(new_cps) == len(cps):
                return False
            _write_json(self.checkpoints_path, new_cps)
            return True

    # ------------------------------------------------------------ presets
    def list_presets(self, engine: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            custom = _read_json(self.presets_path, [])
            all_p = list(custom) + list(BUILTIN_PRESETS)
            if engine:
                all_p = [p for p in all_p if p.get("engine") == engine]
            return all_p

    def save_preset(
        self,
        name: str,
        engine: str,
        goal: str,
        config: dict[str, Any],
        notes: str = "",
        metrics: dict[str, Any] | None = None,
        model_key: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            custom = _read_json(self.presets_path, [])
            # Update if same name + engine exists
            existing = next(
                (p for p in custom if p.get("name") == name and p.get("engine") == engine),
                None,
            )
            if existing:
                existing.update({
                    "goal": goal,
                    "config": dict(config),
                    "notes": notes,
                    "metrics": metrics or existing.get("metrics"),
                    "model_key": model_key or existing.get("model_key", ""),
                    "updated_ts": time.time(),
                })
                entry = existing
            else:
                entry = {
                    "id": f"pre-{uuid.uuid4().hex[:10]}",
                    "name": name,
                    "engine": engine,
                    "goal": goal,
                    "model_key": model_key,
                    "config": dict(config),
                    "notes": notes,
                    "metrics": metrics,
                    "builtin": False,
                    "ts": time.time(),
                    "updated_ts": time.time(),
                }
                custom.insert(0, entry)
            _write_json(self.presets_path, custom[:100])
            return entry

    def delete_preset(self, preset_id: str) -> bool:
        with self._lock:
            custom = _read_json(self.presets_path, [])
            new_c = [p for p in custom if p.get("id") != preset_id]
            if len(new_c) == len(custom):
                return False
            _write_json(self.presets_path, new_c)
            return True

    # ------------------------------------------------------------ experience memory
    def _load_memory(self) -> dict[str, Any]:
        mem = _read_json(self.memory_path, {})
        if not isinstance(mem, dict):
            mem = {}
        mem.setdefault("trials", [])
        mem.setdefault("best", {})       # key: "{engine}|{model_key}|{goal}" -> record
        mem.setdefault("task_best", {})  # key: "{task}" -> record (best model+engine+config)
        mem.setdefault("failures", [])   # recent failure signatures to avoid repeating
        return mem

    def record_trial(
        self,
        *,
        engine: str,
        model_key: str,
        goal: str,
        config: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        score: float = 0.0,
        ok: bool = True,
        task: str = "",
        arch: str = "",
        moe: bool = False,
        hardware_sig: str = "",
        error_signature: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            mem = self._load_memory()
            rec = {
                "id": f"tr-{uuid.uuid4().hex[:10]}",
                "ts": time.time(),
                "engine": engine,
                "model_key": model_key,
                "goal": goal,
                "task": task or goal,
                "arch": arch,
                "moe": bool(moe),
                "hardware_sig": hardware_sig,
                "config": dict(config),
                "metrics": dict(metrics or {}),
                "score": round(float(score), 3),
                "ok": bool(ok),
                "error_signature": error_signature,
                "notes": notes,
            }
            mem["trials"].insert(0, rec)
            mem["trials"] = mem["trials"][:300]

            if ok:
                bkey = f"{engine}|{model_key}|{goal}"
                prev = mem["best"].get(bkey)
                if prev is None or rec["score"] >= prev.get("score", -1e9):
                    mem["best"][bkey] = rec
                if task:
                    tprev = mem["task_best"].get(task)
                    if tprev is None or rec["score"] >= tprev.get("score", -1e9):
                        mem["task_best"][task] = rec
            elif error_signature:
                mem["failures"].insert(0, {
                    "ts": rec["ts"],
                    "engine": engine,
                    "model_key": model_key,
                    "config": dict(config),
                    "error_signature": error_signature,
                    "notes": notes,
                })
                mem["failures"] = mem["failures"][:100]

            _write_json(self.memory_path, mem)
            return rec

    def get_best(self, engine: str, model_key: str, goal: str) -> dict[str, Any] | None:
        with self._lock:
            mem = self._load_memory()
            return mem["best"].get(f"{engine}|{model_key}|{goal}")

    def get_task_best(self, task: str) -> dict[str, Any] | None:
        with self._lock:
            mem = self._load_memory()
            return mem["task_best"].get(task)

    def get_recent_failures(
        self, engine: str, model_key: str = "", limit: int = 10
    ) -> list[dict[str, Any]]:
        with self._lock:
            mem = self._load_memory()
            out = []
            for f in mem["failures"]:
                if f.get("engine") != engine:
                    continue
                if model_key and f.get("model_key") != model_key:
                    continue
                out.append(f)
                if len(out) >= limit:
                    break
            return out

    def memory_summary(self, limit: int = 25) -> dict[str, Any]:
        with self._lock:
            mem = self._load_memory()
            return {
                "trial_count": len(mem["trials"]),
                "best_count": len(mem["best"]),
                "failure_count": len(mem["failures"]),
                "best": list(mem["best"].values())[:limit],
                "task_best": mem["task_best"],
                "recent_trials": mem["trials"][:limit],
                "recent_failures": mem["failures"][:15],
                "vram_calibration": self.calibration_summary(),
            }

    def clear_memory(self) -> None:
        with self._lock:
            _write_json(self.memory_path, {"trials": [], "best": {}, "task_best": {}, "failures": []})

    # ------------------------------------------------------------ VRAM calibration
    def record_vram_sample(
        self,
        *,
        arch: str,
        moe: bool,
        engine: str,
        estimated_mb: float,
        actual_mb: float,
        model_key: str = "",
    ) -> dict[str, Any]:
        if estimated_mb <= 0 or actual_mb <= 0:
            return {}
        with self._lock:
            cal = _read_json(self.calibration_path, {"samples": []})
            samples = cal.get("samples", [])
            ratio = actual_mb / max(1.0, estimated_mb)
            # Ignore extreme outliers (e.g. another app eating GPU during measurement)
            if ratio < 0.35 or ratio > 2.5:
                return {"ignored": True, "reason": "outlier_ratio", "ratio": round(ratio, 3)}
            sample = {
                "ts": time.time(),
                "arch": (arch or "unknown").lower(),
                "moe": bool(moe),
                "engine": engine,
                "model_key": model_key,
                "estimated_mb": round(float(estimated_mb)),
                "actual_mb": round(float(actual_mb)),
                "diff_mb": round(float(actual_mb - estimated_mb)),
                "ratio": round(ratio, 4),
            }
            samples.insert(0, sample)
            cal["samples"] = samples[:200]
            _write_json(self.calibration_path, cal)
            return sample

    def get_vram_calibration(self, arch: str, moe: bool, engine: str = "llama") -> dict[str, Any]:
        """Compute bounded multiplicative/additive VRAM correction from real observations."""
        with self._lock:
            cal = _read_json(self.calibration_path, {"samples": []})
            samples = cal.get("samples", [])
            arch_l = (arch or "unknown").lower()

            # Prefer exact (arch, moe, engine) matches; fall back to (moe, engine)
            exact = [
                s for s in samples
                if s.get("arch") == arch_l and bool(s.get("moe")) == bool(moe) and s.get("engine") == engine
            ]
            pool = exact if len(exact) >= 2 else [
                s for s in samples
                if bool(s.get("moe")) == bool(moe) and s.get("engine") == engine
            ]
            if not pool:
                return {"applied": False, "scale": 1.0, "offset_mb": 0, "samples": 0}

            # Weighted average favoring recent samples
            recent = pool[:12]
            avg_ratio = sum(s["ratio"] for s in recent) / len(recent)
            avg_diff = sum(s["diff_mb"] for s in recent) / len(recent)
            # Shrink towards 1.0 when sample count is small to avoid noisy overfitting
            weight = min(1.0, len(recent) / 5.0)
            scale = 1.0 + (avg_ratio - 1.0) * weight
            scale = max(0.75, min(1.35, scale))
            return {
                "applied": True,
                "scale": round(scale, 4),
                "avg_diff_mb": round(avg_diff),
                "samples": len(recent),
                "scope": "arch+moe" if len(exact) >= 2 else "moe_class",
            }

    def calibration_summary(self) -> dict[str, Any]:
        with self._lock:
            cal = _read_json(self.calibration_path, {"samples": []})
            samples = cal.get("samples", [])
            buckets: dict[str, list[dict[str, Any]]] = {}
            for s in samples:
                k = f"{s.get('engine', 'llama')}:{s.get('arch', 'unknown')}:{'moe' if s.get('moe') else 'dense'}"
                buckets.setdefault(k, []).append(s)
            groups = []
            for k, items in buckets.items():
                avg_r = sum(x["ratio"] for x in items) / len(items)
                avg_d = sum(x["diff_mb"] for x in items) / len(items)
                groups.append({
                    "bucket": k,
                    "samples": len(items),
                    "avg_ratio": round(avg_r, 3),
                    "avg_diff_mb": round(avg_d),
                })
            return {"total_samples": len(samples), "groups": groups[:20], "recent": samples[:10]}
