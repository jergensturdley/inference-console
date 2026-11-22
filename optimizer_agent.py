#!/usr/bin/env python3
"""Hybrid Engine Config Optimization Agent for Inference Console.

Supports all three engines (llama.cpp, Avalon, Strata) with:
- Deep MoE mechanics (attention/router/KV on GPU vs routed experts in RAM; never dismisses MoE
  just because total file size exceeds VRAM).
- Documentation-grounded flag validation & error diagnosis (`--help`, `strata doctor`, bundled guides).
- Dry-Run Advisor (instant, zero restarts) & Active Empirical Auto-Tune (bounded launch -> bench -> iterate).
- Automatic pre-optimization checkpoints & rollback, reusable presets, persistent experience memory,
  real-world VRAM calibration, and Crash/OOM self-healing.
- Optional OpenAI-compatible LLM layer (with fallback to the currently running local engine for
  dry-run advice/debugging).
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from agent_memory import AgentStore

GOALS = ("fastest", "low_latency", "max_context", "balanced", "low_vram", "multi_user")
ENGINES = ("llama", "avalon", "strata", "maya")

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "custom":         {"label": "Custom OpenAI-Compatible",  "url": "",                                                   "api": "openai",    "key_optional": True},
    "openai":         {"label": "OpenAI",                    "url": "https://api.openai.com/v1",                          "api": "openai",    "key_optional": False},
    "anthropic":      {"label": "Anthropic (Claude)",        "url": "https://api.anthropic.com/v1",                       "api": "anthropic", "key_optional": False},
    "openrouter":     {"label": "OpenRouter",                "url": "https://openrouter.ai/api/v1",                       "api": "openai",    "key_optional": False},
    "google":         {"label": "Google Gemini",             "url": "https://generativelanguage.googleapis.com/v1beta/openai", "api": "openai", "key_optional": False},
    "deepseek":       {"label": "DeepSeek",                  "url": "https://api.deepseek.com/v1",                        "api": "openai",    "key_optional": False},
    "xai":            {"label": "xAI (Grok)",                "url": "https://api.x.ai/v1",                                "api": "openai",    "key_optional": False},
    "groq":           {"label": "Groq",                      "url": "https://api.groq.com/openai/v1",                     "api": "openai",    "key_optional": False},
    "together":       {"label": "Together AI",               "url": "https://api.together.xyz/v1",                        "api": "openai",    "key_optional": False},
    "fireworks":      {"label": "Fireworks AI",              "url": "https://api.fireworks.ai/inference/v1",              "api": "openai",    "key_optional": False},
    "mistral":        {"label": "Mistral AI",                "url": "https://api.mistral.ai/v1",                          "api": "openai",    "key_optional": False},
    "moonshot":       {"label": "Moonshot / Kimi",           "url": "https://api.moonshot.ai/v1",                         "api": "openai",    "key_optional": False},
    "zai":            {"label": "Z.ai / Zhipu (GLM)",        "url": "https://api.z.ai/api/paas/v4",                       "api": "openai",    "key_optional": False},
    "nous":           {"label": "Nous Portal (Hermes)",      "url": "https://inference-api.nousresearch.com/v1",          "api": "openai",    "key_optional": False},
    "ollama":         {"label": "Ollama (local)",            "url": "http://127.0.0.1:11434/v1",                          "api": "openai",    "key_optional": True},
    "lmstudio":       {"label": "LM Studio (local)",         "url": "http://127.0.0.1:1234/v1",                           "api": "openai",    "key_optional": True},
    "vllm":           {"label": "vLLM / llama-server (local)", "url": "http://127.0.0.1:8000/v1",                         "api": "openai",    "key_optional": True},
    "local_fallback": {"label": "Active Local Engine Only",  "url": "",                                                   "api": "local",     "key_optional": True},
}

# ----------------------------------------------------------------------------
# Deep MoE mechanics (baked-in domain knowledge)
# ----------------------------------------------------------------------------

def analyze_moe_mechanics(
    model_meta: dict[str, Any],
    usable_vram_mb: float,
    avail_ram_mb: float,
    kv_vram_mb: float = 0.0,
    compute_mb: float = 400.0,
) -> dict[str, Any]:
    """Analyze MoE memory layout & feasibility.

    Crucial rule: NEVER reject an MoE model simply because total file size > VRAM.
    In Mixture-of-Experts GGUFs, embeddings + shared attention + router + KV cache form
    the GPU-critical hot path (~20-26% of weights), while routed FFN experts (~74-80%
    of weights) can sit in system RAM (`--n-cpu-moe` / `-ot ".ffn_.*_exps.=CPU"` / `-fit on`)
    because only `experts_used / experts` (typically 2-8 of 64-128) are touched per token.
    """
    is_moe = bool(model_meta.get("moe"))
    size_mb = float(model_meta.get("size") or 0) / 1048576.0
    blocks = int(model_meta.get("blocks") or 32)
    experts = int(model_meta.get("experts") or 0)
    experts_used = int(model_meta.get("experts_used") or 0)

    if not is_moe:
        fits_vram = (size_mb + kv_vram_mb + compute_mb) <= usable_vram_mb
        return {
            "is_moe": False,
            "feasible": fits_vram or (avail_ram_mb >= size_mb * 0.5),
            "strategy": "dense_full_gpu" if fits_vram else "dense_partial_offload",
            "total_weights_mb": round(size_mb),
            "gpu_hot_path_mb": round(size_mb + kv_vram_mb + compute_mb),
            "cpu_cold_experts_mb": 0,
            "recommended_ncmoe": 0,
            "notes": [
                "Dense architecture: every parameter is active on every token; full GPU offload (-ngl 99) is strongly preferred."
            ],
        }

    # MoE breakdown: shared/attention/router/embed ~24% of weights; routed experts ~76%
    shared_frac = 0.24
    expert_frac = 1.0 - shared_frac
    shared_mb = size_mb * shared_frac
    all_experts_mb = size_mb * expert_frac
    hot_base_mb = shared_mb + kv_vram_mb + compute_mb

    active_ratio = (experts_used / experts) if (experts > 0 and experts_used > 0) else 0.125
    per_layer_expert_mb = (all_experts_mb / max(1, blocks)) if blocks > 0 else 0.0

    # Check if ALL experts fit in VRAM alongside attention + KV
    if (size_mb + kv_vram_mb + compute_mb) <= usable_vram_mb:
        return {
            "is_moe": True,
            "feasible": True,
            "strategy": "moe_full_gpu",
            "experts": experts,
            "experts_used": experts_used,
            "active_expert_ratio": round(active_ratio, 4),
            "total_weights_mb": round(size_mb),
            "gpu_hot_path_mb": round(size_mb + kv_vram_mb + compute_mb),
            "cpu_cold_experts_mb": 0,
            "recommended_ncmoe": 0,
            "notes": [
                f"MoE model ({experts_used or '?'}/{experts or '?'} active experts/tok) fits entirely in VRAM "
                f"({round(size_mb + kv_vram_mb + compute_mb)} / {round(usable_vram_mb)} MB) — full GPU offload yields peak tok/s."
            ],
        }

    # Hybrid MoE split: keep attention + router + KV on GPU, spill only as many expert layers to CPU as needed!
    spare_vram_for_experts = max(0.0, usable_vram_mb - hot_base_mb)
    gpu_expert_layers = int(spare_vram_for_experts // per_layer_expert_mb) if per_layer_expert_mb > 0 else 0
    gpu_expert_layers = max(0, min(blocks, gpu_expert_layers))
    cpu_expert_layers = max(0, blocks - gpu_expert_layers)

    gpu_total_mb = hot_base_mb + gpu_expert_layers * per_layer_expert_mb
    cpu_total_mb = cpu_expert_layers * per_layer_expert_mb

    hot_fits_vram = hot_base_mb <= usable_vram_mb
    cold_fits_ram = cpu_total_mb <= max(avail_ram_mb, 1024.0)
    feasible = hot_fits_vram and cold_fits_ram

    notes = [
        f"MoE mechanics: total weights ({round(size_mb)} MB) exceed VRAM ({round(usable_vram_mb)} MB), "
        f"but only ~{round(active_ratio * 100, 1)}% of experts activate per token. "
        f"Attention + router + KV hot path needs ~{round(hot_base_mb)} MB VRAM.",
    ]
    if gpu_expert_layers > 0 and cpu_expert_layers > 0:
        notes.append(
            f"Optimal hybrid split: keep {gpu_expert_layers}/{blocks} expert layers in spare VRAM and offload "
            f"--n-cpu-moe {cpu_expert_layers} layers (~{round(cpu_total_mb)} MB) to system RAM."
        )
    elif cpu_expert_layers == blocks:
        notes.append(
            f"Full expert CPU offload (--n-cpu-moe {blocks} or moe-split / -fit on): keeps attention & KV on GPU "
            f"(~{round(hot_base_mb)} MB) and routed FFN experts (~{round(cpu_total_mb)} MB) in RAM."
        )
    if not hot_fits_vram:
        notes.append(
            f"Attention + KV hot path (~{round(hot_base_mb)} MB) exceeds usable VRAM ({round(usable_vram_mb)} MB); "
            f"quantize KV cache (q8_0/q4_0), reduce context, or reduce -ngl."
        )
    if not cold_fits_ram:
        notes.append(
            f"Offloaded experts (~{round(cpu_total_mb)} MB) may exceed available system RAM ({round(avail_ram_mb)} MB); "
            f"consider a smaller quant or freeing host RAM."
        )

    return {
        "is_moe": True,
        "feasible": feasible,
        "strategy": "moe_partial_cpu_experts" if gpu_expert_layers > 0 else "moe_full_cpu_experts",
        "experts": experts,
        "experts_used": experts_used,
        "active_expert_ratio": round(active_ratio, 4),
        "total_weights_mb": round(size_mb),
        "gpu_hot_path_mb": round(gpu_total_mb),
        "cpu_cold_experts_mb": round(cpu_total_mb),
        "gpu_expert_layers": gpu_expert_layers,
        "recommended_ncmoe": cpu_expert_layers,
        "notes": notes,
    }


# ----------------------------------------------------------------------------
# Documentation-Grounded Oracle (live --help + structured engine mechanics)
# ----------------------------------------------------------------------------

ENGINE_GUIDES: dict[str, dict[str, str]] = {
    "llama": {
        "moe": (
            "llama.cpp MoE tuning: keep `-ngl 99` so attention, norms, router, and KV cache stay on GPU. "
            "Use `--n-cpu-moe N` (newest builds) or `-ot '.ffn_.*_exps.=CPU'` to offload routed FFN experts to RAM. "
            "Set `-t` near physical cores (not hyperthreads) for CPU expert GEMV bandwidth; enable `-fa on`."
        ),
        "kv_cache": (
            "KV cache flags: `-ctk` (key cache type) and `-ctv` (value cache type). "
            "f16=2B/elem, q8_0=1.06B/elem (~2x savings, near-lossless), q4_0=0.56B/elem (~3.5x savings). "
            "Keys are more sensitive to quantization than values, so `q8_0 / q4_0` is the sweet spot for max context. "
            "Flash Attention (`-fa on`) is required for quantized V cache (`-ctv q4_0`/`q8_0`)."
        ),
        "speed": (
            "Speed tuning: full GPU offload (`-ngl 99`), `-fa on`, tune `-b` (2048-4096) and `-ub` (512-1024) "
            "for prompt prefill throughput, keep `--parallel 1` for single-user max tok/s, and attach a matching "
            "draft/MTP GGUF via `--model-draft` + `--draft-max 8..16` when available."
        ),
        "oom": (
            "OOM recovery order: (1) If MoE and full-GPU, switch to `moe-split` or increase `--n-cpu-moe`; "
            "(2) Enable `-fa on` and quantize KV cache to `-ctk q8_0 -ctv q4_0`; "
            "(3) Reduce `--parallel` to 1; (4) Halve `-c` context; (5) Reduce `-ngl` layer by layer."
        ),
    },
    "avalon": {
        "moe": (
            "Avalon targets large MoE Flash-Next packs. Default `-fit on` automatically keeps dense/attention/KV on GPU "
            "and places MoE experts in system RAM. Explicitly setting `ngl` switches to `-ngl N -fit off`; "
            "`n_cmoe` overrides `--n-cpu-moe` explicitly."
        ),
        "draft": (
            "Avalon speculative decoding uses `draft_n` (`--draft-max`) and `ubd` (`--ubatch-draft` / microbatch). "
            "Setting `draft_n=0` disables drafting if acceptance rate is low or VRAM is tight."
        ),
        "oom": (
            "Avalon OOM recovery: clear manual `ngl` to re-enable `-fit on`, switch KV cache to `q8_0/q4_0`, "
            "reduce `ctx` or `batch`/`ubatch`, or set `mlock=False` if system RAM is constrained."
        ),
    },
    "strata": {
        "plan": (
            "Strata supports read-only `strata launch --plan --output json` to preview exact hardware placement, "
            "memory budget, and downloads before launching."
        ),
        "kv_cache": (
            "Strata `--kv` accepts `fp16`, `int8`, `k8v4` (8-bit keys + 4-bit values, ideal for 128k+ context), "
            "or `q4_0`. Context is passed as words like `8k`, `32k`, `64k`, `128k`, `256k`."
        ),
        "drafting": (
            "Strata supports `--mtp on|off` and `--drafting auto|mtp|dflash2|off`. "
            "Use `--tune quick|thorough` to invoke Strata's internal kernel tuner, or `--set key=value` for advanced overrides."
        ),
    },
    "maya": {
        "moe": (
            "Project Maya runs MoE models (hy4 / glm engines on ROCm) by keeping hot routed experts in VRAM "
            "and streaming remaining experts from system RAM (logged as 'X of Y routed experts in VRAM'). "
            "Configs are JSON files in ~/project-maya with `exe`, `args`, `ctx`, `--mtp`, `reasoning_effort`, and `sampling`."
        ),
        "speed": (
            "Maya speed tuning: enable `--mtp` in `args` when supported by the config, set `reasoning_effort` to `low` "
            "for snappy responses, and keep `ctx` moderate (32k-65k) so more routed experts stay resident in GPU VRAM."
        ),
        "oom": (
            "Maya OOM recovery: disable `--mtp` to reclaim draft buffer VRAM, reduce `--ctx` / context length, "
            "and verify no other engine (llama.cpp / Avalon / Strata) is holding GPU VRAM. Never bind port 8080."
        ),
    },
}

ERROR_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "vram_oom",
        "regex": re.compile(
            r"(out of memory|cudaMalloc failed|HIP out of memory|VK_ERROR_OUT_OF_DEVICE_MEMORY|"
            r"failed to allocate .* buffer|not enough VRAM|ggml_backend_.*_buffer_type_alloc_buffer: failed)",
            re.I,
        ),
        "severity": "error",
        "title": "GPU VRAM Out-of-Memory",
        "cause": "Model weights + KV cache + compute buffer exceeded available GPU VRAM.",
    },
    {
        "id": "kv_alloc_oom",
        "regex": re.compile(
            r"(failed to allocate KV|llama_kv_cache_init: failed|kv_self|context size .* too large|n_ctx .* exceeds)",
            re.I,
        ),
        "severity": "error",
        "title": "KV Cache Allocation Failure",
        "cause": "Requested context window (`-c` / `--context`) and KV cache precision exceeded available memory.",
    },
    {
        "id": "unknown_flag",
        "regex": re.compile(
            r"(error: invalid argument|unknown argument|unrecognized option|invalid option|unexpected argument)[:\s]+([^\s\n]+)?",
            re.I,
        ),
        "severity": "error",
        "title": "Unsupported CLI Flag",
        "cause": "The installed engine binary rejected a CLI flag it does not advertise in `--help`.",
    },
    {
        "id": "fa_required_for_kv",
        "regex": re.compile(r"(flash.?attn.*required|cache_type_v.*flash|V cache quantization requires flash)", re.I),
        "severity": "error",
        "title": "Flash Attention Required for Quantized KV Cache",
        "cause": "Quantized V cache (`-ctv q4_0` / `q8_0`) requires Flash Attention (`-fa on`).",
    },
    {
        "id": "draft_mismatch",
        "regex": re.compile(r"(draft model.*vocab|incompatible draft|model-draft.*failed|speculative.*vocab)", re.I),
        "severity": "error",
        "title": "Draft / Speculative Model Mismatch",
        "cause": "The attached draft/MTP model tokenizer or architecture does not match the target model.",
    },
    {
        "id": "port_in_use",
        "regex": re.compile(r"(address already in use|bind.*failed|port \d+ .* in use)", re.I),
        "severity": "error",
        "title": "Port Already In Use",
        "cause": "Another server process is already bound to the target port.",
    },
    {
        "id": "model_missing",
        "regex": re.compile(r"(failed to open GGUF|No such file or directory.*\.gguf|invalid magic|failed to load model)", re.I),
        "severity": "error",
        "title": "Model File Unreadable or Incomplete",
        "cause": "The GGUF file path is missing, partially downloaded, or corrupted.",
    },
]


class EngineDocOracle:
    """Reads live engine `--help` / doctor output and combines it with structured domain guides."""

    def __init__(self, host: Any):
        self.host = host
        self._cache: dict[str, tuple[float, str]] = {}
        self._ttl = 300.0

    def _run_help(self, key: str, cmd: list[str]) -> str:
        now = time.time()
        if key in self._cache and (now - self._cache[key][0]) < self._ttl:
            return self._cache[key][1]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            txt = (r.stdout or "") + "\n" + (r.stderr or "")
        except Exception as e:
            txt = f"(live help unavailable: {e})"
        self._cache[key] = (now, txt)
        return txt

    def live_help(self, engine: str) -> str:
        if engine == "llama":
            b = self.host.cfg.get("server_binary") or self.host.default_bin
            return self._run_help(f"llama:{b}", [b, "--help"]) if b and Path(b).exists() else ""
        if engine == "avalon":
            b = self.host.avalon_bin
            return self._run_help(f"avalon:{b}", [b, "--help"]) if b and Path(b).exists() else ""
        if engine == "strata":
            b = self.host.strata_bin()
            return self._run_help(f"strata:{b}", [b, "launch", "--help"]) if b else ""
        return ""

    def lookup(self, engine: str, query: str = "") -> dict[str, Any]:
        """Consult bundled guides + live `--help` lines matching `query`."""
        guides = ENGINE_GUIDES.get(engine, {})
        q = (query or "").strip().lower()
        matched_guides = {}
        for topic, text in guides.items():
            if not q or q in topic.lower() or any(w in text.lower() for w in q.split() if len(w) >= 3):
                matched_guides[topic] = text
        if not matched_guides:
            matched_guides = dict(guides)

        help_text = self.live_help(engine)
        help_matches: list[str] = []
        if help_text and q:
            tokens = [w for w in re.split(r"[\s,/]+", q) if len(w) >= 2]
            for line in help_text.splitlines():
                ll = line.lower()
                if any(t in ll for t in tokens):
                    help_matches.append(line.rstrip())
                if len(help_matches) >= 25:
                    break
        elif help_text:
            help_matches = [ln.rstrip() for ln in help_text.splitlines() if ln.strip().startswith("-")][:25]

        return {
            "engine": engine,
            "query": query,
            "guides": matched_guides,
            "live_help_matches": help_matches,
            "live_help_available": bool(help_text and not help_text.startswith("(live help unavailable")),
        }


# ----------------------------------------------------------------------------
# Objective scoring
# ----------------------------------------------------------------------------

def score_trial(goal: str, metrics: dict[str, Any], config: dict[str, Any]) -> float:
    """Compute a comparable scalar score for an empirical benchmark trial."""
    tok_s = float(metrics.get("tok_s") or 0.0)
    prompt_s = float(metrics.get("prompt_s") or 0.0)
    ttft_ms = float(metrics.get("ttft_ms") or 5000.0)
    vram_mb = float(metrics.get("vram_mb") or 0.0)
    ctx = _parse_ctx_int(config.get("ctx") or config.get("context") or 8192)
    parallel = int(config.get("parallel") or 1)

    if goal == "fastest":
        return tok_s * 10.0 + prompt_s * 0.25 - (ttft_ms / 1000.0)
    if goal == "low_latency":
        # Strongly reward fast TTFT and high prefill speed, plus solid generation
        return (5000.0 / max(20.0, ttft_ms)) * 20.0 + prompt_s * 0.5 + tok_s * 2.0
    if goal == "max_context":
        # Strongly reward verified context size, tie-break with speed
        return (ctx / 1024.0) * 15.0 + tok_s * 2.0
    if goal == "low_vram":
        # Reward low VRAM footprint while still requiring usable speed
        return tok_s * 4.0 - (vram_mb / 256.0)
    if goal == "multi_user":
        return (parallel * 40.0) + tok_s * 5.0 + prompt_s * 0.15
    # balanced
    return tok_s * 6.0 + prompt_s * 0.2 + (ctx / 4096.0) * 4.0 - (ttft_ms / 1500.0)


def _parse_ctx_int(val: Any) -> int:
    if val is None or val == "":
        return 8192
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().lower()
    m = re.match(r"^(\d+)\s*k$", s)
    if m:
        return int(m.group(1)) * 1024
    try:
        return int(s)
    except ValueError:
        return 8192


def _ctx_to_strata_word(ctx: int) -> str:
    k = max(4, round(ctx / 1024))
    # Snap to clean power-of-two / common Strata tiers
    for tier in (8, 16, 32, 64, 128, 256, 512):
        if k <= tier:
            return f"{tier}k"
    return "512k"


# ----------------------------------------------------------------------------
# Host adapter protocol (decouples OptimizerAgent from panel.py globals)
# ----------------------------------------------------------------------------

@dataclass
class HostContext:
    cfg: dict[str, Any]
    default_bin: str
    avalon_bin: str
    avalon_profiles: dict[str, Any]
    launch_profiles_path: Path
    vitals: Callable[..., dict[str, Any]]
    scan_models: Callable[[], dict[str, Any]]
    estimate_vram: Callable[..., dict[str, Any]]
    bin_caps: Callable[[str | None], dict[str, Any]]
    find_draft: Callable[[str], str]
    save_config: Callable[[], None]
    mgr: Any
    smgr: Any
    avalon_target: Callable[[], dict[str, Any]]
    strata_bin: Callable[[], str | None]
    strata_status: Callable[..., dict[str, Any]]
    strata_build_args: Callable[[Any, bool], list[str]]
    strata_run: Callable[..., Any]
    strata_launch_body_cls: Any
    strata_stop: Callable[[], dict[str, Any]]
    strata_log_path: Path
    chat_target: Callable[[], tuple[int, str] | None]
    benchmark_run: Callable[[str, int], dict[str, Any]]
    read_json: Callable[[Path, Any], Any]
    write_json: Callable[[Path, Any], None]
    # Phase 2 optional hooks (default None for backward compatibility)
    sync_agents_fn: Callable[[list[str], str], dict[str, Any]] | None = None
    coding_agents_list: Callable[[], list[str]] | None = None
    hf_search_fn: Callable[[str], list[dict[str, Any]]] | None = None
    hf_files_fn: Callable[[str], list[dict[str, Any]]] | None = None
    hf_pull_fn: Callable[[str, list[str], dict[str, int]], dict[str, Any]] | None = None
    downloads_fn: Callable[[], list[dict[str, Any]]] | None = None
    thermal_status_fn: Callable[[], dict[str, Any]] | None = None
    thermal_eval_fn: Callable[[], dict[str, Any]] | None = None
    strata_doctor_fn: Callable[[], dict[str, Any]] | None = None
    maya_mgr: Any = None


# ----------------------------------------------------------------------------
# Comprehensive Clickable Action Menu & Templates (single source of truth)
# ----------------------------------------------------------------------------

AGENT_ACTION_TEMPLATES: list[dict[str, Any]] = [
    # ---- Quant & Download ----
    {
        "id": "hf_quant_empty_gpu",
        "category": "Quant & Download",
        "label": "Calculate & Pull Best Quant (Empty GPU)",
        "description": "Search Hugging Face GGUFs and rank the best quant assuming full GPU VRAM is unoccupied.",
        "instant": False,
        "fields": [
            {"name": "query", "label": "Model name or HF repo", "type": "text", "required": True, "placeholder": "e.g. Qwen3-32B, DeepSeek-R1, unsloth/Qwen3-30B-A3B-GGUF"},
            {"name": "goal", "label": "Target goal", "type": "select", "default": "balanced", "options": ["balanced", "fastest", "max_context", "low_vram"]},
            {"name": "assume_empty_gpu", "label": "Assume empty GPU (ignore current VRAM use)", "type": "checkbox", "default": True},
            {"name": "auto_pull", "label": "Queue top matching quant immediately", "type": "checkbox", "default": False},
        ],
        "prompt_template": "Calculate best quant for {query} ({goal}, empty GPU={assume_empty_gpu})",
    },
    {
        "id": "hf_quant_free_vram",
        "category": "Quant & Download",
        "label": "Calculate Quant (Free VRAM Only)",
        "description": "Find the best quant that fits in currently free VRAM alongside running workloads.",
        "instant": False,
        "fields": [
            {"name": "query", "label": "Model name or HF repo", "type": "text", "required": True, "placeholder": "e.g. Llama-3.1-8B, Qwen2.5-Coder-14B"},
            {"name": "goal", "label": "Target goal", "type": "select", "default": "low_vram", "options": ["low_vram", "balanced", "fastest", "max_context"]},
            {"name": "assume_empty_gpu", "label": "Assume empty GPU", "type": "checkbox", "default": False},
            {"name": "auto_pull", "label": "Queue top matching quant immediately", "type": "checkbox", "default": False},
        ],
        "prompt_template": "Calculate best quant for {query} using currently free VRAM ({goal})",
    },
    {
        "id": "downloads_status",
        "category": "Quant & Download",
        "label": "Check Active Downloads",
        "description": "Show Hugging Face download queue, progress, and recent transfers.",
        "instant": True,
        "fields": [],
        "prompt_template": "Show active Hugging Face downloads",
    },
    # ---- Optimize & Tune ----
    {
        "id": "optimize_engine",
        "category": "Optimize & Tune",
        "label": "Optimize Engine Config",
        "description": "Compute a hardware-tuned config (dry-run) for llama.cpp, Avalon, Strata, or Maya.",
        "instant": False,
        "fields": [
            {"name": "engine", "label": "Engine", "type": "select", "default": "llama", "options": ["llama", "avalon", "strata", "maya"]},
            {"name": "goal", "label": "Goal", "type": "select", "default": "balanced", "options": ["balanced", "fastest", "max_context", "low_vram", "low_latency", "multi_user"]},
            {"name": "model", "label": "Local model (optional)", "type": "model_select", "default": ""},
            {"name": "notes", "label": "Extra constraints (optional)", "type": "text", "placeholder": "e.g. ctx 32k, keep 2gb free"},
        ],
        "prompt_template": "Optimize {engine} for {goal}",
    },
    {
        "id": "start_autotune",
        "category": "Optimize & Tune",
        "label": "Start Active Auto-Tune",
        "description": "Create a safety checkpoint and run benchmarked config trials to find the fastest setup.",
        "instant": False,
        "fields": [
            {"name": "engine", "label": "Engine", "type": "select", "default": "llama", "options": ["llama", "avalon", "strata", "maya"]},
            {"name": "goal", "label": "Goal", "type": "select", "default": "fastest", "options": ["fastest", "balanced", "max_context", "low_vram", "low_latency", "multi_user"]},
            {"name": "model", "label": "Local model (optional)", "type": "model_select", "default": ""},
            {"name": "max_trials", "label": "Max trials", "type": "number", "default": 5, "min": 1, "max": 12},
        ],
        "prompt_template": "Start active auto-tune on {engine} for {goal} ({max_trials} trials)",
    },
    {
        "id": "stop_autotune",
        "category": "Optimize & Tune",
        "label": "Stop / Rollback Auto-Tune",
        "description": "Stop an active auto-tune session, optionally rolling back to its pre-tune checkpoint.",
        "instant": False,
        "fields": [
            {"name": "rollback", "label": "Rollback to pre-tune checkpoint", "type": "checkbox", "default": False},
        ],
        "prompt_template": "Stop auto-tune (rollback={rollback})",
    },
    {
        "id": "route_workload",
        "category": "Optimize & Tune",
        "label": "Route Workload to Best Engine",
        "description": "Pick the best installed model & engine for coding, vision, reasoning, chat, or long context.",
        "instant": False,
        "fields": [
            {"name": "task", "label": "Workload", "type": "select", "default": "coding", "options": ["coding", "chat", "vision", "reasoning", "long_context"]},
            {"name": "apply", "label": "Apply & launch winner now", "type": "checkbox", "default": False},
            {"name": "sync_agents", "label": "Sync coding agents after launch", "type": "checkbox", "default": False},
        ],
        "prompt_template": "Route {task} workload to best engine",
    },
    {
        "id": "tune_speculation",
        "category": "Optimize & Tune",
        "label": "Benchmark Speculative Decoding",
        "description": "Compare baseline vs draft/MTP models on llama.cpp and keep the faster setup.",
        "instant": False,
        "fields": [
            {"name": "model", "label": "Local model (optional)", "type": "model_select", "default": ""},
            {"name": "apply_best", "label": "Apply fastest configuration", "type": "checkbox", "default": True},
        ],
        "prompt_template": "Benchmark speculative decoding",
    },
    # ---- Engines & Benchmarks ----
    {
        "id": "system_status",
        "category": "Engines & Benchmarks",
        "label": "Hardware & Engine Status",
        "description": "Inspect GPU VRAM, system RAM, CPU, and which engines are currently running.",
        "instant": True,
        "fields": [],
        "prompt_template": "Show hardware vitals and running engine status",
    },
    {
        "id": "run_benchmark",
        "category": "Engines & Benchmarks",
        "label": "Run Speed Benchmark",
        "description": "Benchmark prompt & generation tok/s on whichever engine is currently serving.",
        "instant": False,
        "fields": [
            {"name": "prompt_key", "label": "Suite", "type": "select", "default": "short", "options": ["short", "standard", "code"]},
            {"name": "max_tokens", "label": "Max tokens", "type": "number", "default": 128, "min": 16, "max": 1024},
        ],
        "prompt_template": "Run {prompt_key} benchmark ({max_tokens} tokens)",
    },
    {
        "id": "stop_engine",
        "category": "Engines & Benchmarks",
        "label": "Stop an Engine",
        "description": "Stop a specific inference engine to free GPU VRAM.",
        "instant": False,
        "fields": [
            {"name": "engine", "label": "Engine to stop", "type": "select", "default": "llama", "options": ["llama", "avalon", "strata", "maya"]},
        ],
        "prompt_template": "Stop {engine} engine",
    },
    {
        "id": "stop_all_engines",
        "category": "Engines & Benchmarks",
        "label": "Stop All Engines (Free GPU)",
        "description": "Create a safety checkpoint and stop all running engines so the GPU is completely empty.",
        "instant": True,
        "fields": [],
        "prompt_template": "Stop all running engines and free GPU VRAM",
    },
    {
        "id": "strata_doctor",
        "category": "Engines & Benchmarks",
        "label": "Strata Doctor Check",
        "description": "Run Strata CLI health diagnostics.",
        "instant": True,
        "fields": [],
        "prompt_template": "Run Strata doctor diagnostics",
    },
    # ---- Diagnose & Safety ----
    {
        "id": "diagnose_engine",
        "category": "Diagnose & Safety",
        "label": "Diagnose Last Failure",
        "description": "Inspect engine logs for OOM, bad flags, or draft mismatches and propose a fix.",
        "instant": False,
        "fields": [
            {"name": "engine", "label": "Engine", "type": "select", "default": "llama", "options": ["llama", "avalon", "strata", "maya"]},
        ],
        "prompt_template": "Diagnose last failure on {engine}",
    },
    {
        "id": "heal_engine",
        "category": "Diagnose & Safety",
        "label": "Auto-Heal Engine",
        "description": "Diagnose the last failure and immediately apply the safe fix + relaunch.",
        "instant": False,
        "fields": [
            {"name": "engine", "label": "Engine", "type": "select", "default": "llama", "options": ["llama", "avalon", "strata", "maya"]},
        ],
        "prompt_template": "Auto-heal {engine}",
    },
    {
        "id": "thermal_status",
        "category": "Diagnose & Safety",
        "label": "Thermal Guard Status",
        "description": "Check GPU temperature, VRAM pressure, and thermal guard actions.",
        "instant": True,
        "fields": [],
        "prompt_template": "Check thermal guard status",
    },
    {
        "id": "thermal_downshift",
        "category": "Diagnose & Safety",
        "label": "Downshift Active Engine",
        "description": "Checkpoint and downshift the active engine to a cooler / low-VRAM profile.",
        "instant": True,
        "fields": [],
        "prompt_template": "Downshift active engine to low-VRAM/cooler profile",
    },
    {
        "id": "checkpoints_list",
        "category": "Diagnose & Safety",
        "label": "Checkpoints & Rollback",
        "description": "View saved safety checkpoints with one-click restore.",
        "instant": True,
        "fields": [],
        "prompt_template": "Show checkpoints",
    },
    {
        "id": "checkpoint_create",
        "category": "Diagnose & Safety",
        "label": "Create Safety Checkpoint",
        "description": "Snapshot the current engine configuration before making changes.",
        "instant": False,
        "fields": [
            {"name": "engine", "label": "Engine", "type": "select", "default": "llama", "options": ["llama", "avalon", "strata", "maya"]},
            {"name": "label", "label": "Label (optional)", "type": "text", "placeholder": "e.g. Before quant swap"},
        ],
        "prompt_template": "Create checkpoint for {engine}",
    },
    # ---- Library & Coding Agents ----
    {
        "id": "list_models",
        "category": "Library & Coding Agents",
        "label": "List Local Models & Fit",
        "description": "List installed GGUF models with sizes, quants, and MoE/VRAM fit.",
        "instant": True,
        "fields": [],
        "prompt_template": "List local models and VRAM fit",
    },
    {
        "id": "presets_list",
        "category": "Library & Coding Agents",
        "label": "Saved Optimization Presets",
        "description": "View built-in and saved presets with one-click apply.",
        "instant": True,
        "fields": [],
        "prompt_template": "Show saved optimization presets",
    },
    {
        "id": "sync_coding_agents",
        "category": "Library & Coding Agents",
        "label": "Register in Coding Agents",
        "description": "Point omp, opencode, codex, aider, zed, pi, claude-code, hermes, and kimi at the active server.",
        "instant": False,
        "fields": [
            {"name": "source", "label": "Engine source", "type": "select", "default": "auto", "options": ["auto", "llama", "avalon", "strata", "maya"]},
        ],
        "prompt_template": "Register active engine ({source}) in coding agents",
    },
    {
        "id": "agent_memory",
        "category": "Library & Coding Agents",
        "label": "Agent Experience Memory",
        "description": "Show top recorded benchmark winners and VRAM calibrations.",
        "instant": True,
        "fields": [],
        "prompt_template": "Show agent experience memory",
    },
]


# ----------------------------------------------------------------------------
# Main Hybrid Optimization Agent
# ----------------------------------------------------------------------------

class OptimizerAgent:
    """Unified 3-engine hybrid optimization agent."""

    def __init__(self, host: HostContext, store: AgentStore):
        self.host = host
        self.store = store
        self.docs = EngineDocOracle(host)
        self._tune_lock = threading.Lock()
        self._tune_session: dict[str, Any] | None = None
        self._tune_cancel = threading.Event()
        self._chat_history: list[dict[str, Any]] = []
        self._last_llm_error: str = ""

    # ------------------------------------------------------------ hardware & model helpers
    def _hardware_sig(self, vit: dict[str, Any]) -> str:
        g = vit.get("gpu") or {}
        m = vit.get("mem") or {}
        c = vit.get("cpu") or {}
        return f"gpu:{g.get('name', 'none')}:{g.get('total_mb', 0)}|ram:{m.get('total_gb', 0)}|cpu:{c.get('cores', 0)}"

    def _find_model(self, model_path_or_name: str = "") -> dict[str, Any] | None:
        models = (self.host.scan_models() or {}).get("models") or []
        if not models:
            return None
        if not model_path_or_name:
            # Prefer currently running llama.cpp model, else first pinned, else first model
            srv = self.host.mgr.status()
            if srv.get("running") and srv.get("model"):
                for m in models:
                    if m.get("path") == srv.get("model"):
                        return m
            return next((m for m in models if m.get("pinned")), models[0])
        q = model_path_or_name.strip().lower()
        for m in models:
            if m.get("path", "").lower() == q or m.get("name", "").lower() == q:
                return m
        for m in models:
            if q in m.get("name", "").lower() or q in m.get("path", "").lower():
                return m
        return None

    def calibrated_estimate_vram(
        self,
        model_meta: dict[str, Any],
        *,
        ctx: int = 8192,
        ngl: int = 99,
        ctk: str = "f16",
        ctv: str = "f16",
        parallel: int = 1,
        preset: str = "auto",
        ncmoe: int = 0,
        vit: dict[str, Any] | None = None,
        engine: str = "llama",
    ) -> dict[str, Any]:
        est = dict(
            self.host.estimate_vram(
                model_meta,
                ctx=ctx,
                ngl=ngl,
                ctk=ctk,
                ctv=ctv,
                parallel=parallel,
                preset=preset,
                ncmoe=ncmoe,
                vit=vit,
            )
        )
        cal = self.store.get_vram_calibration(
            model_meta.get("arch", "unknown"),
            bool(model_meta.get("moe")),
            engine=engine,
        )
        est["calibration"] = cal
        if cal.get("applied"):
            raw_vram = est.get("total_vram_mb", 0)
            adj_vram = round(raw_vram * cal["scale"])
            est["raw_vram_mb"] = raw_vram
            est["total_vram_mb"] = adj_vram
            est["vram_used_mb"] = adj_vram
            usable = max(1, est.get("usable_mb", 1))
            est["pct"] = min(999, round(100 * adj_vram / usable))
            ratio = adj_vram / usable
            if ratio <= 0.85:
                est["status"], est["badge"], est["label"] = "fit", "Comfortably fits", "Comfortably fits"
            elif ratio <= 1.0:
                est["status"], est["badge"], est["label"] = "tight", "Tight fit", "Tight fit"
            else:
                est["status"], est["badge"], est["label"] = "overflow", "Exceeds memory", "Exceeds memory"
        return est

    # ------------------------------------------------------------ checkpoints & rollback
    def capture_checkpoint(self, engine: str, label: str = "", reason: str = "") -> dict[str, Any]:
        """Capture full restorable state across engines & configs before any mutation."""
        profiles = self.host.read_json(self.host.launch_profiles_path, {})
        maya_status = {}
        maya_configs_backup = {}
        if self.host.maya_mgr:
            try:
                maya_status = self.host.maya_mgr.status()
                for c in self.host.maya_mgr.discover_configs():
                    p = Path(c["path"])
                    if p.is_file():
                        maya_configs_backup[c["path"]] = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        snapshot = {
            "engine": engine,
            "llama_status": self.host.mgr.status(),
            "avalon_status": self.host.smgr.status(),
            "strata_status": self.host.strata_status(refresh=False),
            "maya_status": maya_status,
            "maya_configs": maya_configs_backup,
            "launch_profiles": copy.deepcopy(profiles),
            "avalon_tune": copy.deepcopy(self.host.cfg.get("avalon_tune") or {}),
            "strata_last": copy.deepcopy(self.host.cfg.get("strata_last") or {}),
            "strata_ctx": self.host.cfg.get("strata_ctx", 8192),
            "vram_reserve_mb": self.host.cfg.get("vram_reserve_mb", 512),
            "ctx": self.host.cfg.get("ctx", 8192),
        }
        return self.store.create_checkpoint(
            label=label or f"Auto-checkpoint ({engine})",
            engine=engine,
            snapshot=snapshot,
            reason=reason,
        )

    def restore_checkpoint(self, cp_id: str, relaunch: bool = False) -> dict[str, Any]:
        cp = self.store.get_checkpoint(cp_id)
        if not cp:
            raise ValueError(f"Checkpoint {cp_id} not found")
        snap = cp.get("snapshot") or {}
        if "launch_profiles" in snap:
            self.host.write_json(self.host.launch_profiles_path, snap["launch_profiles"])
        if "avalon_tune" in snap:
            self.host.cfg["avalon_tune"] = copy.deepcopy(snap["avalon_tune"])
        if "strata_last" in snap:
            self.host.cfg["strata_last"] = copy.deepcopy(snap["strata_last"])
        if "strata_ctx" in snap:
            self.host.cfg["strata_ctx"] = snap["strata_ctx"]
        if "vram_reserve_mb" in snap:
            self.host.cfg["vram_reserve_mb"] = snap["vram_reserve_mb"]
        if "ctx" in snap:
            self.host.cfg["ctx"] = snap["ctx"]
        for cpath, raw_txt in (snap.get("maya_configs") or {}).items():
            try:
                p = Path(cpath)
                if p.parent.is_dir():
                    p.write_text(raw_txt, encoding="utf-8")
            except OSError:
                pass
        self.host.save_config()

        relaunch_result = None
        if relaunch:
            eng = snap.get("engine") or cp.get("engine")
            if eng == "llama":
                st = snap.get("llama_status") or {}
                if st.get("running") and st.get("model"):
                    prof = (snap.get("launch_profiles") or {}).get(st["model"]) or {}
                    relaunch_result = self.host.mgr.start(
                        st["model"],
                        preset=st.get("preset") or prof.get("preset") or "auto",
                        ctx=int(prof.get("ctx") or st.get("ctx") or 8192),
                        ngl=int(prof.get("ngl") or 99),
                        threads=int(prof.get("threads") or 0),
                        fa=str(prof.get("fa") or "auto"),
                        ctk=str(prof.get("ctk") or "f16"),
                        ctv=str(prof.get("ctv") or "f16"),
                        parallel=int(prof.get("parallel") or 1),
                        batch=int(prof.get("batch") or 0),
                        ubatch=int(prof.get("ubatch") or 0),
                        ncmoe=int(prof.get("ncmoe") or 0),
                        extra_args=str(prof.get("extra_args") or ""),
                    )
                else:
                    self.host.mgr.stop()
            elif eng == "avalon":
                st = snap.get("avalon_status") or {}
                if st.get("running") and st.get("profile"):
                    tune = (snap.get("avalon_tune") or {}).get(st["profile"])
                    relaunch_result = self.host.smgr.start(st["profile"], tune)
                else:
                    self.host.smgr.stop()
            elif eng == "maya" and self.host.maya_mgr:
                st = snap.get("maya_status") or {}
                if st.get("running") and st.get("config_id"):
                    relaunch_result = self.host.maya_mgr.switch(st["config_id"])
                else:
                    self.host.maya_mgr.stop()

        return {
            "ok": True,
            "restored_checkpoint": cp_id,
            "label": cp.get("label"),
            "engine": cp.get("engine"),
            "relaunch": relaunch_result,
        }

    # ------------------------------------------------------------ Dry-Run Advisor (all 3 engines)
    def advise(
        self,
        *,
        engine: str = "llama",
        goal: str = "balanced",
        model: str = "",
        profile: str = "flagship",
        constraints: dict[str, Any] | None = None,
        user_prompt: str = "",
        use_llm: bool = True,
    ) -> dict[str, Any]:
        engine = (engine or "llama").lower()
        if engine not in ENGINES:
            raise ValueError(f"Unsupported engine: {engine}")
        goal = (goal or "balanced").lower()
        if goal not in GOALS:
            goal = "balanced"
        constraints = dict(constraints or {})

        # Parse simple natural-language hints from user_prompt even without an LLM
        if user_prompt:
            goal, constraints = self._parse_prompt_constraints(user_prompt, goal, constraints)

        vit = self.host.vitals(refresh=False)
        if engine == "llama":
            rec = self._advise_llama(model=model, goal=goal, constraints=constraints, vit=vit)
        elif engine == "avalon":
            rec = self._advise_avalon(profile=profile, goal=goal, constraints=constraints, vit=vit)
        elif engine == "maya":
            rec = self._advise_maya(model=model, goal=goal, constraints=constraints, vit=vit)
        else:
            rec = self._advise_strata(model=model, goal=goal, constraints=constraints, vit=vit)

        # Attach historical best & failure warnings from persistent memory
        model_key = rec.get("model_key", "")
        past_best = self.store.get_best(engine, model_key, goal)
        recent_fails = self.store.get_recent_failures(engine, model_key, limit=5)
        rec["past_best"] = past_best
        rec["recent_failures"] = recent_fails
        if past_best:
            m = past_best.get("metrics") or {}
            rec["notes"].append(
                f"Persistent memory has a verified {goal} winner for this model "
                f"({m.get('tok_s', '?')} tok/s, {m.get('vram_mb', '?')} MB VRAM)."
            )

        # Optional LLM commentary / refinement (only if user gave a custom prompt or explicitly enabled)
        llm_note = None
        if use_llm and user_prompt:
            llm_note = self._llm_refine_advice(rec, user_prompt, vit)
        rec["llm_analysis"] = llm_note
        rec["hardware"] = vit
        return rec

    def _parse_prompt_constraints(
        self, text: str, goal: str, constraints: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        t = text.lower()
        c = dict(constraints)
        if any(k in t for k in ("fastest", "max speed", "highest tok", "tok/s", "quickest")):
            goal = "fastest"
        elif any(k in t for k in ("ttft", "low latency", "prefill", "snappy")):
            goal = "low_latency"
        elif any(k in t for k in ("longest context", "max context", "max ctx", "long context", "huge context")):
            goal = "max_context"
        elif any(k in t for k in ("low vram", "quiet", "cool", "small footprint", "gaming")):
            goal = "low_vram"
        elif any(k in t for k in ("multi", "concurrent", "parallel", "team", "slots")):
            goal = "multi_user"

        m_ctx = re.search(r"(\d+)\s*k\s*(?:ctx|context|tokens?)?", t)
        if m_ctx and "min_ctx" not in c:
            c["min_ctx"] = int(m_ctx.group(1)) * 1024
        m_free = re.search(r"leave\s+(\d+(?:\.\d+)?)\s*gb", t)
        if m_free and "vram_headroom_mb" not in c:
            c["vram_headroom_mb"] = int(float(m_free.group(1)) * 1024)
        m_par = re.search(r"(\d+)\s*(?:parallel|slots|users)", t)
        if m_par and "parallel" not in c:
            c["parallel"] = max(1, min(8, int(m_par.group(1))))
        return goal, c

    def _advise_llama(
        self,
        *,
        model: str,
        goal: str,
        constraints: dict[str, Any],
        vit: dict[str, Any],
    ) -> dict[str, Any]:
        meta = self._find_model(model)
        if not meta:
            raise ValueError("No GGUF models found in configured model_dirs")

        caps = self.host.bin_caps(self.host.cfg.get("server_binary") or self.host.default_bin)
        cores = int((vit.get("cpu") or {}).get("cores") or os.cpu_count() or 8)
        phys_threads = max(4, min(16, cores // 2 if cores >= 12 else cores))
        blocks = int(meta.get("blocks") or 32)
        native_ctx = int(meta.get("ctx_len") or 32768)
        is_moe = bool(meta.get("moe"))
        is_vl = bool(meta.get("vl") and meta.get("mmproj"))

        # Baseline targets by goal
        if goal == "fastest":
            ctk, ctv = "q8_0", "q8_0"
            ctx = min(native_ctx, 16384)
            parallel, batch, ubatch = 1, 2048, 512
        elif goal == "low_latency":
            ctk, ctv = "q8_0", "q8_0"
            ctx = min(native_ctx, 16384)
            parallel, batch, ubatch = 1, 4096, 1024
        elif goal == "max_context":
            ctk, ctv = "q8_0", "q4_0"
            ctx = min(native_ctx, 131072)
            parallel, batch, ubatch = 1, 2048, 512
        elif goal == "low_vram":
            ctk, ctv = "q4_0", "q4_0"
            ctx = min(native_ctx, 8192)
            parallel, batch, ubatch = 1, 1024, 256
        elif goal == "multi_user":
            ctk, ctv = "q8_0", "q4_0"
            ctx = min(native_ctx, 32768)
            parallel, batch, ubatch = 4, 2048, 512
        else:  # balanced
            ctk, ctv = "q8_0", "q8_0"
            ctx = min(native_ctx, 32768 if (meta.get("size", 0) < 16 * 1024**3) else 16384)
            parallel, batch, ubatch = 1, 2048, 512

        if constraints.get("min_ctx"):
            ctx = max(ctx, min(native_ctx or 262144, int(constraints["min_ctx"])))
        if constraints.get("parallel"):
            parallel = max(1, min(8, int(constraints["parallel"])))

        # Compute effective VRAM budget accounting for optional user headroom
        base_fit = self.calibrated_estimate_vram(
            meta, ctx=ctx, ngl=99, ctk=ctk, ctv=ctv, parallel=parallel,
            preset="vision" if is_vl else "dense-gpu", vit=vit, engine="llama",
        )
        extra_headroom = int(constraints.get("vram_headroom_mb") or 0)
        usable_vram = max(512, base_fit.get("usable_mb", 8192) - extra_headroom)
        avail_ram_mb = max(
            1024.0,
            (float((vit.get("mem") or {}).get("total_gb") or 32) - float((vit.get("mem") or {}).get("used_gb") or 8)) * 1024.0,
        )

        moe_info = analyze_moe_mechanics(
            meta,
            usable_vram_mb=usable_vram,
            avail_ram_mb=avail_ram_mb,
            kv_vram_mb=float(base_fit.get("kv_vram_mb") or 0),
            compute_mb=float(base_fit.get("compute_vram_mb") or 400),
        )

        notes: list[str] = list(moe_info.get("notes") or [])
        ngl = 99
        ncmoe = 0
        preset = "vision" if is_vl else "dense-gpu"

        if is_moe:
            if moe_info["strategy"] == "moe_full_gpu":
                preset = "vision" if is_vl else "dense-gpu"
                ncmoe = 0
            else:
                # Prefer --n-cpu-moe if the binary supports it so spare VRAM can hold hot experts
                if caps.get("n_cpu_moe") and moe_info.get("recommended_ncmoe", 0) < blocks:
                    preset = "vision" if is_vl else "dense-gpu"
                    ncmoe = int(moe_info["recommended_ncmoe"])
                else:
                    preset = "moe-split"
                    ncmoe = 0
                phys_threads = max(8, min(24, max(4, cores - 2)))
        else:
            # Dense model fit adjustment
            fit_now = self.calibrated_estimate_vram(
                meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, engine="llama"
            )
            if fit_now["total_vram_mb"] > usable_vram:
                if (ctk, ctv) != ("q8_0", "q4_0"):
                    ctk, ctv = "q8_0", "q4_0"
                    notes.append("Switched KV cache to Q8_0/Q4_0 to fit inside VRAM budget.")
                fit_now = self.calibrated_estimate_vram(
                    meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, engine="llama"
                )
            if fit_now["total_vram_mb"] > usable_vram and goal != "max_context":
                while ctx > 4096 and self.calibrated_estimate_vram(
                    meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, engine="llama"
                )["total_vram_mb"] > usable_vram:
                    ctx //= 2
                notes.append(f"Scaled context window to {ctx:,} tokens to preserve full GPU offload.")
            fit_now = self.calibrated_estimate_vram(
                meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, engine="llama"
            )
            if fit_now["total_vram_mb"] > usable_vram and blocks > 0:
                for cand in range(blocks, -1, -1):
                    if self.calibrated_estimate_vram(
                        meta, ctx=ctx, ngl=cand, ctk=ctk, ctv=ctv, parallel=parallel, preset=preset, vit=vit, engine="llama"
                    )["total_vram_mb"] <= usable_vram:
                        ngl = cand
                        break
                else:
                    ngl = 0
                notes.append(f"Partial GPU offload: {ngl}/{blocks} layers on GPU, rest in system RAM.")

        # Respect binary capabilities (doc-grounded flag gating!)
        fa = "on" if caps.get("flash_attn", True) else "auto"
        if not caps.get("kv_quant", True):
            ctk, ctv = "f16", "f16"
            notes.append("Binary --help does not advertise -ctk/-ctv; keeping f16 KV cache.")
        elif fa != "on" and ctv != "f16":
            ctv = "f16"
            notes.append("Disabled V cache quantization because Flash Attention is not advertised by this binary.")

        draft = ""
        draft_max = 16
        if goal in ("fastest", "low_latency", "balanced") and meta.get("path") and caps.get("draft", True):
            d = self.host.find_draft(meta["path"])
            if d:
                draft = d
                notes.append(f"Auto-detected draft/MTP model ({Path(d).name}) for speculative decoding.")

        final_fit = self.calibrated_estimate_vram(
            meta, ctx=ctx, ngl=ngl, ctk=ctk, ctv=ctv, parallel=parallel,
            preset=preset, ncmoe=ncmoe, vit=vit, engine="llama",
        )

        cfg = {
            "model": meta["path"],
            "preset": preset,
            "ctx": ctx,
            "ngl": ngl,
            "threads": phys_threads,
            "fa": fa,
            "ctk": ctk,
            "ctv": ctv,
            "parallel": parallel,
            "batch": batch,
            "ubatch": ubatch,
            "ncmoe": ncmoe,
            "draft": draft,
            "draft_max": draft_max if draft else 0,
            "jinja": "on",
            "extra_args": "",
        }

        return {
            "engine": "llama",
            "goal": goal,
            "model_key": meta.get("name") or Path(meta["path"]).name,
            "model_path": meta["path"],
            "config": cfg,
            "fit": final_fit,
            "moe": moe_info,
            "notes": notes,
            "docs_consulted": self.docs.lookup("llama", "moe" if is_moe else goal)["guides"],
        }

    def _advise_avalon(
        self,
        *,
        profile: str,
        goal: str,
        constraints: dict[str, Any],
        vit: dict[str, Any],
    ) -> dict[str, Any]:
        if profile not in self.host.avalon_profiles:
            profile = next(iter(self.host.avalon_profiles.keys()), "flagship")
        tgt = self.host.avalon_target() or {}
        cores = int((vit.get("cpu") or {}).get("cores") or os.cpu_count() or 8)
        threads = max(8, min(24, max(4, cores - 2)))

        tune: dict[str, Any] = {"fa": "on", "threads": threads}
        notes: list[str] = [
            "Avalon runs MoE Flash-Next packs with `-fit on` by default (dense attention + KV on GPU, MoE experts in RAM)."
        ]

        if goal == "fastest":
            tune.update({"ctx": 32768, "batch": 2048, "ubatch": 512, "ctk": "q8_0", "ctv": "q8_0", "parallel": 1})
            if tgt.get("draft"):
                tune["draft_n"] = 12
                notes.append("Draft GGUF detected — enabling speculative decoding (`draft_n=12`) for peak tok/s.")
        elif goal == "low_latency":
            tune.update({"ctx": 16384, "batch": 4096, "ubatch": 1024, "ctk": "q8_0", "ctv": "q8_0", "parallel": 1})
            notes.append("Increased batch/ubatch (4096/1024) for faster prompt prefill and lower TTFT.")
        elif goal == "max_context":
            tune.update({"ctx": 131072, "batch": 2048, "ubatch": 512, "ctk": "q8_0", "ctv": "q4_0", "parallel": 1})
            notes.append("Using Q8_0/Q4_0 KV cache quantization to maximize stable context length on GPU.")
        elif goal == "low_vram":
            tune.update({"ctx": 16384, "batch": 1024, "ubatch": 256, "ctk": "q4_0", "ctv": "q4_0", "parallel": 1, "draft_n": 0})
            notes.append("Disabled drafting and compressed KV to Q4_0 to minimize VRAM footprint.")
        elif goal == "multi_user":
            tune.update({"ctx": 65536, "batch": 2048, "ubatch": 512, "ctk": "q8_0", "ctv": "q4_0", "parallel": 4})
            notes.append("Allocated 4 parallel slots with quantized KV cache.")
        else:  # balanced
            tune.update({"ctx": 65536, "batch": 2048, "ubatch": 512, "ctk": "q8_0", "ctv": "q8_0", "parallel": 1})

        if constraints.get("min_ctx"):
            tune["ctx"] = max(int(tune.get("ctx", 32768)), int(constraints["min_ctx"]))
        if constraints.get("parallel"):
            tune["parallel"] = max(1, min(8, int(constraints["parallel"])))

        cmd_preview = ""
        try:
            cmd_preview = " ".join(self.host.smgr.build_args(profile, tune))
        except Exception as e:
            notes.append(f"CLI preview unavailable: {e}")

        model_key = tgt.get("base") or f"avalon:{profile}"
        return {
            "engine": "avalon",
            "goal": goal,
            "profile": profile,
            "model_key": model_key,
            "config": {"profile": profile, **tune},
            "cmd_preview": cmd_preview,
            "moe": {
                "is_moe": True,
                "feasible": True,
                "strategy": "avalon_auto_fit",
                "notes": ["Avalon `-fit on` keeps attention & KV on GPU and streams MoE experts from system RAM."],
            },
            "notes": notes,
            "docs_consulted": self.docs.lookup("avalon", "moe")["guides"],
        }

    def _advise_strata(
        self,
        *,
        model: str,
        goal: str,
        constraints: dict[str, Any],
        vit: dict[str, Any],
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        last_strata = self.host.cfg.get("strata_last") or {}
        if model:
            if "/" in model or model.endswith(".gguf"):
                cfg["model_path"] = model
            else:
                cfg["model"] = model
        elif last_strata.get("preset"):
            cfg["preset"] = str(last_strata["preset"])
        elif last_strata.get("model"):
            cfg["model"] = str(last_strata["model"])

        notes: list[str] = []
        if goal == "fastest":
            cfg.update({"context": "32k", "kv": "int8", "parallel": 1, "drafting": "auto", "mtp": "on", "tune": "quick"})
            notes.append("Enabled Strata auto drafting + MTP + quick tuning with int8 KV cache for peak tok/s.")
        elif goal == "low_latency":
            cfg.update({"context": "32k", "kv": "int8", "parallel": 1, "drafting": "mtp", "mtp": "on"})
            notes.append("Single-slot int8 KV with MTP drafting for low TTFT.")
        elif goal == "max_context":
            cfg.update({"context": "128k", "kv": "k8v4", "parallel": 1, "drafting": "auto"})
            notes.append("Selected `k8v4` KV format (8-bit keys, 4-bit values) to maximize stable context window.")
        elif goal == "low_vram":
            cfg.update({"context": "16k", "kv": "q4_0", "parallel": 1, "drafting": "off", "mtp": "off"})
            notes.append("Compressed KV (`q4_0`) and disabled draft layers to minimize VRAM usage.")
        elif goal == "multi_user":
            cfg.update({"context": "64k", "kv": "k8v4", "parallel": 4})
            notes.append("Configured 4 parallel request slots with `k8v4` KV cache.")
        else:
            cfg.update({"context": "64k", "kv": "int8", "parallel": 1, "drafting": "auto"})
            notes.append("Balanced 64k context with int8 KV cache and automatic drafting.")

        if constraints.get("min_ctx"):
            cfg["context"] = _ctx_to_strata_word(int(constraints["min_ctx"]))
        if constraints.get("parallel"):
            cfg["parallel"] = max(1, min(8, int(constraints["parallel"])))

        # Read-only Strata plan preview if strata CLI is installed
        plan_preview = None
        cmd_preview = ""
        try:
            body = self.host.strata_launch_body_cls(**cfg, plan=True)
            args = self.host.strata_build_args(body, plan=True)
            cmd_preview = "strata " + " ".join(args)
            if self.host.strata_bin():
                r = self.host.strata_run(args, timeout=20)
                if isinstance(r, dict):
                    if r.get("ok") and r.get("json") is not None:
                        plan_preview = r["json"]
                        notes.append("Verified dry-run placement with `strata launch --plan --output json`.")
                    elif r.get("ok") and (r.get("stdout") or "").strip():
                        plan_preview = {"raw": str(r["stdout"])[:1000]}
                    elif r.get("error") or r.get("stderr"):
                        notes.append(f"Strata plan preview note: {str(r.get('error') or r.get('stderr'))[:240]}")
                elif getattr(r, "returncode", 1) == 0 and (getattr(r, "stdout", "") or "").strip():
                    try:
                        plan_preview = json.loads(r.stdout)
                        notes.append("Verified dry-run placement with `strata launch --plan --output json`.")
                    except Exception:
                        plan_preview = {"raw": r.stdout[:1000]}
        except Exception as e:
            notes.append(f"Strata plan preview skipped: {e}")

        model_key = model or (self.host.cfg.get("strata_last") or {}).get("model") or "strata:default"
        return {
            "engine": "strata",
            "goal": goal,
            "model_key": model_key,
            "config": cfg,
            "cmd_preview": cmd_preview,
            "plan_preview": plan_preview,
            "notes": notes,
            "docs_consulted": self.docs.lookup("strata", "kv_cache")["guides"],
        }

    def _advise_maya(
        self,
        *,
        model: str,
        goal: str,
        constraints: dict[str, Any],
        vit: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.host.maya_mgr:
            raise ValueError("Project Maya manager is not configured")
        configs = self.host.maya_mgr.discover_configs() or []
        if not configs:
            raise ValueError("No Project Maya JSON configs found in maya_dir (~/project-maya)")

        chosen = None
        if model:
            ml = model.strip().lower()
            chosen = next(
                (c for c in configs if ml in (c.get("id", "").lower(), c.get("model_name", "").lower(), c.get("path", "").lower())),
                None,
            )
            if not chosen:
                chosen = next((c for c in configs if ml in c.get("id", "").lower() or ml in c.get("model_name", "").lower()), None)
        if not chosen:
            # Pick best available config for goal
            avail = [c for c in configs if c.get("available")] or configs
            if goal == "fastest":
                chosen = sorted(avail, key=lambda c: (bool(c.get("mtp")), -(c.get("dir_size_bytes") or 0)), reverse=True)[0]
            elif goal == "max_context":
                chosen = sorted(avail, key=lambda c: int(c.get("ctx") or 0), reverse=True)[0]
            else:
                chosen = avail[0]

        cur_ctx = int(chosen.get("ctx") or 32768)
        notes: list[str] = [
            f"Project Maya config `{chosen['id']}` ({chosen.get('engine', 'glm')}, {chosen.get('quant') or 'custom quant'}) "
            "uses MoE expert-in-VRAM streaming so models larger than GPU VRAM run cleanly."
        ]

        if goal == "fastest":
            ctx = min(cur_ctx, 32768)
            mtp = True
            reasoning = "low"
            notes.append("Enabled `--mtp` and set `reasoning_effort: low` with 32k ctx to keep more routed experts resident in VRAM.")
        elif goal == "low_latency":
            ctx = min(cur_ctx, 16384)
            mtp = True
            reasoning = "low"
            notes.append("Trimmed context to 16k with `--mtp` enabled for minimal TTFT and maximum VRAM expert residency.")
        elif goal == "max_context":
            ctx = max(cur_ctx, 131072)
            mtp = False
            reasoning = chosen.get("reasoning_effort") or "medium"
            notes.append("Disabled `--mtp` to free draft VRAM for a deeper KV cache window (131k+).")
        elif goal == "low_vram":
            ctx = min(cur_ctx, 16384)
            mtp = False
            reasoning = "low"
            notes.append("Disabled `--mtp` and reduced context to 16k to minimize GPU VRAM and thermals.")
        else:  # balanced / multi_user
            ctx = max(32768, min(cur_ctx, 65536))
            mtp = bool(chosen.get("mtp", True))
            reasoning = chosen.get("reasoning_effort") or "medium"
            if goal == "multi_user" and chosen.get("engine") == "hy4":
                notes.append("Note: Maya `hy4` serves 1 request at a time; prefer a `glm` Maya config or llama.cpp/Strata for multi-user concurrency.")

        if constraints.get("min_ctx"):
            ctx = max(ctx, int(constraints["min_ctx"]))

        cfg = {
            "config_id": chosen["id"],
            "config_path": chosen["path"],
            "model_name": chosen["model_name"],
            "engine_flavor": chosen.get("engine", "glm"),
            "ctx": ctx,
            "mtp": mtp,
            "reasoning_effort": reasoning,
        }

        usable_vram = max(512.0, float((vit.get("gpu") or {}).get("total_mb") or 20480) - float(self.host.cfg.get("vram_reserve_mb") or 512))
        avail_ram_mb = max(1024.0, (float((vit.get("mem") or {}).get("total_gb") or 64) - float((vit.get("mem") or {}).get("used_gb") or 10)) * 1024.0)
        moe_info = analyze_moe_mechanics(
            {"size": chosen.get("dir_size_bytes") or 0, "blocks": 48, "moe": True, "experts": 64, "experts_used": 8},
            usable_vram_mb=usable_vram,
            avail_ram_mb=avail_ram_mb,
            kv_vram_mb=1536.0 if ctx >= 65536 else 768.0,
        )

        return {
            "engine": "maya",
            "goal": goal,
            "model_key": chosen["model_name"],
            "config": cfg,
            "moe": moe_info,
            "notes": notes,
            "docs_consulted": self.docs.lookup("maya", "moe")["guides"],
        }

    def _patch_maya_json_config(self, cfg: dict[str, Any]) -> str:
        """Apply tuned ctx / --mtp / reasoning_effort fields to a Maya JSON config file safely."""
        if not self.host.maya_mgr:
            raise ValueError("Project Maya manager is not configured")
        ref = cfg.get("config_id") or cfg.get("config_path") or ""
        found = self.host.maya_mgr._find_config(ref)
        if not found:
            configs = self.host.maya_mgr.discover_configs() or []
            if not configs:
                raise ValueError("No Project Maya config found")
            found = self.host.maya_mgr._find_config(configs[0]["id"])
        cfg_path, raw, pub = found
        updated = copy.deepcopy(raw)

        # Update --mtp and --ctx / -c in args list if present, plus top-level ctx & reasoning_effort
        args = list(updated.get("args") or []) if isinstance(updated.get("args"), list) else []
        if "mtp" in cfg:
            want_mtp = bool(cfg["mtp"])
            args = [a for a in args if str(a) != "--mtp"]
            if want_mtp:
                args.append("--mtp")
        if cfg.get("ctx"):
            ctx_val = int(cfg["ctx"])
            updated["ctx"] = ctx_val
            # Also update --ctx / -c / --context in args if present
            replaced = False
            for i, a in enumerate(args):
                if str(a) in ("--ctx", "-c", "--context") and i + 1 < len(args):
                    args[i + 1] = str(ctx_val)
                    replaced = True
                    break
            if not replaced and any(str(a).startswith("--ctx=") for a in args):
                args = [f"--ctx={ctx_val}" if str(a).startswith("--ctx=") else a for a in args]
        if isinstance(updated.get("args"), list) or args:
            updated["args"] = args
        if cfg.get("reasoning_effort"):
            updated["reasoning_effort"] = str(cfg["reasoning_effort"])

        tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
        tmp.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        tmp.replace(cfg_path)
        return pub["id"]

    # ------------------------------------------------------------ Apply config (with auto-checkpoint)
    def apply_config(
        self,
        *,
        engine: str,
        config: dict[str, Any],
        launch: bool = True,
        save_preset_name: str = "",
        goal: str = "balanced",
        checkpoint_label: str = "",
    ) -> dict[str, Any]:
        engine = engine.lower()
        cp = self.capture_checkpoint(
            engine=engine,
            label=checkpoint_label or f"Before applying {engine} ({goal})",
            reason="apply_config",
        )
        result: dict[str, Any] = {"ok": True, "engine": engine, "checkpoint_id": cp["id"], "config": config}

        if engine == "llama":
            model_path = config.get("model")
            if not model_path:
                m = self._find_model("")
                if not m:
                    raise ValueError("No model specified and no local GGUF found")
                model_path = m["path"]
            # Save into launch_profiles.json
            profiles = self.host.read_json(self.host.launch_profiles_path, {})
            profiles[model_path] = {k: v for k, v in config.items() if k != "model"}
            self.host.write_json(self.host.launch_profiles_path, profiles)
            if launch:
                result["launch"] = self.host.mgr.start(
                    model_path,
                    preset=config.get("preset", "auto"),
                    ctx=int(config.get("ctx") or 8192),
                    ngl=int(config.get("ngl") if config.get("ngl") is not None else 99),
                    threads=int(config.get("threads") or 0),
                    fa=str(config.get("fa") or "auto"),
                    ctk=str(config.get("ctk") or "f16"),
                    ctv=str(config.get("ctv") or "f16"),
                    parallel=int(config.get("parallel") or 1),
                    batch=int(config.get("batch") or 0),
                    ubatch=int(config.get("ubatch") or 0),
                    ncmoe=int(config.get("ncmoe") or 0),
                    extra_args=str(config.get("extra_args") or ""),
                    draft=str(config.get("draft") or ""),
                    draft_max=int(config.get("draft_max") or 16),
                    jinja=str(config.get("jinja") or "auto"),
                )
        elif engine == "avalon":
            profile = config.get("profile") or "flagship"
            tune = {k: v for k, v in config.items() if k != "profile" and v is not None and v != ""}
            self.host.cfg.setdefault("avalon_tune", {})[profile] = tune
            self.host.save_config()
            if launch:
                result["launch"] = self.host.smgr.start(profile, tune)
        elif engine == "strata":
            if launch:
                body_kwargs = {k: v for k, v in config.items() if k != "plan"}
                body = self.host.strata_launch_body_cls(**body_kwargs, plan=False)
                args = self.host.strata_build_args(body, plan=False)
                r = self.host.strata_run(args, timeout=180)
                if isinstance(r, dict):
                    result["launch"] = r
                else:
                    result["launch"] = {
                        "ok": getattr(r, "returncode", 1) == 0,
                        "returncode": getattr(r, "returncode", 1),
                        "stdout": (getattr(r, "stdout", "") or "")[-2000:],
                        "stderr": (getattr(r, "stderr", "") or "")[-2000:],
                    }
        elif engine == "maya":
            cid = self._patch_maya_json_config(config)
            result["maya_config_id"] = cid
            if launch and self.host.maya_mgr:
                cur = self.host.maya_mgr.status()
                result["launch"] = self.host.maya_mgr.switch(cid) if cur.get("running") else self.host.maya_mgr.start(cid)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        if save_preset_name:
            result["preset"] = self.store.save_preset(
                name=save_preset_name,
                engine=engine,
                goal=goal,
                config=config,
                notes=f"Saved via apply_config ({goal})",
            )
        return result

    # ------------------------------------------------------------ Active Empirical Auto-Tune
    def tune_status(self) -> dict[str, Any]:
        with self._tune_lock:
            if not self._tune_session:
                return {"running": False, "session": None}
            return {"running": self._tune_session.get("state") == "running", "session": copy.deepcopy(self._tune_session)}

    def stop_tune(self, rollback: bool = False) -> dict[str, Any]:
        self._tune_cancel.set()
        cp_id = None
        with self._tune_lock:
            if self._tune_session:
                self._tune_session["cancel_requested"] = True
                cp_id = self._tune_session.get("checkpoint_id")
        res = {"ok": True, "cancel_requested": True}
        if rollback and cp_id:
            res["rollback"] = self.restore_checkpoint(cp_id, relaunch=True)
        return res

    def start_tune(
        self,
        *,
        engine: str = "llama",
        goal: str = "fastest",
        model: str = "",
        profile: str = "flagship",
        constraints: dict[str, Any] | None = None,
        user_prompt: str = "",
        max_trials: int = 5,
        max_seconds: int = 240,
        save_preset_name: str = "",
        sync_run: bool = False,
    ) -> dict[str, Any]:
        engine = (engine or "llama").lower()
        if engine not in ENGINES:
            raise ValueError(f"Unsupported engine: {engine}")
        goal = (goal or "balanced").lower()
        if goal not in GOALS:
            goal = "balanced"
        max_trials = max(1, min(12, int(max_trials or 5)))
        max_seconds = max(30, min(900, int(max_seconds or 240)))

        with self._tune_lock:
            if self._tune_session and self._tune_session.get("state") == "running":
                raise RuntimeError("An auto-tune session is already running")
            self._tune_cancel.clear()

            # Always create an automatic checkpoint BEFORE starting optimization!
            cp = self.capture_checkpoint(
                engine=engine,
                label=f"Pre-tune checkpoint ({engine} · {goal})",
                reason=f"auto_tune:{goal}",
            )
            advice = self.advise(
                engine=engine,
                goal=goal,
                model=model,
                profile=profile,
                constraints=constraints,
                user_prompt=user_prompt,
                use_llm=False,
            )
            candidates = self._build_candidates(engine, goal, advice, max_trials=max_trials)
            session = {
                "id": f"tune-{uuid.uuid4().hex[:8]}",
                "state": "running",
                "engine": engine,
                "goal": goal,
                "model_key": advice.get("model_key", ""),
                "checkpoint_id": cp["id"],
                "started_ts": time.time(),
                "finished_ts": None,
                "max_trials": len(candidates),
                "max_seconds": max_seconds,
                "completed_trials": 0,
                "trials": [],
                "best": None,
                "log": [
                    f"Created safety checkpoint {cp['id']} before starting {engine} auto-tune ({goal}).",
                    f"Planned {len(candidates)} candidate configurations to evaluate empirically.",
                ],
                "save_preset_name": save_preset_name,
                "cancel_requested": False,
            }
            self._tune_session = session

        if sync_run:
            self._run_tune_worker(session["id"], engine, goal, advice, candidates, max_seconds, save_preset_name)
            return self.tune_status()

        t = threading.Thread(
            target=self._run_tune_worker,
            args=(session["id"], engine, goal, advice, candidates, max_seconds, save_preset_name),
            daemon=True,
        )
        t.start()
        return self.tune_status()

    def _build_candidates(
        self, engine: str, goal: str, advice: dict[str, Any], max_trials: int = 5
    ) -> list[dict[str, Any]]:
        base = copy.deepcopy(advice["config"])
        cands: list[dict[str, Any]] = [{"label": "advisor-baseline", "config": base}]

        if engine == "llama":
            if goal == "fastest":
                c2 = copy.deepcopy(base); c2["batch"] = 4096; c2["ubatch"] = 1024
                cands.append({"label": "larger-batch-4096", "config": c2})
                c3 = copy.deepcopy(base); c3["ctk"] = "f16"; c3["ctv"] = "f16"
                cands.append({"label": "f16-kv-cache", "config": c3})
                if base.get("draft"):
                    c4 = copy.deepcopy(base); c4["draft"] = ""; c4["draft_max"] = 0
                    cands.append({"label": "no-speculation-compare", "config": c4})
                if base.get("ncmoe", 0) > 2:
                    c5 = copy.deepcopy(base); c5["ncmoe"] = max(0, int(base["ncmoe"]) - 2)
                    cands.append({"label": "more-moe-on-gpu", "config": c5})
            elif goal == "max_context":
                for mult_label, factor in (("ctx-1.5x", 1.5), ("ctx-2x", 2.0), ("ctx-q4-kv", 1.0)):
                    c = copy.deepcopy(base)
                    if mult_label == "ctx-q4-kv":
                        c["ctk"], c["ctv"] = "q4_0", "q4_0"
                        c["ctx"] = min(262144, int(base.get("ctx", 16384) * 2))
                    else:
                        c["ctx"] = min(262144, int((base.get("ctx", 16384) * factor) // 1024 * 1024))
                    cands.append({"label": mult_label, "config": c})
            elif goal == "low_latency":
                c2 = copy.deepcopy(base); c2["batch"] = 4096; c2["ubatch"] = 1024
                c3 = copy.deepcopy(base); c3["batch"] = 1024; c3["ubatch"] = 512
                cands.extend([{"label": "prefill-4096", "config": c2}, {"label": "tight-ubatch-512", "config": c3}])
            else:
                c2 = copy.deepcopy(base); c2["ctk"] = "q8_0"; c2["ctv"] = "q4_0"
                c3 = copy.deepcopy(base); c3["batch"] = 4096; c3["ubatch"] = 1024
                cands.extend([{"label": "k8v4-kv", "config": c2}, {"label": "batch-4096", "config": c3}])

        elif engine == "avalon":
            if goal == "fastest":
                c2 = copy.deepcopy(base); c2["batch"] = 4096; c2["ubatch"] = 1024
                c3 = copy.deepcopy(base); c3["draft_n"] = 16
                c4 = copy.deepcopy(base); c4["draft_n"] = 0
                cands.extend([
                    {"label": "batch-4096", "config": c2},
                    {"label": "draft-16", "config": c3},
                    {"label": "no-draft-compare", "config": c4},
                ])
            elif goal == "max_context":
                c2 = copy.deepcopy(base); c2["ctx"] = min(262144, int(base.get("ctx", 65536)) * 2); c2["ctk"] = "q4_0"; c2["ctv"] = "q4_0"
                cands.append({"label": "ctx-2x-q4kv", "config": c2})
            else:
                c2 = copy.deepcopy(base); c2["ctk"] = "q8_0"; c2["ctv"] = "q4_0"
                cands.append({"label": "k8v4-kv", "config": c2})

        elif engine == "strata":
            if goal == "fastest":
                c2 = copy.deepcopy(base); c2["drafting"] = "dflash2"
                c3 = copy.deepcopy(base); c3["drafting"] = "off"; c3["mtp"] = "off"
                cands.extend([{"label": "dflash2-draft", "config": c2}, {"label": "no-draft-compare", "config": c3}])
            elif goal == "max_context":
                c2 = copy.deepcopy(base); c2["context"] = "256k"; c2["kv"] = "q4_0"
                cands.append({"label": "256k-q4-kv", "config": c2})
            else:
                c2 = copy.deepcopy(base); c2["kv"] = "k8v4"
                cands.append({"label": "k8v4-kv", "config": c2})

        elif engine == "maya":
            if goal == "fastest":
                c2 = copy.deepcopy(base); c2["mtp"] = False
                c3 = copy.deepcopy(base); c3["mtp"] = True; c3["ctx"] = 16384; c3["reasoning_effort"] = "low"
                cands.extend([{"label": "no-mtp-compare", "config": c2}, {"label": "tight-16k-mtp", "config": c3}])
            elif goal == "max_context":
                c2 = copy.deepcopy(base); c2["ctx"] = min(262144, int(base.get("ctx") or 65536) * 2); c2["mtp"] = False
                cands.append({"label": "ctx-2x-no-mtp", "config": c2})
            else:
                c2 = copy.deepcopy(base); c2["mtp"] = not bool(base.get("mtp", True))
                cands.append({"label": "toggle-mtp", "config": c2})

        # Deduplicate identical configs
        seen: set[str] = set()
        uniq: list[dict[str, Any]] = []
        for c in cands:
            key = json.dumps(c["config"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
            if len(uniq) >= max_trials:
                break
        return uniq

    def _tune_log(self, msg: str) -> None:
        with self._tune_lock:
            if self._tune_session is not None:
                self._tune_session["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
                self._tune_session["log"] = self._tune_session["log"][-200:]

    def _run_tune_worker(
        self,
        session_id: str,
        engine: str,
        goal: str,
        advice: dict[str, Any],
        candidates: list[dict[str, Any]],
        max_seconds: int,
        save_preset_name: str,
    ) -> None:
        t0 = time.time()
        best_trial: dict[str, Any] | None = None
        vit_start = self.host.vitals(refresh=True)
        hw_sig = self._hardware_sig(vit_start)
        model_key = advice.get("model_key", "")
        meta = self._find_model(advice.get("model_path", "")) if engine == "llama" else None

        try:
            for idx, cand in enumerate(candidates):
                if self._tune_cancel.is_set() or (time.time() - t0) >= max_seconds:
                    self._tune_log("Stopping auto-tune loop (cancelled or time budget reached).")
                    break

                label = cand["label"]
                cfg = cand["config"]
                self._tune_log(f"Trial {idx + 1}/{len(candidates)} ({label}): launching {engine}…")

                trial_rec: dict[str, Any] = {
                    "index": idx + 1,
                    "label": label,
                    "config": cfg,
                    "ok": False,
                    "score": -1.0,
                    "metrics": {},
                    "error": "",
                }

                try:
                    launch_ok, err_msg = self._launch_for_trial(engine, cfg)
                    if not launch_ok:
                        diag = self.diagnose_failure(engine=engine, last_config=cfg, use_llm=False)
                        trial_rec["error"] = err_msg or diag.get("title") or "launch failed"
                        trial_rec["diagnosis"] = diag
                        self._tune_log(f"Trial {idx + 1} failed: {trial_rec['error']}")
                        self.store.record_trial(
                            engine=engine,
                            model_key=model_key,
                            goal=goal,
                            config=cfg,
                            ok=False,
                            error_signature=diag.get("pattern_id") or "launch_error",
                            hardware_sig=hw_sig,
                            notes=trial_rec["error"],
                        )
                    else:
                        prompt_key = "short" if goal == "low_latency" else "standard"
                        bench = self.host.benchmark_run(prompt_key, 96)
                        metrics = {
                            "tok_s": bench.get("tok_s", 0),
                            "prompt_s": bench.get("prompt_s", 0),
                            "ttft_ms": bench.get("ttft_ms", 0),
                            "vram_mb": bench.get("vram_mb", 0),
                        }
                        sc = score_trial(goal, metrics, cfg)
                        trial_rec["ok"] = True
                        trial_rec["metrics"] = metrics
                        trial_rec["score"] = round(sc, 2)
                        self._tune_log(
                            f"Trial {idx + 1} OK: {metrics['tok_s']} tok/s | "
                            f"prefill {metrics['prompt_s']} tok/s | TTFT {metrics['ttft_ms']} ms | "
                            f"VRAM {metrics['vram_mb']} MB (score {trial_rec['score']})"
                        )

                        # Real-world VRAM calibration sample!
                        if engine == "llama" and meta and metrics.get("vram_mb", 0) > 0:
                            est_raw = self.host.estimate_vram(
                                meta,
                                ctx=int(cfg.get("ctx") or 8192),
                                ngl=int(cfg.get("ngl") if cfg.get("ngl") is not None else 99),
                                ctk=str(cfg.get("ctk") or "f16"),
                                ctv=str(cfg.get("ctv") or "f16"),
                                parallel=int(cfg.get("parallel") or 1),
                                preset=str(cfg.get("preset") or "auto"),
                                ncmoe=int(cfg.get("ncmoe") or 0),
                            )
                            self.store.record_vram_sample(
                                arch=meta.get("arch", "unknown"),
                                moe=bool(meta.get("moe")),
                                engine="llama",
                                estimated_mb=est_raw.get("total_vram_mb", 0),
                                actual_mb=metrics["vram_mb"],
                                model_key=model_key,
                            )

                        self.store.record_trial(
                            engine=engine,
                            model_key=model_key,
                            goal=goal,
                            config=cfg,
                            metrics=metrics,
                            score=sc,
                            ok=True,
                            arch=(meta or {}).get("arch", ""),
                            moe=bool((meta or {}).get("moe", engine == "avalon")),
                            hardware_sig=hw_sig,
                            notes=f"Auto-tune trial ({label})",
                        )

                        if best_trial is None or trial_rec["score"] > best_trial["score"]:
                            best_trial = trial_rec
                except Exception as e:
                    trial_rec["error"] = str(e)
                    self._tune_log(f"Trial {idx + 1} exception: {e}")

                with self._tune_lock:
                    if self._tune_session and self._tune_session["id"] == session_id:
                        self._tune_session["trials"].append(trial_rec)
                        self._tune_session["completed_trials"] = idx + 1
                        self._tune_session["best"] = best_trial

            # Finalize: apply winner or roll back if nothing succeeded
            with self._tune_lock:
                cp_id = self._tune_session.get("checkpoint_id") if self._tune_session else None

            if best_trial and not self._tune_cancel.is_set():
                self._tune_log(f"Applying winning config ({best_trial['label']}, score {best_trial['score']}).")
                self._launch_for_trial(engine, best_trial["config"])
                if save_preset_name:
                    self.store.save_preset(
                        name=save_preset_name,
                        engine=engine,
                        goal=goal,
                        config=best_trial["config"],
                        notes=f"Auto-tune winner ({best_trial['label']})",
                        metrics=best_trial["metrics"],
                        model_key=model_key,
                    )
                final_state = "done"
            else:
                self._tune_log("No successful trial (or cancelled) — restoring pre-tune checkpoint.")
                if cp_id:
                    try:
                        self.restore_checkpoint(cp_id, relaunch=False)
                    except Exception as e:
                        self._tune_log(f"Checkpoint restore warning: {e}")
                final_state = "cancelled" if self._tune_cancel.is_set() else "failed"

            with self._tune_lock:
                if self._tune_session and self._tune_session["id"] == session_id:
                    self._tune_session["state"] = final_state
                    self._tune_session["finished_ts"] = time.time()
        except Exception as e:
            self._tune_log(f"Auto-tune worker crashed: {e}")
            with self._tune_lock:
                if self._tune_session and self._tune_session["id"] == session_id:
                    self._tune_session["state"] = "failed"
                    self._tune_session["finished_ts"] = time.time()

    def _launch_for_trial(self, engine: str, cfg: dict[str, Any]) -> tuple[bool, str]:
        try:
            if engine == "llama":
                res = self.host.mgr.start(
                    cfg["model"],
                    preset=cfg.get("preset", "auto"),
                    ctx=int(cfg.get("ctx") or 8192),
                    ngl=int(cfg.get("ngl") if cfg.get("ngl") is not None else 99),
                    threads=int(cfg.get("threads") or 0),
                    fa=str(cfg.get("fa") or "auto"),
                    ctk=str(cfg.get("ctk") or "f16"),
                    ctv=str(cfg.get("ctv") or "f16"),
                    parallel=int(cfg.get("parallel") or 1),
                    batch=int(cfg.get("batch") or 0),
                    ubatch=int(cfg.get("ubatch") or 0),
                    ncmoe=int(cfg.get("ncmoe") or 0),
                    extra_args=str(cfg.get("extra_args") or ""),
                    draft=str(cfg.get("draft") or ""),
                    draft_max=int(cfg.get("draft_max") or 16),
                    jinja=str(cfg.get("jinja") or "auto"),
                )
                return bool(res.get("ok", True)), str(res.get("error") or "")
            if engine == "avalon":
                profile = cfg.get("profile") or "flagship"
                tune = {k: v for k, v in cfg.items() if k != "profile"}
                res = self.host.smgr.start(profile, tune)
                return bool(res.get("ok", True)), str(res.get("error") or "")
            if engine == "strata":
                body_kwargs = {k: v for k, v in cfg.items() if k != "plan"}
                body = self.host.strata_launch_body_cls(**body_kwargs, plan=False, restart=True)
                args = self.host.strata_build_args(body, plan=False)
                cmd_str = "strata " + " ".join(args)
                self._tune_log(f"Running: {cmd_str}")
                r = self.host.strata_run(args, timeout=300)
                if isinstance(r, dict):
                    if r.get("ok"):
                        j = r.get("json") if isinstance(r.get("json"), dict) else {}
                        if j:
                            st_last = dict(self.host.cfg.get("strata_last") or {})
                            for k in ("address", "port", "model", "profile", "preset"):
                                if j.get(k) not in (None, ""):
                                    st_last[k] = j[k]
                            if st_last:
                                self.host.cfg["strata_last"] = st_last
                        return True, ""
                    parts = [
                        str(r.get("error") or "").strip(),
                        str(r.get("stderr") or "").strip()[-300:],
                        str(r.get("stdout") or "").strip()[-300:],
                    ]
                    msg = " | ".join(p for p in parts if p) or f"exit rc={r.get('rc')} ({cmd_str})"
                    return False, msg[:500]
                return (r.returncode == 0), (r.stderr or r.stdout or f"exit rc={r.returncode}")[-400:]
            if engine == "maya" and self.host.maya_mgr:
                cid = self._patch_maya_json_config(cfg)
                cur = self.host.maya_mgr.status()
                res = self.host.maya_mgr.switch(cid) if cur.get("running") else self.host.maya_mgr.start(cid)
                return bool(res.get("running", True) or res.get("ok", True)), str(res.get("error") or "")
        except Exception as e:
            return False, str(e)
        return False, "unknown engine"

    # ------------------------------------------------------------ Crash & OOM Self-Healing
    def _engine_log_tail(self, engine: str, lines: int = 120) -> str:
        if engine == "llama":
            return "\n".join(list(getattr(self.host.mgr, "logbuf", []))[-lines:])
        if engine == "avalon":
            return "\n".join(list(getattr(self.host.smgr, "logbuf", []))[-lines:])
        if engine == "maya" and self.host.maya_mgr:
            try:
                t = self.host.maya_mgr.tail_log(max_lines=lines)
                return str(t.get("log") or "") if isinstance(t, dict) else str(t or "")
            except Exception:
                return ""
        if engine == "strata":
            try:
                if self.host.strata_log_path.exists():
                    return "\n".join(self.host.strata_log_path.read_text(errors="replace").splitlines()[-lines:])
            except Exception:
                pass
        return ""

    def diagnose_failure(
        self,
        *,
        engine: str = "llama",
        log_text: str | None = None,
        last_config: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        engine = (engine or "llama").lower()
        tail = log_text if log_text is not None else self._engine_log_tail(engine, 150)

        matched = None
        matched_line = ""
        for pat in ERROR_PATTERNS:
            m = pat["regex"].search(tail)
            if m:
                matched = pat
                matched_line = m.group(0)
                break

        cfg = copy.deepcopy(last_config or {})
        fix_config = copy.deepcopy(cfg)
        steps: list[str] = []
        pattern_id = matched["id"] if matched else "unknown"
        title = matched["title"] if matched else "No explicit crash signature found"
        cause = matched["cause"] if matched else "Inspect engine log tail or check binary/model health."

        # Build MoE-aware, doc-grounded remediation
        if pattern_id in ("vram_oom", "kv_alloc_oom"):
            if engine == "llama":
                meta = self._find_model(cfg.get("model", ""))
                is_moe = bool(meta and meta.get("moe"))
                if is_moe and cfg.get("preset") != "moe-split" and int(cfg.get("ncmoe") or 0) == 0:
                    fix_config["preset"] = "moe-split"
                    steps.append("Detected MoE model OOMing on full GPU — switching to `moe-split` so attention & KV stay on GPU while routed experts offload to RAM.")
                elif is_moe and int(cfg.get("ncmoe") or 0) > 0:
                    blocks = int((meta or {}).get("blocks") or 32)
                    fix_config["ncmoe"] = min(blocks, int(cfg["ncmoe"]) + max(2, blocks // 8))
                    steps.append(f"Increasing `--n-cpu-moe` to {fix_config['ncmoe']} to free GPU VRAM without dropping attention layers.")
                if fix_config.get("ctv") != "q4_0":
                    fix_config["fa"] = "on"
                    fix_config["ctk"] = "q8_0"
                    fix_config["ctv"] = "q4_0"
                    steps.append("Enabled Flash Attention (`-fa on`) + quantized KV cache (`-ctk q8_0 -ctv q4_0`).")
                if int(fix_config.get("parallel") or 1) > 1:
                    fix_config["parallel"] = 1
                    steps.append("Reduced `--parallel` to 1 slot.")
                if pattern_id == "kv_alloc_oom" or (not is_moe and int(fix_config.get("ctx") or 8192) > 4096):
                    fix_config["ctx"] = max(4096, int(fix_config.get("ctx") or 16384) // 2)
                    steps.append(f"Halved context window to {fix_config['ctx']:,} tokens.")
                elif not is_moe and int(fix_config.get("ngl") or 99) > 0:
                    blocks = int((meta or {}).get("blocks") or 32)
                    cur_ngl = min(blocks, int(fix_config.get("ngl") or blocks))
                    fix_config["ngl"] = max(0, cur_ngl - max(4, blocks // 6))
                    steps.append(f"Reduced GPU layer offload to `-ngl {fix_config['ngl']}`.")
            elif engine == "avalon":
                fix_config.pop("ngl", None)  # restore -fit on
                fix_config["ctk"] = "q8_0"
                fix_config["ctv"] = "q4_0"
                fix_config["fa"] = "on"
                fix_config["ctx"] = max(8192, int(fix_config.get("ctx") or 32768) // 2)
                steps.append("Restored Avalon `-fit on`, quantized KV to Q8_0/Q4_0, and halved context.")
            elif engine == "strata":
                fix_config["kv"] = "q4_0"
                fix_config["parallel"] = 1
                fix_config["context"] = "16k"
                steps.append("Switched Strata to `--kv q4_0 --parallel 1 --context 16k`.")
            elif engine == "maya":
                fix_config["mtp"] = False
                fix_config["ctx"] = max(8192, int(fix_config.get("ctx") or 32768) // 2)
                fix_config["reasoning_effort"] = "low"
                steps.append("Disabled Maya `--mtp` draft buffer and halved context to free VRAM for routed experts.")

        elif pattern_id == "fa_required_for_kv":
            fix_config["fa"] = "on"
            steps.append("Enabled Flash Attention (`-fa on`) required by quantized V cache.")

        elif pattern_id == "draft_mismatch":
            if engine == "llama":
                fix_config["draft"] = ""
                fix_config["draft_max"] = 0
            elif engine == "avalon":
                fix_config["draft_n"] = 0
            elif engine == "strata":
                fix_config["drafting"] = "off"
                fix_config["mtp"] = "off"
            steps.append("Disabled speculative drafting due to draft/target vocabulary or architecture mismatch.")

        elif pattern_id == "unknown_flag":
            fix_config["extra_args"] = ""
            steps.append("Cleared custom extra flags and gated all arguments against live binary `--help`.")

        doc_ref = self.docs.lookup(engine, "oom" if "oom" in pattern_id else pattern_id)
        llm_diag = None
        if use_llm and (pattern_id == "unknown" or not steps):
            llm_diag = self._llm_diagnose_log(engine, tail, cfg, doc_ref)

        return {
            "engine": engine,
            "pattern_id": pattern_id,
            "title": title,
            "cause": cause,
            "matched_line": matched_line,
            "remediation_steps": steps,
            "fix_config": fix_config if steps else None,
            "docs_consulted": doc_ref["guides"],
            "llm_diagnosis": llm_diag,
            "log_tail": tail[-2500:],
        }

    def heal_failure(
        self,
        *,
        engine: str = "llama",
        last_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        diag = self.diagnose_failure(engine=engine, last_config=last_config, use_llm=False)
        if not diag.get("fix_config"):
            return {"ok": False, "diagnosis": diag, "error": "No automated fix available for this failure"}
        applied = self.apply_config(
            engine=engine,
            config=diag["fix_config"],
            launch=True,
            goal="heal",
            checkpoint_label=f"Before self-heal ({engine}: {diag['pattern_id']})",
        )
        return {"ok": True, "diagnosis": diag, "applied": applied}

    # ------------------------------------------------------------ Optional LLM Provider Layer (Hermes-style presets + Custom + Model Polling)
    def _provider_api_style(self, provider_id: str, url: str) -> str:
        p = PROVIDER_PRESETS.get(provider_id or "")
        if p:
            return p["api"]
        if "anthropic.com" in (url or "").lower():
            return "anthropic"
        return "openai"

    def _custom_headers(self) -> dict[str, str]:
        raw = self.host.cfg.get("agent_llm_headers")
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items() if k and v is not None}
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items() if k and v is not None}
            except Exception:
                pass
        return {}

    def llm_status(self) -> dict[str, Any]:
        provider = (self.host.cfg.get("agent_llm_provider") or "").strip() or "custom"
        preset = PROVIDER_PRESETS.get(provider) or PROVIDER_PRESETS["custom"]
        url = (self.host.cfg.get("agent_llm_url") or "").strip() or preset.get("url", "")
        model = (self.host.cfg.get("agent_llm_model") or "").strip()
        key = (self.host.cfg.get("agent_llm_key") or "").strip()
        fallback = bool(self.host.cfg.get("agent_llm_fallback_local", True))
        local = self.host.chat_target()
        if provider != "local_fallback" and url and model:
            return {
                "available": True,
                "mode": "external",
                "provider": provider,
                "provider_label": preset["label"],
                "api_style": self._provider_api_style(provider, url),
                "url": url,
                "model": model,
                "key_set": bool(key),
            }
        if (fallback or provider == "local_fallback") and local:
            return {
                "available": True,
                "mode": "local_fallback",
                "provider": "local_fallback",
                "provider_label": "Active Local Engine",
                "port": local[0],
                "model": "active-local-engine",
            }
        return {
            "available": False,
            "mode": "deterministic_only",
            "provider": provider,
            "provider_label": preset["label"],
            "url": url,
            "model": model,
            "key_set": bool(key),
        }

    def fetch_provider_models(
        self,
        *,
        provider: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Always-on model polling for any configured or custom provider (OpenAI-compat, Anthropic, Ollama, or local)."""
        prov = (provider if provider is not None else (self.host.cfg.get("agent_llm_provider") or "custom")).strip() or "custom"
        preset = PROVIDER_PRESETS.get(prov) or PROVIDER_PRESETS["custom"]
        base_url = (url if url is not None else (self.host.cfg.get("agent_llm_url") or "")).strip() or preset.get("url", "")
        key = (api_key if api_key is not None else (self.host.cfg.get("agent_llm_key") or "")).strip()
        if key in ("", "__keep__"):
            key = (self.host.cfg.get("agent_llm_key") or "").strip()

        if prov == "local_fallback" or (not base_url and self.host.chat_target()):
            tgt = self.host.chat_target()
            if not tgt:
                return {"ok": False, "provider": prov, "models": [], "error": "No active local engine running"}
            base_url = f"http://127.0.0.1:{tgt[0]}/v1"
            key = tgt[1] or ""

        if not base_url:
            return {"ok": False, "provider": prov, "models": [], "error": "Enter a Base URL to poll models"}

        api_style = self._provider_api_style(prov, base_url)
        clean_base = base_url.rstrip("/")
        if clean_base.endswith("/chat/completions"):
            clean_base = clean_base[: -len("/chat/completions")]
        elif clean_base.endswith("/messages"):
            clean_base = clean_base[: -len("/messages")]

        headers = self._custom_headers()
        if api_style == "anthropic":
            if key:
                headers["x-api-key"] = key
            headers.setdefault("anthropic-version", "2023-06-01")
        elif key:
            headers["Authorization"] = f"Bearer {key}"

        models: list[str] = []
        last_err = ""
        try:
            r = httpx.get(f"{clean_base}/models", headers=headers, timeout=httpx.Timeout(6.0, read=12.0))
            r.raise_for_status()
            data = r.json()
            raw_list = data.get("data") if isinstance(data, dict) else (data if isinstance(data, list) else [])
            if isinstance(data, dict) and not raw_list and isinstance(data.get("models"), list):
                raw_list = data["models"]
            for item in raw_list or []:
                if isinstance(item, str):
                    models.append(item)
                elif isinstance(item, dict):
                    mid = item.get("id") or item.get("name") or item.get("model")
                    if mid:
                        models.append(str(mid))
        except Exception as e:
            last_err = str(e)
            # Ollama native /api/tags fallback if /v1/models fails
            if prov == "ollama" or ":11434" in clean_base:
                try:
                    root = clean_base[:-3] if clean_base.endswith("/v1") else clean_base
                    r2 = httpx.get(f"{root}/api/tags", timeout=httpx.Timeout(4.0, read=8.0))
                    r2.raise_for_status()
                    for item in (r2.json().get("models") or []):
                        if isinstance(item, dict) and item.get("name"):
                            models.append(str(item["name"]))
                    last_err = ""
                except Exception as e2:
                    last_err = str(e2)

        uniq = sorted(set(models))
        return {
            "ok": bool(uniq) or not last_err,
            "provider": prov,
            "url": clean_base,
            "count": len(uniq),
            "models": uniq[:300],
            "error": last_err if not uniq else "",
        }

    def test_provider(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        st = self.llm_status()
        if not st.get("available"):
            return {"ok": False, "latency_ms": 0, "error": "No LLM provider configured (set Base URL + Model, or start a local engine)"}
        reply = self._llm_complete("Reply with the single word OK.", "Ping", max_tokens=16)
        ms = round((time.perf_counter() - t0) * 1000)
        if reply:
            return {"ok": True, "latency_ms": ms, "reply": reply[:120], "status": st}
        err = self._last_llm_error or "Provider request failed — check Base URL, API key, and model ID"
        return {"ok": False, "latency_ms": ms, "error": err, "status": st}

    def _llm_complete(self, system_prompt: str, user_message: str, max_tokens: int = 500) -> str | None:
        self._last_llm_error = ""
        st = self.llm_status()
        if not st["available"]:
            return None
        if st["mode"] == "external":
            base = st["url"].rstrip("/")
            headers = {"Content-Type": "application/json", **self._custom_headers()}
            key = (self.host.cfg.get("agent_llm_key") or "").strip()
            api_style = st.get("api_style", "openai")
            if api_style == "anthropic":
                endpoint = base if base.endswith("/messages") else f"{base}/messages"
                if key:
                    headers["x-api-key"] = key
                headers.setdefault("anthropic-version", "2023-06-01")
                payload = {
                    "model": st["model"],
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_message}],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                }
            else:
                endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": st["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                }
        else:
            port, key = self.host.chat_target() or (None, "")
            if not port:
                return None
            endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            api_style = "openai"
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            }
        try:
            r = httpx.post(endpoint, headers=headers, json=payload, timeout=httpx.Timeout(10.0, read=45.0))
            if r.status_code == 400 and api_style == "openai":
                body_low = (r.text or "").lower()
                if "max_completion_tokens" in body_low or "temperature" in body_low:
                    retry_payload = dict(payload)
                    if "max_completion_tokens" in body_low and "max_tokens" in retry_payload:
                        retry_payload["max_completion_tokens"] = retry_payload.pop("max_tokens")
                    if "temperature" in body_low:
                        retry_payload.pop("temperature", None)
                    r = httpx.post(endpoint, headers=headers, json=retry_payload, timeout=httpx.Timeout(10.0, read=45.0))
            if r.status_code >= 400:
                snippet = (r.text or "").strip().replace("\n", " ")[:220]
                self._last_llm_error = f"HTTP {r.status_code} from {endpoint}: {snippet or r.reason_phrase}"
                return None
            data = r.json()
            if api_style == "anthropic":
                blocks = data.get("content") or []
                txt = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text").strip()
                if not txt:
                    self._last_llm_error = "Provider returned an empty Anthropic content response"
                return txt or None
            msg = ((data.get("choices") or [{}])[0].get("message") or {})
            raw_content = msg.get("content")
            if isinstance(raw_content, list):
                txt = "".join(
                    (b.get("text", "") if isinstance(b, dict) else str(b)) for b in raw_content
                ).strip()
            else:
                txt = (raw_content or "").strip()
            if not txt:
                txt = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            if not txt:
                self._last_llm_error = "Provider returned an empty message content"
            return txt or None
        except Exception as e:
            self._last_llm_error = f"{type(e).__name__}: {e}"
            return None

    def _llm_refine_advice(self, rec: dict[str, Any], user_prompt: str, vit: dict[str, Any]) -> str | None:
        sys_p = (
            "You are the Inference Console Optimization Agent. Give concise, grounded advice for local LLM "
            "engines (llama.cpp, Avalon, Strata). Never claim an MoE model won't run just because total "
            "weights exceed VRAM — explain attention/KV on GPU vs routed experts in RAM. Only cite flags "
            "present in the provided docs/config."
        )
        ctx_blob = json.dumps(
            {
                "engine": rec.get("engine"),
                "goal": rec.get("goal"),
                "config": rec.get("config"),
                "moe": rec.get("moe"),
                "fit": rec.get("fit"),
                "docs": rec.get("docs_consulted"),
                "hardware": vit,
            },
            indent=1,
        )
        return self._llm_complete(sys_p, f"User goal: {user_prompt}\n\nComputed plan:\n{ctx_blob}\n\nSummarize trade-offs and any tweaks in <=6 bullet points.")

    def _llm_diagnose_log(
        self, engine: str, tail: str, cfg: dict[str, Any], doc_ref: dict[str, Any]
    ) -> str | None:
        sys_p = (
            "You are a local inference engine debugger. Read the engine log tail and docs, identify the "
            "exact root cause, and propose the smallest verified flag fix. Do not invent flags."
        )
        return self._llm_complete(
            sys_p,
            f"Engine: {engine}\nConfig: {json.dumps(cfg)}\nDocs: {json.dumps(doc_ref.get('guides'))}\nLog tail:\n{tail[-2000:]}",
            max_tokens=400,
        )

    # ------------------------------------------------------------ Conversational Agent Sidebar & Action Templates
    @staticmethod
    def _extract_hf_query(msg: str) -> str:
        """Extract a model name or HF org/repo from a natural-language quant/pull prompt."""
        raw = (msg or "").strip()
        if not raw:
            return ""
        # 1. Explicit Hugging Face org/repo token (excluding units like tok/s, gpu/cpu)
        bad_slashes = {"tok/s", "mb/s", "gb/s", "gpu/cpu", "cpu/gpu", "vram/ram", "ram/vram"}
        for m in re.finditer(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b", raw):
            tok = m.group(1)
            if tok.lower() not in bad_slashes and not tok.startswith("http"):
                return tok

        # 2. Strip trailing hardware/assumption clauses
        cleaned = re.sub(
            r"\b(?:assuming|using|with|given|when|if|on my|for my|to fit|that fits|fits my|fits on)\b.*$",
            "",
            raw,
            flags=re.I,
        ).strip()
        cleaned = re.sub(
            r"\bfor\s+(?:fastest|balanced|max[\s_-]?context|max[\s_-]?ctx|low[\s_-]?vram|speed|coding|chat|reasoning)\b.*$",
            "",
            cleaned,
            flags=re.I,
        ).strip()

        # 3. Strip conversational filler words while preserving model identifiers
        stop = {
            "the", "a", "an", "model", "models", "quant", "quants", "quantization", "quantisation",
            "gguf", "ggufs", "correct", "best", "optimal", "right", "fitting", "fit", "fits",
            "calculate", "calc", "compute", "figure", "out", "find", "search", "recommend",
            "suggest", "pick", "choose", "pull", "download", "fetch", "get", "grab", "queue",
            "please", "can", "could", "would", "you", "me", "my", "gpu", "vram", "ram",
            "empty", "free", "idle", "unoccupied", "nothing", "else", "is", "are", "occupying",
            "and", "to", "for", "of", "on", "in", "that", "what", "which", "size", "weights",
            "from", "hugging", "face", "huggingface", "hf", "repo", "file", "files", "it",
        }
        tokens = re.findall(r"[A-Za-z0-9._+-]+", cleaned)
        kept = [t for t in tokens if t.lower() not in stop]
        return " ".join(kept).strip()

    def _active_engines_summary(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            st = self.host.mgr.status() or {}
            out.append({"engine": "llama", "running": bool(st.get("running")), "model": st.get("model") or "", "port": st.get("port")})
        except Exception:
            out.append({"engine": "llama", "running": False, "model": ""})
        try:
            av = self.host.smgr.status() or {}
            out.append({"engine": "avalon", "running": bool(av.get("running")), "model": av.get("profile") or "", "port": av.get("port")})
        except Exception:
            out.append({"engine": "avalon", "running": False, "model": ""})
        try:
            sr = self.host.strata_status(refresh=False) or {}
            out.append({"engine": "strata", "running": bool(sr.get("running")), "model": sr.get("model") or sr.get("preset") or "", "port": sr.get("port")})
        except Exception:
            out.append({"engine": "strata", "running": False, "model": ""})
        if self.host.maya_mgr:
            try:
                my = self.host.maya_mgr.status() or {}
                out.append({"engine": "maya", "running": bool(my.get("running")), "model": my.get("config_id") or "", "port": my.get("port")})
            except Exception:
                out.append({"engine": "maya", "running": False, "model": ""})
        return out

    def _format_hf_recommendations(
        self,
        rec_data: dict[str, Any],
        *,
        auto_pull: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        hb = rec_data.get("hardware_budget") or {}
        recs = rec_data.get("recommendations") or []
        gpu_name = hb.get("gpu_name") or "GPU"
        total_mb = hb.get("gpu_total_mb") or 0
        used_mb = hb.get("gpu_used_mb") or 0
        res_mb = hb.get("vram_reserve_mb") or 512
        usable_mb = hb.get("usable_vram_mb") or 0
        ram_mb = hb.get("avail_ram_mb") or 0
        empty_gpu = hb.get("assume_empty_gpu", True)

        mode_note = (
            f"assuming empty GPU (ignoring {used_mb} MB currently used)"
            if empty_gpu
            else f"using currently free VRAM only ({used_mb} MB currently used)"
        )
        header = (
            f"**HF Quant Calculator ({rec_data.get('goal', 'balanced')})** for `{rec_data.get('query', '')}`\n"
            f"Budget: {gpu_name} · {total_mb} MB total − {res_mb} MB reserve = **{usable_mb} MB usable VRAM** "
            f"({mode_note}) · {round(ram_mb / 1024.0, 1)} GB free RAM"
        )
        actions: list[dict[str, Any]] = []
        if not recs:
            msg = (
                header
                + "\n\nNo fitting GGUF quants were found for that query within your VRAM+RAM budget. "
                "Try a broader model name (e.g. `Qwen3-32B`, `unsloth/Qwen3-30B-A3B-GGUF`) or switch goal to `low_vram`."
            )
            actions.append({"label": "Open Quant Calculator", "type": "open_template", "action_id": "hf_quant_empty_gpu"})
            return msg, actions

        lines = []
        for idx, r in enumerate(recs[:5], 1):
            ram_part = f" + {r['ram_est_mb']} MB RAM experts" if r.get("ram_est_mb") else ""
            extras = []
            if r.get("mmproj"):
                extras.append("+mmproj")
            if r.get("draft"):
                extras.append("+draft")
            extra_str = (" " + " ".join(extras)) if extras else ""
            lines.append(
                f"{idx}. `{r['repo']}` · **{r['quant']}** ({r['size_gb']} GB) — {r['fit_label']} "
                f"(~{r['gpu_est_mb']} MB GPU{ram_part}){extra_str}"
            )

        body = header + "\n\n" + "\n".join(lines)

        if auto_pull and self.host.hf_pull_fn:
            top = recs[0]
            try:
                pulled = self.host.hf_pull_fn(top["repo"], top["pull_files"], top.get("pull_sizes") or {})
                started = (pulled or {}).get("started") or top["pull_files"]
                body += f"\n\nQueued download for **{top['repo']} ({top['quant']})**: {len(started)} file(s)."
                actions.append({"label": "View Downloads", "type": "open_tab", "tab": "models"})
            except Exception as e:
                body += f"\n\nAuto-pull failed: {e}"
                actions.append({
                    "label": f"Retry Pull #{1}: {top['quant']} ({top['size_gb']} GB)",
                    "type": "hf_pull",
                    "repo": top["repo"],
                    "files": top["pull_files"],
                    "sizes": top.get("pull_sizes") or {},
                })
        else:
            for idx, r in enumerate(recs[:3], 1):
                actions.append({
                    "label": f"Queue Pull #{idx}: {r['quant']} ({r['size_gb']} GB)",
                    "type": "hf_pull",
                    "repo": r["repo"],
                    "files": r["pull_files"],
                    "sizes": r.get("pull_sizes") or {},
                })

        if empty_gpu and used_mb > 1024:
            actions.append({"label": "Stop All Engines (Free GPU)", "type": "killswitch"})
        return body, actions

    def execute_action(
        self,
        action_id: str,
        params: dict[str, Any] | None = None,
        engine: str = "llama",
        model: str = "",
    ) -> dict[str, Any]:
        """Deterministically execute a structured action template from the Agent Chat Action Menu."""
        p = dict(params or {})
        eng = (p.get("engine") or engine or "llama").strip().lower()
        mod = (p.get("model") or model or "").strip()
        actions: list[dict[str, Any]] = []
        reply = ""

        if action_id in ("hf_quant_empty_gpu", "hf_quant_free_vram"):
            query = str(p.get("query") or "").strip()
            goal = str(p.get("goal") or ("low_vram" if action_id == "hf_quant_free_vram" else "balanced"))
            default_empty = action_id == "hf_quant_empty_gpu"
            assume_empty = bool(p.get("assume_empty_gpu", default_empty))
            auto_pull = bool(p.get("auto_pull", False))
            if not query:
                vit = self.host.vitals(refresh=False)
                gpu = vit.get("gpu") or {}
                total_mb = round(float(gpu.get("total_mb") or 8192))
                res_mb = round(float(self.host.cfg.get("vram_reserve_mb") or 512))
                usable = max(512, total_mb - res_mb)
                reply = (
                    f"Specify a model name or Hugging Face repo to calculate the best quant.\n"
                    f"Assuming an empty GPU ({gpu.get('name') or 'GPU'}), you have **{usable} MB usable VRAM** "
                    f"({total_mb} MB total − {res_mb} MB reserve)."
                )
                actions.append({"label": "Open Quant Calculator", "type": "open_template", "action_id": action_id})
            else:
                try:
                    rec_data = self.recommend_hf(query=query, goal=goal, assume_empty_gpu=assume_empty)
                    reply, actions = self._format_hf_recommendations(rec_data, auto_pull=auto_pull)
                except Exception as e:
                    reply = f"HF quant calculation failed for `{query}`: {e}"

        elif action_id == "downloads_status":
            if self.host.downloads_fn:
                dls = self.host.downloads_fn() or []
                if not dls:
                    reply = "No active or recent Hugging Face downloads."
                else:
                    lines = []
                    for d in dls[:8]:
                        pct = f" ({round(100 * d['done'] / d['total'])}%)" if d.get("total") else ""
                        lines.append(f"- `{d.get('file') or d.get('id')}` — **{d.get('status')}**{pct}")
                    reply = "**Hugging Face Downloads**\n" + "\n".join(lines)
                actions.append({"label": "Open Models & Downloads Tab", "type": "open_tab", "tab": "models"})
            else:
                reply = "Download status hook is not available."

        elif action_id == "optimize_engine":
            goal = str(p.get("goal") or "balanced")
            notes = str(p.get("notes") or "")
            goal, constraints = self._parse_prompt_constraints(notes, goal, {})
            try:
                rec = self.advise(engine=eng, goal=goal, model=mod, constraints=constraints, user_prompt=notes, use_llm=True)
                reply = (
                    f"**{eng.upper()} recommendation ({rec['goal']})** for `{rec['model_key']}`:\n"
                    + "\n".join(f"- {n}" for n in rec.get("notes", [])[:6])
                )
                if rec.get("llm_analysis"):
                    reply += "\n\n" + rec["llm_analysis"]
                actions.append({"label": f"Apply & Launch ({eng})", "type": "apply", "engine": eng, "goal": rec["goal"], "config": rec["config"]})
                actions.append({"label": f"Start Auto-Tune ({rec['goal']})", "type": "start_tune", "engine": eng, "goal": rec["goal"], "model": rec.get("model_path") or mod})
            except Exception as e:
                reply = f"Could not optimize {eng}: {e}"

        elif action_id == "start_autotune":
            goal = str(p.get("goal") or "fastest")
            max_trials = int(p.get("max_trials") or 5)
            try:
                st = self.start_tune(engine=eng, goal=goal, model=mod, max_trials=max_trials)
                reply = (
                    f"**Active Auto-Tune started** on `{eng}` (goal: `{goal}`, max {max_trials} trials).\n"
                    f"Safety checkpoint `{st.get('checkpoint_id', '')}` created."
                )
                actions.append({"label": "Open Agent Tab", "type": "open_tab", "tab": "agent"})
                actions.append({"label": "Stop Auto-Tune", "type": "stop_tune", "rollback": False})
                actions.append({"label": "Stop & Rollback", "type": "stop_tune", "rollback": True})
            except Exception as e:
                reply = f"Could not start auto-tune on {eng}: {e}"

        elif action_id == "stop_autotune":
            rb = bool(p.get("rollback", False))
            st = self.stop_tune(rollback=rb)
            reply = f"Auto-tune stop requested (rollback={rb}). Current status: `{st.get('state', 'idle')}`."

        elif action_id == "route_workload":
            task = str(p.get("task") or "coding")
            do_apply = bool(p.get("apply", False))
            do_sync = bool(p.get("sync_agents", False))
            try:
                r = self.route_workload(task=task, apply=do_apply, sync_agents=do_sync)
                w = r.get("winner") or {}
                reply = (
                    f"**Workload Router ({task})**\n"
                    f"Best match: `{w.get('model_name') or w.get('model_path') or 'default'}` on **{w.get('engine', 'llama')}** "
                    f"(goal: `{w.get('goal', 'balanced')}`, score: {w.get('score', 0)}).\n"
                    + (f"Applied: yes (checkpoint `{r['applied'].get('checkpoint_id', '')}`)" if r.get("applied") else "Dry-run preview (not yet launched).")
                )
                if not do_apply and w.get("config"):
                    actions.append({"label": f"Apply & Launch ({w.get('engine', 'llama')})", "type": "apply", "engine": w.get("engine", "llama"), "goal": w.get("goal", "balanced"), "config": w["config"]})
                actions.append({"label": "Register in Coding Agents", "type": "sync_agents", "source": "auto"})
            except Exception as e:
                reply = f"Workload routing failed: {e}"

        elif action_id == "tune_speculation":
            apply_best = bool(p.get("apply_best", True))
            try:
                r = self.tune_speculation(model=mod, apply_best=apply_best)
                trials = r.get("trials") or []
                lines = [
                    f"- {t['label']}: {t['tok_s']} tok/s ({'+' if t.get('speedup_pct', 0) >= 0 else ''}{t.get('speedup_pct', 0)}%)"
                    if t.get("ok") else f"- {t['label']}: failed ({t.get('error', '')})"
                    for t in trials
                ]
                reply = f"**Speculative Decoding Tuner**\n{r.get('verdict', '')}\n" + "\n".join(lines)
            except Exception as e:
                reply = f"Speculative decoding tuner failed: {e}"

        elif action_id == "system_status":
            vit = self.host.vitals(refresh=False)
            gpu = vit.get("gpu") or {}
            mem = vit.get("mem") or {}
            cpu = vit.get("cpu") or {}
            engines = self._active_engines_summary()
            eng_lines = [
                f"- **{e['engine']}**: {'RUNNING' if e['running'] else 'stopped'}"
                + (f" (`{e['model']}` on :{e['port']})" if e["running"] and e.get("port") else (f" (`{e['model']}`)" if e["running"] and e.get("model") else ""))
                for e in engines
            ]
            total_mb = round(float(gpu.get("total_mb") or 0))
            used_mb = round(float(gpu.get("used_mb") or 0))
            free_mb = max(0, total_mb - used_mb)
            reply = (
                f"**Hardware & Engine Status**\n"
                f"- GPU: {gpu.get('name') or 'Unknown'} · {used_mb} / {total_mb} MB VRAM ({free_mb} MB free) · {gpu.get('temp', '?')}°C · {gpu.get('util', '?')}% util\n"
                f"- RAM: {round(float(mem.get('used_gb') or 0), 1)} / {round(float(mem.get('total_gb') or 0), 1)} GB\n"
                f"- CPU: {cpu.get('cores', '?')} cores · {cpu.get('percent', '?')}%\n\n"
                f"**Engines**\n" + "\n".join(eng_lines)
            )
            running = [e for e in engines if e["running"]]
            for e in running:
                actions.append({"label": f"Stop {e['engine']}", "type": "stop_engine", "engine": e["engine"]})
            if running:
                actions.append({"label": "Run Short Benchmark", "type": "benchmark", "prompt_key": "short", "max_tokens": 128})
                actions.append({"label": "Register in Coding Agents", "type": "sync_agents", "source": "auto"})
            actions.append({"label": "Calculate Best Quant (Empty GPU)", "type": "open_template", "action_id": "hf_quant_empty_gpu"})

        elif action_id == "run_benchmark":
            pk = str(p.get("prompt_key") or "short")
            mt = int(p.get("max_tokens") or 128)
            try:
                res = self.host.benchmark_run(pk, mt)
                gen = res.get("gen_tps") or res.get("tok_s") or res.get("tokens_per_second") or 0
                pp = res.get("prompt_tps") or res.get("prefill_tps") or 0
                reply = f"**Benchmark complete (`{pk}`, {mt} tok)**\n- Generation: **{round(float(gen), 2)} tok/s**" + (f"\n- Prefill: **{round(float(pp), 2)} tok/s**" if pp else "")
                actions.append({"label": "Open Benchmarks Tab", "type": "open_tab", "tab": "bench"})
            except Exception as e:
                reply = f"Benchmark failed (make sure an engine is running first): {e}"

        elif action_id == "stop_engine":
            try:
                if eng == "llama":
                    self.host.mgr.stop()
                elif eng == "avalon":
                    self.host.smgr.stop()
                elif eng == "strata":
                    self.host.strata_stop()
                elif eng == "maya" and self.host.maya_mgr:
                    self.host.maya_mgr.stop()
                else:
                    raise ValueError(f"Unknown or unavailable engine: {eng}")
                reply = f"Stopped **{eng}** engine."
            except Exception as e:
                reply = f"Failed to stop {eng}: {e}"

        elif action_id == "stop_all_engines":
            res = self.emergency_killswitch()
            stopped = res.get("stopped") or []
            reply = (
                f"**Stopped all running engines** ({', '.join(stopped) if stopped else 'none were running'}).\n"
                f"Safety checkpoint `{res.get('checkpoint_id', '')}` saved. GPU VRAM is now free."
            )
            if res.get("checkpoint_id"):
                actions.append({"label": f"Restore checkpoint ({res['checkpoint_id']})", "type": "restore_checkpoint", "checkpoint_id": res["checkpoint_id"]})
            actions.append({"label": "Calculate Best Quant (Empty GPU)", "type": "open_template", "action_id": "hf_quant_empty_gpu"})

        elif action_id == "strata_doctor":
            if self.host.strata_doctor_fn:
                try:
                    d = self.host.strata_doctor_fn() or {}
                    out = d.get("output") or d.get("detail") or json.dumps(d)
                    reply = f"**Strata Doctor**\n{str(out)[:1400]}"
                except Exception as e:
                    reply = f"Strata doctor failed: {e}"
            else:
                reply = "Strata doctor hook is not wired."

        elif action_id == "diagnose_engine":
            diag = self.diagnose_failure(engine=eng, use_llm=True)
            reply = f"**{diag['title']}** ({eng})\n{diag['cause']}"
            if diag.get("remediation_steps"):
                reply += "\n\nProposed fix:\n" + "\n".join(f"- {s}" for s in diag["remediation_steps"])
                actions.append({"label": f"Fix & Relaunch ({eng})", "type": "apply", "engine": eng, "config": diag["fix_config"]})
            if diag.get("llm_diagnosis"):
                reply += "\n\n" + diag["llm_diagnosis"]

        elif action_id == "heal_engine":
            res = self.heal_failure(engine=eng)
            diag = res.get("diagnosis") or {}
            reply = f"**Auto-Heal ({eng})**: {diag.get('title', 'Completed')}\n"
            if res.get("healed"):
                reply += f"Applied fix and relaunched (checkpoint `{(res.get('applied') or {}).get('checkpoint_id', '')}`)."
            else:
                reply += res.get("detail") or diag.get("cause") or "No automatic fix was required."

        elif action_id == "thermal_status":
            if self.host.thermal_status_fn:
                st = self.host.thermal_status_fn() or {}
                reply = (
                    f"**Thermal Guard**: `{st.get('state', 'unknown')}` · enabled={st.get('enabled', False)} · "
                    f"auto_act={st.get('auto_act', False)} · eff_temp={st.get('eff_temp_c', '?')}°C"
                )
            else:
                vit = self.host.vitals(refresh=False)
                gpu = vit.get("gpu") or {}
                reply = f"**GPU Thermals**: {gpu.get('temp', '?')}°C · {gpu.get('util', '?')}% util"
            actions.append({"label": "Evaluate Thermals Now", "type": "thermal_evaluate"})
            actions.append({"label": "Downshift Active Engine", "type": "downshift"})
            actions.append({"label": "Emergency Stop All", "type": "killswitch"})

        elif action_id == "thermal_downshift":
            res = self.downshift_active_engine()
            reply = f"**Thermal Downshift**: {res.get('detail', 'Done')}"

        elif action_id == "checkpoints_list":
            cps = self.store.list_checkpoints(limit=6)
            if not cps:
                reply = "No checkpoints saved yet. One is created automatically before Apply or Auto-Tune."
            else:
                lines = [f"- `{c['id']}` · {c['label']} ({c['engine']})" for c in cps]
                reply = "**Recent Checkpoints**\n" + "\n".join(lines)
                for c in cps[:2]:
                    actions.append({"label": f"Restore {c['id']} ({c['engine']})", "type": "restore_checkpoint", "checkpoint_id": c["id"]})
            actions.append({"label": f"Create Checkpoint ({eng})", "type": "create_checkpoint", "engine": eng})

        elif action_id == "checkpoint_create":
            lbl = str(p.get("label") or f"Chat checkpoint ({eng})")
            cp = self.capture_checkpoint(engine=eng, label=lbl, reason="manual_chat")
            reply = f"Created checkpoint `{cp['id']}` for **{eng}**: {cp['label']}."
            actions.append({"label": f"Restore {cp['id']}", "type": "restore_checkpoint", "checkpoint_id": cp["id"]})

        elif action_id == "list_models":
            scan = self.host.scan_models() or {}
            models = scan.get("models") or []
            if not models:
                reply = "No local GGUF models found. Use **Calculate & Pull Best Quant** to download one from Hugging Face."
                actions.append({"label": "Calculate & Pull Best Quant", "type": "open_template", "action_id": "hf_quant_empty_gpu"})
            else:
                lines = []
                for m in models[:12]:
                    sz = round(float(m.get("size") or 0) / (1024**3), 2)
                    tag = "MoE" if m.get("moe") else "Dense"
                    lines.append(f"- `{m.get('name')}` · {m.get('quant') or '?'} · {sz} GB · {tag}")
                reply = f"**Local Models ({len(models)} total)**\n" + "\n".join(lines)
                top_m = models[0]
                actions.append({"label": f"Optimize {top_m.get('name', 'top model')[:24]}", "type": "send_chat", "message": f"Optimize llama.cpp for {top_m.get('name', '')}"})
                actions.append({"label": "Open Models Tab", "type": "open_tab", "tab": "models"})

        elif action_id == "presets_list":
            presets = self.store.list_presets()
            if not presets:
                reply = "No optimization presets saved."
            else:
                lines = [f"- **{pr['name']}** ({pr['engine']} · {pr['goal']})" for pr in presets[:10]]
                reply = "**Optimization Presets**\n" + "\n".join(lines)
                for pr in presets[:3]:
                    actions.append({"label": f"Apply {pr['name'][:22]}", "type": "apply", "engine": pr["engine"], "goal": pr["goal"], "config": pr["config"]})

        elif action_id == "sync_coding_agents":
            src = str(p.get("source") or "auto")
            if self.host.sync_agents_fn and self.host.coding_agents_list:
                ags = self.host.coding_agents_list()
                try:
                    res = self.host.sync_agents_fn(ags, src)
                    ok_cnt = sum(1 for v in (res.get("results") or {}).values() if isinstance(v, dict) and v.get("ok"))
                    reply = f"**Synced {ok_cnt}/{len(ags)} coding agents** (source: `{src}`)."
                except Exception as e:
                    reply = f"Coding-agent sync failed (is an engine running?): {e}"
            else:
                reply = "Coding-agent sync hook is not wired."

        elif action_id == "agent_memory":
            mem = self.store.memory_summary(limit=8)
            best = mem.get("best") or []
            if not best:
                reply = "No benchmarked experience records saved yet. Run Active Auto-Tune to populate memory."
            else:
                lines = [f"- `{b.get('model_key')}` ({b.get('engine')} · {b.get('goal')}): **{b.get('tok_s', 0)} tok/s**" for b in best[:6]]
                reply = f"**Agent Experience Memory ({mem.get('total_records', 0)} records)**\n" + "\n".join(lines)

        else:
            reply = f"Unknown action template `{action_id}`."

        return {"reply": reply, "actions": actions, "action_id": action_id, "llm": self.llm_status()}

    def chat(
        self,
        message: str = "",
        engine: str = "llama",
        model: str = "",
        action_id: str = "",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action_id:
            res = self.execute_action(action_id=action_id, params=params, engine=engine, model=model)
            entry = {"ts": time.time(), "user": message or f"[{action_id}]", "reply": res["reply"], "actions": res["actions"]}
            self._chat_history.append(entry)
            self._chat_history = self._chat_history[-40:]
            return res

        msg = (message or "").strip()
        if not msg:
            return {
                "reply": "Pick any workflow from the **Actions** menu or ask me to calculate & pull a quant, optimize an engine, diagnose a crash, run a benchmark, or manage checkpoints.",
                "actions": [{"label": "Calculate & Pull Best Quant (Empty GPU)", "type": "open_template", "action_id": "hf_quant_empty_gpu"}],
                "llm": self.llm_status(),
            }

        low = msg.lower()
        actions: list[dict[str, Any]] = []
        reply_parts: list[str] = []

        # Detect engine override in message
        for eng_kw, eng_id in (("llama.cpp", "llama"), ("llama", "llama"), ("avalon", "avalon"), ("strata", "strata"), ("maya", "maya")):
            if eng_kw in low:
                engine = eng_id
                break

        # 1. Hugging Face Quant Calculation & Pull intent (checked BEFORE generic 'recommend' -> advise!)
        hf_quant_kws = ("quant", "quantiz", "pull", "download", "hugging face", "huggingface", "hf ", "gguf")
        is_download_status = any(k in low for k in ("download status", "active download", "show download", "check download"))
        if is_download_status:
            res = self.execute_action("downloads_status", {}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])
        elif any(k in low for k in hf_quant_kws):
            goal, _ = self._parse_prompt_constraints(msg, "balanced", {})
            free_only_kws = ("currently free", "free vram only", "remaining vram", "alongside", "with current")
            assume_empty = not any(k in low for k in free_only_kws)
            query = self._extract_hf_query(msg)
            if not query:
                vit = self.host.vitals(refresh=False)
                gpu = vit.get("gpu") or {}
                mem = vit.get("mem") or {}
                total_mb = round(float(gpu.get("total_mb") or 8192))
                used_mb = round(float(gpu.get("used_mb") or 0))
                res_mb = round(float(self.host.cfg.get("vram_reserve_mb") or 512))
                usable_mb = max(512, total_mb - res_mb) if assume_empty else max(512, total_mb - used_mb - res_mb)
                free_ram_gb = round(max(1.0, float(mem.get("total_gb") or 32) - float(mem.get("used_gb") or 8)), 1)
                mode_label = "assuming nothing else is occupying the GPU" if assume_empty else "using currently free VRAM only"
                reply_parts.append(
                    f"**Ready to calculate & pull the best quant ({mode_label})**\n"
                    f"- GPU: {gpu.get('name') or 'GPU'} · **{usable_mb} MB usable VRAM** "
                    f"({total_mb} MB total − {res_mb} MB reserve"
                    + (f"; ignoring {used_mb} MB currently used" if assume_empty else f" − {used_mb} MB used")
                    + f") · {free_ram_gb} GB free RAM\n\n"
                    "Which model or Hugging Face repo would you like me to calculate for? "
                    "Type a model name (e.g. *\"Calculate and pull best quant for Qwen3-32B\"*) or click **Open Quant Calculator** below."
                )
                actions.append({"label": "Open Quant Calculator", "type": "open_template", "action_id": "hf_quant_empty_gpu" if assume_empty else "hf_quant_free_vram"})
                actions.append({"label": "Try Qwen3-32B", "type": "send_chat", "message": "Calculate and pull best quant for Qwen3-32B assuming empty GPU"})
                actions.append({"label": "Try Qwen3-30B-A3B", "type": "send_chat", "message": "Calculate and pull best quant for Qwen3-30B-A3B assuming empty GPU"})
            else:
                try:
                    rec_data = self.recommend_hf(query=query, goal=goal, assume_empty_gpu=assume_empty)
                    body, hf_acts = self._format_hf_recommendations(rec_data, auto_pull=False)
                    reply_parts.append(body)
                    actions.extend(hf_acts)
                except Exception as e:
                    reply_parts.append(f"Could not calculate HF quants for `{query}`: {e}")

        # 2. Diagnose / Heal
        elif any(k in low for k in ("diagnose", "crash", "oom", "error", "failed", "why did it fail", "fix", "heal")):
            if "auto-heal" in low or "auto heal" in low or low.startswith("heal"):
                res = self.execute_action("heal_engine", {"engine": engine}, engine=engine, model=model)
                reply_parts.append(res["reply"])
                actions.extend(res["actions"])
            else:
                diag = self.diagnose_failure(engine=engine, use_llm=True)
                reply_parts.append(f"**{diag['title']}** ({engine})\n{diag['cause']}")
                if diag["remediation_steps"]:
                    reply_parts.append("Proposed fix:\n" + "\n".join(f"- {s}" for s in diag["remediation_steps"]))
                    actions.append({
                        "label": f"Fix & Relaunch ({engine})",
                        "type": "apply",
                        "engine": engine,
                        "config": diag["fix_config"],
                    })
                if diag.get("llm_diagnosis"):
                    reply_parts.append(diag["llm_diagnosis"])

        # 3. Checkpoints
        elif any(k in low for k in ("checkpoint", "rollback", "revert", "undo")):
            res = self.execute_action("checkpoints_list", {"engine": engine}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])

        # 4. Stop / Unload / Free GPU
        elif any(k in low for k in ("stop all", "killswitch", "free gpu", "clear vram", "unload all")):
            res = self.execute_action("stop_all_engines", {}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])
        elif re.search(r"\b(stop|unload)\s+(llama|avalon|strata|maya)\b", low):
            res = self.execute_action("stop_engine", {"engine": engine}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])

        # 5. Workload Router
        elif any(k in low for k in ("route ", "workload", "best model for coding", "best model for vision", "best model for reasoning", "best model for chat")):
            task = "coding"
            for t in ("vision", "reasoning", "long_context", "chat", "coding"):
                if t.replace("_", " ") in low or t in low:
                    task = t
                    break
            res = self.execute_action("route_workload", {"task": task, "apply": False, "sync_agents": False}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])

        # 6. Speculative Decoding
        elif any(k in low for k in ("speculat", "draft model")):
            res = self.execute_action("tune_speculation", {"model": model, "apply_best": True}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])

        # 7. Benchmark
        elif any(k in low for k in ("benchmark", "speed test", "tok/s test")):
            pk = "code" if "code" in low else ("standard" if "standard" in low else "short")
            res = self.execute_action("run_benchmark", {"prompt_key": pk, "max_tokens": 128}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])

        # 8. Coding-Agent Sync
        elif any(k in low for k in ("sync coding", "register coding", "coding agent", "register in", "opencode", "claude-code", "codex", "aider")):
            res = self.execute_action("sync_coding_agents", {"source": "auto"}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])

        # 9. Thermal Guard
        elif any(k in low for k in ("thermal", "overheat", "downshift", "gpu temp")):
            act = "thermal_downshift" if "downshift" in low else "thermal_status"
            res = self.execute_action(act, {}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])

        # 10. Local Models / Presets / Memory / Status
        elif any(k in low for k in ("list models", "local models", "installed models", "my models")):
            res = self.execute_action("list_models", {}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])
        elif any(k in low for k in ("presets", "saved preset")):
            res = self.execute_action("presets_list", {}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])
        elif any(k in low for k in ("memory", "past trials", "history")):
            res = self.execute_action("agent_memory", {}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])
        elif any(k in low for k in ("status", "vitals", "what is running", "what's running", "gpu usage", "vram usage")):
            res = self.execute_action("system_status", {}, engine=engine, model=model)
            reply_parts.append(res["reply"])
            actions.extend(res["actions"])

        # 11. Optimize / Advise / Auto-Tune
        elif any(k in low for k in ("optimi", "config", "tune", "fastest", "longest", "context", "max ctx", "low vram", "moe", "recommend")):
            goal, constraints = self._parse_prompt_constraints(msg, "balanced", {})
            try:
                rec = self.advise(engine=engine, goal=goal, model=model, constraints=constraints, user_prompt=msg, use_llm=True)
                reply_parts.append(
                    f"**{engine.upper()} recommendation ({rec['goal']})** for `{rec['model_key']}`:\n"
                    + "\n".join(f"- {n}" for n in rec.get("notes", [])[:5])
                )
                if rec.get("llm_analysis"):
                    reply_parts.append(rec["llm_analysis"])
                actions.append({
                    "label": f"Apply & Launch ({engine})",
                    "type": "apply",
                    "engine": engine,
                    "goal": rec["goal"],
                    "config": rec["config"],
                })
                actions.append({
                    "label": f"Start Active Auto-Tune ({rec['goal']})",
                    "type": "start_tune",
                    "engine": engine,
                    "goal": rec["goal"],
                    "model": rec.get("model_path") or model,
                })
            except Exception as e:
                reply_parts.append(f"Could not compute advice for {engine}: {e}")

        # 12. General LLM fallback with full live app context
        else:
            vit = self.host.vitals(refresh=False)
            mem = self.store.memory_summary(limit=5)
            engines_st = self._active_engines_summary()
            local_models = [
                {"name": m.get("name"), "quant": m.get("quant"), "size_gb": round(float(m.get("size") or 0) / (1024**3), 2), "moe": bool(m.get("moe"))}
                for m in ((self.host.scan_models() or {}).get("models") or [])[:10]
            ]
            caps_summary = [f"{a['id']}: {a['label']}" for a in AGENT_ACTION_TEMPLATES]
            llm_reply = self._llm_complete(
                "You are the Inference Console Agent assistant. You can help with ANY workflow in the app: "
                "calculating and pulling Hugging Face GGUF quants (assuming empty GPU or free VRAM), "
                "optimizing/auto-tuning llama.cpp, Avalon, Strata, and Maya, diagnosing crashes, benchmarking, "
                "workload routing, speculative decoding, thermal guard, checkpoints, and coding-agent sync. "
                "Be concise, technical, and grounded in the live hardware, models, and engine state provided.",
                f"Hardware: {json.dumps(vit)}\n"
                f"Engines: {json.dumps(engines_st)}\n"
                f"Local models: {json.dumps(local_models)}\n"
                f"Available app actions: {json.dumps(caps_summary)}\n"
                f"Memory summary: {json.dumps(mem['best'][:3])}\n"
                f"User: {msg}",
                max_tokens=1024,
            )
            if llm_reply:
                reply_parts.append(llm_reply)
            else:
                st = self.llm_status()
                if st.get("available"):
                    lbl = st.get("provider_label") or st.get("provider") or "provider"
                    mod = st.get("model") or ""
                    err = self._last_llm_error or "empty response"
                    reply_parts.append(
                        f"Configured LLM ({lbl}{f' · `{mod}`' if mod else ''}) failed: {err}\n"
                        "Check **Agent → LLM Provider** (Base URL, API key, Model ID) and click **Test**."
                    )
                else:
                    reply_parts.append(
                        "Deterministic mode active (open the **Actions** menu above for every app workflow, or configure an LLM in **Agent → LLM Provider**). Try asking:\n"
                        "- *\"Calculate and pull best quant for Qwen3-32B assuming empty GPU\"*\n"
                        "- *\"Optimize llama.cpp for fastest speed\"*\n"
                        "- *\"Show hardware and engine status\"*\n"
                        "- *\"Diagnose last crash\"*"
                    )
                    actions.append({"label": "Calculate & Pull Best Quant", "type": "open_template", "action_id": "hf_quant_empty_gpu"})
                    actions.append({"label": "Hardware & Engine Status", "type": "send_chat", "message": "Show hardware and engine status"})

        reply = "\n\n".join(p for p in reply_parts if p)
        entry = {"ts": time.time(), "user": msg, "reply": reply, "actions": actions}
        self._chat_history.append(entry)
        self._chat_history = self._chat_history[-40:]
        return {"reply": reply, "actions": actions, "llm": self.llm_status()}

    # ------------------------------------------------------------ Phase 2: Thermal Guard callbacks
    def downshift_active_engine(self) -> dict[str, Any]:
        """Create a checkpoint and downshift the currently running engine to a cooler/low_vram config."""
        srv = self.host.mgr.status()
        if srv.get("running") and srv.get("model"):
            rec = self.advise(engine="llama", goal="low_vram", model=srv["model"], use_llm=False)
            applied = self.apply_config(
                engine="llama",
                config=rec["config"],
                launch=True,
                goal="low_vram",
                checkpoint_label="Before thermal/VRAM downshift (llama)",
            )
            return {"ok": True, "engine": "llama", "detail": "Downshifted llama.cpp to low_vram profile", "applied": applied}

        av = self.host.smgr.status()
        if av.get("running"):
            prof = av.get("profile") or "flagship"
            rec = self.advise(engine="avalon", goal="low_vram", profile=prof, use_llm=False)
            applied = self.apply_config(
                engine="avalon",
                config=rec["config"],
                launch=True,
                goal="low_vram",
                checkpoint_label="Before thermal/VRAM downshift (avalon)",
            )
            return {"ok": True, "engine": "avalon", "detail": "Downshifted Avalon to low_vram profile", "applied": applied}

        st = self.host.strata_status(refresh=False)
        if st.get("running"):
            rec = self.advise(engine="strata", goal="low_vram", use_llm=False)
            applied = self.apply_config(
                engine="strata",
                config=rec["config"],
                launch=True,
                goal="low_vram",
                checkpoint_label="Before thermal/VRAM downshift (strata)",
            )
            return {"ok": True, "engine": "strata", "detail": "Downshifted Strata to low_vram profile", "applied": applied}

        if self.host.maya_mgr and self.host.maya_mgr.status().get("running"):
            ms = self.host.maya_mgr.status()
            rec = self.advise(engine="maya", goal="low_vram", model=ms.get("config_id") or "", use_llm=False)
            applied = self.apply_config(
                engine="maya",
                config=rec["config"],
                launch=True,
                goal="low_vram",
                checkpoint_label="Before thermal/VRAM downshift (maya)",
            )
            return {"ok": True, "engine": "maya", "detail": "Downshifted Maya to low_vram profile", "applied": applied}

        return {"ok": True, "engine": "none", "detail": "No active inference engine running to downshift"}

    def emergency_killswitch(self) -> dict[str, Any]:
        """Create a safety checkpoint and stop all active inference engines."""
        stopped = []
        cp = self.capture_checkpoint(engine="llama", label="Pre-killswitch emergency checkpoint", reason="thermal_killswitch")
        if self.host.mgr.status().get("running"):
            self.host.mgr.stop()
            stopped.append("llama")
        if self.host.smgr.status().get("running"):
            self.host.smgr.stop()
            stopped.append("avalon")
        if self.host.strata_status(refresh=False).get("running"):
            try:
                self.host.strata_stop()
                stopped.append("strata")
            except Exception:
                pass
        if self.host.maya_mgr and self.host.maya_mgr.status().get("running"):
            try:
                self.host.maya_mgr.stop()
                stopped.append("maya")
            except Exception:
                pass
        return {"ok": True, "checkpoint_id": cp["id"], "stopped": stopped}

    # ------------------------------------------------------------ Phase 2: Workload-Aware Router & Coding-Agent Sync
    def route_workload(
        self,
        *,
        task: str = "coding",
        apply: bool = False,
        sync_agents: bool = False,
        agents: list[str] | None = None,
    ) -> dict[str, Any]:
        """Select the best local (model, engine, config) for a workload, using persistent memory + MoE-aware scoring."""
        task = (task or "coding").lower()
        task_to_goal = {
            "coding": "balanced",
            "long_doc": "max_context",
            "fast_chat": "low_latency",
            "vision": "balanced",
            "reasoning": "balanced",
        }
        goal = task_to_goal.get(task, "balanced")
        vit = self.host.vitals(refresh=False)
        models = (self.host.scan_models() or {}).get("models") or []

        # 1. Check persistent memory for a verified task winner first
        mem_winner = self.store.get_task_best(task)
        candidates: list[dict[str, Any]] = []

        usable_vram = max(
            512.0,
            float((vit.get("gpu") or {}).get("total_mb") or 8192) - float(self.host.cfg.get("vram_reserve_mb") or 512),
        )
        avail_ram_mb = max(
            1024.0,
            (float((vit.get("mem") or {}).get("total_gb") or 32) - float((vit.get("mem") or {}).get("used_gb") or 8)) * 1024.0,
        )

        for m in models:
            moe_info = analyze_moe_mechanics(m, usable_vram_mb=usable_vram, avail_ram_mb=avail_ram_mb, kv_vram_mb=1024)
            if not moe_info["feasible"]:
                continue
            name_l = (m.get("name") or "").lower()
            size_gb = (m.get("size") or 0) / (1024**3)
            ctx_len = int(m.get("ctx_len") or 8192)
            is_vl = bool(m.get("vl") and m.get("mmproj"))
            is_moe = bool(m.get("moe"))

            score = 0.0
            reasons: list[str] = []

            if task == "vision":
                if not is_vl:
                    continue
                score += 60.0
                reasons.append("Vision-capable model with mmproj detected")
            elif task == "coding":
                if any(k in name_l for k in ("coder", "code", "qwen", "deepseek", "devstral", "codestral")):
                    score += 35.0
                    reasons.append("Coding-strong model family")
                if ctx_len >= 32768:
                    score += 20.0
                    reasons.append(f"Large native context ({ctx_len:,})")
                if moe_info["strategy"] in ("dense_full_gpu", "moe_full_gpu"):
                    score += 18.0
                    reasons.append("Full GPU speed")
                elif is_moe:
                    score += 14.0
                    reasons.append("MoE hybrid split keeps attention+KV on GPU")
            elif task == "long_doc":
                score += min(60.0, (ctx_len / 2048.0))
                reasons.append(f"Native context {ctx_len:,} tokens")
                if is_moe:
                    score += 10.0
            elif task == "fast_chat":
                if moe_info["strategy"] in ("dense_full_gpu", "moe_full_gpu"):
                    score += 40.0
                    reasons.append("Fits fully in VRAM for instant responses")
                score += max(0.0, 25.0 - size_gb)
                if m.get("path") and self.host.find_draft(m["path"]):
                    score += 15.0
                    reasons.append("Draft/MTP model available for speculative decoding")
            else:  # reasoning
                if any(k in name_l for k in ("r1", "think", "reason", "qwq", "qwen3", "flash-next")):
                    score += 30.0
                    reasons.append("Reasoning-optimized family")
                score += min(30.0, size_gb * 0.8)

            # Bonus if persistent memory has a verified winner for this model
            if mem_winner and mem_winner.get("model_key") == m.get("name"):
                score += 25.0
                reasons.append(f"Persistent memory verified winner for `{task}`")

            candidates.append({
                "engine": "llama",
                "model": m,
                "model_key": m.get("name") or Path(m["path"]).name,
                "score": round(score, 2),
                "reasons": reasons,
                "moe": moe_info,
            })

        # Also consider Avalon Flash-Next for reasoning / coding / long_doc if available
        av_tgt = self.host.avalon_target() or {}
        if task in ("reasoning", "coding", "long_doc") and av_tgt.get("model") and Path(self.host.avalon_bin).exists():
            av_score = 48.0 + (20.0 if task in ("reasoning", "long_doc") else 10.0)
            if mem_winner and mem_winner.get("engine") == "avalon":
                av_score += 25.0
            candidates.append({
                "engine": "avalon",
                "profile": "flagship",
                "model_key": av_tgt.get("base") or "avalon:flagship",
                "score": round(av_score, 2),
                "reasons": ["Avalon Flash-Next MoE pack available with `-fit on` GPU/RAM split"],
            })

        # Also consider Project Maya configs for coding / long_doc / reasoning / fast_chat
        if self.host.maya_mgr and task in ("reasoning", "coding", "long_doc", "fast_chat"):
            try:
                for mc in self.host.maya_mgr.discover_configs():
                    if not mc.get("available"):
                        continue
                    ms = 44.0
                    mreasons = [f"Project Maya ({mc.get('engine', 'glm')}, {mc.get('quant') or 'MoE'}) available"]
                    if task in ("reasoning", "coding"):
                        ms += 14.0
                    if task == "long_doc" and int(mc.get("ctx") or 0) >= 65536:
                        ms += 20.0
                        mreasons.append(f"Context {int(mc['ctx']):,}")
                    if mc.get("mtp") and task in ("fast_chat", "coding"):
                        ms += 8.0
                        mreasons.append("MTP drafting enabled")
                    if mem_winner and mem_winner.get("engine") == "maya" and mem_winner.get("model_key") == mc.get("model_name"):
                        ms += 25.0
                    candidates.append({
                        "engine": "maya",
                        "maya_config": mc,
                        "model_key": mc.get("model_name") or mc["id"],
                        "score": round(ms, 2),
                        "reasons": mreasons,
                    })
            except Exception:
                pass

        if not candidates:
            raise ValueError(f"No feasible local model found for workload '{task}'")

        candidates.sort(key=lambda x: x["score"], reverse=True)
        winner = candidates[0]

        if winner["engine"] == "llama":
            advice = self.advise(
                engine="llama",
                goal=goal,
                model=winner["model"]["path"],
                constraints={"min_ctx": 32768} if task in ("coding", "long_doc") else None,
                use_llm=False,
            )
        elif winner["engine"] == "maya":
            advice = self.advise(
                engine="maya",
                goal=goal,
                model=winner["maya_config"]["id"],
                constraints={"min_ctx": 32768} if task in ("coding", "long_doc") else None,
                use_llm=False,
            )
        else:
            advice = self.advise(engine="avalon", goal=goal, profile=winner.get("profile", "flagship"), use_llm=False)

        out: dict[str, Any] = {
            "task": task,
            "goal": goal,
            "selected": {
                "engine": winner["engine"],
                "model_key": winner["model_key"],
                "score": winner["score"],
                "reasons": winner["reasons"],
            },
            "alternatives": [
                {"engine": c["engine"], "model_key": c["model_key"], "score": c["score"], "reasons": c["reasons"]}
                for c in candidates[1:4]
            ],
            "advice": advice,
            "applied": None,
            "synced_agents": None,
        }

        if apply:
            out["applied"] = self.apply_config(
                engine=winner["engine"],
                config=advice["config"],
                launch=True,
                goal=goal,
                checkpoint_label=f"Before workload route ({task} -> {winner['model_key']})",
            )
            self.store.record_trial(
                engine=winner["engine"],
                model_key=winner["model_key"],
                goal=goal,
                task=task,
                config=advice["config"],
                score=winner["score"],
                ok=True,
                notes=f"Routed for workload `{task}`",
            )
            if sync_agents and self.host.sync_agents_fn:
                target_agents = agents or (self.host.coding_agents_list() if self.host.coding_agents_list else [])
                if target_agents:
                    src = "panel" if winner["engine"] == "llama" else winner["engine"]
                    out["synced_agents"] = self.host.sync_agents_fn(target_agents, src)

        return out

    # ------------------------------------------------------------ Phase 2: Speculative Decoding Auto-Pairing
    def tune_speculation(
        self,
        *,
        model: str = "",
        draft_candidates: list[str] | None = None,
        apply_best: bool = True,
    ) -> dict[str, Any]:
        """Benchmark baseline (no draft) vs candidate draft/MTP models; disable speculation if net tok/s drops."""
        meta = self._find_model(model)
        if not meta:
            raise ValueError("No target GGUF model found for speculative decoding tuner")

        cp = self.capture_checkpoint(
            engine="llama",
            label=f"Pre-speculation tune ({meta.get('name', 'model')})",
            reason="tune_speculation",
        )
        base_advice = self.advise(engine="llama", goal="fastest", model=meta["path"], use_llm=False)
        base_cfg = copy.deepcopy(base_advice["config"])
        base_cfg["draft"] = ""
        base_cfg["draft_max"] = 0

        # Discover draft candidates: explicit -> find_draft() beside weights -> smaller same-arch models
        discovered: list[str] = []
        if draft_candidates:
            discovered.extend([d for d in draft_candidates if d])
        else:
            side = self.host.find_draft(meta["path"])
            if side:
                discovered.append(side)
            all_models = (self.host.scan_models() or {}).get("models") or []
            target_arch = (meta.get("arch") or "").lower()
            target_size = int(meta.get("size") or 0)
            for m in all_models:
                if m.get("path") == meta.get("path"):
                    continue
                if target_arch and (m.get("arch") or "").lower() == target_arch and int(m.get("size") or 0) <= target_size * 0.25:
                    discovered.append(m["path"])

        # Deduplicate
        uniq_drafts: list[str] = []
        for d in discovered:
            if d not in uniq_drafts:
                uniq_drafts.append(d)

        trials: list[dict[str, Any]] = []

        # 1. Baseline (speculation OFF)
        ok_b, err_b = self._launch_for_trial("llama", base_cfg)
        if not ok_b:
            self.restore_checkpoint(cp["id"], relaunch=False)
            raise RuntimeError(f"Baseline launch without draft failed: {err_b}")
        b_bench = self.host.benchmark_run("standard", 96)
        base_tps = float(b_bench.get("tok_s") or 0.0)
        best = {
            "label": "no-draft (baseline)",
            "draft": "",
            "draft_max": 0,
            "config": base_cfg,
            "tok_s": base_tps,
            "ttft_ms": b_bench.get("ttft_ms", 0),
            "speedup_pct": 0.0,
            "ok": True,
        }
        trials.append(best)

        # 2. Evaluate each draft candidate with draft_max in (8, 16)
        for dp in uniq_drafts[:3]:
            for dmax in (8, 16):
                cand_cfg = copy.deepcopy(base_cfg)
                cand_cfg["draft"] = dp
                cand_cfg["draft_max"] = dmax
                ok_c, err_c = self._launch_for_trial("llama", cand_cfg)
                if not ok_c:
                    trials.append({
                        "label": f"{Path(dp).name} (max={dmax})",
                        "draft": dp,
                        "draft_max": dmax,
                        "ok": False,
                        "error": err_c,
                    })
                    continue
                c_bench = self.host.benchmark_run("standard", 96)
                c_tps = float(c_bench.get("tok_s") or 0.0)
                speedup = round(((c_tps - base_tps) / max(0.1, base_tps)) * 100.0, 1)
                rec = {
                    "label": f"{Path(dp).name} (max={dmax})",
                    "draft": dp,
                    "draft_max": dmax,
                    "config": cand_cfg,
                    "tok_s": c_tps,
                    "ttft_ms": c_bench.get("ttft_ms", 0),
                    "speedup_pct": speedup,
                    "ok": True,
                }
                trials.append(rec)
                if c_tps > best["tok_s"] * 1.02:  # require >=2% real speedup over baseline/previous
                    best = rec

        speculation_enabled = bool(best.get("draft"))
        verdict = (
            f"Enabled speculative decoding with {Path(best['draft']).name} (--draft-max {best['draft_max']}): "
            f"+{best['speedup_pct']}% net speedup ({base_tps} -> {best['tok_s']} tok/s)."
            if speculation_enabled
            else f"Kept speculative decoding OFF (baseline {base_tps} tok/s beat all draft candidates)."
        )

        if apply_best:
            self._launch_for_trial("llama", best["config"])
            self.store.record_trial(
                engine="llama",
                model_key=meta.get("name") or Path(meta["path"]).name,
                goal="fastest",
                config=best["config"],
                metrics={"tok_s": best["tok_s"], "ttft_ms": best["ttft_ms"]},
                score=score_trial("fastest", {"tok_s": best["tok_s"], "ttft_ms": best["ttft_ms"]}, best["config"]),
                ok=True,
                notes=verdict,
            )

        return {
            "ok": True,
            "checkpoint_id": cp["id"],
            "model": meta.get("name"),
            "drafts_tested": len(uniq_drafts[:3]),
            "speculation_enabled": speculation_enabled,
            "verdict": verdict,
            "winner": best,
            "trials": trials,
        }

    # ------------------------------------------------------------ Phase 2: Hardware-Aware HF Model/Quant Recommender
    def recommend_hf(
        self,
        *,
        query: str,
        goal: str = "balanced",
        max_repos: int = 4,
        assume_empty_gpu: bool = True,
    ) -> dict[str, Any]:
        """Search Hugging Face GGUF repos and rank best-fitting quants using MoE-aware VRAM+RAM fit math.

        When assume_empty_gpu=True (default), calculates usable VRAM from total GPU VRAM minus
        vram_reserve_mb, ignoring currently occupied VRAM (for planning a dedicated run).
        When assume_empty_gpu=False, subtracts currently used GPU VRAM so the quant fits alongside
        existing GPU workloads.
        """
        if not self.host.hf_search_fn or not self.host.hf_files_fn:
            raise RuntimeError("HF search hooks are not wired")
        vit = self.host.vitals(refresh=False)
        gpu = vit.get("gpu") or {}
        mem = vit.get("mem") or {}
        total_mb = float(gpu.get("total_mb") or 8192)
        used_mb = float(gpu.get("used_mb") or 0)
        reserve_mb = float(self.host.cfg.get("vram_reserve_mb") or 512)
        if assume_empty_gpu:
            usable_vram = max(512.0, total_mb - reserve_mb)
        else:
            usable_vram = max(512.0, total_mb - used_mb - reserve_mb)
        avail_ram_mb = max(
            1024.0,
            (float(mem.get("total_gb") or 32) - float(mem.get("used_gb") or 8)) * 1024.0,
        )

        q = query.strip()
        direct_repo = "/" in q and " " not in q
        if "gguf" not in q.lower():
            q_search = f"{q} GGUF"
        else:
            q_search = q

        repos: list[dict[str, Any]] = []
        if direct_repo:
            repos.append({"id": q, "downloads": 0})
        try:
            searched = self.host.hf_search_fn(q_search) or []
            seen = {r.get("id") for r in repos}
            for r in searched:
                if r.get("id") not in seen:
                    repos.append(r)
                    seen.add(r.get("id"))
        except Exception:
            if not direct_repo:
                raise
        moe_name_re = re.compile(r"(moe|mixtral|a\d+b|deepseek|qwen3.*next|flash.?next|\d+x\d+b)", re.I)
        quant_quality = {
            "Q8_0": 95, "Q6_K": 90, "Q5_K_M": 85, "Q5_K_S": 82,
            "Q4_K_M": 78, "IQ4_NL": 77, "IQ4_XS": 75, "Q4_K_S": 74, "Q4_0": 70,
            "Q3_K_M": 62, "IQ3_M": 60, "IQ3_XS": 56, "Q3_K_S": 54,
            "Q2_K": 42, "IQ2_XS": 38, "IQ2_XXS": 34,
        }

        recommendations: list[dict[str, Any]] = []
        for repo in repos[:max_repos]:
            rid = repo.get("id") or ""
            if not rid:
                continue
            try:
                files = self.host.hf_files_fn(rid) or []
            except Exception:
                continue
            if not files:
                continue

            size_by_path = {f["path"]: int(f.get("size") or 0) for f in files if f.get("path")}
            mmprojs = [f["path"] for f in files if "mmproj" in f["path"].lower()]
            drafts = [f["path"] for f in files if any(k in f["path"].lower() for k in ("mtp", "draft"))]
            weights = [
                f for f in files
                if "mmproj" not in f["path"].lower()
                and not any(k in f["path"].lower() for k in ("mtp", "draft"))
            ]

            # Group multi-shard GGUFs (e.g. -00001-of-00003.gguf)
            grouped: dict[str, dict[str, Any]] = {}
            for wf in weights:
                base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", ".gguf", wf["path"], flags=re.I)
                g = grouped.setdefault(base, {"base": base, "files": [], "size": 0})
                g["files"].append(wf["path"])
                g["size"] += int(wf.get("size") or 0)

            repo_is_moe = bool(moe_name_re.search(rid))
            for g in grouped.values():
                size_mb = g["size"] / 1048576.0
                if size_mb < 50:
                    continue
                is_moe = repo_is_moe or bool(moe_name_re.search(g["base"]))
                kv_budget_mb = 2048.0 if goal == "max_context" else 1024.0
                moe_info = analyze_moe_mechanics(
                    {"size": g["size"], "blocks": 32, "moe": is_moe, "experts": 64 if is_moe else 0, "experts_used": 8 if is_moe else 0},
                    usable_vram_mb=usable_vram,
                    avail_ram_mb=avail_ram_mb,
                    kv_vram_mb=kv_budget_mb,
                )
                if not moe_info["feasible"]:
                    continue

                # Extract quant label
                q_match = re.search(r"(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_0|Q\d_1|F16|BF16)", g["base"], re.I)
                q_label = q_match.group(1).upper() if q_match else "GGUF"
                qual = quant_quality.get(q_label, 65)

                full_gpu = moe_info["strategy"] in ("dense_full_gpu", "moe_full_gpu")
                fit_label = "Full GPU VRAM" if full_gpu else ("MoE Hybrid (GPU+RAM)" if is_moe else "Partial Offload")

                score = float(qual)
                if goal == "fastest":
                    score += 40.0 if full_gpu else -15.0
                elif goal == "max_context":
                    headroom = usable_vram - moe_info["gpu_hot_path_mb"]
                    score += min(35.0, max(0.0, headroom / 256.0))
                elif goal == "low_vram":
                    score += max(0.0, 30.0 - (moe_info["gpu_hot_path_mb"] / 512.0))
                else:  # balanced
                    score += 20.0 if full_gpu else (12.0 if is_moe else -10.0)

                pull_files = list(g["files"])
                if mmprojs:
                    pull_files.append(mmprojs[0])
                if drafts and goal in ("fastest", "balanced"):
                    pull_files.append(drafts[0])
                pull_sizes = {p: size_by_path.get(p, 0) for p in pull_files}

                recommendations.append({
                    "repo": rid,
                    "downloads": repo.get("downloads", 0),
                    "quant": q_label,
                    "base_file": g["base"],
                    "size_gb": round(g["size"] / (1024**3), 2),
                    "is_moe": is_moe,
                    "fit_label": fit_label,
                    "gpu_est_mb": moe_info["gpu_hot_path_mb"],
                    "ram_est_mb": moe_info["cpu_cold_experts_mb"],
                    "mmproj": mmprojs[0] if mmprojs else "",
                    "draft": drafts[0] if drafts else "",
                    "pull_files": pull_files,
                    "pull_sizes": pull_sizes,
                    "score": round(score, 1),
                    "notes": moe_info["notes"][:2],
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query": query,
            "goal": goal,
            "hardware_budget": {
                "gpu_name": gpu.get("name") or "GPU",
                "gpu_total_mb": round(total_mb),
                "gpu_used_mb": round(used_mb),
                "vram_reserve_mb": round(reserve_mb),
                "usable_vram_mb": round(usable_vram),
                "avail_ram_mb": round(avail_ram_mb),
                "assume_empty_gpu": bool(assume_empty_gpu),
            },
            "recommendations": recommendations[:10],
        }
