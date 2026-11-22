#!/usr/bin/env python3
"""Graduated Thermal & Shared-GPU Guard for Inference Console.

Escalation ladder (configurable thresholds & opt-in auto-actions):
  Level 0 (normal):     Healthy temps & VRAM. Auto-restores fans/power cap once cooled below cool_temp_c.
  Level 1 (cool_hw):    >= warn_temp_c -> alert + ramp GPU fans to 100% and/or lower power cap (-15%) if writable.
  Level 2 (downshift):  >= throttle_temp_c (or external VRAM pressure) -> checkpoint + downshift to cooler/low_vram profile.
  Level 3 (killswitch): >= kill_temp_c -> checkpoint + emergency stop active inference engine(s).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_GUARD_CONFIG: dict[str, Any] = {
    "enabled": False,
    "auto_act": False,            # False = alert + recommend; True = execute graduated ladder automatically
    "warn_temp_c": 80,            # Level 1: max fans + power throttle
    "throttle_temp_c": 88,        # Level 2: downshift profile / lower VRAM
    "kill_temp_c": 95,            # Level 3: emergency killswitch
    "cool_temp_c": 70,            # Hysteresis: restore auto fan & power cap once temp <= cool_temp_c
    "power_throttle_pct": 15,     # Reduce GPU power cap by this % at Level 1
    "vram_pressure_pct": 92,      # Shared-GPU external VRAM pressure alert threshold
    "poll_sec": 5,
}


class GPUHardwareController:
    """Safe, permission-checked wrapper for AMD hwmon sysfs & NVIDIA nvidia-smi fan/power controls."""

    def __init__(self, hwmon_root: Path = Path("/sys/class/hwmon")):
        self.hwmon_root = hwmon_root
        self._saved_hw: dict[str, Any] = {}

    def _find_amd_hwmon(self) -> Path | None:
        if not self.hwmon_root.exists():
            return None
        for hp in sorted(self.hwmon_root.glob("hwmon*")):
            try:
                if (hp / "name").read_text().strip() == "amdgpu" and (hp / "fan1_input").exists():
                    return hp
            except Exception:
                continue
        return None

    def probe_capabilities(self) -> dict[str, Any]:
        amd = self._find_amd_hwmon()
        if amd:
            pwm_en = amd / "pwm1_enable"
            pwm = amd / "pwm1"
            pcap = amd / "power1_cap"
            fan_w = bool(pwm_en.exists() and pwm.exists() and os.access(pwm_en, os.W_OK) and os.access(pwm, os.W_OK))
            pwr_w = bool(pcap.exists() and os.access(pcap, os.W_OK))
            hint = ""
            if not (fan_w and pwr_w):
                hint = (
                    f"Sysfs controls under {amd} are read-only for this user. "
                    "To enable fan/power control without root, add a udev rule granting group write access "
                    "to pwm1_enable, pwm1, and power1_cap (e.g. SUBSYSTEM==\"hwmon\", ATTR{name}==\"amdgpu\", ...)."
                )
            return {
                "vendor": "amdgpu",
                "hwmon": str(amd),
                "fan_writable": fan_w,
                "power_writable": pwr_w,
                "setup_hint": hint,
            }

        if shutil.which("nvidia-smi"):
            # Writing nvidia-smi -pl typically requires root or CAP_SYS_ADMIN / sudoers
            pwr_w = (os.geteuid() == 0) if hasattr(os, "geteuid") else False
            hint = "" if pwr_w else (
                "NVIDIA power-limit throttling (`nvidia-smi -pl`) requires root or a passwordless sudoers rule "
                "for `nvidia-smi -pl`."
            )
            return {
                "vendor": "nvidia",
                "hwmon": "",
                "fan_writable": False,
                "power_writable": pwr_w,
                "setup_hint": hint,
            }

        return {
            "vendor": "none",
            "hwmon": "",
            "fan_writable": False,
            "power_writable": False,
            "setup_hint": "No writable discrete GPU fan/power interface detected on this host.",
        }

    def set_max_fans(self) -> dict[str, Any]:
        caps = self.probe_capabilities()
        if caps["vendor"] == "amdgpu" and caps["fan_writable"]:
            hp = Path(caps["hwmon"])
            try:
                if "pwm1_enable" not in self._saved_hw:
                    self._saved_hw["pwm1_enable"] = (hp / "pwm1_enable").read_text().strip()
                if "pwm1" not in self._saved_hw and (hp / "pwm1").exists():
                    self._saved_hw["pwm1"] = (hp / "pwm1").read_text().strip()
                (hp / "pwm1_enable").write_text("1")
                (hp / "pwm1").write_text("255")
                return {"ok": True, "vendor": "amdgpu", "action": "max_fans", "detail": "Set amdgpu pwm1=255 (100% fan)"}
            except Exception as e:
                return {"ok": False, "action": "max_fans", "error": str(e)}
        return {"ok": False, "action": "max_fans", "error": caps["setup_hint"] or "Fan control unavailable"}

    def throttle_power(self, pct: int = 15) -> dict[str, Any]:
        pct = max(5, min(40, int(pct or 15)))
        caps = self.probe_capabilities()
        if caps["vendor"] == "amdgpu" and caps["power_writable"]:
            hp = Path(caps["hwmon"])
            try:
                cur_uw = int((hp / "power1_cap").read_text().strip())
                if "power1_cap" not in self._saved_hw:
                    self._saved_hw["power1_cap"] = str(cur_uw)
                base_uw = int(self._saved_hw["power1_cap"])
                min_uw = int((hp / "power1_cap_min").read_text().strip()) if (hp / "power1_cap_min").exists() else int(base_uw * 0.5)
                target_uw = max(min_uw, int(base_uw * (100 - pct) / 100))
                (hp / "power1_cap").write_text(str(target_uw))
                return {
                    "ok": True,
                    "vendor": "amdgpu",
                    "action": "throttle_power",
                    "detail": f"Lowered amdgpu power1_cap from {base_uw // 1_000_000}W to {target_uw // 1_000_000}W (-{pct}%)",
                }
            except Exception as e:
                return {"ok": False, "action": "throttle_power", "error": str(e)}

        if caps["vendor"] == "nvidia" and caps["power_writable"]:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.limit,power.min_limit", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=3,
                ).splitlines()[0]
                cur_w, min_w = [float(x.strip()) for x in out.split(",")]
                if "nvidia_pl" not in self._saved_hw:
                    self._saved_hw["nvidia_pl"] = cur_w
                base_w = float(self._saved_hw["nvidia_pl"])
                target_w = max(min_w, round(base_w * (100 - pct) / 100.0, 1))
                subprocess.run(["nvidia-smi", "-pl", str(target_w)], check=True, capture_output=True, timeout=5)
                return {
                    "ok": True,
                    "vendor": "nvidia",
                    "action": "throttle_power",
                    "detail": f"Lowered NVIDIA power limit from {base_w}W to {target_w}W (-{pct}%)",
                }
            except Exception as e:
                return {"ok": False, "action": "throttle_power", "error": str(e)}

        return {"ok": False, "action": "throttle_power", "error": caps["setup_hint"] or "Power cap control unavailable"}

    def restore_defaults(self) -> dict[str, Any]:
        if not self._saved_hw:
            return {"ok": True, "action": "restore_hw", "detail": "No active hardware overrides to restore"}
        caps = self.probe_capabilities()
        restored = []
        errors = []
        if caps["vendor"] == "amdgpu":
            hp = Path(caps["hwmon"])
            if "pwm1_enable" in self._saved_hw:
                try:
                    (hp / "pwm1_enable").write_text(self._saved_hw["pwm1_enable"])
                    restored.append(f"pwm1_enable={self._saved_hw['pwm1_enable']}")
                except Exception as e:
                    errors.append(f"fan: {e}")
            if "power1_cap" in self._saved_hw:
                try:
                    (hp / "power1_cap").write_text(self._saved_hw["power1_cap"])
                    restored.append(f"power1_cap={int(self._saved_hw['power1_cap']) // 1_000_000}W")
                except Exception as e:
                    errors.append(f"power: {e}")
        elif caps["vendor"] == "nvidia" and "nvidia_pl" in self._saved_hw:
            try:
                subprocess.run(["nvidia-smi", "-pl", str(self._saved_hw["nvidia_pl"])], check=True, capture_output=True, timeout=5)
                restored.append(f"nvidia_pl={self._saved_hw['nvidia_pl']}W")
            except Exception as e:
                errors.append(f"nvidia_pl: {e}")

        if not errors:
            self._saved_hw.clear()
        return {"ok": not errors, "action": "restore_hw", "restored": restored, "errors": errors}


class ThermalGuard:
    """Graduated thermal & shared-GPU watchdog with auto-restore and manual overrides."""

    def __init__(
        self,
        *,
        get_config: Callable[[], dict[str, Any]],
        save_config: Callable[[dict[str, Any]], None],
        vitals_fn: Callable[..., dict[str, Any]],
        downshift_fn: Callable[[], dict[str, Any]],
        killswitch_fn: Callable[[], dict[str, Any]],
        hw: GPUHardwareController | None = None,
    ):
        self._get_cfg = get_config
        self._save_cfg = save_config
        self._vitals = vitals_fn
        self._downshift = downshift_fn
        self._killswitch = killswitch_fn
        self.hw = hw or GPUHardwareController()
        self._lock = threading.RLock()
        self._level = 0
        self._state = "normal"
        self._hw_active = False
        self._events: list[dict[str, Any]] = []
        self._last_eval: dict[str, Any] = {}
        self._stop_ev = threading.Event()
        self._thread: threading.Thread | None = None

    def config(self) -> dict[str, Any]:
        c = dict(DEFAULT_GUARD_CONFIG)
        raw = self._get_cfg() or {}
        if isinstance(raw, dict):
            c.update(raw)
        return c

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            c = self.config()
            for k in DEFAULT_GUARD_CONFIG:
                if k in patch and patch[k] is not None:
                    c[k] = patch[k]
            # Keep ladder monotonic
            c["cool_temp_c"] = max(40, min(85, int(c["cool_temp_c"])))
            c["warn_temp_c"] = max(c["cool_temp_c"] + 3, min(95, int(c["warn_temp_c"])))
            c["throttle_temp_c"] = max(c["warn_temp_c"] + 2, min(100, int(c["throttle_temp_c"])))
            c["kill_temp_c"] = max(c["throttle_temp_c"] + 2, min(110, int(c["kill_temp_c"])))
            self._save_cfg(c)
            if c["enabled"]:
                self.start_loop()
            else:
                self.stop_loop()
            return c

    def _log_event(self, level: int, action: str, detail: str, ok: bool = True) -> None:
        self._events.insert(0, {
            "ts": time.time(),
            "level": level,
            "action": action,
            "detail": detail,
            "ok": ok,
        })
        self._events = self._events[:50]

    def evaluate_once(self, vit: dict[str, Any] | None = None) -> dict[str, Any]:
        """Inspect current GPU vitals and advance/de-escalate the graduated ladder."""
        with self._lock:
            cfg = self.config()
            v = vit if vit is not None else self._vitals(refresh=True)
            g = (v or {}).get("gpu") or {}
            temp = g.get("temp")
            temp_mem = g.get("temp_mem")
            eff_temp = max([t for t in (temp, temp_mem) if isinstance(t, (int, float))], default=0)
            used_mb = float(g.get("used_mb") or 0)
            total_mb = float(g.get("total_mb") or 0)
            vram_pct = round(100.0 * used_mb / total_mb, 1) if total_mb > 0 else 0.0

            actions_taken: list[dict[str, Any]] = []
            recommended: list[str] = []

            if eff_temp >= cfg["kill_temp_c"]:
                target_level, state = 3, "killswitch"
                recommended = ["Emergency killswitch: stop active inference engines immediately"]
                if cfg["auto_act"] and self._level < 3:
                    res = self._killswitch()
                    actions_taken.append({"action": "killswitch", "result": res})
                    self._log_event(3, "killswitch", f"GPU temp {eff_temp}°C >= {cfg['kill_temp_c']}°C — stopped engines", bool(res.get("ok", True)))
                self._level, self._state = target_level, state

            elif eff_temp >= cfg["throttle_temp_c"] or vram_pct >= cfg["vram_pressure_pct"]:
                target_level, state = 2, "downshift"
                reason = f"GPU temp {eff_temp}°C" if eff_temp >= cfg["throttle_temp_c"] else f"VRAM pressure {vram_pct}%"
                recommended = [f"Max fans + power throttle, then downshift to low-VRAM/cooler profile ({reason})"]
                if cfg["auto_act"]:
                    if not self._hw_active:
                        r1 = self.hw.set_max_fans()
                        r2 = self.hw.throttle_power(cfg["power_throttle_pct"])
                        self._hw_active = bool(r1.get("ok") or r2.get("ok"))
                        actions_taken.extend([r1, r2])
                    if self._level < 2:
                        r3 = self._downshift()
                        actions_taken.append({"action": "downshift", "result": r3})
                        self._log_event(2, "downshift", f"{reason} — downshifted profile", bool(r3.get("ok", True)))
                self._level, self._state = target_level, state

            elif eff_temp >= cfg["warn_temp_c"]:
                target_level, state = 1, "cool_hw"
                recommended = [f"Ramp GPU fans to 100% and lower power cap by {cfg['power_throttle_pct']}% ({eff_temp}°C)"]
                if cfg["auto_act"] and self._level < 1:
                    r1 = self.hw.set_max_fans()
                    r2 = self.hw.throttle_power(cfg["power_throttle_pct"])
                    self._hw_active = bool(r1.get("ok") or r2.get("ok"))
                    actions_taken.extend([r1, r2])
                    self._log_event(
                        1,
                        "cool_hw",
                        f"GPU temp {eff_temp}°C >= {cfg['warn_temp_c']}°C — fan={r1.get('ok')} power={r2.get('ok')}",
                        self._hw_active,
                    )
                self._level, self._state = target_level, state

            else:
                # Hysteresis: only restore hardware overrides once temp drops <= cool_temp_c
                if self._level > 0 and eff_temp <= cfg["cool_temp_c"]:
                    if self._hw_active:
                        r = self.hw.restore_defaults()
                        actions_taken.append(r)
                        self._hw_active = False
                        self._log_event(0, "restore_hw", f"GPU cooled to {eff_temp}°C <= {cfg['cool_temp_c']}°C — restored auto fan/power", bool(r.get("ok")))
                    self._level = 0
                    self._state = "normal"
                elif self._level == 0:
                    self._state = "normal"

            self._last_eval = {
                "ts": time.time(),
                "level": self._level,
                "state": self._state,
                "temp_c": temp,
                "temp_mem_c": temp_mem,
                "eff_temp_c": eff_temp,
                "vram_pct": vram_pct,
                "hw_override_active": self._hw_active,
                "recommended": recommended,
                "actions_taken": actions_taken,
            }
            return self._last_eval

    def run_manual_action(self, action: str) -> dict[str, Any]:
        with self._lock:
            cfg = self.config()
            if action == "max_fans":
                r = self.hw.set_max_fans()
                if r.get("ok"):
                    self._hw_active = True
                self._log_event(1, "manual:max_fans", r.get("detail") or r.get("error", ""), bool(r.get("ok")))
                return r
            if action == "throttle_power":
                r = self.hw.throttle_power(cfg["power_throttle_pct"])
                if r.get("ok"):
                    self._hw_active = True
                self._log_event(1, "manual:throttle_power", r.get("detail") or r.get("error", ""), bool(r.get("ok")))
                return r
            if action == "restore_hw":
                r = self.hw.restore_defaults()
                if r.get("ok"):
                    self._hw_active = False
                    self._level = 0
                    self._state = "normal"
                self._log_event(0, "manual:restore_hw", str(r.get("restored") or r.get("detail") or ""), bool(r.get("ok")))
                return r
            if action == "downshift":
                r = self._downshift()
                self._log_event(2, "manual:downshift", str(r.get("detail") or r.get("error") or "downshifted"), bool(r.get("ok", True)))
                return r
            if action == "killswitch":
                r = self._killswitch()
                self._level = 3
                self._state = "killswitch"
                self._log_event(3, "manual:killswitch", "Emergency stop executed", bool(r.get("ok", True)))
                return r
            return {"ok": False, "error": f"Unknown thermal guard action: {action}"}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config": self.config(),
                "level": self._level,
                "state": self._state,
                "hw_override_active": self._hw_active,
                "hw_caps": self.hw.probe_capabilities(),
                "last_eval": self._last_eval,
                "events": self._events[:20],
            }

    def start_loop(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_ev.clear()

        def _loop():
            while not self._stop_ev.is_set():
                cfg = self.config()
                if cfg.get("enabled"):
                    try:
                        self.evaluate_once()
                    except Exception:
                        pass
                self._stop_ev.wait(max(2, int(cfg.get("poll_sec") or 5)))

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_loop(self) -> None:
        self._stop_ev.set()
