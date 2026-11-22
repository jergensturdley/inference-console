#!/usr/bin/env python3
"""Panel restart tests for panel.py (argv rebuild, session mirror, failed execv).

Runnable either way:

    python3 tests/test_panel_restart.py
    python3 -m pytest tests/test_panel_restart.py
"""
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOME_TMP = tempfile.mkdtemp(prefix="inference-console-restart-")
os.environ["HOME"] = HOME_TMP          # must be set before panel is imported
sys.path.insert(0, str(REPO))

import panel  # noqa: E402


class SelfArgvTests(unittest.TestCase):
    def test_interpreter_first_and_script_resolved(self):
        argv = panel._self_argv()
        self.assertEqual(argv[0], sys.executable)
        self.assertTrue(Path(argv[1]).is_absolute(), "script path must survive a cwd change")
        self.assertTrue(Path(argv[1]).exists())

    def test_missing_interpreter_is_reported_not_silently_accepted(self):
        orig = panel._self_argv
        panel._self_argv = lambda: ["/definitely/not/here/python", "panel.py"]
        try:
            with self.assertRaises(panel.HTTPException) as ctx:
                panel.api_panel_restart(panel.RestartBody(delay_ms=0))
            self.assertIn("interpreter not found", str(ctx.exception.detail))
        finally:
            panel._self_argv = orig


class SessionMirrorTests(unittest.TestCase):
    def test_round_trip_keeps_only_live_tokens(self):
        panel._sessions.clear()
        panel._sessions["live"] = time.time() + 60
        panel._sessions["dead"] = time.time() - 60
        panel._save_sessions()
        self.assertEqual(panel._load_sessions(), {"live": panel._sessions["live"]})

    def test_mirror_is_owner_only(self):
        panel._sessions.clear()
        panel._sessions["tok"] = time.time() + 60
        panel._save_sessions()
        mode = stat.S_IMODE(panel.SESSIONS_PATH.stat().st_mode)
        self.assertEqual(mode, 0o600, f"session file mode {oct(mode)}")

    def test_corrupt_mirror_does_not_raise(self):
        panel.SESSIONS_PATH.write_text("[not a dict")
        self.assertEqual(panel._load_sessions(), {})


class RestartTests(unittest.TestCase):
    def test_endpoint_schedules_execv_with_the_current_argv(self):
        calls = []
        orig = panel._restart_self
        panel._restart_self = lambda argv, delay: calls.append((argv, delay))
        panel._sessions["tok"] = time.time() + 60
        try:
            out = panel.api_panel_restart(panel.RestartBody(delay_ms=1))
            for _ in range(40):
                if calls:
                    break
                time.sleep(0.05)
            self.assertTrue(out["ok"])
            self.assertEqual(calls[0][0][0], sys.executable)
            self.assertAlmostEqual(calls[0][1], 0.001)
            self.assertIn("tok", panel._load_sessions(), "restart must keep the tab signed in")
        finally:
            panel._restart_self = orig

    def test_failed_execv_leaves_the_panel_serving(self):
        orig = panel.os.execv

        def boom(*a, **k):
            raise OSError("no such file")

        panel.os.execv = boom
        try:
            panel._restart_self([sys.executable, "panel.py"], 0)   # must not raise or exit
        finally:
            panel.os.execv = orig

    def test_delay_is_clamped(self):
        orig = panel._restart_self
        seen = []
        panel._restart_self = lambda argv, delay: seen.append(delay)
        try:
            self.assertEqual(panel.api_panel_restart(panel.RestartBody(delay_ms=60000))["delay_s"], 5.0)
            for _ in range(40):
                if seen:
                    break
                time.sleep(0.05)
        finally:
            panel._restart_self = orig


class PingTests(unittest.TestCase):
    def test_ping_shape(self):
        out = panel.api_panel_ping()
        for k in ("ok", "pid", "started_at", "uptime_s", "argv", "port", "restartable"):
            self.assertIn(k, out)
        self.assertEqual(out["pid"], os.getpid())


if __name__ == "__main__":
    unittest.main(verbosity=2)