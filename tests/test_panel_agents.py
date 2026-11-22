#!/usr/bin/env python3
"""Coding-agent registration tests for panel.py (JSONC configs, comment-preserving YAML,
server-reported alias/context). Runnable either way:

    python3 tests/test_panel_agents.py
    python3 -m pytest tests/test_panel_agents.py
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOME_TMP = tempfile.mkdtemp(prefix="inference-console-agents-")
os.environ["HOME"] = HOME_TMP          # must be set before panel is imported
sys.path.insert(0, str(REPO))

import panel  # noqa: E402

HOME = Path(HOME_TMP)
ALIAS = "qwen3.8-flash-next"
BASE = "http://127.0.0.1:8086"


class JsoncTests(unittest.TestCase):
    def test_comments_and_trailing_commas_parse(self):
        text = '// lead\n{\n  /* block */\n  "a": [1, 2,], "b": "// not a comment",\n}\n'
        self.assertEqual(panel._jsonc_load(text), {"a": [1, 2], "b": "// not a comment"})

    def test_zed_settings_with_comments_and_existing_models(self):
        path = HOME / ".config/zed/settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('// Zed settings\n{\n  "language_models": {"openai": '
                        '{"available_models": [{"name": "old-model"}]}},\n}\n')
        panel._w_zed(ALIAS, BASE, 65536)
        cfg = json.loads(path.read_text())
        names = [m["name"] for m in cfg["language_models"]["openai"]["available_models"]]
        self.assertEqual(names, [ALIAS, "old-model"])          # new first, nothing dropped
        self.assertEqual(cfg["language_models"]["openai"]["api_url"], BASE + "/v1")

    def test_second_sync_does_not_duplicate_model(self):
        path = HOME / ".config/zed/settings.json"
        panel._w_zed(ALIAS, BASE, 65536)
        panel._w_zed(ALIAS, BASE, 65536)
        names = [m["name"] for m in json.loads(path.read_text())["language_models"]["openai"]["available_models"]]
        self.assertEqual(names.count(ALIAS), 1)


class YamlUpsertTests(unittest.TestCase):
    def test_preserves_comments_and_other_keys(self):
        path = HOME / ".hermes/config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# top comment\nproviders:\n  other:\n    name: Other  # trailing\n"
                        "toolsets:\n  - hermes-cli\n")
        panel._w_hermes(ALIAS, BASE, 262144, "")
        text = path.read_text()
        self.assertIn("# top comment", text)
        self.assertIn("name: Other  # trailing", text)
        self.assertIn("- hermes-cli", text)                    # untouched style
        cfg = panel.yaml.safe_load(text)
        self.assertEqual(cfg["providers"][panel.PANEL_PROVIDER_ID]["default_model"], ALIAS)
        self.assertEqual(cfg["providers"]["other"]["name"], "Other")

    def test_second_sync_updates_in_place(self):
        path = HOME / ".hermes/config.yaml"
        panel._w_hermes(ALIAS, BASE, 262144, "")
        first = path.read_text()
        panel._w_hermes(ALIAS, BASE, 262144, "")
        self.assertEqual(path.read_text(), first)              # idempotent, no growth
        self.assertEqual(first.count("inference-console:"), 1)

    def test_missing_file_creates_parent(self):
        path = HOME / ".omp/agent/models.yml"
        panel._yaml_upsert(path, "providers", "x", {"baseUrl": "http://h/v1"})
        self.assertEqual(panel.yaml.safe_load(path.read_text())["providers"]["x"]["baseUrl"],
                         "http://h/v1")

    def test_unparseable_yaml_reports_instead_of_clobbering(self):
        path = HOME / "broken.yml"
        path.write_text("key: [unclosed\n")
        with self.assertRaises(panel.HTTPException):
            panel._yaml_upsert(path, "providers", "x", {"a": 1})
        self.assertEqual(path.read_text(), "key: [unclosed\n")


class _FakeLlama(BaseHTTPRequestHandler):
    """/v1/models + /props as a real llama-server reports them."""
    def do_GET(self):
        body = {"/v1/models": {"data": [{"id": "/m/models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"}]},
                "/props": {"default_generation_settings": {"n_ctx": 262144}}}.get(self.path)
        if body is None:
            self.send_error(404)
            return
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        pass


class ServerFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _FakeLlama)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_alias_is_derived_from_reported_gguf_path(self):
        alias, ctx = panel._server_facts(self.port)
        self.assertEqual(alias, "Qwen3.8-Flash-Next-UD-IQ4_XS")
        self.assertEqual(ctx, 262144)

    def test_dead_port_is_silent(self):
        self.assertEqual(panel._server_facts(9, timeout=0.3), (None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)