"""Offline tests for the web UI server and the model-server request format.

    python -m unittest discover -s tests
"""
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ui"))

import server  # noqa: E402
from benchmark_core import openai_schema_format  # noqa: E402


class FakeApp:
    """Stands in for the model server so the tests need no network."""

    def __init__(self):
        self.calls = []

    def health(self):
        return {"ok": True, "url": "fake", "model": server.MODEL, "model_available": True}

    def predict(self, method, ticket):
        self.calls.append((method, ticket))
        return {"method": method, "code": "J", "intent": "declined_cash_withdrawal", "raw": "J",
                "lenient": "J", "input_tokens": 100, "seconds": 0.01}


class TestSchemaFormat(unittest.TestCase):
    def test_json_object_with_schema_becomes_json_schema(self):
        schema = {"type": "object", "properties": {"rule": {"type": "string"}}, "required": ["rule"]}
        out = openai_schema_format({"type": "json_object", "schema": schema})
        self.assertEqual(out["type"], "json_schema")
        self.assertEqual(out["json_schema"]["schema"], schema)

    def test_other_formats_pass_through(self):
        plain = {"type": "json_object"}
        self.assertIs(openai_schema_format(plain), plain)


class TestRunPayload(unittest.TestCase):
    def test_published_run_matches_saved_metrics(self):
        payload = server.run_payload("qwen_3b")
        self.assertEqual(set(payload["methods"]), set(server.METHODS))
        self.assertEqual(len(payload["tickets"]), 120)
        for method, data in payload["methods"].items():
            saved = json.loads((ROOT / "results/qwen_3b" / method / "test_metrics.json").read_text())
            self.assertAlmostEqual(data["metrics"]["accuracy"], saved["accuracy"])
            self.assertEqual(len(data["predictions"]), 120)
            # Lenient scoring accepts a superset of strict answers, so it can only be higher.
            self.assertGreaterEqual(data["lenient_accuracy"], saved["accuracy"])

    def test_hand_written_prompts_cost_nothing_to_fit(self):
        payload = server.run_payload("qwen_3b")
        self.assertEqual(payload["methods"]["manual_zero"]["optimization"]["tokens"], 0)
        self.assertGreater(payload["methods"]["dspy_copro"]["optimization"]["tokens"], 0)

    def test_unknown_run_raises(self):
        with self.assertRaises(KeyError):
            server.run_payload("nope")


class TestHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FakeApp()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(cls.app))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_index_served(self):
        status, ctype, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"Banking77 Prompt Lab", body)

    def test_meta_lists_published_run_and_samples(self):
        _, _, body = self.get("/api/meta")
        meta = json.loads(body)
        self.assertIn("qwen_3b", [r["id"] for r in meta["runs"]])
        self.assertEqual(len(meta["intents"]), 12)
        self.assertEqual(len(meta["samples"]), 120)

    def test_run_endpoint(self):
        _, _, body = self.get("/api/run?id=qwen_3b")
        self.assertEqual(json.loads(body)["id"], "qwen_3b")

    def test_unknown_paths_404(self):
        for path in ("/api/run?id=nope", "/missing"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.get(path)
            self.assertEqual(ctx.exception.code, 404)

    def test_predict_calls_model(self):
        status, out = self.post("/api/predict", {"method": "manual_few", "ticket": "ATM declined my card"})
        self.assertEqual(status, 200)
        self.assertEqual(out["code"], "J")
        self.assertIn(("manual_few", "ATM declined my card"), self.app.calls)

    def test_predict_rejects_bad_input(self):
        for body in ({"method": "nope", "ticket": "x"}, {"method": "manual_few", "ticket": "  "},
                     {"method": "manual_few", "ticket": "x" * 1001}):
            status, out = self.post("/api/predict", body)
            self.assertEqual(status, 400)
            self.assertIn("error", out)

    def test_refit_progress_shape(self):
        _, _, body = self.get("/api/refit-progress")
        progress = json.loads(body)
        self.assertEqual(set(progress), set(server.METHODS))
        self.assertTrue(all({"fitted", "tested"} <= set(v) for v in progress.values()))


if __name__ == "__main__":
    unittest.main()
