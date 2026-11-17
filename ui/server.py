"""Local web UI for the Banking77 prompt-optimization benchmark.

    python ui/server.py --url http://127.0.0.1:11434/v1      # Ollama
    python ui/server.py --url http://127.0.0.1:18976/v1      # llama.cpp server from the README

Then open http://127.0.0.1:8077. Results are read from the run folders on disk;
the live classifier sends one request per method to the model server.
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from benchmark_core import DEMOS, INTENTS, MANUAL, Backend, metrics, parse_lenient, prompt, read_splits  # noqa: E402

METHODS = ("manual_zero", "manual_few", "dspy_bootstrap", "dspy_copro", "textgrad", "adalflow")
MODEL = "qwen2.5-3b-instruct-q4_k_m"
HERE = Path(__file__).resolve().parent

# Run folders the dashboard can show, in display order. Missing or partial folders are skipped.
RUNS = [
    ("rerun_refit", "3B · re-fitted here (Ollama)", "rerun_refit"),
    ("rerun_saved", "3B · saved prompts re-tested here (Ollama)", "rerun_saved"),
    ("qwen_3b", "3B · published (llama.cpp)", "results/qwen_3b"),
    ("qwen_0_5b", "0.5B · published (llama.cpp)", "results/qwen_0_5b"),
]


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def complete_run(folder):
    return all((folder / m / "test_metrics.json").exists() for m in METHODS)


def ledger_summary(rows):
    return {
        "calls": len(rows),
        "tokens": sum(r["total_tokens"] for r in rows),
        "seconds": round(sum(r["seconds"] for r in rows), 1),
    }


def run_payload(run_id):
    entry = next((r for r in RUNS if r[0] == run_id), None)
    if entry is None:
        raise KeyError(run_id)
    folder = ROOT / entry[2]
    fallback = ROOT / "results" / "qwen_3b"  # a re-test reuses the published fit, so reuse its cost
    test_rows = {row["id"]: row for row in read_splits()["test"]}
    methods = {}
    for method in METHODS:
        d = folder / method
        spec = read_json(d / "classifier.json", {})
        scores = read_json(d / "test_metrics.json", {})
        preds = read_jsonl(d / "predictions.jsonl")
        opt_rows = read_jsonl(d / "optimization_calls.jsonl")
        cost_source = "this run"
        if not (d / "optimization_calls.jsonl").exists() and method not in ("manual_zero", "manual_few"):
            opt_rows = read_jsonl(fallback / method / "optimization_calls.jsonl")
            cost_source = "published fit"
        lenient = metrics(preds, [parse_lenient(p["raw"]) for p in preds]) if preds else None
        methods[method] = {
            "spec": spec,
            "metrics": scores,
            "lenient_accuracy": lenient and lenient["accuracy"],
            "lenient_macro_f1": lenient and lenient["macro_f1"],
            "optimization": {**ledger_summary(opt_rows), "source": cost_source},
            "test_cost": ledger_summary(read_jsonl(d / "test_calls.jsonl")),
            "trace": read_json(d / "optimization_trace.json", []),
            "predictions": [
                {"id": p["id"], "gold": p["code"], "pred": p["prediction"], "raw": p["raw"],
                 "lenient": parse_lenient(p["raw"])}
                for p in preds
            ],
        }
    return {"id": run_id, "label": entry[1], "folder": entry[2], "methods": methods,
            "tickets": {k: v["ticket"] for k, v in test_rows.items()}}


class App:
    def __init__(self, url, model):
        self.url = url.rstrip("/")
        self.model = model

    def health(self):
        try:
            r = requests.get(f"{self.url}/models", timeout=3)
            r.raise_for_status()
            names = [m.get("id") for m in r.json().get("data", [])]
            return {"ok": True, "url": self.url, "model": self.model,
                    "model_available": any(n == self.model or n.split(":")[0] == self.model
                                           for n in names if n) if names else None}
        except requests.RequestException as exc:
            return {"ok": False, "url": self.url, "model": self.model, "error": type(exc).__name__}

    def predict(self, method, ticket):
        spec = read_json(ROOT / "results" / "qwen_3b" / method / "classifier.json")
        backend = Backend(self.url, self.model)
        start = time.perf_counter()
        code, raw, tokens = backend.classify(spec["instruction"], ticket, spec["demos"], spec["prefix"])
        return {"method": method, "code": code, "intent": INTENTS.get(code), "raw": raw,
                "lenient": parse_lenient(raw), "input_tokens": tokens,
                "seconds": round(time.perf_counter() - start, 2)}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, status, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path in ("/", "/index.html"):
                return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/meta":
                runs = [{"id": r[0], "label": r[1]} for r in RUNS if complete_run(ROOT / r[2])]
                return self.send(200, {
                    "intents": INTENTS, "methods": METHODS, "runs": runs, "manual": MANUAL,
                    "demos": DEMOS, "health": app.health(),
                    "prompt_example": prompt(MANUAL, "<ticket>", DEMOS),
                    "samples": read_splits()["test"],
                })
            if path == "/api/health":
                return self.send(200, app.health())
            if path == "/api/run":
                try:
                    return self.send(200, run_payload(params.get("id", "qwen_3b")))
                except KeyError:
                    return self.send(404, {"error": "unknown run"})
            if path == "/api/refit-progress":
                folder = ROOT / "rerun_refit"
                return self.send(200, {m: {
                    "fitted": (folder / m / "classifier.json").exists(),
                    "tested": (folder / m / "test_metrics.json").exists(),
                } for m in METHODS})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/predict":
                return self.send(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            method, ticket = body.get("method"), (body.get("ticket") or "").strip()
            if method not in METHODS or not ticket or len(ticket) > 1000:
                return self.send(400, {"error": "need a known method and a ticket under 1000 characters"})
            try:
                return self.send(200, app.predict(method, ticket))
            except requests.RequestException as exc:
                return self.send(502, {"error": f"model server unreachable ({type(exc).__name__})"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible model server")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--port", type=int, default=8077)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(App(args.url, args.model)))
    print(f"Banking77 UI on http://127.0.0.1:{args.port}  (model server {args.url})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
