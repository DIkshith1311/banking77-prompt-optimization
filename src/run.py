"""Fit, evaluate, or use a saved Banking77 intent classifier."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_core import (
    Backend, DEMOS, INTENTS, LocalModel, MANUAL, ROOT, metrics, prepare_data, prompt, read_splits, write_json,
)

METHODS = ("manual_zero", "manual_few", "dspy_bootstrap", "dspy_copro", "textgrad", "adalflow")
MODEL = "qwen2.5-3b-instruct-q4_k_m"


def evaluate(backend, spec, rows, workers):
    def predict(row):
        label, raw, tokens = backend.classify(
            spec["instruction"], row["ticket"], spec["demos"], spec["prefix"],
        )
        return {**row, "prediction": label, "raw": raw, "prompt_tokens": tokens}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(pool.map(predict, rows))
    return metrics(rows, [item["prediction"] for item in predictions]), predictions


def fit(args, method, directory, splits):
    ledger = directory / "optimization_calls.jsonl"
    ledger.write_text("", encoding="utf-8")
    backend = Backend(args.url, args.model, ledger, method, args.local_model)
    backend.phase = "optimization"
    spec = {"instruction": MANUAL, "demos": [], "prefix": "Intent"}
    trace = []
    if method == "manual_few":
        spec["demos"] = DEMOS
    elif method not in ("manual_zero", "manual_few"):
        from optimizers import optimize

        def validate(candidate):
            scores, _ = evaluate(backend, candidate, splits["validation"], args.workers)
            return scores["accuracy"], scores["macro_f1"]

        spec, trace = optimize(method, backend, splits, validate, args.workers)
    write_json(directory / "classifier.json", {**spec, "method": method, "model": args.model})
    write_json(directory / "optimization_trace.json", trace)


def test(args, method, directory, splits):
    spec = json.loads((directory / "classifier.json").read_text(encoding="utf-8"))
    if spec["model"] != args.model:
        raise ValueError(f"{method} was fitted with a different model")
    ledger = directory / "test_calls.jsonl"
    ledger.write_text("", encoding="utf-8")
    backend = Backend(args.url, args.model, ledger, method, args.local_model)
    backend.phase = "test"
    scores, predictions = evaluate(backend, spec, splits["test"], args.workers)
    backend.phase = "length_probe"
    messages = prompt(spec["instruction"], "", spec["demos"], spec["prefix"])
    _, probe = backend.complete(messages, "empty_ticket_prompt_length", 1)
    write_json(directory / "rendered_prompt.json", messages)
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    scores["n_test"] = len(predictions)
    scores["prompt_tokens_static"] = probe["input_tokens"]
    scores["prompt_tokens_mean_test"] = sum(p["prompt_tokens"] for p in predictions) / len(predictions)
    write_json(directory / "test_metrics.json", scores)
    print(f"{method}  accuracy={scores['accuracy']:.4f}  macro F1={scores['macro_f1']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18971/v1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--gguf", type=Path, help="Run a local GGUF with llama-cpp-python")
    parser.add_argument("--threads", type=int, default=6, help="Threads for local inference")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "qwen_3b")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("fit", "test", "all"):
        command = sub.add_parser(action)
        command.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
        command.add_argument("--workers", type=int, default=4)
    predict = sub.add_parser("predict")
    predict.add_argument("method", choices=METHODS)
    predict.add_argument("ticket")
    data = sub.add_parser("prepare-data")
    data.add_argument("official_train", type=Path)
    data.add_argument("official_test", type=Path)
    data.add_argument("--destination", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.action == "prepare-data":
        prepare_data(args.official_train, args.official_test, args.destination)
        return
    args.local_model = LocalModel(args.gguf, args.threads) if args.gguf else None
    if args.action == "predict":
        spec = json.loads((args.out / args.method / "classifier.json").read_text(encoding="utf-8"))
        if spec["model"] != args.model:
            parser.error("The classifier was fitted with a different --model.")
        backend = Backend(args.url, args.model, local_model=args.local_model)
        code, raw, tokens = backend.classify(
            spec["instruction"], args.ticket, spec["demos"], spec["prefix"],
        )
        print(json.dumps({"code": code, "intent": INTENTS.get(code), "raw": raw, "input_tokens": tokens}))
        return
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.action in ("all", "fit"):
        args.out.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("DSPY_CACHEDIR", str(args.out / ".dspy_cache"))
        os.environ.setdefault("ADALFLOW_ROOT_PATH", str(args.out / ".adalflow_cache"))
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    splits = read_splits()
    if args.action in ("all", "fit"):
        for method in args.methods:
            directory = args.out / method
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "classifier.json").exists():
                parser.error(f"Classifier already exists in {directory}. Choose a fresh --out directory.")
            print(f"Fitting {method}", flush=True)
            fit(args, method, directory, {key: splits[key] for key in ("train", "validation")})
    if args.action in ("all", "test"):
        for method in args.methods:
            test(args, method, args.out / method, {"test": splits["test"]})


if __name__ == "__main__":
    main()
