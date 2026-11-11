"""Shared data, model transport, prompt rendering, and metrics."""
import csv
import hashlib
import json
import random
import re
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SEED = 2718
INTENTS = dict(zip("ABCDEFGHIJKL", [
    "card_arrival", "card_delivery_estimate", "card_not_working",
    "contactless_not_working", "declined_card_payment", "pending_card_payment",
    "card_payment_not_recognised", "transaction_charged_twice",
    "cash_withdrawal_not_recognised", "declined_cash_withdrawal",
    "pending_transfer", "transfer_not_received_by_recipient",
], strict=True))
CATALOG = "\n".join(f"{code}. {intent.replace('_', ' ')}" for code, intent in INTENTS.items())
MANUAL = "Classify this customer-support ticket by its primary banking intent."
DEMOS = [
    ("How many business days should my replacement card take to arrive?", "B"),
    ("My plastic card has still not turned up after three weeks.", "A"),
    ("Tapping my card at the reader no longer works, but inserting it does.", "D"),
    ("There is an unfamiliar debit-card purchase on my statement.", "G"),
    ("The ATM said my attempt to take out cash was declined.", "J"),
    ("The transfer shows complete, but my landlord never got the money.", "L"),
]


def prompt(instruction, ticket, demos=(), prefix="Intent"):
    system = (
        f"{instruction.strip()}\n\nChoose one intent code from this list:\n{CATALOG}"
        "\n\nReturn exactly one capital letter from A to L."
    )
    if demos:
        examples = "\n".join(f"Ticket: {text}\n{prefix}: {code}" for text, code in demos)
        system += "\n\nExamples:\n" + examples
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Ticket: {ticket}\n{prefix}:"},
    ]


def parse(text):
    """Strict closed-set scoring, with whitespace ignored."""
    value = text.strip()
    return value if value in INTENTS else "INVALID"


def parse_lenient(text):
    """Accept a bare intent code or a leading code followed by '.', ')' or ':'."""
    match = re.match(r"^([A-L])(?=$|[.):])", text.strip())
    return match[1] if match else "INVALID"


def read_splits():
    result = {}
    for split in ("train", "validation", "test"):
        with (ROOT / "data" / f"{split}.jsonl").open(encoding="utf-8") as file:
            result[split] = [json.loads(line) for line in file]
    return result


def prepare_data(train_path, test_path, target):
    """Freeze balanced, disjoint samples of the publisher's official splits."""
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    splits = {"train": [], "validation": [], "test": []}
    code_by_label = {label: code for code, label in INTENTS.items()}
    sources = {"train": train_path, "test": test_path}
    pools = {}
    for split, path in sources.items():
        with path.open(encoding="utf-8-sig", newline="") as file:
            pools[split] = [
                {"id": f"{split}-{index}", "ticket": row["text"].strip(),
                 "code": code_by_label[row["category"]]}
                for index, row in enumerate(csv.DictReader(file))
                if row["category"] in code_by_label
            ]

    def normalized(row):
        return " ".join(row["ticket"].casefold().split())

    test_text = {normalized(row) for row in pools["test"]}
    used = set()
    for source, destinations in [("train", [("train", 8), ("validation", 3)]),
                                 ("test", [("test", 10)])]:
        for code in INTENTS:
            candidates = [row for row in pools[source] if row["code"] == code]
            rng.shuffle(candidates)
            for destination, count in destinations:
                for row in candidates:
                    key = normalized(row)
                    if key not in used and (source == "test" or key not in test_text):
                        used.add(key)
                        splits[destination].append(row)
                        count -= 1
                        if count == 0:
                            break
                if count:
                    raise ValueError(f"Insufficient unique data for {destination}/{code}")
    for split, rows in splits.items():
        rng.shuffle(rows)
        (target / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {"seed": SEED, "source": "https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data",
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in sources.items()},
                "sizes": {name: len(rows) for name, rows in splits.items()}, "intents": INTENTS}
    write_json(target / "manifest.json", manifest)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Backend:
    """All generation, including optimizer feedback, passes through one meter."""

    def __init__(self, url, model, ledger=None, method="inference", local_model=None):
        self.url = url.rstrip("/")
        self.model = model
        self.ledger = ledger
        self.method = method
        self.phase = "inference"
        self.lock = threading.Lock()
        self.events = []
        self.local_model = local_model

    def complete(self, messages, purpose, max_tokens, temperature=0.0, seed=SEED,
                 response_format=None):
        start = time.perf_counter()
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "seed": seed, "max_tokens": max_tokens, "stream": False}
        if response_format:
            body["response_format"] = response_format
        if self.local_model is None:
            response = requests.post(f"{self.url}/chat/completions", json=body, timeout=240)
            response.raise_for_status()
            data = response.json()
        else:
            with self.local_model.lock:
                data = self.local_model.model.create_chat_completion(**body)
        usage = data["usage"]
        event = {
            "method": self.method, "phase": self.phase, "purpose": purpose,
            "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"], "seconds": round(time.perf_counter() - start, 4),
        }
        with self.lock:
            self.events.append(event)
            if self.ledger:
                with self.ledger.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event) + "\n")
        return data["choices"][0]["message"]["content"] or "", event

    def classify(self, instruction, ticket, demos=(), prefix="Intent"):
        raw, event = self.complete(prompt(instruction, ticket, demos, prefix), "classify", 8)
        return parse(raw), raw, event["input_tokens"]


class LocalModel:
    def __init__(self, path, threads):
        from llama_cpp import Llama

        self.lock = threading.Lock()
        self.model = Llama(
            model_path=str(path), n_ctx=4096, n_threads=threads, n_threads_batch=threads,
            n_batch=512, chat_format="chatml", verbose=False,
        )


def metrics(rows, predictions):
    pairs = list(zip((row["code"] for row in rows), predictions, strict=True))
    if not pairs:
        raise ValueError("Metrics need at least one example")
    f1 = {}
    for code in INTENTS:
        tp = sum(gold == pred == code for gold, pred in pairs)
        fp = sum(gold != code and pred == code for gold, pred in pairs)
        fn = sum(gold == code and pred != code for gold, pred in pairs)
        f1[code] = (2 * tp / (2 * tp + fp + fn)) if tp or fp or fn else 0.0
    return {"accuracy": sum(gold == pred for gold, pred in pairs) / len(pairs),
            "macro_f1": sum(f1.values()) / len(f1), "per_intent_f1": f1,
            "invalid": sum(pred == "INVALID" for _, pred in pairs)}
