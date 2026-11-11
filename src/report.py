"""Audit benchmark evidence and rebuild the Markdown comparison."""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from benchmark_core import INTENTS, ROOT, SEED, metrics, parse, parse_lenient, prompt, read_splits
from run import METHODS

TITLES = dict(zip(METHODS, [
    "Manual zero-shot", "Manual few-shot", "DSPy BootstrapFewShot", "DSPy COPRO",
    "TextGrad", "AdalFlow",
], strict=True))
BOOTSTRAP_RESAMPLES = 20_000


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(results, method, test):
    folder = results / method
    predictions = load_lines(folder / "predictions.jsonl")
    assert len(test) == len(predictions)
    for row, prediction in zip(test, predictions, strict=True):
        assert all(prediction[key] == value for key, value in row.items())
        assert parse(prediction["raw"]) == prediction["prediction"]
    scores = metrics(test, [row["prediction"] for row in predictions])
    saved = load_json(folder / "test_metrics.json")
    assert all(saved[key] == value for key, value in scores.items())
    assert saved["n_test"] == len(test)
    calls = load_lines(folder / "optimization_calls.jsonl")
    test_calls = load_lines(folder / "test_calls.jsonl")
    assert all(row["method"] == method and row["phase"] == "optimization" for row in calls)
    assert all(row["method"] == method for row in test_calls)
    assert all(row["total_tokens"] == row["input_tokens"] + row["output_tokens"]
               for row in calls + test_calls)
    inference = [row for row in test_calls if row["phase"] == "test"]
    probe = [row for row in test_calls if row["phase"] == "length_probe"]
    assert len(inference) == len(test) and len(probe) == 1 and len(test_calls) == len(test) + 1
    assert sum(row["input_tokens"] for row in inference) == sum(row["prompt_tokens"] for row in predictions)
    assert saved["prompt_tokens_mean_test"] == sum(row["input_tokens"] for row in inference) / len(test)
    assert probe[0]["input_tokens"] == saved["prompt_tokens_static"]
    fitted = load_json(folder / "classifier.json")
    assert fitted["method"] == method
    assert prompt(fitted["instruction"], "", fitted["demos"], fitted["prefix"]) == load_json(
        folder / "rendered_prompt.json",
    )
    lenient = [parse_lenient(row["raw"]) for row in predictions]
    recoveries = [(row, code) for row, code in zip(predictions, lenient, strict=True)
                  if row["prediction"] == "INVALID" and code != "INVALID"]
    saved.update({
        "lenient_accuracy": metrics(test, lenient)["accuracy"],
        "lenient_parsed": len(recoveries),
        "lenient_recovered_correct": sum(row["code"] == code for row, code in recoveries),
    })
    return {"scores": saved, "predictions": predictions, "calls": calls, "spec": fitted,
            "trace": load_json(folder / "optimization_trace.json")}


def accuracy_interval(method, baseline, test):
    """Paired stratified 95% percentile interval for an accuracy difference."""
    rng = random.Random(SEED)
    grouped = {code: [] for code in INTENTS}
    for left, right in zip(method["predictions"], baseline["predictions"], strict=True):
        gold = left["code"]
        grouped[gold].append(int(left["prediction"] == gold) - int(right["prediction"] == gold))
    values = sorted(
        100 * sum(sum(rng.choices(group, k=len(group))) for group in grouped.values()) / len(test)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    lower = values[math.ceil(0.025 * BOOTSTRAP_RESAMPLES) - 1]
    upper = values[math.ceil(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return f"{lower:+.1f} to {upper:+.1f}"


def build(results, runtime="llama-cpp-python 0.3.19, CPU inference"):
    splits = read_splits()
    assert all(len(set(row["id"] for row in rows)) == len(rows) for rows in splits.values())
    assert all(Counter(row["code"] for row in rows) == {code: n for code in INTENTS}
               for rows, n in zip(splits.values(), (8, 3, 10), strict=True))
    texts = [" ".join(row["ticket"].casefold().split()) for rows in splits.values() for row in rows]
    assert len(set(texts)) == len(texts)
    test = splits["test"]
    evidence = {method: audit(results, method, test) for method in METHODS}
    model_ids = {row["spec"]["model"] for row in evidence.values()}
    assert len(model_ids) == 1
    model, = model_ids
    zero = evidence["manual_zero"]["scores"]
    few = evidence["manual_few"]["scores"]
    best_method = max(METHODS, key=lambda name: (evidence[name]["scores"]["accuracy"],
                                                evidence[name]["scores"]["macro_f1"]))
    best = evidence[best_method]["scores"]
    best_optimized = max(METHODS[2:], key=lambda name: (evidence[name]["scores"]["accuracy"],
                                                       evidence[name]["scores"]["macro_f1"]))
    top = (f"**{TITLES[best_method]}** had the highest held-out accuracy at "
           f"**{best['accuracy']:.1%}** ({round(best['accuracy'] * len(test))}/{len(test)}) "
           f"with **{best['macro_f1']:.3f}** macro F1.")
    if best_method == "manual_few":
        top += (f" The highest-accuracy optimized prompt was {TITLES[best_optimized]} "
                f"at {evidence[best_optimized]['scores']['accuracy']:.1%}.")
    else:
        top += (f" The manual few-shot baseline scored {few['accuracy']:.1%} accuracy "
                f"and {few['macro_f1']:.3f} macro F1.")
    lines = [
        "# Programmatic prompt optimization for Banking77",
        "",
        "## Held-out results",
        "",
        top,
        "",
        "| Classifier | Strict accuracy | Lenient accuracy | Strict macro F1 | Strict invalid / 120 | Final prompt tokens | Mean test input tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, item in evidence.items():
        score = item["scores"]
        lines.append(f"| {TITLES[method]} | {score['accuracy']:.1%} | "
                     f"{score['lenient_accuracy']:.1%} | {score['macro_f1']:.3f} | "
                     f"{score['invalid']} | {score['prompt_tokens_static']} | "
                     f"{score['prompt_tokens_mean_test']:.1f} |")
    lines.extend([
        "",
        "Strict scoring accepts a single A–L code after trimming whitespace. Lenient scoring also "
        "accepts a leading uppercase A–L code followed immediately by a period, closing parenthesis, "
        "or colon, ignoring the remaining text. The lenient column rescores the same saved outputs. "
        "Prompt selection, macro F1, and accuracy comparisons use strict scoring.",
        "",
        "Final prompt length is the model-reported input-token count for the full saved chat "
        "template with an empty ticket. It includes the instruction, codebook, examples, labels, "
        "and chat framing. Mean test input adds the actual ticket text. The length probe is excluded "
        "from both optimization and test inference counts.",
        "",
        "## Change from the manual prompts",
        "",
        "| Optimizer | Δ accuracy vs. zero-shot (pp) | Δ macro F1 vs. zero-shot | "
        "Δ accuracy vs. few-shot (pp) | Δ macro F1 vs. few-shot | "
        "95% interval for accuracy change vs. few-shot (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method in METHODS[2:]:
        item = evidence[method]
        score = item["scores"]
        lines.append(f"| {TITLES[method]} | {100 * (score['accuracy'] - zero['accuracy']):+.1f} | "
                     f"{score['macro_f1'] - zero['macro_f1']:+.3f} | "
                     f"{100 * (score['accuracy'] - few['accuracy']):+.1f} | "
                     f"{score['macro_f1'] - few['macro_f1']:+.3f} | "
                     f"{accuracy_interval(item, evidence['manual_few'], test)} |")
    lines.extend([
        "",
        f"The intervals resample paired tickets within each intent {BOOTSTRAP_RESAMPLES:,} times "
        f"with seed {SEED}, using nearest-rank 2.5th and 97.5th percentiles. They describe "
        "test-sample variability for these fitted prompts. The intervals are unadjusted for multiple comparisons.",
        "",
        "## Optimization cost",
        "",
        "| Classifier | LLM calls | Input tokens | Output tokens | Total tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for method, item in evidence.items():
        calls = item["calls"]
        tokens = [sum(event[key] for event in calls)
                  for key in ("input_tokens", "output_tokens", "total_tokens")]
        lines.append(f"| {TITLES[method]} | {len(calls):,} | "
                     + " | ".join(f"{n:,}" for n in tokens) + " |")
    lines.extend([
        "",
        "Each completed generation counts as one call. Optimization includes training predictions, "
        "instruction proposals, textual feedback, and candidate validation when used by that method. "
        "Test inference and the length probe have separate ledger phases. "
        "The backend counts full input tokens per call, including any reusable prefix. "
        "The totals cover the final benchmark runs. Development and integration trials are excluded.",
        "",
        "## Experimental design",
        "",
        f"All classifiers used `{model}` with {runtime}, "
        "the ChatML template, greedy classification, and an eight-token classification output cap. "
        "The runner accepts exactly one capital intent code after trimming whitespace. "
        "Accuracy counts every invalid output as an error. Macro F1 averages all 12 intents equally.",
        "",
        "The frozen balanced subset has 96 training and 36 validation tickets sampled from the "
        "official training split and 120 held-out tickets sampled from the official test split. "
        "Sampling used seed 2718. Case-folded, whitespace-normalized exact duplicates were excluded "
        "across the selected rows, and training and validation texts matching any test-pool text "
        "were excluded. Closely related card, payment, withdrawal, and transfer intents make up the subset.",
        "",
        "| Classifier | How the prompt was obtained |",
        "| --- | --- |",
        "| Manual zero-shot | One hand-written instruction and a fixed A–L codebook. |",
        "| Manual few-shot | The zero-shot prompt plus six hand-written demonstrations. |",
        "| DSPy BootstrapFewShot | A typed signature and Predict module. The optimizer selects up to six correctly predicted traces from the training split. |",
        "| DSPy COPRO | The same module with native instruction search, breadth three and depth two. Candidate accuracy is measured on validation. The output cue is fixed across methods. |",
        "| TextGrad | BlackboxLLM, a trainable instruction variable, TextLoss from labeled training errors, backward feedback, and TGD updates. |",
        "| AdalFlow | A declarative Component with a Generator, trainable Parameter, exact-match EvalFnToTextLoss, BackwardEngine, and native TGDOptimizer propose/step/revert. |",
        "",
        "TextGrad and AdalFlow each run two rounds with feedback from up to two training errors per round. "
        "TextGrad generates a routing rule from the textual gradients and labeled errors. "
        "AdalFlow uses its textual gradients to select a rule from the labeled mistakes and codebook. "
        "Both adapters append the selected guidance to the instruction. "
        "The starting prompt is the first incumbent. A proposal replaces the incumbent only "
        "when its strict validation accuracy improves, or when accuracy ties and macro F1 improves. "
        "An exact tie keeps the incumbent. Every final prompt is evaluated on the same held-out set.",
        "",
        "### What changed during fitting",
        "",
    ])
    for method in METHODS[2:]:
        item = evidence[method]
        purposes = Counter(row["purpose"] for row in item["calls"])
        changes = []
        if item["spec"]["demos"]:
            changes.append(f"{len(item['spec']['demos'])} selected demonstrations")
        if "\nDecision guidance: " in item["spec"]["instruction"]:
            changes.append("generated routing guidance")
        elif item["spec"]["instruction"] != evidence["manual_zero"]["spec"]["instruction"]:
            changes.append("rewritten instruction")
        if not changes:
            changes.append("original zero-shot prompt retained")
        counts = ", ".join(f"`{key}` × {count}" for key, count in sorted(purposes.items()))
        rounds = [entry for entry in item["trace"] if "round" in entry]
        accepted = f" Accepted {sum(bool(e.get('accepted')) for e in rounds)} of {len(rounds)} rounds." if rounds else ""
        lines.append(f"- **{TITLES[method]}** used {' and '.join(changes)}.{accepted} Call breakdown was {counts}.")
        if rounds and "initial_validation" in item["trace"][0]:
            start_accuracy, start_f1 = item["trace"][0]["initial_validation"]
            path = [f"{start_accuracy:.1%} / {start_f1:.3f} (start)"]
            path.extend(
                f"{round_['validation'][0]:.1%} / {round_['validation'][1]:.3f} "
                f"({'kept' if round_['accepted'] else 'rejected'})"
                if "validation" in round_ else round_.get("reason", "no scored proposal")
                for round_ in rounds
            )
            lines.append(f"  Strict validation accuracy / macro F1 was {' → '.join(path)}.")
    lines.extend(["", "### Saved routing guidance", ""])
    copro_instruction = " ".join(evidence["dspy_copro"]["spec"]["instruction"].split()).replace("`", "'")
    lines.append(f"- **DSPy COPRO** selected `{copro_instruction}`.")
    if "A-J" in copro_instruction and "L" in INTENTS:
        lines.append(
            "  The generated instruction mentions A–J. The fixed codebook and scorer still cover A–L."
        )
    for method in ("textgrad", "adalflow"):
        guidance = evidence[method]["spec"]["instruction"].split("\nDecision guidance: ")[1:]
        for rule in guidance:
            excerpt = " ".join(rule.split()).replace("`", "'")
            lines.append(f"- **{TITLES[method]}** appended `{excerpt}`.")
    lines.extend([
        "",
        "## Per-intent F1",
        "",
        "| Intent | Zero-shot | Few-shot | Bootstrap | COPRO | TextGrad | AdalFlow |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for code, intent in INTENTS.items():
        cells = " | ".join(f"{item['scores']['per_intent_f1'][code]:.3f}" for item in evidence.values())
        lines.append(f"| `{intent}` | {cells} |")
    errors = Counter((row["code"], row["prediction"]) for row in evidence[best_method]["predictions"]
                     if row["code"] != row["prediction"])
    lines.extend(["", "These were the three most frequent error patterns for the highest-accuracy classifier."])
    lines.append("")
    for (gold, predicted), count in errors.most_common(3):
        lines.append(f"- `{INTENTS[gold]}` → `{INTENTS.get(predicted, predicted)}` on {count} tickets.")
    lines.extend([
        "",
        "## Conclusion and limits",
        "",
        "The zero-shot baseline is the closer comparison for the instruction-only methods. "
        "Manual few-shot and DSPy BootstrapFewShot add demonstrations, so differences against "
        "manual few-shot also reflect access to examples and how those examples were chosen. "
        "The results apply to this model, balanced subset, prompt design, and optimizer budgets. "
        "TextGrad and AdalFlow had just two rounds and up to two error examples per round.",
        "",
    ])
    zero_prompt = results / "manual_zero" / "rendered_prompt.json"
    adal_prompt = results / "adalflow" / "rendered_prompt.json"
    if zero_prompt.read_bytes() == adal_prompt.read_bytes():
        adal = evidence["adalflow"]
        zero_run = evidence["manual_zero"]
        different = sum(left["raw"] != right["raw"] for left, right in
                        zip(adal["predictions"], zero_run["predictions"], strict=True))
        scores = adal["scores"]
        lines.extend([
            "### Identical-prompt repeatability",
            "",
            "AdalFlow retained the starting prompt. Its rendered prompt is byte-identical to "
            f"manual zero-shot, yet their raw replies differed on {different} of {len(test)} tickets. "
            f"AdalFlow and manual zero-shot scored {scores['accuracy']:.1%} and "
            f"{zero['accuracy']:.1%} strict accuracy, with {scores['invalid']} and "
            f"{zero['invalid']} invalid replies and macro F1 of {scores['macro_f1']:.3f} "
            f"and {zero['macro_f1']:.3f}, respectively.",
            "",
            "The differences show run-to-run variation with the same prompt. Parallel batching "
            "in llama.cpp is a plausible contributor. Its role was untested. Small score gaps "
            "should be interpreted against that variation. The benchmark used a single seed, "
            "and the paired bootstrap intervals measure test-ticket sampling variation for the "
            "saved predictions. Estimating inference variability would require repeated runs.",
            "",
        ])
    lines.extend([
        "See the [README](README.md) for reproduction steps and primary sources. Saved prompt "
        "specifications, optimizer traces, per-ticket predictions, and usage ledgers are in [3B results](results/qwen_3b/).",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "qwen_3b")
    parser.add_argument("--output", type=Path, default=ROOT / "Benchmark Report.md")
    args = parser.parse_args()
    args.output.write_text(build(args.results), encoding="utf-8")
    print(f"Audited all {len(METHODS)} methods and wrote {args.output}")


if __name__ == "__main__":
    main()
