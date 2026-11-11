"""Audit both model runs and generate the final comparative report."""
import argparse
from pathlib import Path

from benchmark_core import ROOT, metrics, parse_lenient, read_splits
from report import TITLES, accuracy_interval, audit, build
from run import METHODS


def comparison(large_dir, small_dir, large_label, runtime):
    test = read_splits()["test"]
    small = {name: audit(small_dir, name, test) for name in METHODS}
    large = {name: audit(large_dir, name, test) for name in METHODS}
    lines = [
        "## Comparison with the 0.5B run",
        "",
        "Both runs use the same tickets, intents, manual prompts, and output format. The original "
        "0.5B run used Q5_K_M weights and llama-cpp-python on CPU. "
        f"The {large_label} run used {runtime}. The revised {large_label} TextGrad and AdalFlow "
        "runs also use keep-best validation selection. Differences reflect these configurations together.",
        "",
        f"| Classifier | 0.5B accuracy | {large_label} accuracy | Δ accuracy (pp) | Paired 95% interval (pp) | 0.5B macro F1 | {large_label} macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        before = small[method]["scores"]
        after = large[method]["scores"]
        delta = (after["accuracy"] - before["accuracy"]) * 100
        interval = accuracy_interval(large[method], small[method], test)
        lines.append(
            f"| {TITLES[method]} | {before['accuracy']:.1%} | {after['accuracy']:.1%} | "
            f"{delta:+.1f} | {interval} | {before['macro_f1']:.3f} | {after['macro_f1']:.3f} |"
        )
    old_few = small["manual_few"]["scores"]
    new_few = large["manual_few"]["scores"]
    old_wrong = len(test) - round(old_few["accuracy"] * len(test)) - old_few["invalid"]
    new_wrong = len(test) - round(new_few["accuracy"] * len(test)) - new_few["invalid"]
    lines.extend([
        "",
        "Paired intervals resample held-out tickets within each intent. The two manual rows "
        "reuse the same prompts and compare inference. The four optimized rows compare the "
        "end-to-end systems fitted for each model.",
        "",
        f"With the same manual few-shot prompt, invalid replies changed from {old_few['invalid']} "
        f"to {new_few['invalid']}, and valid but incorrect labels changed from {old_wrong} to "
        f"{new_wrong}.",
        "",
        "See the [original 0.5B report](<results/qwen_0_5b/Original Benchmark Report.md>) for its full optimization ledger totals.",
        "",
    ])
    return "\n".join(lines)


def plot_accuracy_cost(evidence, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    colors = ("#475569", "#0f766e", "#6472b6", "#2363a5", "#a15126", "#955681")
    markers = ("s", "D", "^", "o", "X", "P")
    fig, ax = plt.subplots(figsize=(8.5, 4.7), layout="constrained")
    for method, color, marker in zip(METHODS, colors, markers, strict=True):
        item = evidence[method]
        tokens = sum(call["total_tokens"] for call in item["calls"])
        ax.scatter(tokens, item["scores"]["accuracy"], color=color, marker=marker,
                   s=85, zorder=3, label=TITLES[method])
    ax.axhline(evidence["manual_few"]["scores"]["accuracy"], color=colors[1],
               linestyle="--", linewidth=1, alpha=0.45)
    ax.set(title="Held-out accuracy vs. optimization cost", xlabel="Optimization tokens",
           ylabel="Strict accuracy", ylim=(0, 1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def review_analysis(evidence, before_dir, test):
    zero = evidence["manual_zero"]
    copro = evidence["dspy_copro"]
    few = evidence["manual_few"]
    revised_tg = evidence["textgrad"]["scores"]
    lines = ["## Analysis and viva notes", ""]
    if before_dir.is_dir():
        before = {name: audit(before_dir, name, test) for name in ("textgrad", "adalflow")}
        lines.extend([
            "### Effect of keep-best selection",
            "",
            "| Method | Previous strict | Previous lenient | Revised strict | Revised lenient |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for method, item in before.items():
            old = item["scores"]
            new = evidence[method]["scores"]
            lines.append(
                f"| {TITLES[method]} | {old['accuracy']:.1%} | {old['lenient_accuracy']:.1%} | "
                f"{new['accuracy']:.1%} | {new['lenient_accuracy']:.1%} |"
            )
        tg = before["textgrad"]["scores"]
        adal = before["adalflow"]["scores"]
        lenient_f1 = metrics(test, [parse_lenient(row["raw"])
                                   for row in before["textgrad"]["predictions"]])["per_intent_f1"]
        lines.extend([
            "",
            "The previous selection policy accepted the first scored rewrite regardless of its "
            "validation score. The new runs start with the original prompt as incumbent and can "
            "return it after spending the optimization budget. Starting validation scores differed "
            "across runs with the same starting prompt and split. The before-and-after figures "
            "reflect the policy change and run-to-run variation. The saved traces show each decision. "
            "[Previous 3B outputs and traces](results/qwen_3b_before_keep_best/)",
            "",
            "### Why the previous TextGrad run had 36 invalid replies",
            "",
            "The decoder generated freely up to eight tokens. The prompt requested a single letter, "
            "and the strict parser enforced that format during scoring.",
            "",
            f"Of the {tg['invalid']} strict-invalid replies, {tg['lenient_parsed']} had a parseable "
            f"leading code, including `C.` and `H. transaction charged twice`. "
            f"{tg['lenient_recovered_correct']} of those codes matched the gold labels. "
            "The remaining reply began with “None of the above.” "
            f"AdalFlow had {adal['lenient_parsed']} similarly recoverable formats and "
            f"{adal['lenient_recovered_correct']} additional correct predictions.",
            "",
            "The previous TextGrad prompt referred to undefined “conditions” and routed cues for "
            "declined payments, pending payments, and duplicate charges to G, the unrecognized card "
            "payment label. This conflation plausibly contributed to the poor per-intent results. "
            "Under strict scoring, C, F, G, and I each had zero F1 in that run. Lenient parsing "
            f"raises C to {lenient_f1['C']:.3f} and F to {lenient_f1['F']:.3f}, while G and I "
            f"remain at {lenient_f1['G']:.3f} and {lenient_f1['I']:.3f}. The effect of any "
            "individual rule is unmeasured.",
            "",
            f"The revised TextGrad prompt produced {revised_tg['invalid']} strict-invalid replies. "
            f"Lenient scoring recovered {revised_tg['lenient_recovered_correct']} correct labels. "
            f"TextGrad and manual zero-shot scored {revised_tg['accuracy']:.1%} and "
            f"{zero['scores']['accuracy']:.1%} under strict parsing and "
            f"{revised_tg['lenient_accuracy']:.1%} and {zero['scores']['lenient_accuracy']:.1%} "
            "under lenient parsing.",
            "",
        ])
    gains = {
        code: sum(r["prediction"] == code == r["code"] for r in copro["predictions"])
        - sum(r["prediction"] == code == r["code"] for r in zero["predictions"])
        for code in ("A", "F", "I", "J", "L")
    }
    net = round((copro["scores"]["accuracy"] - zero["scores"]["accuracy"]) * len(test))
    interval = accuracy_interval(copro, few, test)
    manual_demos = few["spec"]["demos"]
    bootstrap_demos = evidence["dspy_bootstrap"]["spec"]["demos"]
    bootstrap_labels = len({code for _, code in bootstrap_demos})
    manual_labels = len({code for _, code in manual_demos})
    lines.extend([
        "### Where COPRO gained and lost",
        "",
        f"Intent L accounted for {gains['L']:+d} correct tickets out of the {net:+d} net change "
        f"over zero-shot. Its strict F1 moved from {zero['scores']['per_intent_f1']['L']:.3f} to "
        f"{copro['scores']['per_intent_f1']['L']:.3f}. Intent A contributed {gains['A']:+d} correct "
        f"tickets and J contributed {gains['J']:+d}. F and I lost {abs(gains['F'])} and "
        f"{abs(gains['I'])} correct tickets, respectively, and each fell to zero F1. "
        "The generated A–J wording adds a separate conflict with the A–L codebook.",
        "",
        f"The paired 95% interval for COPRO minus manual few-shot was {interval} percentage points. "
        "It spans zero. A conclusion of equivalence would require a prespecified margin and "
        "an interval contained within it.",
        "",
        "### Why manual few-shot led",
        "",
        "The hand-written demonstrations showed both the decision boundaries and the desired "
        "one-letter response. Manual few-shot produced valid codes on every test ticket. "
        "Both DSPy prompts also had zero invalid replies, so their remaining gap to few-shot "
        "came from label choices. "
        f"The {len(manual_demos)} manual demonstrations covered {manual_labels} distinct labels. "
        f"BootstrapFewShot's {len(bootstrap_demos)} demonstrations covered {bootstrap_labels}. "
        "The examples are a plausible advantage in this setup. "
        "Isolating their effect would require matching the example sets and optimization budgets.",
        "",
        "Under these short optimization budgets on the 3B model, COPRO was the closest optimized "
        "method to manual few-shot. The zero-shot comparison is the more direct benchmark for "
        "instruction-only optimization. The results describe these fitted prompts and this dataset split.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", type=Path, default=ROOT / "results" / "qwen_3b")
    parser.add_argument("--small", type=Path, default=ROOT / "results" / "qwen_0_5b")
    parser.add_argument("--pre-keep-best", type=Path, default=ROOT / "results" / "qwen_3b_before_keep_best")
    parser.add_argument("--output", type=Path, default=ROOT / "Benchmark Report.md")
    parser.add_argument("--runtime", default="llama.cpp b8200 on CPU")
    parser.add_argument("--large-label", default="3B")
    parser.add_argument("--quantization", default="Q4_K_M")
    args = parser.parse_args()
    report = build(args.large, runtime=args.runtime)
    insertion = "## Change from the manual prompts"
    platform = f"{args.quantization} weights and {args.runtime}"
    report = report.replace(
        insertion, comparison(args.large, args.small, args.large_label, platform) + "\n" + insertion, 1,
    )
    report = report.replace(
        "# Programmatic prompt optimization for Banking77",
        "# Programmatic Prompt Optimization for Banking77 Intent Classification",
        1,
    )
    test = read_splits()["test"]
    evidence = {method: audit(args.large, method, test) for method in METHODS}
    plot_name = ("Accuracy vs Optimization Cost.png" if args.output.name == "Benchmark Report.md"
                 else f"{args.output.stem} - Accuracy vs Optimization Cost.png")
    plot_path = args.output.with_name(plot_name)
    plot_accuracy_cost(evidence, plot_path)
    report = report.replace(
        "## Experimental design",
        f"![Strict held-out accuracy versus total optimization tokens](<{plot_name}>)\n\n"
        "The plot includes all optimization calls, including rejected proposals. "
        "Manual prompts have zero optimization-token cost.\n\n## Experimental design",
        1,
    )
    report = report.replace(
        "## Conclusion and limits", review_analysis(evidence, args.pre_keep_best, test)
        + "\n## Conclusion and limits", 1,
    )
    args.output.write_text(report, encoding="utf-8")
    print("Audited both runs and wrote", args.output)


if __name__ == "__main__":
    main()
