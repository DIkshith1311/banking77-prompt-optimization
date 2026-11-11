# Programmatic prompt optimization for Banking77

## Held-out results

**AdalFlow** had the highest held-out accuracy at **19.2%** (23/120) with **0.125** macro F1. The manual few-shot baseline scored 18.3% accuracy and 0.140 macro F1.

| Classifier | Accuracy | Macro F1 | Invalid / 120 | Final prompt tokens | Mean test input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Manual zero-shot | 10.0% | 0.055 | 27 | 123 | 135.8 |
| Manual few-shot | 18.3% | 0.140 | 5 | 236 | 248.8 |
| DSPy BootstrapFewShot | 13.3% | 0.076 | 10 | 257 | 269.8 |
| DSPy COPRO | 11.7% | 0.079 | 23 | 122 | 134.8 |
| TextGrad | 13.3% | 0.087 | 25 | 147 | 159.8 |
| AdalFlow | 19.2% | 0.125 | 21 | 147 | 159.8 |

Final prompt length is the model-reported input-token count for the full saved chat template with an empty ticket. It includes the instruction, codebook, examples, labels, and chat framing. Mean test input adds the actual ticket text. The length probe is excluded from both optimization and test inference counts.

## Change from the manual prompts

| Optimizer | Δ accuracy vs. zero-shot (pp) | Δ macro F1 vs. zero-shot | Δ accuracy vs. few-shot (pp) | Δ macro F1 vs. few-shot | 95% interval for accuracy change vs. few-shot (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DSPy BootstrapFewShot | +3.3 | +0.022 | -5.0 | -0.063 | -10.8 to +0.8 |
| DSPy COPRO | +1.7 | +0.024 | -6.7 | -0.061 | -13.3 to +0.0 |
| TextGrad | +3.3 | +0.032 | -5.0 | -0.053 | -11.7 to +1.7 |
| AdalFlow | +9.2 | +0.070 | +0.8 | -0.015 | -6.7 to +7.5 |

The intervals resample paired tickets within each intent 2,000 times. They describe test-sample variability for these fitted prompts and are unadjusted for multiple comparisons.

## Optimization cost

| Classifier | LLM calls | Input tokens | Output tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: |
| Manual zero-shot | 0 | 0 | 0 | 0 |
| Manual few-shot | 0 | 0 | 0 | 0 |
| DSPy BootstrapFewShot | 32 | 7,261 | 32 | 7,293 |
| DSPy COPRO | 221 | 36,991 | 547 | 37,538 |
| TextGrad | 146 | 25,979 | 986 | 26,965 |
| AdalFlow | 123 | 24,888 | 452 | 25,340 |

Each completed generation counts as one call. Optimization includes training predictions, instruction proposals, textual feedback, and candidate validation when used by that method. Test inference and the length probe have separate ledger phases. The backend counts full input tokens per call, including any reusable prefix. The totals cover the final benchmark runs. Development and integration trials are excluded.

## Experimental design

All classifiers used `qwen2.5-0.5b-instruct-q5_k_m` with llama-cpp-python 0.3.19, CPU inference, the ChatML template, greedy classification, and an eight-token classification output cap. The runner accepts exactly one capital intent code after trimming whitespace. Accuracy counts every invalid output as an error. Macro F1 averages all 12 intents equally.

The frozen balanced subset has 96 training and 36 validation tickets sampled from the official training split and 120 held-out tickets sampled from the official test split. Sampling used seed 2718. Case-folded, whitespace-normalized exact duplicates were excluded across the selected rows, and training and validation texts matching any test-pool text were excluded. Closely related card, payment, withdrawal, and transfer intents make up the subset.

| Classifier | How the prompt was obtained |
| --- | --- |
| Manual zero-shot | One hand-written instruction and a fixed A–L codebook. |
| Manual few-shot | The zero-shot prompt plus six hand-written demonstrations. |
| DSPy BootstrapFewShot | A typed signature and Predict module. The optimizer selects up to six correctly predicted traces from the training split. |
| DSPy COPRO | The same module with native instruction search, breadth three and depth two. Candidate accuracy is measured on validation. The output cue is fixed across methods. |
| TextGrad | BlackboxLLM, a trainable instruction variable, TextLoss from labeled training errors, backward feedback, and TGD updates. |
| AdalFlow | A declarative Component with a Generator, trainable Parameter, exact-match EvalFnToTextLoss, BackwardEngine, and native TGDOptimizer propose/step/revert. |

TextGrad and AdalFlow each run two rounds with feedback from up to two training errors per round. TextGrad generates a routing rule from the textual gradients and labeled errors. AdalFlow uses its textual gradients to select a rule from the labeled mistakes and codebook. Both adapters append the selected guidance to the instruction. They select the first valid rewrite, then keep a later rewrite if its validation tuple (accuracy, macro F1) equals or exceeds the first. The initial validation score is recorded for context. Every final prompt is evaluated on the same held-out set.

### What changed during fitting

- **DSPy BootstrapFewShot** used 6 selected demonstrations. Call breakdown was `classify` × 32.
- **DSPy COPRO** used rewritten instruction. Call breakdown was `classify` × 216, `copro_instruction_proposal` × 5.
- **TextGrad** used generated routing guidance. Accepted 2 of 2 rounds. Call breakdown was `classify` × 132, `textgrad_feedback` × 12, `textgrad_update` × 2.
- **AdalFlow** used generated routing guidance. Accepted 1 of 2 rounds. Call breakdown was `adalflow_feedback` × 8, `adalflow_update` × 2, `classify` × 113.

### Saved routing guidance

- **TextGrad** appended `If the variable D is`.
- **TextGrad** appended `Classify this customer-support ticket by its primary banking intent.`.
- **AdalFlow** appended `For tickets about card delivery estimate, choose B. Distinguish this from contactless not working (D).`.

## Per-intent F1

| Intent | Zero-shot | Few-shot | Bootstrap | COPRO | TextGrad | AdalFlow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `card_arrival` | 0.000 | 0.370 | 0.143 | 0.125 | 0.000 | 0.286 |
| `card_delivery_estimate` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.640 |
| `card_not_working` | 0.318 | 0.091 | 0.000 | 0.359 | 0.375 | 0.000 |
| `contactless_not_working` | 0.000 | 0.105 | 0.095 | 0.000 | 0.000 | 0.340 |
| `declined_card_payment` | 0.000 | 0.091 | 0.211 | 0.000 | 0.100 | 0.000 |
| `pending_card_payment` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `card_payment_not_recognised` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `transaction_charged_twice` | 0.000 | 0.000 | 0.269 | 0.000 | 0.000 | 0.000 |
| `cash_withdrawal_not_recognised` | 0.170 | 0.250 | 0.200 | 0.125 | 0.269 | 0.233 |
| `declined_cash_withdrawal` | 0.000 | 0.323 | 0.000 | 0.000 | 0.167 | 0.000 |
| `pending_transfer` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `transfer_not_received_by_recipient` | 0.167 | 0.444 | 0.000 | 0.333 | 0.133 | 0.000 |

These were the three most frequent error patterns for the highest-accuracy classifier.

- `declined_card_payment` → `contactless_not_working` on 9 tickets.
- `card_not_working` → `contactless_not_working` on 8 tickets.
- `pending_transfer` → `INVALID` on 8 tickets.

## Limits

This is one run on a small CPU-friendly model, one balanced subset, and short optimizer budgets. The measured effects combine the algorithm, prompt design, and budget. The comparison applies to this 12-intent protocol and its balanced class distribution.

See the [README](../../README.md) for primary sources. Saved prompt specifications, optimizer traces, per-ticket predictions, and usage ledgers for this run are in [0.5B results](./).
