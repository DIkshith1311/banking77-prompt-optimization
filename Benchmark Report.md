# Programmatic Prompt Optimization for Banking77 Intent Classification

## Held-out results

**Manual few-shot** had the highest held-out accuracy at **70.0%** (84/120) with **0.661** macro F1. The highest-accuracy optimized prompt was DSPy COPRO at 65.8%.

| Classifier | Strict accuracy | Lenient accuracy | Strict macro F1 | Strict invalid / 120 | Final prompt tokens | Mean test input tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Manual zero-shot | 53.3% | 58.3% | 0.489 | 8 | 123 | 135.8 |
| Manual few-shot | 70.0% | 70.0% | 0.661 | 0 | 236 | 248.8 |
| DSPy BootstrapFewShot | 56.7% | 56.7% | 0.522 | 0 | 225 | 237.8 |
| DSPy COPRO | 65.8% | 65.8% | 0.604 | 0 | 185 | 197.8 |
| TextGrad | 45.0% | 60.0% | 0.471 | 32 | 251 | 263.8 |
| AdalFlow | 53.3% | 59.2% | 0.494 | 10 | 123 | 135.8 |

Strict scoring accepts a single A–L code after trimming whitespace. Lenient scoring also accepts a leading uppercase A–L code followed immediately by a period, closing parenthesis, or colon, ignoring the remaining text. The lenient column rescores the same saved outputs. Prompt selection, macro F1, and accuracy comparisons use strict scoring.

Final prompt length is the model-reported input-token count for the full saved chat template with an empty ticket. It includes the instruction, codebook, examples, labels, and chat framing. Mean test input adds the actual ticket text. The length probe is excluded from both optimization and test inference counts.

## Comparison with the 0.5B run

Both runs use the same tickets, intents, manual prompts, and output format. The original 0.5B run used Q5_K_M weights and llama-cpp-python on CPU. The 3B run used Q4_K_M weights and llama.cpp b8200 on CPU. The revised 3B TextGrad and AdalFlow runs also use keep-best validation selection. Differences reflect these configurations together.

| Classifier | 0.5B accuracy | 3B accuracy | Δ accuracy (pp) | Paired 95% interval (pp) | 0.5B macro F1 | 3B macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Manual zero-shot | 10.0% | 53.3% | +43.3 | +36.7 to +50.0 | 0.055 | 0.489 |
| Manual few-shot | 18.3% | 70.0% | +51.7 | +43.3 to +60.0 | 0.140 | 0.661 |
| DSPy BootstrapFewShot | 13.3% | 56.7% | +43.3 | +35.8 to +50.8 | 0.076 | 0.522 |
| DSPy COPRO | 11.7% | 65.8% | +54.2 | +46.7 to +61.7 | 0.079 | 0.604 |
| TextGrad | 13.3% | 45.0% | +31.7 | +23.3 to +40.0 | 0.087 | 0.471 |
| AdalFlow | 19.2% | 53.3% | +34.2 | +27.5 to +40.8 | 0.125 | 0.494 |

Paired intervals resample held-out tickets within each intent. The two manual rows reuse the same prompts and compare inference. The four optimized rows compare the end-to-end systems fitted for each model.

With the same manual few-shot prompt, invalid replies changed from 5 to 0, and valid but incorrect labels changed from 93 to 36.

See the [original 0.5B report](<results/qwen_0_5b/Original Benchmark Report.md>) for its full optimization ledger totals.

## Change from the manual prompts

| Optimizer | Δ accuracy vs. zero-shot (pp) | Δ macro F1 vs. zero-shot | Δ accuracy vs. few-shot (pp) | Δ macro F1 vs. few-shot | 95% interval for accuracy change vs. few-shot (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DSPy BootstrapFewShot | +3.3 | +0.033 | -13.3 | -0.139 | -19.2 to -7.5 |
| DSPy COPRO | +12.5 | +0.115 | -4.2 | -0.057 | -9.2 to +0.8 |
| TextGrad | -8.3 | -0.019 | -25.0 | -0.191 | -32.5 to -17.5 |
| AdalFlow | +0.0 | +0.005 | -16.7 | -0.167 | -23.3 to -10.0 |

The intervals resample paired tickets within each intent 20,000 times with seed 2718, using nearest-rank 2.5th and 97.5th percentiles. They describe test-sample variability for these fitted prompts. The intervals are unadjusted for multiple comparisons.

## Optimization cost

| Classifier | LLM calls | Input tokens | Output tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: |
| Manual zero-shot | 0 | 0 | 0 | 0 |
| Manual few-shot | 0 | 0 | 0 | 0 |
| DSPy BootstrapFewShot | 9 | 2,019 | 18 | 2,037 |
| DSPy COPRO | 221 | 41,743 | 968 | 42,711 |
| TextGrad | 146 | 30,072 | 1,324 | 31,396 |
| AdalFlow | 127 | 24,537 | 805 | 25,342 |

Each completed generation counts as one call. Optimization includes training predictions, instruction proposals, textual feedback, and candidate validation when used by that method. Test inference and the length probe have separate ledger phases. The backend counts full input tokens per call, including any reusable prefix. The totals cover the final benchmark runs. Development and integration trials are excluded.

![Strict held-out accuracy versus total optimization tokens](<Accuracy vs Optimization Cost.png>)

The plot includes all optimization calls, including rejected proposals. Manual prompts have zero optimization-token cost.

## Experimental design

All classifiers used `qwen2.5-3b-instruct-q4_k_m` with llama.cpp b8200 on CPU, the ChatML template, greedy classification, and an eight-token classification output cap. The runner accepts exactly one capital intent code after trimming whitespace. Accuracy counts every invalid output as an error. Macro F1 averages all 12 intents equally.

The frozen balanced subset has 96 training and 36 validation tickets sampled from the official training split and 120 held-out tickets sampled from the official test split. Sampling used seed 2718. Case-folded, whitespace-normalized exact duplicates were excluded across the selected rows, and training and validation texts matching any test-pool text were excluded. Closely related card, payment, withdrawal, and transfer intents make up the subset.

| Classifier | How the prompt was obtained |
| --- | --- |
| Manual zero-shot | One hand-written instruction and a fixed A–L codebook. |
| Manual few-shot | The zero-shot prompt plus six hand-written demonstrations. |
| DSPy BootstrapFewShot | A typed signature and Predict module. The optimizer selects up to six correctly predicted traces from the training split. |
| DSPy COPRO | The same module with native instruction search, breadth three and depth two. Candidate accuracy is measured on validation. The output cue is fixed across methods. |
| TextGrad | BlackboxLLM, a trainable instruction variable, TextLoss from labeled training errors, backward feedback, and TGD updates. |
| AdalFlow | A declarative Component with a Generator, trainable Parameter, exact-match EvalFnToTextLoss, BackwardEngine, and native TGDOptimizer propose/step/revert. |

TextGrad and AdalFlow each run two rounds with feedback from up to two training errors per round. TextGrad generates a routing rule from the textual gradients and labeled errors. AdalFlow uses its textual gradients to select a rule from the labeled mistakes and codebook. Both adapters append the selected guidance to the instruction. The starting prompt is the first incumbent. A proposal replaces the incumbent only when its strict validation accuracy improves, or when accuracy ties and macro F1 improves. An exact tie keeps the incumbent. Every final prompt is evaluated on the same held-out set.

### What changed during fitting

- **DSPy BootstrapFewShot** used 6 selected demonstrations. Call breakdown was `classify` × 9.
- **DSPy COPRO** used rewritten instruction. Call breakdown was `classify` × 216, `copro_instruction_proposal` × 5.
- **TextGrad** used generated routing guidance. Accepted 1 of 2 rounds. Call breakdown was `classify` × 132, `textgrad_feedback` × 12, `textgrad_update` × 2.
  Strict validation accuracy / macro F1 was 38.9% / 0.336 (start) → 38.9% / 0.330 (rejected) → 38.9% / 0.367 (kept).
- **AdalFlow** used original zero-shot prompt retained. Accepted 0 of 2 rounds. Call breakdown was `adalflow_feedback` × 8, `adalflow_update` × 2, `classify` × 117.
  Strict validation accuracy / macro F1 was 44.4% / 0.385 (start) → 44.4% / 0.360 (rejected) → 41.7% / 0.376 (rejected).

### Saved routing guidance

- **DSPy COPRO** selected `Classify the customer-support ticket based on the primary banking intent using the provided intent codebook (A-J). Use clear distinctions for labels that are frequently confused, such as A (card arrival) and L (transfer not received by recipient), ensuring that their implications are clearly differentiated. Prioritize the output format as a single-letter code from the given list (A-J).`.
  The generated instruction mentions A–J. The fixed codebook and scorer still cover A–L.
- **TextGrad** appended `Classify this customer-support ticket by its primary banking intent. Existing codes: A. card arrival B. card delivery estimate C. card not working D. contactless not working E. declined card payment F. pending card payment G. card payment not recognised H. transaction charged twice I. cash withdrawal not recognised J. declined cash withdrawal K. pending transfer L. transfer not received by recipient New rule: If the ticket contains the phrase 'track the card you sent', classify as **A. card arrival**. Otherwise, classify as the closest matching intent code.`.

## Per-intent F1

| Intent | Zero-shot | Few-shot | Bootstrap | COPRO | TextGrad | AdalFlow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `card_arrival` | 0.000 | 0.783 | 0.235 | 0.667 | 0.154 | 0.154 |
| `card_delivery_estimate` | 0.533 | 0.667 | 0.560 | 0.545 | 0.533 | 0.571 |
| `card_not_working` | 0.900 | 0.769 | 0.783 | 0.889 | 0.667 | 0.857 |
| `contactless_not_working` | 0.889 | 0.889 | 0.889 | 0.889 | 0.824 | 0.889 |
| `declined_card_payment` | 0.533 | 0.522 | 0.727 | 0.818 | 0.485 | 0.533 |
| `pending_card_payment` | 0.182 | 0.909 | 0.154 | 0.000 | 0.333 | 0.182 |
| `card_payment_not_recognised` | 0.286 | 0.571 | 0.533 | 0.429 | 0.308 | 0.308 |
| `transaction_charged_twice` | 0.947 | 0.900 | 0.783 | 0.783 | 0.900 | 0.889 |
| `cash_withdrawal_not_recognised` | 0.375 | 0.462 | 0.636 | 0.000 | 0.000 | 0.286 |
| `declined_cash_withdrawal` | 0.600 | 0.818 | 0.429 | 0.571 | 0.421 | 0.636 |
| `pending_transfer` | 0.625 | 0.645 | 0.541 | 0.714 | 0.690 | 0.625 |
| `transfer_not_received_by_recipient` | 0.000 | 0.000 | 0.000 | 0.947 | 0.333 | 0.000 |

These were the three most frequent error patterns for the highest-accuracy classifier.

- `transfer_not_received_by_recipient` → `pending_transfer` on 10 tickets.
- `card_delivery_estimate` → `card_arrival` on 4 tickets.
- `declined_card_payment` → `card_not_working` on 4 tickets.

## Analysis and viva notes

### Effect of keep-best selection

| Method | Previous strict | Previous lenient | Revised strict | Revised lenient |
| --- | ---: | ---: | ---: | ---: |
| TextGrad | 31.7% | 46.7% | 45.0% | 60.0% |
| AdalFlow | 51.7% | 51.7% | 53.3% | 59.2% |

The previous selection policy accepted the first scored rewrite regardless of its validation score. The new runs start with the original prompt as incumbent and can return it after spending the optimization budget. Starting validation scores differed across runs with the same starting prompt and split. The before-and-after figures reflect the policy change and run-to-run variation. The saved traces show each decision. [Previous 3B outputs and traces](results/qwen_3b_before_keep_best/)

### Why the previous TextGrad run had 36 invalid replies

The decoder generated freely up to eight tokens. The prompt requested a single letter, and the strict parser enforced that format during scoring.

Of the 36 strict-invalid replies, 35 had a parseable leading code, including `C.` and `H. transaction charged twice`. 18 of those codes matched the gold labels. The remaining reply began with “None of the above.” AdalFlow had 5 similarly recoverable formats and 0 additional correct predictions.

The previous TextGrad prompt referred to undefined “conditions” and routed cues for declined payments, pending payments, and duplicate charges to G, the unrecognized card payment label. This conflation plausibly contributed to the poor per-intent results. Under strict scoring, C, F, G, and I each had zero F1 in that run. Lenient parsing raises C to 0.692 and F to 0.182, while G and I remain at 0.000 and 0.000. The effect of any individual rule is unmeasured.

The revised TextGrad prompt produced 32 strict-invalid replies. Lenient scoring recovered 18 correct labels. TextGrad and manual zero-shot scored 45.0% and 53.3% under strict parsing and 60.0% and 58.3% under lenient parsing.

### Where COPRO gained and lost

Intent L accounted for +9 correct tickets out of the +15 net change over zero-shot. Its strict F1 moved from 0.000 to 0.947. Intent A contributed +7 correct tickets and J contributed +4. F and I lost 1 and 3 correct tickets, respectively, and each fell to zero F1. The generated A–J wording adds a separate conflict with the A–L codebook.

The paired 95% interval for COPRO minus manual few-shot was -9.2 to +0.8 percentage points. It spans zero. A conclusion of equivalence would require a prespecified margin and an interval contained within it.

### Why manual few-shot led

The hand-written demonstrations showed both the decision boundaries and the desired one-letter response. Manual few-shot produced valid codes on every test ticket. Both DSPy prompts also had zero invalid replies, so their remaining gap to few-shot came from label choices. The 6 manual demonstrations covered 6 distinct labels. BootstrapFewShot's 6 demonstrations covered 5. The examples are a plausible advantage in this setup. Isolating their effect would require matching the example sets and optimization budgets.

Under these short optimization budgets on the 3B model, COPRO was the closest optimized method to manual few-shot. The zero-shot comparison is the more direct benchmark for instruction-only optimization. The results describe these fitted prompts and this dataset split.

## Conclusion and limits

The zero-shot baseline is the closer comparison for the instruction-only methods. Manual few-shot and DSPy BootstrapFewShot add demonstrations, so differences against manual few-shot also reflect access to examples and how those examples were chosen. The results apply to this model, balanced subset, prompt design, and optimizer budgets. TextGrad and AdalFlow had just two rounds and up to two error examples per round.

### Identical-prompt repeatability

AdalFlow retained the starting prompt. Its rendered prompt is byte-identical to manual zero-shot, yet their raw replies differed on 7 of 120 tickets. AdalFlow and manual zero-shot scored 53.3% and 53.3% strict accuracy, with 10 and 8 invalid replies and macro F1 of 0.494 and 0.489, respectively.

The differences show run-to-run variation with the same prompt. Parallel batching in llama.cpp is a plausible contributor. Its role was untested. Small score gaps should be interpreted against that variation. The benchmark used a single seed, and the paired bootstrap intervals measure test-ticket sampling variation for the saved predictions. Estimating inference variability would require repeated runs.

See the [README](README.md) for reproduction steps and primary sources. Saved prompt specifications, optimizer traces, per-ticket predictions, and usage ledgers are in [3B results](results/qwen_3b/).
