# Programmatic Prompt Optimization for Banking77 Intent Classification

The final evaluation compares hand-written zero-shot and few-shot prompts with DSPy BootstrapFewShot, DSPy COPRO, TextGrad, and AdalFlow. It uses Qwen2.5-3B-Instruct on the same 12 Banking77 intents and 120 held-out tickets as the original 0.5B run.

- [Final report and model comparison](<Benchmark Report.md>)
- [Accuracy versus optimization cost](<Accuracy vs Optimization Cost.png>)
- [3B results and fitted classifiers](results/qwen_3b/) include the final instruction, demonstrations, rendered prompt, test predictions, optimizer trace, and per-call token ledger for each method.
- [Previous 3B TextGrad and AdalFlow outputs](results/qwen_3b_before_keep_best/) preserve the runs before the validation selection fix.
- [Original 0.5B report](<results/qwen_0_5b/Original Benchmark Report.md>) and [original evidence](results/qwen_0_5b/)
- [Dataset manifest](data/manifest.json) records the splits, label map, seed, source, and checksums. The bundled subset is adapted from the PolyAI Banking77 release under [CC BY 4.0](data/LICENSE.txt).

## Install

Use Python 3.12 and the Windows CPU build of [llama.cpp b8200](https://github.com/ggml-org/llama.cpp/releases/tag/b8200). The Qwen2.5-3B-Instruct Q4_K_M weights are a separate download of about 1.93 GB. The GGUF comes from the [Ollama 3B model distribution](https://ollama.com/library/qwen2.5:3b).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
curl.exe -fL "https://registry.ollama.ai/v2/library/qwen2.5/blobs/sha256:5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6" -o qwen2.5-3b-instruct.gguf
Get-FileHash qwen2.5-3b-instruct.gguf -Algorithm SHA256
```

The expected SHA-256 is `5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6`.

## Classify a ticket

Start the model server in one terminal.

```powershell
.\llama\llama-server.exe -m .\qwen2.5-3b-instruct.gguf --alias qwen2.5-3b-instruct-q4_k_m --host 127.0.0.1 --port 18976 --device none -ngl 0
```

Run a saved classifier from a second terminal.

```powershell
.\.venv\Scripts\python.exe .\src\run.py --url http://127.0.0.1:18976/v1 predict manual_few "My ATM cash withdrawal was declined."
```

Methods are `manual_zero`, `manual_few`, `dspy_bootstrap`, `dspy_copro`, `textgrad`, and `adalflow`. Inference uses the saved classifier specification and the same renderer used for scoring. The output includes the intent, raw response, and input token count. An invalid code produces a null intent.

## Reproduce the comparison

```powershell
.\.venv\Scripts\python.exe .\src\run_server.py --server-exe .\llama\llama-server.exe --gguf qwen2.5-3b-instruct.gguf --out rerun --server-log rerun_server.log
.\.venv\Scripts\python.exe .\src\compare.py --large rerun --output RERUN_REPORT.md
```

The launcher uses six CPU threads and four request slots by default. Use `--phase fit` or `--phase test` for separate stages and `--methods` to run a subset. Use a fresh output directory when fitting. The launcher starts and shuts down the local model server.

To rebuild the frozen data, download the publisher's two Banking77 CSV files and run `python .\src\run.py prepare-data train.csv test.csv --destination rebuilt_data`. The split files use zero-based source row identifiers. Strict accuracy counts every malformed or out-of-set prediction as wrong. The report also rescores saved raw replies with a lenient leading-code parser. TextGrad and AdalFlow select against the original prompt using strict validation accuracy and break accuracy ties by macro F1.

## Project structure

Source files are in `src/`. The dataset, license, and model manifest are in `data/`.

| File | Purpose |
| --- | --- |
| `benchmark_core.py` | Dataset sampling, manual prompts, strict response parser, model transport, and token ledger |
| `optimizers.py` | Native DSPy signatures and modules, TextGrad feedback graph, and AdalFlow declarative pipeline and optimizer |
| `run.py` | Fit, benchmark, and single-ticket CLI |
| `run_server.py` | Start the pinned local server and run the larger-model benchmark |
| `report.py`, `compare.py` | Audit saved predictions and ledgers and build the comparative report |

DSPy uses a typed signature with `Predict` inside a `Module`. The benchmark requires a single-letter answer and caps generation at eight tokens, so the module has no `ChainOfThought` reasoning field.

AdalFlow runs automatic proposal generation through its native `TGDOptimizer` in an explicit `propose`/`step`/`revert` loop. The loop applies the shared validation keep-best rule and records proposal decisions and model usage directly, taking the place of `Trainer`.

## Sources

- [Banking77 publisher data](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
- [DSPy optimizers](https://dspy.ai/learn/optimization/optimizers/)
- [TextGrad project](https://github.com/zou-group/textgrad)
- [AdalFlow classification pipeline](https://github.com/SylphAI-Inc/AdalFlow/blob/main/docs/source/use_cases/classification.rst)
- [Qwen2.5-0.5B-Instruct GGUF model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
- [llama-cpp-python 0.3.19](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.3.19)
