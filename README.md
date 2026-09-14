# DISTRACT-BENCH

**Research question:** does giving a large reasoning model a bigger token budget make it *more* wrong, not less, once the prompt contains a plausible but irrelevant distractor?

CSE 6201 course project, IUT, Dhaka, Bangladesh. Supervised by Dr. Hasan Mahmud, SSL Lab.

This document explains what each part of the codebase does, why it exists, how to run it, and — importantly — several places where the implementation had to deviate from the original research design because of real-world constraints (retired model APIs, structural limits of hosted inference, free-tier quotas). Every deviation is also documented inline, at the point it occurs, in the relevant `.py` file.

---

## 1. Why this project exists

Prior work (Gema et al. 2025, NoisyBench/Lee et al. 2026, Wang et al. 2025) has shown that large reasoning models get *worse* — not better — when given more "thinking" tokens in the presence of distractor text. That's the inverse scaling paradox: more compute, lower accuracy, under exactly the kind of noisy, plausible-but-irrelevant context a model encounters in real deployments (retrieved documents, tool outputs, conversation history).

Nobody has yet:
1. Plotted accuracy-vs-token-budget curves across multiple *distractor intensities*, for multiple model families, on the same benchmark.
2. Built one benchmark spanning math, factual QA, and commonsense reasoning with parametric distractor control.
3. Compared reasoning models against a matched non-reasoning baseline under the same conditions.

DISTRACT-BENCH is a small-scale (600-instance) attempt at (1) and (2), plus (3) via a non-reasoning Llama baseline, within a 5-week course-project timeline and free-tier API budgets.

---

## 2. Setup

### 2.1 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Built and tested against **Python 3.14.4** on macOS (Apple Silicon). Every dependency in `requirements.txt` is pinned to a version confirmed (via `pip index versions`) to publish wheels for 3.14 — if you're on an older Python, the pins should still work, but this wasn't tested against them.

### 2.2 API keys

```bash
cp .env.example .env
```

Then fill in `.env` with:
- `GOOGLE_API_KEY` — from [Google AI Studio](https://aistudio.google.com/). Used for the two Gemini reasoning models and the distractor generator.
- `DEEPSEEK_API_KEY` — from the [DeepSeek platform](https://platform.deepseek.com/). Used for the DeepSeek reasoning model.
- `GROQ_API_KEY` — from [Groq Console](https://console.groq.com/). Used for the non-reasoning Llama baseline.

Never commit `.env` — it's gitignored. `config.py` loads it via `python-dotenv` and every API client raises a clear `RuntimeError` naming the missing key if you forget a step.

### 2.3 Local model (for the mechanistic-analysis stage)

The attention/probe stage (§3, Stage 4) additionally downloads and runs `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` locally (~3GB). No API key needed for this — it runs on-device. See §5.1 (Known Limitations) for why this stage needs a 4th, local model at all.

### 2.4 Run the tests

```bash
python -m pytest tests/ -v
```

All test files use mocked API clients — no network access or API keys required to run them. Run this before starting any stage below, and again after touching any file in `experiments/`, `distractors/`, or `models/`.

---

## 3. Running the pipeline

Every script below is run as a module (`python -m package.module`) from the project root, not as a bare script path — this ensures `import config` and the project's other absolute imports resolve correctly regardless of your current directory. The four stages run in dependency order — each one's output feeds the next.

| Stage | Command | Produces |
|---|---|---|
| 1. Build the benchmark | `python -m data.sample_instances` | `results/benchmark/sampled_instances.json` |
| | `python -m distractors.build_benchmark` | `results/benchmark/benchmark_with_distractors.json` |
| 2. Run the scaling-curve experiment | `python -m experiments.run_scaling_curves` | `results/scaling_curves/scaling_results.jsonl` |
| 3. Compute metrics and plots | `python -m analysis.compute_metrics` | printed summary tables |
| | `python -m analysis.plot_curves` | `results/scaling_curves/plots/*.png` |
| 4. Mechanistic analysis (local model) | `python -m analysis.attention_entropy` | `results/attention/*` |
| | `python -m analysis.linear_probe` | `results/probe/*` |

### Stage 1 — Build the benchmark

`sample_instances.py` downloads GSM8K, TriviaQA, and CommonsenseQA from the HuggingFace Hub (cached under `hf_cache/` after the first run), draws 200 instances per task (600 total, seed=42), and normalizes them into one common schema.

`build_benchmark.py` is the expensive step: for each of the 600 instances, it generates 3 candidate distractors per intensity level (low/medium/high) using Gemini Flash-Lite, scores each candidate's effect on Gemini (the reference model) via repeated sampling, and keeps the strongest one per intensity. This is **~23,000 API calls** against Gemini's free tier (1,500 requests/day) — expect this step alone to take several days. It saves progress after every completed instance and safely resumes from where it left off if interrupted; just re-run the same command.

Before running it, you can sanity-check each model client standalone once your `.env` is set — each of these runs one real API call and prints the parsed response:

```bash
python -m models.gemini_client
python -m models.deepseek_client
python -m models.groq_client
```

### Stage 2 — Run the scaling-curve experiment

This is the core deliverable: 600 instances × 7 conditions × 6 token budgets × 3 models = **75,600 API calls**, spread across all three providers. Like the benchmark build, it saves incrementally (one line per completed call) and resumes automatically — safe to interrupt and restart at any time; it will skip everything already saved.

Run `python -m pytest tests/test_clients.py tests/test_correctness.py -v` before starting this — a bug in either would silently corrupt every accuracy number this stage produces.

### Stage 3 — Metrics and plots

`compute_metrics.py` turns the raw 75,600 rows into accuracy, PDR, RPD, and inversion-point tables. `plot_curves.py` renders one 300 DPI figure per (task type, model), grayscale-coded by distractor intensity per the project's color scheme (clean=black, low=light grey, medium=dark grey, high=black dashed; early/late position shown via marker shape).

### Stage 4 — Attention entropy and linear probes

**Read §5.1 before running this stage** — it uses a 4th, local, open-weight model, not the three API models from Stages 1-3. First run `python -m models.local_client` once to download the local model and confirm one smoke-test generation works.

### Not yet implemented — Mitigation ablation and write-up

RARE/DPO/prompting comparison and the paper draft are the remaining deliverables, to be added once Stages 1-4's data is in hand.

---

## 4. What each output file contains

| Path | Produced by | Contents |
|---|---|---|
| `results/benchmark/sampled_instances.json` | `data.sample_instances` | 600 normalized instances, no distractors yet |
| `results/benchmark/benchmark_with_distractors.json` | `distractors.build_benchmark` | Same 600 instances, each with a `distractors` key: best text + confidence drop per intensity |
| `results/scaling_curves/scaling_results.jsonl` | `experiments.run_scaling_curves` | One JSON object per API call: instance, condition, budget, model, answer, correctness, timing |
| `results/scaling_curves/plots/*.png` | `analysis.plot_curves` | One scaling-curve figure per (task type, model) |
| `results/attention/*` | `analysis.attention_entropy` | Entropy comparison data/plots and attention heatmaps (local model only) |
| `results/probe/*` | `analysis.linear_probe` | Per-layer probe metrics and t-SNE plots (local model only) |
| `results/experiment.log` | every script, via `config.get_logger` | Timestamped log of every run, across all stages |

## 5. Known limitations (and why)

These aren't oversights — each one is a documented, deliberate response to a real constraint discovered while building this project. The corresponding `.py` file has the full reasoning inline; this is the summary.

### 5.1 The mechanistic-analysis stage uses a different, smaller model

Gemini, DeepSeek, and Groq are all **hosted inference APIs** — a chat-completions endpoint returns text, never attention weights, hidden states, or gradients, for any provider. This is structural, not a library limitation. The attention-entropy and linear-probe stage (§3, Stage 4) therefore runs against a 4th model, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, downloaded and run **locally** via `transformers` (see `models/local_client.py`). It's a distillation of DeepSeek-R1 — the same lineage as one of the three main models — but it is genuinely a different, much smaller (1.5B parameter) model. **This stage's mechanistic findings describe the local model's internals, not Gemini's or the full DeepSeek-R1 API model's** — this should be stated explicitly wherever these results are reported in the paper.

### 5.2 Distractors are adaptive to one reference model only

The adaptive distractor-selection method (Wang et al. 2025) scores candidate distractors by their measured effect on a target model. Scoring against all three models separately would triple the already-expensive scoring API budget (~23,000 calls) to ~70,000+ calls just for the benchmark-build stage. Instead, every distractor is scored against Gemini only (`config.DISTRACTOR_SCORING_MODEL_KEY`) and the same winning text is reused for all three models in the main experiment. A distractor optimized against Gemini may be a weaker or stronger test for DeepSeek/Llama than a model-specific one would be — this should be reported as a limitation when comparing cross-model results.

### 5.3 "Confidence drop" is measured empirically, not via token probabilities

None of the three provider APIs expose usable per-token log-probabilities for a multi-step chain-of-thought answer in a way that cleanly gives P(correct answer). Instead, `distractors/scorer.py` measures confidence as the fraction of `config.NUM_SCORING_SAMPLES` (3) independent samples that are correct — the same correctness check used everywhere else in the project. This is a standard self-consistency-style proxy, but it is coarser-grained than a true softmax probability.

### 5.4 Model IDs are not what the original research design specified

At project start (2026-09-14), every model the original design named was already retired or scheduled to retire mid-project:

- **Gemini 2.0 Flash Thinking** — fully shut down (2026-06-01).
- **Gemini 2.5 Flash** — scheduled to shut down 2026-10-16, about a month into this project.
- **`google.generativeai` (the Python package)** — deprecated by Google, and its `GenerationConfig` never had a `thinking_budget` field at all.
- **`gemini-1.5-flash`** — fully shut down.
- **DeepSeek's `deepseek-reasoner`** — retired 2026-07-24; the current model (`deepseek-flash`) controls thinking via a qualitative `reasoning_effort` parameter, not a numeric budget.
- **Groq's `llama-3.3-70b-versatile`** — decommissioned 2026-08-16.

`config.py`'s `MODEL_SPECS` documents each substitution inline. The Groq baseline in particular resolves its model id **at runtime** against Groq's live model list (`models/groq_client.py`), rather than trusting a hardcoded id, because Groq has been deprecating models roughly monthly throughout 2026 — a second hardcoded guess would face the same risk. **Before running the experiment stage, re-verify these model ids are still current** (provider deprecation pages change fast); the fail-fast validation in each client will raise a clear error naming the problem if not.

### 5.5 DeepSeek's "token budget" is a `max_tokens` ceiling, not a thinking-specific cap

Unlike Gemini's `thinking_budget` (which only caps reasoning, never the final answer), DeepSeek's current API has no numeric reasoning-specific cap. `models/deepseek_client.py` uses `max_tokens` as the budget control instead — a DeepSeek response cut off mid-reasoning has no room left to write an answer at all, which is a stronger and qualitatively different truncation effect than Gemini's. Keep this in mind interpreting DeepSeek's low-budget data points.

### 5.6 GSM8K uses the `test` split, not `validation`

GSM8K ships only `train` and `test` splits — there is no `validation` split to draw from. `test` (the standard held-out evaluation split for GSM8K in essentially every paper) is used instead, playing the same contamination-avoidance role `validation` plays for TriviaQA and CommonsenseQA.

---

## 6. References

- Gema, A. et al. (2025). Extended reasoning length degrades accuracy under distractors in Large Reasoning Models.
- Lee, C. et al. (2026). NoisyBench: measuring performance drops under contextual noise.
- Wang, Y. et al. (2025). Adaptive tree-search distractor generation; RARE/DPO/prompting-based mitigation comparison.
- Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. (Attention rollout, used in `analysis/attention_entropy.py`.)
- Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems. (GSM8K.)
