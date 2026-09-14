"""
config.py

WHAT: Single source of truth for every path, hyperparameter, and constant
      used across DISTRACT-BENCH.

WHY: This project runs 75,600 API calls across 5 weeks, driven by many
     separate scripts (data sampling, distractor generation, the scaling
     curve experiment, analysis). If each script hardcoded its own copy of
     e.g. TOKEN_BUDGETS or the random seed, a change in one place (say,
     dropping the 8192 budget because of rate limits in week 3) could
     silently desync from the value used to build the benchmark in week 1,
     corrupting every downstream comparison. Centralizing every constant
     here, with the reasoning for its value recorded next to it, means a
     future reader (or the supervisor, Dr. Hasan Mahmud) can audit every
     experimental choice from one file.

RESEARCH STEP: Week 1, Step 2. Read by every other module in the project.

INPUT: The .env file (via python-dotenv) for API keys. No other input.

OUTPUT: Importable Python constants. Also creates the results/ directory
        tree on import so downstream scripts can assume it exists.

DEPENDS ON: Imported by every module in data/, distractors/, models/,
            experiments/, and analysis/. Depends only on python-dotenv
            and the standard library.

DESIGN DECISIONS:
- All paths are computed relative to this file's location (PROJECT_ROOT),
  not the current working directory, so scripts behave identically
  whether run as `python data/sample_instances.py` from the repo root or
  `python sample_instances.py` from inside data/.
- Values that come directly from the project's research design document
  (token budgets, distractor intensities, sample sizes) are recorded here
  verbatim with a citation to the motivating paper where one exists.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env before reading any API key below. `override=False` (the
# default) means real shell-exported environment variables always win over
# .env — useful when running on a shared machine where a key is already
# exported.
load_dotenv()

# ────────────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────────────

# Root of the repository, resolved from this file's own location rather
# than os.getcwd(). WHY: every other path below is derived from this, so
# getting it wrong once here (e.g. via a relative "." path) would silently
# break every script run from a different working directory.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_CACHE_DIR = PROJECT_ROOT / "hf_cache"
# WHY a dedicated cache dir instead of the default ~/.cache/huggingface:
# keeps the ~600MB+ of downloaded dataset shards inside the project
# directory, so `git clean` / moving the project / grading on a different
# machine doesn't leave orphaned multi-GB caches in the grader's home dir.

RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_DIR = RESULTS_DIR / "benchmark"        # Week 1 output
SCALING_CURVES_DIR = RESULTS_DIR / "scaling_curves"  # Week 3 output
ATTENTION_DIR = RESULTS_DIR / "attention"        # Week 2 & 4 output
PROBE_DIR = RESULTS_DIR / "probe"                # Week 4 output
LOG_FILE = RESULTS_DIR / "experiment.log"

SAMPLED_INSTANCES_PATH = BENCHMARK_DIR / "sampled_instances.json"
BENCHMARK_WITH_DISTRACTORS_PATH = BENCHMARK_DIR / "benchmark_with_distractors.json"
SCALING_RESULTS_PATH = SCALING_CURVES_DIR / "scaling_results.jsonl"
# WHY .jsonl (one JSON object per line) for the 75,600-row scaling result
# table specifically, while the benchmark itself is a single .json array:
# the scaling experiment must support incremental append-and-resume (see
# INCREMENTAL SAVING below). Appending a line to a .jsonl file is O(1) and
# crash-safe; appending an element to a single large JSON array requires
# rewriting the whole file, which is both slow at 75,600 rows and unsafe
# if the process is killed mid-write.

# Create the results tree eagerly on import so every script (including a
# freshly cloned repo) can assume these directories exist without each one
# repeating `os.makedirs(..., exist_ok=True)`.
for _dir in (DATA_CACHE_DIR, BENCHMARK_DIR, SCALING_CURVES_DIR, ATTENTION_DIR, PROBE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ────────────────────────────────────────────────────────────────────────

# Fixed seed used for every random operation in the project: dataset
# sampling (data/sample_instances.py), train/test splitting in the linear
# probe (analysis/linear_probe.py), and sklearn's LogisticRegression.
# WHY 42: no significance beyond being the project's single fixed seed,
# used everywhere so that "seed=42" unambiguously identifies the exact
# sample of 600 instances and the exact probe train/test split used in
# the paper — required for another student (or Dr. Mahmud) to reproduce
# figures from a re-run.
RANDOM_SEED = 42

# ────────────────────────────────────────────────────────────────────────
# API KEYS (never hardcode — read from environment only)
# ────────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ────────────────────────────────────────────────────────────────────────
# DATASETS (Week 1, Steps 3-5)
# ────────────────────────────────────────────────────────────────────────

# Held-out split used for every dataset, to avoid drawing instances from
# the split most likely to have been seen (in some paraphrased or exact
# form) during a target model's own training or RLHF pipeline.
# WHY per-task rather than one global "validation" split: GSM8K ships only
# "train" and "test" splits (confirmed by inspecting the loaded dataset —
# no "validation" split exists). GSM8K's "test" split is the standard
# held-out evaluation split used by essentially every GSM8K paper, so it
# plays the same contamination-avoidance role "validation" plays for
# TriviaQA and CommonsenseQA (both of which do ship a real validation
# split, confirmed the same way). DEVIATION FROM INITIAL SPEC: the
# project design said "validation split" for all three datasets; GSM8K
# has no such split, so "test" is substituted for math only.
DATASET_SPLIT_BY_TASK = {
    "math": "test",
    "factual": "validation",
    "commonsense": "validation",
}

# Number of instances sampled per task type, and the resulting total.
# WHY 200: large enough for the accuracy metrics per condition (200
# instances x 7 conditions x 6 budgets = 8,400 calls per model per task)
# to have a usable confidence interval on a proportion (a 95% CI on a
# binomial proportion at n=200 has half-width ~7pp at p=0.5, tightening
# as accuracy moves toward 0 or 1) while keeping the total call budget
# (75,600) achievable on free-tier rate limits within the 5-week plan.
N_PER_TASK = 200
TASK_TYPES = ("math", "factual", "commonsense")
TOTAL_INSTANCES = N_PER_TASK * len(TASK_TYPES)  # 600

# HuggingFace dataset identifiers, one entry per task type.
# "hf_name"/"hf_config" are passed directly to datasets.load_dataset().
DATASET_SPECS = {
    "math": {
        "hf_name": "openai/gsm8k",
        # DEVIATION FROM INITIAL SPEC: named "gsm8k" in the project design.
        # The bare "gsm8k" repo id is a legacy (pre-namespace) HF Hub
        # dataset id; the installed datasets==5.0.1 / huggingface_hub==1.31.0
        # pair raises HfUriError on legacy non-namespaced ids (confirmed by
        # running load_dataset("gsm8k", "main") directly — see
        # tests/test_clients.py for the regression check). "openai/gsm8k"
        # is the current canonical namespaced mirror of the same data.
        "hf_config": "main",
        # WHY "main": GSM8K's only config; contains the standard
        # (question, answer) pairs with the final numeric answer after
        # a "####" delimiter in the reasoning chain.
    },
    "factual": {
        "hf_name": "mandarjoshi/trivia_qa",
        # DEVIATION FROM INITIAL SPEC: named "trivia_qa" in the project
        # design; same legacy-repo-id issue as gsm8k above. Retargeted to
        # the canonical namespaced mirror.
        "hf_config": "rc.nocontext",
        # DEVIATION FROM INITIAL SPEC: the project design named config
        # "rc". "rc" ships each question bundled with its full retrieved
        # web/Wikipedia supporting documents (~936MB for the validation
        # split alone, confirmed via the HF Hub API). This project's
        # normalized schema discards that field (`context` is set to ""
        # for TriviaQA — see data/normalize.py) because our own generated
        # distractors, not the dataset's bundled documents, are what gets
        # injected into the prompt. "rc.nocontext" is the same questions
        # with the same `answer.value`/`answer.aliases` fields but without
        # the supporting documents (~7MB for validation) — identical data
        # for every field this project actually reads, at ~130x less
        # download/storage cost.
    },
    "commonsense": {
        "hf_name": "tau/commonsense_qa",
        # DEVIATION FROM INITIAL SPEC: named "commonsense_qa" in the
        # project design; same legacy-repo-id issue as gsm8k above.
        # Retargeted to the canonical namespaced mirror maintained by TAU
        # (the dataset's original authors).
        "hf_config": None,
        # WHY None: commonsense_qa has a single default config.
    },
}

# ────────────────────────────────────────────────────────────────────────
# DISTRACTOR GENERATION (Week 1, Steps 6-8)
# ────────────────────────────────────────────────────────────────────────

DISTRACTOR_INTENSITIES = ("low", "medium", "high")
INJECTION_POSITIONS = ("early", "late")

# Expected confidence-drop ranges per intensity, from the research design.
# WHY recorded here: used by distractors/scorer.py as a sanity-check log
# (not a hard filter) — if a "high" intensity candidate produces a smaller
# confidence drop than this range, that is logged as a warning because it
# suggests the generator produced a weak distractor for that instance,
# which is useful to know when manually spot-checking Week 1 output.
EXPECTED_CONFIDENCE_DROP = {
    "low": (0.02, 0.08),
    "medium": (0.10, 0.20),
    "high": (0.25, 0.50),
}

# Number of candidate distractors generated per intensity level before
# picking the one with the highest measured confidence drop.
# WHY 3: this is the adaptive tree-search width from Wang et al. 2025 —
# their method generates multiple candidates and keeps the most effective
# one per instance rather than accepting the generator's first attempt,
# since a single LLM-generated distractor can vary widely in how
# effectively it distracts the *target* model for a *specific* question.
NUM_DISTRACTOR_CANDIDATES = 3

# Model used to *generate* candidate distractor text (not a model under
# evaluation). WHY Gemini Flash-Lite specifically: fast, free-tier, and
# cheaper than the target Gemini models it feeds into — appropriate for a
# role that only needs to produce short paragraphs, not multi-step
# reasoning. DEVIATION FROM INITIAL SPEC: named "gemini-1.5-flash" in the
# project design; confirmed via web search (2026-09-14) that all Gemini
# 1.0 and 1.5 models are fully shut down (404 on every request).
# Retargeted to gemini-3.5-flash-lite (released 2026-07-21, no shutdown
# date announced).
DISTRACTOR_GENERATOR_MODEL = "gemini-3.5-flash-lite"

# Thinking budget used for distractor-generation calls specifically (NOT
# config.TOKEN_BUDGETS, which is the experiment's independent variable).
# WHY 0 (thinking disabled): the generator's task is to paraphrase a
# short 2-3 sentence paragraph from a template, not to solve a problem —
# spending thinking tokens on this role would only add latency/quota cost
# for a task that doesn't benefit from extended reasoning.
GENERATOR_TOKEN_BUDGET = 0

# ────────────────────────────────────────────────────────────────────────
# DISTRACTOR SCORING — measuring "confidence drop" without token logprobs
# ────────────────────────────────────────────────────────────────────────
#
# WHY THIS SECTION EXISTS: the project design's scorer.py step says to
# "measure confidence drop on the correct answer," implying access to the
# model's probability of the correct answer token(s). None of this
# project's three providers expose usable token-level logprobs for a
# multi-step chain-of-thought answer through their chat/generation APIs
# in a way that cleanly marginalizes over reasoning paths to a single
# P(correct) number. Instead, "confidence" is operationalized empirically
# via repeated sampling: query the target model N times at a fixed
# budget and measure the fraction of samples that are correct. This is a
# standard self-consistency-style confidence proxy, and — critically —
# it is measured with exactly the same check_correctness() function
# (experiments/correctness.py) used everywhere else in the project, so
# "confidence" and "accuracy" are the same underlying measurement at
# different sample sizes (accuracy at n=1 per instance across many
# instances vs. this "confidence" at n=NUM_SCORING_SAMPLES per instance).

# Reference model that candidate distractors are scored against.
# WHY a single model rather than scoring separately per target model
# (which the "adaptive, tailored per target model" framing of Wang et al.
# 2025 would imply): scoring against all three target models would
# require 3x the scoring API calls and a separate distractor benchmark
# per model, which is not achievable within this project's free-tier
# quota (Gemini: 1500 requests/day) and 5-week timeline. This is a
# documented limitation, not an oversight — see README.md's "Known
# Limitations" section — and should be reported as such in the paper:
# distractors are "adaptive" with respect to gemini_primary only, then
# reused unchanged for deepseek_reasoning and llama_baseline, so any
# cross-model comparison must account for the possibility that a
# distractor optimized against Gemini is a weaker (or stronger) test for
# the other two models than a model-specific one would be.
DISTRACTOR_SCORING_MODEL_KEY = "gemini_primary"

# Number of independent samples drawn per confidence measurement.
# WHY 3: balances measurement granularity (n=1 can only measure
# confidence as 0% or 100%, which cannot distinguish between three "high"
# intensity candidates that are all wrong) against API quota — this
# scoring step already multiplies out to a non-trivial number of extra
# calls (see distractors/scorer.py's module docstring for the exact
# count), and n=3 gives 4 distinguishable confidence levels (0, 1/3, 2/3,
# 1) at 3x rather than 5x+ the call cost of n=1.
NUM_SCORING_SAMPLES = 3

# Fixed token budget used only for scoring calls (not swept). WHY a
# single mid-range value rather than sweeping budgets during scoring:
# scoring exists only to *rank* 3 candidate distractors relative to each
# other for a given instance, not to characterize the scaling curve
# itself (that is what the main experiment, using the full
# config.TOKEN_BUDGETS sweep, is for) — one representative budget is
# sufficient for relative ranking and keeps scoring cost independent of
# the number of budgets swept later.
SCORING_TOKEN_BUDGET = 1024

# Injection position used when scoring candidates. WHY "early" only
# (rather than scoring both positions and averaging): the scorer's job is
# to pick the better *distractor text* for a given intensity, not the
# better position — position is a separate, orthogonal condition that the
# main experiment already sweeps for every instance regardless of which
# text was selected here. Scoring only one representative position halves
# the scoring call budget without narrowing what the main experiment
# actually measures.
SCORING_POSITION = "early"

# ────────────────────────────────────────────────────────────────────────
# LOCAL MODEL FOR ATTENTION / HIDDEN-STATE ANALYSIS (Week 4, Steps 19-20)
# ────────────────────────────────────────────────────────────────────────
#
# WHY THIS SECTION EXISTS AT ALL: the project design's Week 4 steps
# (attention entropy, linear probes) require raw attention weight
# matrices and intermediate hidden-state vectors. None of this project's
# three API-based models (Gemini, DeepSeek, Groq) expose either through
# their hosted chat-completions endpoints — this is a structural property
# of calling a hosted inference API, not something any client-side code
# change can work around. Confirmed while building models/gemini_client.py,
# models/deepseek_client.py, and models/groq_client.py (Week 2), and
# raised with the user before Week 4 work began. Per the user's decision,
# a 4th, local, open-weight model is loaded directly via `transformers`
# and run on-device (Apple Silicon MPS backend) for Week 4 analysis ONLY.
# The main scaling-curve experiment (config.MODEL_SPECS, Weeks 2-3) is
# completely unaffected and still uses only the three API models.
LOCAL_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# WHY this specific model: (a) open-weight and small enough (1.5B
# parameters, ~28 transformer layers per its published config) to run
# fully on-device on the project's available hardware (Apple M4 Pro, 24GB
# unified memory) without quantization; (b) it is a distillation of
# DeepSeek-R1 itself — the same reasoning-model *lineage* already used as
# this project's second API-based model family (models/deepseek_client.py)
# — so Week 4's mechanistic findings are at least drawn from a model
# related to one already central to the study, rather than an unrelated
# architecture; (c) 28 layers comfortably covers the three probe depths
# the research design specifies (layers 8, 16, 24 — see
# LINEAR_PROBE_LAYERS below).
# LIMITATION (see README.md's Known Limitations for the full statement):
# this is a genuinely different, much smaller model than gemini_primary
# or the deepseek_reasoning API model — Week 4's attention/probe findings
# describe THIS model's internals, and should not be presented as
# describing Gemini's or the full DeepSeek-R1 API model's internals.

# Number of sample instances used for Week 4's attention visualization
# and linear probe analysis (out of the full 600). WHY 50, matching the
# project design's Week 2 deliverable ("attention visualizations for 50
# sample instances"): running the local model with eager attention
# (required to materialize attention weight tensors at all — see
# analysis/attention_entropy.py) and saving hidden states at 3 layers is
# far more memory- and disk-intensive per instance than an API call, so
# the full 600-instance benchmark is not used for this analysis; 50 is
# the number the original research design itself specifies for this
# exact deliverable.
LOCAL_ANALYSIS_SAMPLE_SIZE = 50

# Which decoder layer indices to extract hidden states from for the
# linear probe, per the project design's explicit instruction ("Save
# hidden states at layers 8, 16, 24 (early, mid, late depth)").
LINEAR_PROBE_LAYERS = (8, 16, 24)

# Max new tokens generated by the local model during Week 4 analysis.
# WHY a separate, small constant rather than reusing config.TOKEN_BUDGETS:
# Week 4's attention/probe analysis is not the scaling-budget sweep
# itself (that used the three API models, Weeks 2-3) — it only needs one
# representative generation per instance, at a length short enough to
# keep per-instance attention-tensor memory bounded (attention tensors
# scale with sequence length squared per layer per head).
LOCAL_MAX_NEW_TOKENS = 256

# The 7 experimental conditions: clean, plus each of 3 intensities crossed
# with each of 2 injection positions (3 x 2 + 1 = 7).
CONDITIONS = ("clean",) + tuple(
    f"{intensity}_{position}"
    for intensity in DISTRACTOR_INTENSITIES
    for position in INJECTION_POSITIONS
)

# ────────────────────────────────────────────────────────────────────────
# TOKEN BUDGETS — the x-axis of every inverse scaling curve
# ────────────────────────────────────────────────────────────────────────

# Token budgets for the inverse scaling curve experiment. Range 256-8192,
# doubling each step.
# WHY these values: covers the range where Gema et al. 2025 observed
# inversion (accuracy decreasing as allotted reasoning length increases)
# in frontier reasoning models under distractor noise.
# WHY stop at 8192: above this, free-tier rate limits (Gemini: 1500
# requests/day; see RATE LIMITS below) make the full 75,600-call design
# impractical within the 5-week plan — each doubling of the top budget
# would add another full sweep's worth of calls without materially
# widening the range where prior work reports inversion.
# WHY start at 256: below this, Gemini Thinking was observed in early
# testing to truncate reasoning chains before producing a final answer,
# which would conflate "too little budget to finish" with the inverse
# scaling effect this project is trying to isolate.
TOKEN_BUDGETS = (256, 512, 1024, 2048, 4096, 8192)

# ────────────────────────────────────────────────────────────────────────
# MODELS (Week 2, Steps 9-12)
# ────────────────────────────────────────────────────────────────────────

MODEL_SPECS = {
    "gemini_primary": {
        "provider": "gemini",
        "model_id": "gemini-3.5-flash",
        "is_reasoning_model": True,
        # DEVIATION FROM INITIAL SPEC — IMPORTANT: the project design
        # named "Gemini 2.0 Flash Thinking" (model id
        # gemini-2.0-flash-thinking-exp) as the primary reasoning model.
        # Verified via web search (2026-09-14) that this experimental
        # model, and the entire Gemini 2.0 Flash generation, were already
        # shut down (2026-06-01). The design's secondary model, Gemini
        # 2.5 Flash, is also confirmed scheduled for shutdown on
        # 2026-10-16 — about one month from this project's start, which
        # would break the experiment mid-run (Week 3-5) if used. Retargeted
        # to gemini-3.5-flash (released 2026-05-19, no shutdown date
        # announced as of 2026-09-14) as the primary model. It also uses
        # the new `google-genai` SDK's ThinkingConfig.thinking_budget
        # field (see models/gemini_client.py) rather than the
        # `google.generativeai` package named in the original spec, which
        # has no thinking-budget support at all in its GenerationConfig
        # (confirmed by inspecting the installed package) and is itself
        # fully deprecated by its maintainers as of this project's start.
        "requests_per_minute": 60,
        "requests_per_day": 1500,
    },
    "gemini_secondary": {
        "provider": "gemini",
        "model_id": "gemini-3.6-flash",
        "is_reasoning_model": True,
        # DEVIATION FROM INITIAL SPEC: same reasoning as gemini_primary
        # above. gemini-3.6-flash (released 2026-07-21, no shutdown date
        # announced) replaces "Gemini 2.5 Flash" as the intra-family
        # comparison point — the research purpose is unchanged (checking
        # whether inverse scaling persists across a same-family model
        # upgrade), only the specific model generation being compared.
        "requests_per_minute": 60,
        "requests_per_day": 1500,
    },
    "deepseek_reasoning": {
        "provider": "deepseek",
        "model_id": "deepseek-flash",
        "is_reasoning_model": True,
        # DEVIATION FROM INITIAL SPEC: the project design named
        # "deepseek-reasoner". Verified via web search (2026-09-14) that
        # DeepSeek retired the legacy "deepseek-chat"/"deepseek-reasoner"
        # aliases on 2026-07-24; both now point through to DeepSeek-V4.1-
        # Flash under the current model id "deepseek-flash". Its thinking
        # mode is also now controlled by a qualitative `reasoning_effort`
        # parameter (low/high/max) rather than a raw token count — see
        # models/deepseek_client.py's DESIGN DECISIONS for how
        # TOKEN_BUDGETS is still applied as a hard `max_tokens` ceiling
        # so the budget sweep remains a numeric, comparable variable
        # across all three model families.
        # WHY DeepSeek at all: second model *family* (not just second
        # model) — an open-weight model trained with RLVR (reinforcement
        # learning from verifiable rewards) rather than Google's training
        # pipeline, needed to claim the inverse scaling effect
        # generalizes across training approaches, not just within
        # Gemini.
        "requests_per_minute": 60,
        "requests_per_day": None,  # DeepSeek's free tier is not a hard daily cap.
    },
    "llama_baseline": {
        "provider": "groq",
        "model_id": None,  # Resolved at runtime — see model_id_candidates below.
        "model_id_candidates": (
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        # DEVIATION FROM INITIAL SPEC: the project design named
        # "llama-3.3-70b-versatile" as a fixed model id. Verified via web
        # search (2026-09-14) that Groq decommissioned
        # llama-3.3-70b-versatile on 2026-08-16, and that Groq has been
        # deprecating models on a roughly monthly cadence throughout 2026
        # — a second hardcoded guess would be just as likely to be dead
        # by the time this runs. models/groq_client.py instead queries
        # Groq's live model list at client construction and picks the
        # first entry of `model_id_candidates` that is actually being
        # served, raising a clear, actionable error (naming what *is*
        # available) if none of them are — this is a general defense
        # against Groq's decommission cadence, not just a one-time fix
        # for llama-3.3-70b-versatile specifically.
        "is_reasoning_model": False,
        # WHY non-reasoning baseline matters: every candidate above is an
        # instruct-tuned model with no explicit thinking-token mechanism,
        # so token "budget" for this model is necessarily a no-op (see
        # models/groq_client.py). Its clean vs. noisy accuracy across the
        # same distractor conditions is the control that lets us claim
        # inverse scaling is specific to models that spend extra compute
        # on explicit reasoning, not a generic property of "any LLM given
        # more output tokens." If Groq's live roster ever leaves only
        # reasoning-capable models (e.g. openai/gpt-oss-*) as candidates,
        # this control breaks and model_id_candidates must be revisited.
        "requests_per_minute": 30,
        "requests_per_day": 14400,
    },
}

# ────────────────────────────────────────────────────────────────────────
# RATE LIMIT HANDLING / RETRY (applies to every API client)
# ────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 5
# WHY 5: enough to ride out a transient rate-limit window (free tiers
# reset per-minute) without a single flaky instance stalling the whole
# 75,600-call run for an unbounded number of attempts.

MAX_BACKOFF_SECONDS = 60
# WHY 60: matches the coarsest rate-limit reset window across all three
# providers (per-minute limits) — waiting longer than one reset window
# before giving up on an instance would just be wasted time within a
# multi-day scheduled run.

BACKOFF_BASE_SECONDS = 1
# Exponential backoff: wait = min(MAX_BACKOFF_SECONDS, BACKOFF_BASE_SECONDS * 2**attempt).
# WHY exponential starting at 1s: a fixed short retry delay (e.g. always
# 1s) would re-trigger the same per-minute rate limit repeatedly in a
# burst; exponential growth spreads retries out automatically without
# needing to know the exact remaining quota.

# ────────────────────────────────────────────────────────────────────────
# INCREMENTAL SAVING
# ────────────────────────────────────────────────────────────────────────

SAVE_EVERY_N_CALLS = 20
# WHY 20: at ~1 call per few seconds, 20 calls is roughly 1-2 minutes of
# work — small enough that a crash loses only a couple of minutes of API
# calls (which cost quota, not money, but quota is the scarce resource on
# free tiers), large enough that per-call disk I/O overhead is negligible
# against 75,600 total calls spread across weeks 2-3.

# ────────────────────────────────────────────────────────────────────────
# LOGGING
# ────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    WHAT: Returns a module-level logger that writes to both the console
          and results/experiment.log.

    WHY: The engineering requirements call for logging (not print) so
         that a multi-day, multi-week experiment run leaves a persistent,
         timestamped record of every API call's outcome — needed both to
         debug failures after the fact and to resume a crashed run by
         checking which instances already have logged results.

    Args:
        name (str): Usually __name__ of the calling module, so log lines
                     are attributable to the file that emitted them.

    Returns:
        logging.Logger: Configured with a console handler and a file
                         handler pointed at LOG_FILE, both using the same
                         timestamped format.

    Note:
        Safe to call repeatedly with the same name (e.g. if a module is
        re-imported) — guards against attaching duplicate handlers, which
        would otherwise duplicate every log line.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (e.g. this module was imported more than
        # once in the same process) — avoid attaching duplicate handlers,
        # which would otherwise print/write every log line multiple times.
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
