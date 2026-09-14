"""
save_results.py

WHAT: Append-only saving and crash-resume lookup for the scaling-curve
      experiment's per-call results.

WHY: The main experiment (experiments/run_scaling_curves.py) makes 75,600
     API calls across weeks 2-3. The engineering requirements mandate
     append-mode saving (never overwrite completed results) and the
     ability to skip already-completed work after a crash/restart — this
     file is the one place that logic lives, so run_scaling_curves.py's
     main loop can call `already_done(...)` and `append_result(...)`
     without re-implementing file I/O itself.

RESEARCH STEP: Week 2-3, Step 16.

INPUT: One result dict per call (see `build_result_row` for the schema).

OUTPUT: results/scaling_curves/scaling_results.jsonl — one JSON object
        per line, appended incrementally.

DEPENDS ON: config.py (output path, save-every-N constant). Used by
            experiments/run_scaling_curves.py. Read by
            analysis/compute_metrics.py and analysis/plot_curves.py.

DESIGN DECISIONS:
- Saving to JSON Lines (.jsonl), not a single JSON array or CSV.
  WHY JSONL over a single JSON array: appending a line is O(1) and safe
  to do after every single API call without rewriting the whole file —
  required for the "save every N calls, never overwrite" requirement at
  75,600 total rows, where rewriting a growing JSON array on every save
  would mean the last save alone rewrites nearly the entire file.
  WHY JSONL over CSV: each row already carries a nested `raw_metadata`
  dict (provider-specific fields — see models/base_client.py's
  ModelResponse) that does not flatten cleanly into CSV columns without
  either lossy flattening or a separate side-table; JSON preserves it
  exactly as returned by the model client.
- Resume/dedup keys on (instance_id, condition, token_budget, model_key)
  — the four values that fully identify one intended API call in the
  experiment design (config.CONDITIONS x config.TOKEN_BUDGETS x
  config.MODEL_SPECS x the 600 instances) — rather than on some derived
  row hash, so a completed-work lookup is a direct, readable tuple
  membership check.
"""

import json
import time

import config

logger = config.get_logger(__name__)


def build_result_row(
    instance: dict,
    condition: str,
    model_key: str,
    model_response,
    is_correct: bool,
) -> dict:
    """
    WHAT: Assembles one saved result row from an experiment call's inputs
          and outputs.

    WHY: A dedicated builder (rather than constructing the dict inline in
         run_scaling_curves.py) keeps the saved schema defined in exactly
         one place — every consumer (analysis/compute_metrics.py,
         analysis/plot_curves.py) can rely on this exact key set.

    Args:
        instance (dict): The normalized instance dict this call was made
                          for. Only `id` and `task_type` are stored (not
                          the full instance) to avoid repeating the same
                          question/answer text across every one of the
                          252 rows (7 conditions x 6 budgets x 3 models
                          - see config.py) this one instance generates —
                          the full instance is always recoverable by
                          joining on `instance_id` back to
                          results/benchmark/benchmark_with_distractors.json.
        condition (str): One of config.CONDITIONS.
        model_key (str): One of config.MODEL_SPECS's keys (e.g.
                          "gemini_primary") — the project's own stable
                          identifier for a model role, kept separate from
                          `model_response.model_id` (the provider's
                          actual model string) since the latter can
                          change if config.MODEL_SPECS is updated
                          mid-project (e.g. a Groq baseline model being
                          re-resolved after a deprecation — see
                          models/groq_client.py) while `model_key`'s
                          meaning ("the non-reasoning baseline role")
                          stays stable across that change.
        model_response (models.base_client.ModelResponse): The result of
                                                             the API call.
        is_correct (bool): The result of
                            experiments/correctness.check_correctness()
                            for this call.

    Returns:
        dict: One row, ready to be passed to append_result().
    """
    return {
        "instance_id": instance["id"],
        "task_type": instance["task_type"],
        "condition": condition,
        "token_budget": model_response.token_budget,
        "model_key": model_key,
        "model_id": model_response.model_id,
        "answer_text": model_response.answer_text,
        "reasoning_text": model_response.reasoning_text,
        "is_correct": is_correct,
        "elapsed_seconds": model_response.elapsed_seconds,
        "raw_metadata": model_response.raw_metadata,
        "saved_at": time.time(),
    }


def append_result(result: dict) -> None:
    """
    WHAT: Appends one result row as a single JSON line to
          config.SCALING_RESULTS_PATH.

    WHY: Kept as a one-line function (rather than inlined at call sites)
         so every append goes through the same open-append-close
         sequence — opening in "a" mode and closing immediately after
         each write (rather than holding the file open for the whole
         75,600-call run) means a crash at any point leaves the file in
         a valid, fully-flushed state with no risk of a partially
         buffered write being lost.

    Args:
        result (dict): One row, normally from build_result_row().
    """
    with open(config.SCALING_RESULTS_PATH, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_completed_keys() -> set[tuple[str, str, int, str]]:
    """
    WHAT: Reads every already-saved result row and returns the set of
          (instance_id, condition, token_budget, model_key) tuples they
          cover.

    WHY: This is the resume mechanism required by the engineering
         requirements — experiments/run_scaling_curves.py calls this once
         at startup and checks membership before making each API call,
         so a restarted run after a crash skips every call already
         completed rather than re-spending API quota on them.

    Returns:
        set[tuple[str, str, int, str]]: Empty set if
                                        config.SCALING_RESULTS_PATH does
                                        not exist yet (first run).
    """
    if not config.SCALING_RESULTS_PATH.exists():
        return set()

    completed = set()
    with open(config.SCALING_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed.add((row["instance_id"], row["condition"], row["token_budget"], row["model_key"]))
    return completed
