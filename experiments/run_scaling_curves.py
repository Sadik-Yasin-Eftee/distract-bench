"""
run_scaling_curves.py

WHAT: The main data-collection loop for the inverse scaling experiment —
      for every (instance, condition, token_budget, model) combination,
      builds the prompt, queries the model, checks correctness, and
      saves the result.

WHY: This is the project's core deliverable: the 75,600-call sweep
     (600 instances x 7 conditions x 6 token budgets x 3 models) that
     produces the accuracy-vs-budget curves the whole research question
     depends on. Every other Week 2-3 file (prompt_builder.py,
     correctness.py, save_results.py, the model clients) exists to be
     called from exactly this loop.

RESEARCH STEP: Week 2-3, Step 15.

INPUT: results/benchmark/benchmark_with_distractors.json (from
       distractors/build_benchmark.py, Week 1's final output).

OUTPUT: results/scaling_curves/scaling_results.jsonl, one row appended
        per completed API call (experiments/save_results.py).

DEPENDS ON: config.py, experiments/prompt_builder.py,
            experiments/correctness.py, experiments/save_results.py,
            models/gemini_client.py, models/deepseek_client.py,
            models/groq_client.py. Read by analysis/compute_metrics.py
            and analysis/plot_curves.py.

DESIGN DECISIONS:
- Loop order is instance -> condition -> model -> token_budget (outermost
  to innermost). WHY this order specifically: instance outermost means a
  crash/restart resumes with the least wasted partial progress on the
  "current" instance's distractor context already loaded; model as the
  second-innermost (rather than outermost) means the three model clients
  are each constructed once, before the loop, and then reused across
  every instance/condition/budget rather than paying model-list
  validation calls (see e.g. GeminiClient.__init__) repeatedly.
- For the non-reasoning baseline (config.MODEL_SPECS["llama_baseline"],
  is_reasoning_model=False), the loop still iterates over every
  token_budget value and saves a row for each — even though
  models/groq_client.py documents that token_budget has no effect on the
  actual API call — because the *comparison* this baseline exists to
  support (analysis/plot_curves.py plotting a flat accuracy line across
  budgets against the two reasoning models' curves) requires the same
  number of independent samples per budget as the reasoning models have,
  not a single measurement reused six times, so that its confidence
  interval at each budget point is computed the same way.
- A ClientError/APIStatusError from *model construction* (e.g. a
  decommissioned model id — see each client's fail-fast validation) is
  allowed to propagate immediately and stop the whole script, since no
  instance/condition/budget combination for that model can succeed
  either way; a per-call exception during the loop itself is caught,
  logged with full context, and causes an immediate save-and-exit rather
  than being swallowed — per the "never silently swallow exceptions"
  engineering requirement — leaving the completed rows so far safely on
  disk for the next resumed run to pick up from.
"""

import json

import config
from experiments.correctness import check_correctness
from experiments.prompt_builder import build_prompt
from experiments.save_results import build_result_row, append_result, load_completed_keys
from models.gemini_client import GeminiClient
from models.deepseek_client import DeepSeekClient
from models.groq_client import GroqClient

logger = config.get_logger(__name__)


def _load_benchmark() -> list:
    """
    WHAT: Loads results/benchmark/benchmark_with_distractors.json.

    WHY: A dedicated loader gives a clear, specific error message if
         Week 1's distractor-generation step (distractors/build_benchmark.py)
         has not been completed yet, rather than a generic
         FileNotFoundError surfacing deep inside the experiment loop.

    Returns:
        list[dict]: The 600 instances, each with a "distractors" key.

    Raises:
        FileNotFoundError: If the benchmark file does not exist — the
                           error message names the script that produces
                           it, since this is the single most likely cause
                           of this error for someone running the weeks
                           out of order.
    """
    if not config.BENCHMARK_WITH_DISTRACTORS_PATH.exists():
        raise FileNotFoundError(
            f"{config.BENCHMARK_WITH_DISTRACTORS_PATH} not found. Run "
            f"`python -m distractors.build_benchmark` (Week 1, Step 8) first."
        )
    with open(config.BENCHMARK_WITH_DISTRACTORS_PATH) as f:
        return json.load(f)


def _build_model_clients() -> dict:
    """
    WHAT: Constructs one client per entry in config.MODEL_SPECS.

    WHY: Constructing all three clients once, up front, means each
         client's fail-fast model-id validation (see each client's
         __init__) runs before any of the 75,600 calls are attempted —
         a dead model id for any of the three fails the whole run
         immediately with a clear message, rather than after burning
         through however many instances complete before that model is
         first used in the loop.

    Returns:
        dict: {model_key (str): ModelClient instance} for every key in
              config.MODEL_SPECS.
    """
    clients = {}
    for model_key, spec in config.MODEL_SPECS.items():
        if spec["provider"] == "gemini":
            clients[model_key] = GeminiClient(spec["model_id"])
        elif spec["provider"] == "deepseek":
            clients[model_key] = DeepSeekClient(spec["model_id"])
        elif spec["provider"] == "groq":
            clients[model_key] = GroqClient(spec["model_id_candidates"])
        else:
            raise ValueError(f"Unknown provider {spec['provider']!r} for model_key={model_key!r}")
    return clients


def run_scaling_curves() -> None:
    """
    WHAT: The main experiment loop — see module docstring for the full
          (instance, condition, model, token_budget) sweep this covers.

    WHY: This is the Week 2-3 entry point — running this module directly
         (`python -m experiments.run_scaling_curves`) is how
         results/scaling_curves/scaling_results.jsonl gets populated.
         Safe to interrupt and re-run at any point: already-completed
         (instance, condition, token_budget, model_key) combinations are
         skipped via experiments.save_results.load_completed_keys().
    """
    instances = _load_benchmark()
    completed = load_completed_keys()
    clients = _build_model_clients()

    total_calls = len(instances) * len(config.CONDITIONS) * len(config.TOKEN_BUDGETS) * len(clients)
    made_calls = 0

    for instance in instances:
        for condition in config.CONDITIONS:
            if condition == "clean":
                distractor_text = None
            else:
                intensity = condition.rsplit("_", 1)[0]
                distractor_text = instance["distractors"][intensity]["text"]
            prompt = build_prompt(instance, condition, distractor_text)

            for model_key, client in clients.items():
                for token_budget in config.TOKEN_BUDGETS:
                    made_calls += 1
                    key = (instance["id"], condition, token_budget, model_key)
                    if key in completed:
                        continue

                    try:
                        response = client.query(prompt, token_budget)
                        is_correct = check_correctness(instance["task_type"], response.answer_text, instance)
                    except Exception:
                        logger.exception(
                            "Non-recoverable error on instance=%s condition=%s "
                            "model_key=%s token_budget=%d — progress saved, exiting.",
                            instance["id"], condition, model_key, token_budget,
                        )
                        raise

                    row = build_result_row(instance, condition, model_key, response, is_correct)
                    append_result(row)
                    completed.add(key)

                    if made_calls % config.SAVE_EVERY_N_CALLS == 0:
                        logger.info(
                            "Progress: %d/%d combinations attempted, %d rows saved so far",
                            made_calls, total_calls, len(completed),
                        )

    logger.info("Scaling curve experiment complete: %d rows in %s", len(completed), config.SCALING_RESULTS_PATH)


if __name__ == "__main__":
    run_scaling_curves()
