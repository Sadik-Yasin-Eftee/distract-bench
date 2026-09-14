"""
build_benchmark.py

WHAT: Orchestrates distractor generation (generator.py) and scoring
      (scorer.py) across all 600 sampled instances, producing the final
      Week 1 deliverable: every instance annotated with its best
      adaptive distractor text per intensity level.

WHY: generator.py and scorer.py each operate on one (instance, intensity)
     pair at a time — this file is the loop that drives them across the
     full benchmark, with the incremental saving and crash-resume
     behavior the engineering requirements call for, since this is the
     single most API-call-expensive step in the entire project (see
     scorer.py's module docstring: ~30 scoring calls + 9 generation calls
     per instance, ~23,400 total for 600 instances).

RESEARCH STEP: Week 1, Step 8. Final step of Week 1.

INPUT: results/benchmark/sampled_instances.json (from
       data/sample_instances.py).

OUTPUT: results/benchmark/benchmark_with_distractors.json — the same 600
        instances, each with an added "distractors" key:
        {"low": {"text": str, "confidence_drop": float},
         "medium": {...}, "high": {...}}

DEPENDS ON: config.py, distractors/generator.py, distractors/scorer.py,
            models/gemini_client.py, experiments/prompt_builder.py (via
            scorer.py). Read by experiments/run_scaling_curves.py in
            Week 2-3 to build the actual experiment prompts.

DESIGN DECISIONS:
- Incremental saving happens once per fully-completed instance, not
  every config.SAVE_EVERY_N_CALLS individual API calls. WHY: this
  step's unit of work (one instance: 1 clean-confidence measurement +
  3 intensities x [candidate generation + scoring]) already spans ~39
  API calls per instance — more than SAVE_EVERY_N_CALLS=20 on its own —
  and an instance's distractors are only meaningful as a complete set
  (partial per-intensity results for one instance are not useful to
  Week 2's prompt builder, which needs all three intensities present).
  Saving per-instance is therefore both simpler and no worse, in
  crash-recovery terms, than trying to checkpoint mid-instance.
- Resume-on-restart works by loading any existing output file and
  skipping instance ids that already have a "distractors" key with all
  three intensities present — this means re-running this script after a
  crash (or even after successfully finishing) is always safe and cheap:
  it recomputes nothing already done.
"""

import json

import config
from distractors.generator import generate_candidates
from distractors.scorer import measure_confidence, score_candidates
from experiments.prompt_builder import build_prompt
from models.gemini_client import GeminiClient

logger = config.get_logger(__name__)


def _load_sampled_instances() -> list:
    """
    WHAT: Loads results/benchmark/sampled_instances.json.

    WHY: A dedicated loader keeps the "where does the base instance list
         come from" concern separate from this file's own output path
         (results/benchmark/benchmark_with_distractors.json), even though
         both currently live under the same results/benchmark/ directory.

    Returns:
        list[dict]: The 600 normalized instances from data/sample_instances.py.

    Raises:
        FileNotFoundError: If sample_instances.py has not been run yet —
                           raised with the original message rather than
                           caught, since there is no sensible fallback:
                           this script cannot proceed without a base
                           instance list.
    """
    with open(config.SAMPLED_INSTANCES_PATH) as f:
        return json.load(f)


def _load_existing_benchmark() -> dict:
    """
    WHAT: Loads any previously-saved benchmark-with-distractors file into
          a dict keyed by instance id, for resume support.

    WHY: Enables the crash-recovery / resume behavior required by the
         engineering requirements — a fresh run and a resumed run share
         this exact same lookup.

    Returns:
        dict: {instance_id (str): instance dict (with "distractors" key)}
              for every instance already fully processed in a prior run.
              Empty dict if no output file exists yet (first run).
    """
    if not config.BENCHMARK_WITH_DISTRACTORS_PATH.exists():
        return {}
    with open(config.BENCHMARK_WITH_DISTRACTORS_PATH) as f:
        existing = json.load(f)
    return {inst["id"]: inst for inst in existing}


def _is_complete(instance: dict) -> bool:
    """
    WHAT: Checks whether an instance already has a full set of
          distractors (all 3 intensities) from a prior run.

    WHY: Used both to decide what to skip on resume and, implicitly, to
         define what "done" means for this script — an instance with
         only 1 or 2 of 3 intensities saved (which should not normally
         happen given per-instance saving, but could from a manually
         edited file) is treated as incomplete and reprocessed from
         scratch rather than trusted partially.

    Args:
        instance (dict): One instance dict, possibly already containing
                          a "distractors" key.

    Returns:
        bool: True if "distractors" is present with all of
              config.DISTRACTOR_INTENSITIES as keys.
    """
    distractors = instance.get("distractors")
    if not distractors:
        return False
    return all(intensity in distractors for intensity in config.DISTRACTOR_INTENSITIES)


def _save_benchmark(instances_by_id: dict) -> None:
    """
    WHAT: Writes the current full set of processed instances to
          config.BENCHMARK_WITH_DISTRACTORS_PATH.

    WHY: Called once per completed instance during the main loop (see
         module DESIGN DECISIONS) — a separate function keeps the
         "preserve original instance order" logic (sorting by id's
         embedded index isn't needed since dict insertion order already
         matches processing order, which matches the original sampled
         order since every instance is processed exactly once in order)
         in one place.

    Args:
        instances_by_id (dict): {instance_id: instance dict}, in the
                                 order instances should appear in the
                                 saved file.
    """
    with open(config.BENCHMARK_WITH_DISTRACTORS_PATH, "w") as f:
        json.dump(list(instances_by_id.values()), f, indent=2, ensure_ascii=False)


def process_instance(instance: dict, generator_client: GeminiClient, scoring_client: GeminiClient) -> dict:
    """
    WHAT: Generates and scores distractors for all 3 intensities for one
          instance, returning the instance with its "distractors" key
          populated.

    WHY: A dedicated per-instance function is the unit both the main loop
         and a future parallelization (not implemented here — see
         README.md's Known Limitations — this project runs the benchmark
         build sequentially given free-tier per-minute rate limits make
         parallel calls counterproductive) would call.

    Args:
        instance (dict): Normalized instance dict (data/normalize.py),
                          without a "distractors" key yet.
        generator_client (GeminiClient): Client for config.DISTRACTOR_GENERATOR_MODEL.
        scoring_client (GeminiClient): Client for config.DISTRACTOR_SCORING_MODEL_KEY.

    Returns:
        dict: `instance` with an added "distractors" key:
              {intensity: {"text": str, "confidence_drop": float}, ...}
              for every intensity in config.DISTRACTOR_INTENSITIES.
    """
    clean_prompt = build_prompt(instance, "clean")
    clean_confidence = measure_confidence(instance, clean_prompt, scoring_client)
    logger.info("instance=%s clean_confidence=%.2f", instance["id"], clean_confidence)

    distractors = {}
    for intensity in config.DISTRACTOR_INTENSITIES:
        candidates = generate_candidates(instance, intensity, generator_client)
        best_text, drop = score_candidates(instance, candidates, intensity, scoring_client, clean_confidence)
        distractors[intensity] = {"text": best_text, "confidence_drop": drop}
        logger.info(
            "instance=%s intensity=%s confidence_drop=%.2f",
            instance["id"], intensity, drop,
        )

    return {**instance, "distractors": distractors}


def build_benchmark() -> None:
    """
    WHAT: Main orchestration loop: loads sampled instances, skips
          already-completed ones, processes the rest, saving after every
          completed instance.

    WHY: This is the Week 1, Step 8 entry point — running this module
         directly (`python -m distractors.build_benchmark`) is how the
         Week 1 deliverable (results/benchmark/benchmark_with_distractors.json)
         gets produced.
    """
    sampled_instances = _load_sampled_instances()
    completed_by_id = _load_existing_benchmark()

    generator_client = GeminiClient(config.DISTRACTOR_GENERATOR_MODEL)
    scoring_client = GeminiClient(config.MODEL_SPECS[config.DISTRACTOR_SCORING_MODEL_KEY]["model_id"])

    for instance in sampled_instances:
        existing = completed_by_id.get(instance["id"])
        if existing is not None and _is_complete(existing):
            logger.info("Skipping already-completed instance=%s", instance["id"])
            continue

        try:
            processed = process_instance(instance, generator_client, scoring_client)
        except Exception:
            logger.exception(
                "Non-recoverable error processing instance=%s — saving progress and exiting.",
                instance["id"],
            )
            _save_benchmark(completed_by_id)
            raise

        completed_by_id[instance["id"]] = processed
        _save_benchmark(completed_by_id)

    logger.info(
        "Benchmark build complete: %d/%d instances have distractors.",
        len(completed_by_id), len(sampled_instances),
    )


if __name__ == "__main__":
    build_benchmark()
