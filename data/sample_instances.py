"""
sample_instances.py

WHAT: Draws a fixed, reproducible sample of config.N_PER_TASK instances
      per task type, normalizes them, and saves the combined 600-instance
      benchmark base to results/benchmark/sampled_instances.json.

WHY: This is the Week 1 deliverable that every later stage builds on:
     distractor generation (Week 1), model evaluation (Week 2-3), and
     analysis (Week 3-4) all operate on this exact fixed set of 600
     instances. Sampling must be reproducible (fixed seed) so that
     re-running this script — on this machine or another — produces byte-
     identical instance ids and content, which is what lets a saved
     scaling-curve result be unambiguously matched back to the instance
     that produced it weeks later.

RESEARCH STEP: Week 1, Step 5.

INPUT: None directly (calls data.load_datasets.load_all_datasets(), which
       downloads/caches from the HuggingFace Hub on first run).

OUTPUT: results/benchmark/sampled_instances.json — a JSON array of 600
        instance dicts in the common schema (data/normalize.py), ordered
        math instances first, then factual, then commonsense.

DEPENDS ON: config.py, data/load_datasets.py, data/normalize.py. Read by
            distractors/build_benchmark.py in Week 1's next step.

DESIGN DECISIONS:
- Sampling uses Python's `random.Random(config.RANDOM_SEED)` rather than
  numpy's RNG or `dataset.shuffle(seed=...)`. WHY: `random.sample` over a
  plain Python list of row indices is a one-line, dependency-free way to
  get a reproducible subset, and avoids any version-to-version change in
  numpy's/HF datasets' own shuffling algorithm silently changing which
  600 rows get selected between package upgrades.
- The full JSON array is written in one shot (not incrementally). WHY:
  unlike the 75,600-row scaling experiment (experiments/save_results.py),
  600 instances normalize in well under a second — there is no meaningful
  crash-recovery benefit to incremental saving here, and a single JSON
  array (rather than JSONL) is easier to hand-inspect and load as a whole
  for the smaller downstream steps that consume it in Week 1.
"""

import json
import random

import config
from data.load_datasets import load_all_datasets
from data.normalize import normalize_row

logger = config.get_logger(__name__)


def sample_task(task_type: str, dataset, n: int, rng: random.Random) -> list:
    """
    WHAT: Draws n row indices from `dataset` without replacement using
          `rng`, then normalizes each selected row.

    WHY: Sampling indices first (rather than dataset.shuffle().select())
         keeps the operation a plain, auditable Python list operation
         whose output depends only on `len(dataset)`, `n`, and the RNG
         state — easy to unit-test and reason about independently of the
         `datasets` library's own internals.

    Args:
        task_type (str): One of config.TASK_TYPES, passed through to
                          normalize_row() to select the right normalizer.
        dataset: The raw HF Dataset for this task (from load_all_datasets()).
        n (int): Number of instances to sample. Raises if n > len(dataset).
        rng (random.Random): Seeded RNG shared across all three task types
                              so the overall sampling is reproducible from
                              one seed rather than three independent ones.

    Returns:
        list[dict]: n normalized instances for this task, in the order
                    their source indices were drawn (not dataset order).

    Raises:
        ValueError: If n exceeds the number of available rows.
    """
    if n > len(dataset):
        raise ValueError(
            f"Requested {n} instances for task_type={task_type!r} but the "
            f"dataset only has {len(dataset)} rows."
        )

    indices = rng.sample(range(len(dataset)), k=n)
    split = config.DATASET_SPLIT_BY_TASK[task_type]

    instances = []
    for position, source_index in enumerate(indices):
        row = dataset[source_index]
        instances.append(normalize_row(task_type, row, index=position, split=split))
    return instances


def build_sampled_instances() -> list:
    """
    WHAT: Loads all three raw datasets and samples config.N_PER_TASK
          instances from each, in a fixed task order (math, factual,
          commonsense).

    WHY: A single `random.Random(config.RANDOM_SEED)` instance is created
         once and threaded through all three calls to sample_task(),
         rather than re-seeding per task, so the overall draw is a
         deterministic function of one seed and the fixed task order —
         changing the order of config.TASK_TYPES would change the sample,
         which is why that tuple's order is itself part of what "seed=42"
         reproducibility commits to.

    Returns:
        list[dict]: config.TOTAL_INSTANCES (600) normalized instances,
                    math instances first, then factual, then commonsense.
    """
    rng = random.Random(config.RANDOM_SEED)
    raw_datasets = load_all_datasets()

    all_instances = []
    for task_type in config.TASK_TYPES:
        task_instances = sample_task(
            task_type, raw_datasets[task_type], config.N_PER_TASK, rng
        )
        logger.info("Sampled %d instances for task_type=%s", len(task_instances), task_type)
        all_instances.extend(task_instances)

    return all_instances


def save_sampled_instances(instances: list) -> None:
    """
    WHAT: Writes the full instance list to
          config.SAMPLED_INSTANCES_PATH as a single JSON array.

    WHY: A dedicated save function (rather than inlining json.dump at the
         call site) gives distractors/build_benchmark.py and any other
         future writer of this same file one documented place to match
         if the output path or format ever changes.

    Args:
        instances (list[dict]): The instances to save, normally the
                                 output of build_sampled_instances().
    """
    with open(config.SAMPLED_INSTANCES_PATH, "w") as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d instances to %s", len(instances), config.SAMPLED_INSTANCES_PATH)


if __name__ == "__main__":
    instances = build_sampled_instances()
    save_sampled_instances(instances)

    # Sanity-print a few instances so a human can visually confirm the
    # sample before distractor generation (Week 1, Step 6) spends API
    # quota on it.
    print(f"\nTotal instances: {len(instances)}")
    for task_type in config.TASK_TYPES:
        count = sum(1 for inst in instances if inst["task_type"] == task_type)
        print(f"  {task_type}: {count}")
    print("\nFirst instance of each task type:")
    seen_types = set()
    for inst in instances:
        if inst["task_type"] not in seen_types:
            print(inst)
            seen_types.add(inst["task_type"])
