"""
load_datasets.py

WHAT: Loads the three raw HuggingFace datasets (GSM8K, TriviaQA, CommonsenseQA)
      for their configured held-out split, caching locally, and validates
      that the fields the rest of the pipeline depends on are present.

WHY: Every later stage (normalize.py, sample_instances.py, and eventually
     the distractor generator and model evaluation) assumes a specific set
     of raw fields exists per task type. Failing fast here — with a clear
     error naming the missing field — is far cheaper than discovering a
     schema mismatch three scripts downstream, after distractor generation
     has already spent API quota on malformed instances.

RESEARCH STEP: Week 1, Step 3.

INPUT: None (downloads from the HuggingFace Hub on first call; reads from
       config.DATA_CACHE_DIR on subsequent calls).

OUTPUT: A dict mapping task_type -> datasets.Dataset (the raw HF dataset
        object for that task's configured split).

DEPENDS ON: config.py (dataset ids/configs/splits, cache dir). Used by
            data/normalize.py.

DESIGN DECISIONS:
- The three "obvious" bare repo ids named in the initial project design
  ("gsm8k", "trivia_qa", "commonsense_qa") are legacy pre-namespace HF Hub
  ids. The installed datasets==5.0.1 / huggingface_hub==1.31.0 pair raises
  huggingface_hub.errors.HfUriError on them (confirmed by direct testing).
  config.py points DATASET_SPECS at the canonical namespaced mirrors
  instead (openai/gsm8k, mandarjoshi/trivia_qa, tau/commonsense_qa) — same
  data, working load path.
- GSM8K has no "validation" split (only train/test — confirmed by
  inspecting the loaded dataset's split names). config.DATASET_SPLIT_BY_TASK
  substitutes "test" for math only; TriviaQA and CommonsenseQA both use
  their real "validation" split.
"""

from datasets import load_dataset, Dataset

import config

logger = config.get_logger(__name__)

# Required raw fields per task type, checked after loading. These are the
# exact column names observed on the Hub for each dataset (verified by
# loading each split once and printing ds.column_names during development
# of this file) — not guesses from documentation, which can drift from
# the actual parquet schema.
REQUIRED_FIELDS = {
    "math": {"question", "answer"},
    "factual": {"question", "answer"},
    "commonsense": {"question", "question_concept", "choices", "answerKey"},
}


def load_raw_dataset(task_type: str) -> Dataset:
    """
    WHAT: Loads the raw HuggingFace dataset for one task type's configured
          held-out split, using the on-disk cache after the first call.

    WHY: Wrapping datasets.load_dataset() per task type (rather than
         calling it inline in three places) means the repo-id/config/split
         lookup and the post-load field validation happen in exactly one
         place, so a schema check added here protects every caller.

    Args:
        task_type (str): One of config.TASK_TYPES ("math", "factual",
                          "commonsense"). Selects which entry of
                          config.DATASET_SPECS / DATASET_SPLIT_BY_TASK to
                          use.

    Returns:
        Dataset: The HF Dataset object for that task's configured split,
                 with columns unchanged from the Hub (normalization into
                 the project's common schema happens separately, in
                 data/normalize.py, so this function's output is still in
                 each dataset's own native format).

    Raises:
        ValueError: If task_type is not a recognized key of
                    config.DATASET_SPECS.
        ValueError: If the loaded dataset is missing a field this project
                    depends on downstream — this indicates the Hub schema
                    has changed since this file was written, or the wrong
                    config/split was loaded.
    """
    if task_type not in config.DATASET_SPECS:
        raise ValueError(
            f"Unknown task_type {task_type!r}; expected one of {config.TASK_TYPES}"
        )

    spec = config.DATASET_SPECS[task_type]
    split = config.DATASET_SPLIT_BY_TASK[task_type]

    logger.info(
        "Loading %s (config=%s, split=%s) for task_type=%s",
        spec["hf_name"], spec["hf_config"], split, task_type,
    )
    dataset = load_dataset(
        spec["hf_name"],
        spec["hf_config"],
        split=split,
        cache_dir=str(config.DATA_CACHE_DIR),
    )

    missing = REQUIRED_FIELDS[task_type] - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"{spec['hf_name']} ({task_type}) is missing required fields "
            f"{missing}; found columns {dataset.column_names}. The Hub "
            f"schema may have changed since REQUIRED_FIELDS was written."
        )

    logger.info("Loaded %s: %d rows", spec["hf_name"], len(dataset))
    return dataset


def load_all_datasets() -> dict:
    """
    WHAT: Loads all three raw datasets (math, factual, commonsense) in one
          call.

    WHY: data/sample_instances.py needs all three at once to draw its
         200-per-task sample; this avoids repeating the same three-call
         boilerplate there.

    Returns:
        dict: {task_type (str): Dataset} for every entry in
              config.TASK_TYPES.
    """
    return {task_type: load_raw_dataset(task_type) for task_type in config.TASK_TYPES}


if __name__ == "__main__":
    # Manual smoke test: load everything and print one example per task so
    # a human can visually confirm the schema looks right before the
    # normalization step (Step 4) depends on it.
    datasets_by_task = load_all_datasets()
    for task_type, ds in datasets_by_task.items():
        print(f"\n=== {task_type}: {len(ds)} rows, columns={ds.column_names} ===")
        print(ds[0])
