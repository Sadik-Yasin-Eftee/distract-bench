"""
normalize.py

WHAT: Converts each dataset's native row format into the project's common
      instance schema.

WHY: GSM8K, TriviaQA, and CommonsenseQA have three completely different
     native schemas (free-text answer embedded in a reasoning chain vs. a
     value+aliases dict vs. a multiple-choice letter). Every downstream
     module (distractor generation, prompt building, correctness checking)
     needs to treat an instance the same way regardless of which dataset
     it came from — e.g. prompt_builder.py must not special-case three
     different answer field names. Normalizing once, here, means every
     later module only ever sees the one schema below.

RESEARCH STEP: Week 1, Step 4.

INPUT: Raw datasets.Dataset objects as returned by
       data.load_datasets.load_raw_dataset().

OUTPUT: Lists of dicts matching the common schema:
        {
            "id": str,            # "<task>_<index>", e.g. "gsm8k_0042"
            "task_type": str,     # "math" | "factual" | "commonsense"
            "question": str,
            "answer": str,        # ground truth, always a string
            "choices": dict|None, # {"A": "...", ...} for commonsense only
            "context": str,       # "" for all three tasks (see below)
            "split": str,         # the actual HF split used (config.py)
        }

DEPENDS ON: config.py (task types, split names). Used by
            data/sample_instances.py.

DESIGN DECISIONS:
- `context` is "" for every task, not just TriviaQA. GSM8K and
  CommonsenseQA never had a supporting-context field to begin with, and
  TriviaQA's is deliberately dropped (see config.py's DATASET_SPECS
  comment on rc.nocontext) because this project injects its own generated
  distractors rather than using a dataset's bundled context — keeping the
  field in the schema (rather than removing it) matches the schema given
  in the project design and leaves a documented slot if a future
  extension wants to reintroduce dataset-native context.
- GSM8K's ground truth answer is extracted from the "#### <number>"
  suffix that every GSM8K answer field ends with, via regex, matching
  experiments/correctness.py's own extraction of the *model's* answer —
  using the same extraction logic on both sides means a correct model
  answer is compared against a ground truth string in the identical
  format.
"""

import re

import config

logger = config.get_logger(__name__)

# Matches the trailing "#### 42" (or "#### 4,200" / "#### -3.5") that every
# GSM8K answer field ends with. Captures the numeric literal, allowing an
# optional leading minus sign, thousands commas, and a decimal point.
# WHY this exact pattern: verified against data/load_datasets.py's printed
# example ("...#### 18") and against GSM8K's documented answer format
# (Cobbe et al. 2021) that guarantees exactly one "####" per answer.
_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def _normalize_math(row: dict, index: int, split: str) -> dict:
    """
    WHAT: Normalizes one GSM8K row.

    WHY: GSM8K's ground truth lives at the end of a full worked solution
         (the `answer` field is the whole chain-of-thought, not just the
         number) — this extracts just the final numeric literal so the
         normalized `answer` field is directly comparable to a model's
         extracted numeric answer via exact string/float match.

    Args:
        row (dict): One row from the raw GSM8K dataset, with `question`
                     and `answer` (chain-of-thought + "#### N") fields.
        index (int): Row's position in the sampled subset, used to build
                      a stable, human-readable instance id.
        split (str): The HF split this row was drawn from (recorded on
                      the instance so it is traceable back to its source).

    Returns:
        dict: One instance in the common schema.

    Raises:
        ValueError: If the "#### <number>" pattern is not found — this
                    would mean GSM8K's answer format changed from the
                    documented one this project relies on.
    """
    match = _GSM8K_FINAL_ANSWER_RE.search(row["answer"])
    if not match:
        raise ValueError(
            f"GSM8K row has no '#### <number>' final answer: {row['answer']!r}"
        )
    final_answer = match.group(1).replace(",", "")

    return {
        "id": f"gsm8k_{index:04d}",
        "task_type": "math",
        "question": row["question"],
        "answer": final_answer,
        "choices": None,
        "context": "",
        "split": split,
    }


def _normalize_factual(row: dict, index: int, split: str) -> dict:
    """
    WHAT: Normalizes one TriviaQA row.

    WHY: TriviaQA ships `answer` as a dict with a canonical `value` plus a
         list of acceptable `aliases` (e.g. "David Seville" with alias
         variants) — this keeps `value` as the normalized `answer` field
         for display/logging, while the full alias list is preserved
         under a project-specific key so experiments/correctness.py can
         still do alias-based matching rather than a single-string exact
         match, which would incorrectly mark a model correct-but-differently
         -phrased answer as wrong.

    Args:
        row (dict): One row from the raw TriviaQA dataset, with `question`
                     and `answer` ({"value": ..., "aliases": [...]}).
        index (int): Row's position in the sampled subset.
        split (str): The HF split this row was drawn from.

    Returns:
        dict: One instance in the common schema, plus an extra
              `answer_aliases` key (list[str]) not part of the schema
              given in the project design but required for correct
              TriviaQA scoring — see experiments/correctness.py.
    """
    return {
        "id": f"triviaqa_{index:04d}",
        "task_type": "factual",
        "question": row["question"],
        "answer": row["answer"]["value"],
        "answer_aliases": list(row["answer"]["aliases"]),
        "choices": None,
        "context": "",
        "split": split,
    }


def _normalize_commonsense(row: dict, index: int, split: str) -> dict:
    """
    WHAT: Normalizes one CommonsenseQA row.

    WHY: CommonsenseQA ships choices as two parallel lists
         (`choices["label"]` = ["A","B","C","D","E"],
         `choices["text"]` = [...]) rather than a mapping — this zips them
         into the {"A": "...", ...} dict shape the project schema
         specifies, which is what prompt_builder.py needs to render
         "A) ... B) ..." option text directly.

    Args:
        row (dict): One row from the raw CommonsenseQA dataset, with
                     `question`, `choices` ({"label": [...], "text": [...]}),
                     and `answerKey` (a single letter).
        index (int): Row's position in the sampled subset.
        split (str): The HF split this row was drawn from.

    Returns:
        dict: One instance in the common schema, with `choices` populated
              and `answer` set to the single-letter ground truth.
    """
    choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
    return {
        "id": f"commonsenseqa_{index:04d}",
        "task_type": "commonsense",
        "question": row["question"],
        "answer": row["answerKey"],
        "choices": choices,
        "context": "",
        "split": split,
    }


_NORMALIZERS = {
    "math": _normalize_math,
    "factual": _normalize_factual,
    "commonsense": _normalize_commonsense,
}


def normalize_row(task_type: str, row: dict, index: int, split: str) -> dict:
    """
    WHAT: Dispatches one raw row to its task-specific normalizer.

    WHY: Gives sample_instances.py a single entry point that doesn't need
         to know which of the three task-specific functions to call.

    Args:
        task_type (str): One of config.TASK_TYPES.
        row (dict): One raw row from the corresponding HF dataset.
        index (int): Row's position in the sampled subset (used for the
                      instance id).
        split (str): The HF split this row was drawn from.

    Returns:
        dict: One instance in the common schema (see module docstring).

    Raises:
        ValueError: If task_type is not recognized.
    """
    if task_type not in _NORMALIZERS:
        raise ValueError(f"Unknown task_type {task_type!r}; expected one of {config.TASK_TYPES}")
    return _NORMALIZERS[task_type](row, index, split)


if __name__ == "__main__":
    # Manual smoke test: normalize the first row of each task type using
    # data/load_datasets.py, so a human can visually confirm the schema
    # transform is correct before sample_instances.py depends on it for
    # all 600 instances.
    from data.load_datasets import load_all_datasets

    raw = load_all_datasets()
    for task_type, ds in raw.items():
        split = config.DATASET_SPLIT_BY_TASK[task_type]
        normalized = normalize_row(task_type, ds[0], index=0, split=split)
        print(f"\n=== {task_type} ===")
        print(normalized)
