"""
correctness.py

WHAT: Three task-specific functions that check whether a raw model output
      string is correct, plus the answer-extraction logic each one needs.

WHY: Model outputs are free text — even a Gemini/DeepSeek reasoning model
     given a strict "Answer:" prompt template may still wrap its final
     answer in a sentence ("The answer is 42.") or restate the option text
     next to the letter ("D) mall"). Every one of the 75,600 scaling-curve
     calls depends on these functions to turn that free text into a single
     correct/incorrect bit — a bug here would silently corrupt every
     accuracy number in the paper, which is why each checker has a
     dedicated unit test file (tests/test_correctness.py) that must pass
     before Week 2's experiments are allowed to run.

RESEARCH STEP: Week 2, Step 13. Called by experiments/run_scaling_curves.py
               once per API response.

INPUT: (model_output: str, ground_truth: str, ...task-specific extras).

OUTPUT: bool — True if the model's output is judged correct.

DEPENDS ON: Nothing outside the standard library. Used by
            experiments/run_scaling_curves.py. Tested by
            tests/test_correctness.py.

DESIGN DECISIONS:
- Each checker returns a plain bool, not a confidence score. WHY: the
  project's primary metric (accuracy) is a simple correct/incorrect count
  per config.py's METRICS section — a graded partial-credit score would
  need its own aggregation rule that the research design never specifies,
  so a bool keeps every checker's contract identical and unambiguous.
- Extraction functions are exposed separately from the boolean checkers
  (e.g. extract_math_answer() vs. check_math()), so run_scaling_curves.py
  can log the *extracted* answer alongside the raw model output — without
  that, a wrong answer and a failed extraction would look identical in
  the saved results, which would make it impossible to later tell "the
  model was wrong" apart from "our regex missed a validly-stated answer."
"""

import re
import string


# ────────────────────────────────────────────────────────────────────────
# MATH (GSM8K)
# ────────────────────────────────────────────────────────────────────────

# Matches a numeric literal (optional leading minus, optional thousands
# commas, optional decimal part). Used both to scan the model's free-text
# output for its final answer and, in data/normalize.py, to pull the
# ground-truth number out of GSM8K's "#### N" suffix.
_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")

# Preferred patterns, tried in order, for locating *the* final answer in
# a longer response rather than just grabbing the first number that
# appears (which is often restated from the question itself, not the
# model's computed result).
_ANSWER_PREFIX_RE = re.compile(
    r"(?:answer|result)\s*(?:is|:)\s*\$?(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE
)


def extract_math_answer(model_output: str) -> str | None:
    """
    WHAT: Extracts the final numeric answer from a free-text model
          response to a GSM8K-style question.

    WHY: A reasoning model's output under a large token budget can contain
         many intermediate numbers from its working (e.g. "16 - 3 - 4 = 9"
         before the final "9 * 2 = 18"). Preferring an explicit
         "answer is X" / "answer: X" phrase over "the last number in the
         text" avoids picking up a number that appears after the true
         final answer merely because the model added a trailing remark
         (e.g. "18. This assumes she sells all 9 eggs.").

    Args:
        model_output (str): The raw text returned by the model for the
                             "Answer:"-suffixed prompt (see
                             experiments/prompt_builder.py).

    Returns:
        str | None: The extracted number as a string with commas removed
                    (e.g. "18", "-3.5"), or None if no numeric literal is
                    found anywhere in the output — this is a real failure
                    mode (e.g. the model was cut off by its token budget
                    before writing any answer) that run_scaling_curves.py
                    must be able to distinguish from a wrong-but-present
                    answer.
    """
    prefix_match = _ANSWER_PREFIX_RE.search(model_output)
    if prefix_match:
        return prefix_match.group(1).replace(",", "")

    # Fall back to the last numeric literal anywhere in the text — chosen
    # over the *first* because GSM8K questions themselves often contain
    # numbers, and a model's final computed answer, absent an explicit
    # "answer is" phrase, is far more likely to be the last number it
    # writes than the first.
    all_numbers = _NUMBER_RE.findall(model_output)
    if not all_numbers:
        return None
    return all_numbers[-1].replace(",", "")


def check_math(model_output: str, ground_truth: str) -> bool:
    """
    WHAT: Checks a GSM8K-style answer for exact numeric match.

    WHY: GSM8K ground truths are exact integers (or occasionally decimals)
         with no ambiguity in "close enough" — exact match is the
         standard scoring convention in every GSM8K paper this project
         builds on, so using anything looser (e.g. rounding tolerance)
         would make accuracy numbers incomparable to prior work.

    Args:
        model_output (str): Raw model response text.
        ground_truth (str): The normalized numeric answer string from
                             data/normalize.py (e.g. "18").

    Returns:
        bool: True if the extracted answer, parsed as a float, exactly
              equals the ground truth parsed as a float. False if
              extraction fails or the numbers differ.

    Note:
        Comparison is done on parsed floats rather than raw strings so
        that "18" and "18.0" (which a model might write either way) are
        treated as equal.
    """
    extracted = extract_math_answer(model_output)
    if extracted is None:
        return False
    try:
        return float(extracted) == float(ground_truth)
    except ValueError:
        # Either string failed to parse as a float — treat as incorrect
        # rather than raising, since a malformed extraction is a scoring
        # outcome (wrong/unparseable), not a program error.
        return False


# ────────────────────────────────────────────────────────────────────────
# FACTUAL (TriviaQA)
# ────────────────────────────────────────────────────────────────────────

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_text(text: str) -> str:
    """
    WHAT: Lowercases and strips punctuation from a string.

    WHY: TriviaQA aliases and model outputs differ trivially in casing and
         punctuation ("David Seville" vs. "david seville." vs. "David
         Seville, aka Ross Bagdasarian") — normalizing both sides the same
         way before substring matching avoids penalizing a correct answer
         for surface formatting differences that were never part of what
         TriviaQA's alias list is meant to capture.

    Args:
        text (str): Raw text to normalize.

    Returns:
        str: Lowercased text with all `string.punctuation` characters
             removed and surrounding whitespace stripped.
    """
    return text.lower().translate(_PUNCTUATION_TABLE).strip()


def check_factual(model_output: str, aliases: list[str]) -> bool:
    """
    WHAT: Checks a TriviaQA-style answer via alias-based substring
          matching.

    WHY: TriviaQA questions can have dozens of valid phrasings for the
         same entity (see data/normalize.py's example: "Amnesty
         International" has 9 aliases including "Amnesty.org" and
         "International Amnesty"). A free-text model response will
         usually embed the answer inside a full sentence
         ("The charity is Amnesty International, founded in 1961.") —
         substring containment (is any normalized alias a substring of
         the normalized model output) is the standard TriviaQA scoring
         approach precisely because it tolerates that surrounding text
         without requiring an exact whole-string match.

    Args:
        model_output (str): Raw model response text.
        aliases (list[str]): The instance's `answer_aliases` list from
                              data/normalize.py (includes the canonical
                              `answer` value itself as one of the
                              aliases in TriviaQA's own data).

    Returns:
        bool: True if any normalized alias appears as a substring of the
              normalized model output. False if `aliases` is empty or no
              alias matches.

    Note:
        A short alias (e.g. a single common word) could in principle
        produce a false positive substring match inside unrelated text.
        This is a known, accepted limitation of alias-based TriviaQA
        scoring in general, not specific to this implementation — see
        tests/test_correctness.py for an explicit edge-case test of this
        behavior.
    """
    normalized_output = _normalize_text(model_output)
    return any(_normalize_text(alias) in normalized_output for alias in aliases if alias)


# ────────────────────────────────────────────────────────────────────────
# COMMONSENSE (CommonsenseQA)
# ────────────────────────────────────────────────────────────────────────

_VALID_LETTERS = set("ABCDE")

# Preferred pattern: an explicit "answer is X" / "answer: X" phrase,
# tried before falling back to scanning for a bare letter, for the same
# reason as _ANSWER_PREFIX_RE above — a reasoning model's working may
# mention other option letters before settling on its final choice.
_LETTER_PREFIX_RE = re.compile(r"(?:answer|option)\s*(?:is|:)\s*\(?([A-E])\)?", re.IGNORECASE)

# Fallback pattern: a standalone letter A-E, optionally followed by ")"
# or "." — matches formats like "D) mall" or a bare trailing "D.".
_STANDALONE_LETTER_RE = re.compile(r"\b([A-E])\b[).]?")


def extract_commonsense_answer(model_output: str) -> str | None:
    """
    WHAT: Extracts the single-letter (A-E) answer choice from a free-text
          model response to a CommonsenseQA question.

    WHY: Mirrors extract_math_answer()'s two-tier strategy (explicit
         "answer is X" phrase first, positional fallback second) for the
         same reason: a reasoning model's chain of thought can mention
         multiple option letters while reasoning through the choices
         before committing to a final one.

    Args:
        model_output (str): The raw text returned by the model for the
                             "Answer:"-suffixed multiple-choice prompt.

    Returns:
        str | None: The extracted letter in uppercase (e.g. "A"), or None
                    if no A-E letter is found anywhere in the output.
    """
    prefix_match = _LETTER_PREFIX_RE.search(model_output)
    if prefix_match:
        return prefix_match.group(1).upper()

    # Fall back to the *last* standalone letter in the text, matching
    # extract_math_answer()'s "last, not first" rationale: a model's
    # final committed choice is more likely to be the last A-E letter it
    # writes than the first, which may just be the option label it is
    # currently reasoning about.
    all_letters = _STANDALONE_LETTER_RE.findall(model_output)
    if not all_letters:
        return None
    return all_letters[-1].upper()


def check_commonsense(model_output: str, ground_truth: str) -> bool:
    """
    WHAT: Checks a CommonsenseQA-style answer for exact letter match.

    WHY: CommonsenseQA has exactly one correct option letter per question
         — exact match is the only sensible scoring rule for a 5-way
         multiple-choice task.

    Args:
        model_output (str): Raw model response text.
        ground_truth (str): The normalized single-letter answer from
                             data/normalize.py (e.g. "A").

    Returns:
        bool: True if the extracted letter equals the ground truth
              letter (case-insensitive). False if extraction fails or
              the letters differ.
    """
    extracted = extract_commonsense_answer(model_output)
    if extracted is None:
        return False
    return extracted.upper() == ground_truth.upper()


# ────────────────────────────────────────────────────────────────────────
# DISPATCH
# ────────────────────────────────────────────────────────────────────────

def check_correctness(task_type: str, model_output: str, instance: dict) -> bool:
    """
    WHAT: Dispatches to the correct task-specific checker based on
          `task_type`.

    WHY: Gives experiments/run_scaling_curves.py a single call site
         instead of repeating this same three-way branch at every place a
         result needs scoring. Kept as an explicit if/elif rather than a
         dict-of-callables because each task's checker takes a different
         second argument (`instance["answer"]` vs.
         `instance["answer_aliases"]`), so a uniform dispatch table would
         need per-entry argument-adapting lambdas anyway — no simpler than
         this.

    Args:
        task_type (str): One of "math", "factual", "commonsense".
        model_output (str): Raw model response text.
        instance (dict): The normalized instance dict (data/normalize.py)
                          this response was generated for — used to read
                          `answer` (all tasks) or `answer_aliases`
                          (factual only).

    Returns:
        bool: True if the model's output is judged correct for this
              instance.

    Raises:
        ValueError: If task_type is not recognized.
    """
    if task_type == "math":
        return check_math(model_output, instance["answer"])
    elif task_type == "factual":
        return check_factual(model_output, instance["answer_aliases"])
    elif task_type == "commonsense":
        return check_commonsense(model_output, instance["answer"])
    else:
        raise ValueError(f"Unknown task_type {task_type!r}")
