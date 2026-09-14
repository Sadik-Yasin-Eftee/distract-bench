"""
prompt_builder.py

WHAT: Builds the final prompt string sent to a model for a given
      (instance, condition) pair, across all 7 experimental conditions.

WHY: The scaling-curve experiment (experiments/run_scaling_curves.py)
     needs to construct the exact same prompt shape for every instance,
     condition, and model — the only thing that should vary between the
     "clean" and "high_early" runs of the *same* instance is whether a
     distractor paragraph is present and where, not incidental formatting
     differences. Centralizing prompt construction here, in one function
     per condition-position, guarantees that.

RESEARCH STEP: Week 2, Step 14. Called by experiments/run_scaling_curves.py
               once per (instance, condition) pair, for every token budget
               and model (the budget/model choice doesn't affect the
               prompt text itself, only how it's sent — see models/).

INPUT: A normalized instance dict (data/normalize.py) and a condition
       string (one of config.CONDITIONS), plus the instance's distractor
       text dict (from distractors/build_benchmark.py) when the condition
       is not "clean".

OUTPUT: A single prompt string, ready to send to any of the three model
        clients unchanged.

DEPENDS ON: config.py (CONDITIONS, DISTRACTOR_INTENSITIES,
            INJECTION_POSITIONS). Used by experiments/run_scaling_curves.py.

DESIGN DECISIONS:
- Commonsense choices are rendered as "A) text" lines, one per line,
  between the question and "Answer:" — this is the standard
  multiple-choice rendering used in essentially every CommonsenseQA
  evaluation harness, chosen so a model's letter-based answer
  (experiments/correctness.py's check_commonsense) has an unambiguous
  A-E option set to select from in the prompt itself.
- The "Additional context:" label is used for late injection to
  distinguish it from the "Context:" label used for early injection —
  this is a naming asymmetry from the project's own prompt templates
  (not something this file introduces), preserved here exactly rather
  than silently unifying the two labels, since Wang et al. 2025's
  adaptive-distractor method scores candidates against these exact
  templates during distractor generation (distractors/scorer.py) — the
  templates used for scoring and for the final experiment must match
  exactly, or the "adaptive" property (candidates chosen for their
  measured effect) breaks.
"""

import config


def _format_choices(choices: dict) -> str:
    """
    WHAT: Renders a CommonsenseQA choices dict as "A) text" lines.

    WHY: A separate helper (rather than inlining the loop in every
         condition-building function below) keeps the choices format
         defined in exactly one place — if the rendering ever needs to
         change (e.g. to "(A) text"), every condition/position
         combination picks up the change automatically.

    Args:
        choices (dict): {"A": "bank", "B": "library", ...} as produced by
                         data/normalize.py's commonsense normalizer.

    Returns:
        str: Newline-joined "A) bank\nB) library\n..." block, in A-E
             order (relies on Python dict insertion order, which
             normalize.py populates in the label order CommonsenseQA
             itself ships, i.e. already A-E).
    """
    return "\n".join(f"{label}) {text}" for label, text in choices.items())


def _question_block(instance: dict) -> str:
    """
    WHAT: Renders "Question: ...\n<choices if any>" for one instance.

    WHY: This exact block appears identically in all three prompt
         templates (clean, early, late) — factoring it out means a
         formatting change only needs to happen once, and guarantees the
         question text is never accidentally rendered differently between
         conditions of the same instance.

    Args:
        instance (dict): Normalized instance dict; uses `question` (all
                          tasks) and `choices` (commonsense only, None
                          otherwise).

    Returns:
        str: "Question: <question>" alone for math/factual, or with an
             appended choices block for commonsense.
    """
    block = f"Question: {instance['question']}"
    if instance["choices"] is not None:
        block += "\n" + _format_choices(instance["choices"])
    return block


def build_prompt(instance: dict, condition: str, distractor_text: str | None = None) -> str:
    """
    WHAT: Builds the full prompt string for one (instance, condition)
          pair.

    WHY: A single function covering all 7 conditions (rather than 7
         separate builder functions) keeps the branching on intensity vs.
         position in one place, matching how experiments/run_scaling_curves.py
         actually iterates: for each instance, for each condition in
         config.CONDITIONS, build one prompt.

    Args:
        instance (dict): Normalized instance dict (data/normalize.py).
        condition (str): One of config.CONDITIONS — "clean", or
                          "{intensity}_{position}" e.g. "high_early".
        distractor_text (str | None): The distractor paragraph to inject.
                                       Required (non-None) for every
                                       condition except "clean"; ignored
                                       if condition == "clean".

    Returns:
        str: The complete prompt, ending in "Answer:" with no trailing
             content after it, ready to send to a model client.

    Raises:
        ValueError: If condition is not in config.CONDITIONS, or if a
                    non-clean condition is requested without
                    distractor_text.
    """
    if condition not in config.CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {config.CONDITIONS}")

    question_block = _question_block(instance)

    if condition == "clean":
        return f"{question_block}\nAnswer:"

    if distractor_text is None:
        raise ValueError(
            f"condition={condition!r} requires distractor_text, got None"
        )

    # Every non-clean condition is "{intensity}_{position}" — position is
    # always the last underscore-separated token (intensity itself never
    # contains an underscore, per config.DISTRACTOR_INTENSITIES).
    position = condition.rsplit("_", 1)[1]

    if position == "early":
        return f"Context:\n{distractor_text}\n\n{question_block}\nAnswer:"
    elif position == "late":
        return f"{question_block}\n\nAdditional context:\n{distractor_text}\n\nAnswer:"
    else:
        # Unreachable given condition is validated against config.CONDITIONS
        # above, whose non-clean entries are only ever built from
        # config.INJECTION_POSITIONS ("early", "late") — kept as an
        # explicit error rather than silently falling through, so a future
        # change to config.INJECTION_POSITIONS that adds a third position
        # fails loudly here instead of building a malformed prompt.
        raise ValueError(f"Unrecognized injection position {position!r} in condition {condition!r}")


if __name__ == "__main__":
    # Manual smoke test: build all 7 conditions for one math instance and
    # one commonsense instance, so a human can visually confirm every
    # template renders correctly before it's relied on to build 75,600
    # actual prompts.
    math_instance = {
        "id": "gsm8k_0000",
        "task_type": "math",
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes muffins with 4. She sells the rest for $2 each. How much does she make daily?",
        "answer": "18",
        "choices": None,
        "context": "",
        "split": "test",
    }
    commonsense_instance = {
        "id": "commonsenseqa_0000",
        "task_type": "commonsense",
        "question": "A revolving door serves as a security measure at a what?",
        "answer": "A",
        "choices": {"A": "bank", "B": "library", "C": "department store", "D": "mall", "E": "new york"},
        "context": "",
        "split": "validation",
    }

    print("=== clean (math) ===")
    print(build_prompt(math_instance, "clean"))

    print("\n=== high_early (math) ===")
    print(build_prompt(math_instance, "high_early", distractor_text="Janet's ducks actually lay 12 eggs per day, not 16, according to a nearby farm's records."))

    print("\n=== medium_late (commonsense, with choices) ===")
    print(build_prompt(commonsense_instance, "medium_late", distractor_text="Revolving doors are commonly discussed in building safety codes and architecture manuals."))
