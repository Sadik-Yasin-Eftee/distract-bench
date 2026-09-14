"""
test_correctness.py

WHAT: Unit tests for experiments/correctness.py's three task-specific
      checkers.

WHY: The engineering requirements call for these checkers to be tested
     before every week's experiments run, since a silent regression here
     (e.g. a regex that stops matching "Answer: 42." after a prompt
     template tweak) would corrupt every accuracy number computed
     downstream without raising any visible error.

RESEARCH STEP: Ongoing — run via `pytest tests/test_correctness.py`
               before each week's experiment scripts.

DEPENDS ON: experiments/correctness.py.
"""

from experiments.correctness import (
    check_math,
    check_factual,
    check_commonsense,
    extract_math_answer,
    extract_commonsense_answer,
)


# ────────────────────────────────────────────────────────────────────────
# MATH
# ────────────────────────────────────────────────────────────────────────

def test_math_exact_match():
    assert check_math("The answer is 18.", "18") is True


def test_math_no_prefix_uses_last_number():
    # No "answer is" phrase; the model just writes the working and ends
    # on the final number.
    assert check_math("9 * 2 = 18", "18") is True


def test_math_wrong_answer():
    assert check_math("The answer is 20.", "18") is False


def test_math_handles_commas_and_decimals():
    assert check_math("The final answer is 4,200.", "4200") is True
    assert check_math("Answer: 3.5", "3.5") is True


def test_math_no_number_present():
    assert check_math("I cannot determine the answer.", "18") is False


def test_math_prefers_answer_phrase_over_last_number():
    # A trailing remark after the true answer should not override the
    # explicit "answer is" phrase.
    text = "The answer is 18. Note: 2026 is the current year."
    assert extract_math_answer(text) == "18"


# ────────────────────────────────────────────────────────────────────────
# FACTUAL
# ────────────────────────────────────────────────────────────────────────

def test_factual_exact_alias_match():
    assert check_factual("David Seville", ["David Seville"]) is True


def test_factual_substring_in_sentence():
    text = "The man behind The Chipmunks was David Seville, a stage name."
    assert check_factual(text, ["David Seville"]) is True


def test_factual_alternate_alias_matches():
    aliases = ["Amnesty International", "Amnesty.org", "International Amnesty"]
    assert check_factual("It was founded as Amnesty.org.", aliases) is True


def test_factual_case_and_punctuation_insensitive():
    assert check_factual("it's DAVID SEVILLE!", ["David Seville"]) is True


def test_factual_wrong_answer():
    assert check_factual("It was Alvin Seville.", ["David Seville"]) is False


def test_factual_empty_aliases_never_matches():
    assert check_factual("Anything at all", []) is False


# ────────────────────────────────────────────────────────────────────────
# COMMONSENSE
# ────────────────────────────────────────────────────────────────────────

def test_commonsense_exact_letter():
    assert check_commonsense("A", "A") is True


def test_commonsense_explicit_answer_phrase():
    assert check_commonsense("The answer is B.", "B") is True


def test_commonsense_letter_with_paren_option_text():
    assert check_commonsense("D) mall", "D") is True


def test_commonsense_wrong_letter():
    assert check_commonsense("The answer is C.", "A") is False


def test_commonsense_case_insensitive():
    assert check_commonsense("answer: a", "A") is True


def test_commonsense_prefers_final_letter_when_multiple_mentioned():
    # A reasoning model may mention several option letters while working
    # through the choices before settling on its final answer.
    text = "Option A seems plausible, but B is a better fit. Answer: B"
    assert extract_commonsense_answer(text) == "B"


def test_commonsense_no_letter_present():
    assert check_commonsense("I'm not sure.", "A") is False
