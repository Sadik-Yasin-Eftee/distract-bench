"""
test_generator.py

WHAT: Unit tests for distractors/generator.py and distractors/scorer.py,
      using a mocked GeminiClient so no API quota is spent.

WHY: distractor generation/scoring is the most API-call-expensive part of
     this project (see scorer.py's module docstring: ~30 calls per
     instance just for scoring). Testing the *logic* — prompt template
     filling, candidate collection, confidence-drop computation and
     selection — against a mock catches bugs (e.g. an off-by-one in which
     candidate is "best," or the high-intensity template silently missing
     the ground-truth answer) before they cost real quota running against
     all 600 instances.

RESEARCH STEP: Ongoing — run via `pytest tests/test_generator.py` before
               distractors/build_benchmark.py's Week 1 run.

DEPENDS ON: distractors/generator.py, distractors/scorer.py.
"""

from unittest.mock import MagicMock

import config
from distractors.generator import generate_candidates, _build_generation_prompt
from distractors.scorer import measure_confidence, score_candidates


_INSTANCE = {
    "id": "gsm8k_0000",
    "task_type": "math",
    "question": "What is 9 + 9?",
    "answer": "18",
    "choices": None,
    "context": "",
    "split": "test",
}


# ────────────────────────────────────────────────────────────────────────
# GENERATOR
# ────────────────────────────────────────────────────────────────────────

def test_high_intensity_prompt_includes_answer():
    prompt = _build_generation_prompt(_INSTANCE, "high")
    assert "18" in prompt
    assert "What is 9 + 9?" in prompt


def test_low_intensity_prompt_excludes_answer():
    prompt = _build_generation_prompt(_INSTANCE, "low")
    # The low-intensity template never mentions the ground truth at all —
    # confirming this guards against a copy-paste bug that accidentally
    # reuses the high-intensity template for another intensity level.
    assert "Correct answer" not in prompt


def test_unknown_intensity_raises():
    try:
        _build_generation_prompt(_INSTANCE, "extreme")
        assert False, "expected ValueError"
    except ValueError:
        pass


def _mock_generator_client(answers):
    """Returns a fake GeminiClient-like object whose .query() yields the
    given answer_text values in sequence, one per call."""
    client = MagicMock()
    client.query.side_effect = [
        MagicMock(answer_text=text) for text in answers
    ]
    return client


def test_generate_candidates_returns_all_three():
    client = _mock_generator_client([
        "Ducks are common farm animals found worldwide.",
        "Many farmers raise poultry for both eggs and meat.",
        "Egg production varies by breed and season.",
    ])
    candidates = generate_candidates(_INSTANCE, "low", client)
    assert len(candidates) == 3
    assert client.query.call_count == 3


def test_generate_candidates_skips_empty_responses():
    client = _mock_generator_client(["", "A valid paragraph here.", "  "])
    candidates = generate_candidates(_INSTANCE, "low", client)
    # Two of three responses were empty/whitespace-only and should be
    # dropped rather than kept as blank candidates.
    assert candidates == ["A valid paragraph here."]


def test_generate_candidates_uses_configured_token_budget():
    client = _mock_generator_client(["x", "y", "z"])
    generate_candidates(_INSTANCE, "low", client)
    _, kwargs = client.query.call_args
    assert kwargs["token_budget"] == config.GENERATOR_TOKEN_BUDGET


# ────────────────────────────────────────────────────────────────────────
# SCORER
# ────────────────────────────────────────────────────────────────────────

def _mock_target_client(correct_pattern):
    """Returns a fake client whose .query() answer_text alternates
    between '18' (correct) and '0' (wrong) per `correct_pattern`, a list
    of bools consumed in order across successive calls."""
    client = MagicMock()
    client.query.side_effect = [
        MagicMock(answer_text=("18" if is_correct else "0"))
        for is_correct in correct_pattern
    ]
    return client


def test_measure_confidence_all_correct():
    client = _mock_target_client([True, True, True])
    confidence = measure_confidence(_INSTANCE, "Question: What is 9 + 9?\nAnswer:", client)
    assert confidence == 1.0


def test_measure_confidence_partial():
    client = _mock_target_client([True, False, True])
    confidence = measure_confidence(_INSTANCE, "Question: What is 9 + 9?\nAnswer:", client)
    assert abs(confidence - (2 / 3)) < 1e-9


def test_score_candidates_picks_highest_drop():
    # 3 candidates scored with NUM_SCORING_SAMPLES=3 each: candidate A
    # keeps the model fully correct (drop=0), candidate B drops it to
    # 1/3 correct (drop=2/3), candidate C drops it to 0/3 (drop=1.0) —
    # C should be selected as the strongest distractor.
    client = _mock_target_client(
        [True, True, True]       # candidate A: 3/3 correct
        + [True, False, False]   # candidate B: 1/3 correct
        + [False, False, False]  # candidate C: 0/3 correct
    )
    best, drop = score_candidates(
        _INSTANCE,
        candidates=["candidate A text", "candidate B text", "candidate C text"],
        intensity="high",
        target_client=client,
        clean_confidence=1.0,
    )
    assert best == "candidate C text"
    assert abs(drop - 1.0) < 1e-9


def test_score_candidates_raises_on_empty_list():
    client = MagicMock()
    try:
        score_candidates(_INSTANCE, [], "high", client, clean_confidence=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
