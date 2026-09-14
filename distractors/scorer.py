"""
scorer.py

WHAT: Scores candidate distractor paragraphs by their measured effect on
      a reference target model, and picks the strongest one per intensity.

WHY: This is the "measure and select" half of Wang et al. 2025's adaptive
     tree-search method — a distractor is "adaptive" precisely because it
     is chosen for its demonstrated effect on the target model, not
     accepted on the generator's first attempt. See config.py's
     DISTRACTOR SCORING section for why "confidence" here is an empirical
     accuracy-over-N-samples proxy rather than a token-probability
     measurement, which none of this project's three provider APIs
     expose in a directly usable form for multi-step reasoning output.

RESEARCH STEP: Week 1, Step 7.

INPUT: A normalized instance dict, a list of candidate distractor
       strings (from distractors/generator.py), and the intensity level
       they were generated for.

OUTPUT: (best_candidate_text: str, confidence_drop: float) — the
        candidate with the highest measured confidence drop and that
        drop's value.

DEPENDS ON: config.py (scoring constants), models/gemini_client.py (the
            reference target model), experiments/prompt_builder.py (to
            build the clean and noisy prompts scored), experiments/correctness.py
            (to check each sample). Used by distractors/build_benchmark.py.

DESIGN DECISIONS:
- API call cost of scoring one instance's full candidate set: 1 clean
  baseline measurement (config.NUM_SCORING_SAMPLES calls) +
  3 intensities * 3 candidates * config.NUM_SCORING_SAMPLES calls for the
  noisy measurements = NUM_SCORING_SAMPLES * (1 + 9) = 30 calls per
  instance at NUM_SCORING_SAMPLES=3, all against the single reference
  model (config.DISTRACTOR_SCORING_MODEL_KEY). At 600 instances this is
  18,000 calls — a real, deliberate cost against Gemini's free-tier daily
  quota (1500/day), meaning distractor generation alone takes multiple
  days to build even before the main 75,600-call experiment starts. This
  is recorded here explicitly so a future reader does not mistake it for
  an oversight: it is the direct consequence of using empirical sampling
  instead of logprobs to measure confidence (see config.py).
- The clean baseline confidence is measured once per instance and reused
  across all 3 intensities' scoring (rather than remeasured per
  intensity), since it does not depend on intensity — this is the single
  biggest lever build_benchmark.py has for reducing the call count above
  without changing the scoring methodology itself.
"""

import config
from experiments.correctness import check_correctness
from experiments.prompt_builder import build_prompt
from models.gemini_client import GeminiClient

logger = config.get_logger(__name__)


def measure_confidence(instance: dict, prompt: str, target_client: GeminiClient) -> float:
    """
    WHAT: Estimates the target model's "confidence" in the correct answer
          for a given prompt, as the fraction of config.NUM_SCORING_SAMPLES
          independent samples that are correct.

    WHY: See module and config.py DESIGN DECISIONS — this project's three
         provider APIs do not expose usable per-token probabilities for
         multi-step reasoning output, so confidence is instead measured
         empirically via repeated sampling, using the exact same
         correctness check (experiments/correctness.py) used everywhere
         else in the project.

    Args:
        instance (dict): Normalized instance dict; used to check each
                          sample's correctness.
        prompt (str): The exact prompt to send — either the clean prompt
                       or a prompt with a candidate distractor injected,
                       built by experiments/prompt_builder.py.
        target_client (GeminiClient): The reference model being scored
                                       against (config.DISTRACTOR_SCORING_MODEL_KEY).

    Returns:
        float: Fraction of samples judged correct, in [0.0, 1.0].
    """
    correct_count = 0
    for _ in range(config.NUM_SCORING_SAMPLES):
        response = target_client.query(prompt, token_budget=config.SCORING_TOKEN_BUDGET)
        if check_correctness(instance["task_type"], response.answer_text, instance):
            correct_count += 1
    return correct_count / config.NUM_SCORING_SAMPLES


def score_candidates(
    instance: dict,
    candidates: list[str],
    intensity: str,
    target_client: GeminiClient,
    clean_confidence: float,
) -> tuple[str, float]:
    """
    WHAT: Measures each candidate's confidence drop and returns the
          candidate with the highest drop.

    WHY: Implements the "keep the variant with the highest confidence
         drop" step of the adaptive method — a dedicated function (rather
         than inlining this in build_benchmark.py) keeps the selection
         rule (max drop, ties broken by first-generated) independently
         testable — see tests/test_generator.py.

    Args:
        instance (dict): Normalized instance dict.
        candidates (list[str]): Candidate distractor texts from
                                 distractors/generator.py.generate_candidates().
        intensity (str): The intensity level these candidates were
                          generated for — used only to log a warning
                          against config.EXPECTED_CONFIDENCE_DROP, not to
                          change scoring behavior.
        target_client (GeminiClient): The reference model being scored
                                       against.
        clean_confidence (float): This instance's confidence on the
                                   clean (no-distractor) prompt, from
                                   measure_confidence() — passed in
                                   rather than remeasured here, since it
                                   is shared across all 3 intensities
                                   (see module DESIGN DECISIONS).

    Returns:
        tuple[str, float]: (best candidate text, its confidence drop).
                            Confidence drop is clean_confidence minus the
                            candidate's noisy confidence; can be negative
                            if a candidate happened to make the model
                            *more* often correct across the sampled runs,
                            which is a real (if unlikely) possible
                            outcome of a noisy few-sample estimate and is
                            not clamped to zero, so it remains visible in
                            saved results rather than being hidden.

    Raises:
        ValueError: If `candidates` is empty (distractors/generator.py
                    returned no usable candidates for this instance).
    """
    if not candidates:
        raise ValueError(
            f"No candidates to score for instance={instance['id']} intensity={intensity}"
        )

    best_candidate = None
    best_drop = float("-inf")
    for candidate_text in candidates:
        noisy_prompt = build_prompt(
            instance, f"{intensity}_{config.SCORING_POSITION}", distractor_text=candidate_text
        )
        noisy_confidence = measure_confidence(instance, noisy_prompt, target_client)
        drop = clean_confidence - noisy_confidence
        if drop > best_drop:
            best_drop = drop
            best_candidate = candidate_text

    expected_low, expected_high = config.EXPECTED_CONFIDENCE_DROP[intensity]
    if best_drop < expected_low:
        logger.warning(
            "instance=%s intensity=%s: best confidence drop %.2f is below the "
            "expected range [%.2f, %.2f] — the generated distractors may be weak "
            "for this instance.",
            instance["id"], intensity, best_drop, expected_low, expected_high,
        )

    return best_candidate, best_drop


if __name__ == "__main__":
    # Manual smoke test: requires GOOGLE_API_KEY to be set in .env.
    # Scores 3 hand-written candidates for one real instance so a human
    # can confirm confidence measurement and selection work end-to-end
    # before build_benchmark.py depends on it for all 600 instances.
    example_instance = {
        "id": "gsm8k_0000",
        "task_type": "math",
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes muffins with 4. She sells the rest for $2 each. How much does she make daily?",
        "answer": "18",
        "choices": None,
        "context": "",
        "split": "test",
    }
    candidates = [
        "Ducks are known to lay eggs seasonally, with peak production in spring.",
        "Janet's ducks actually produce 12 eggs a day according to a nearby farm survey, not 16.",
        "A dozen eggs typically costs $3 at most farmers' markets across the region.",
    ]

    client = GeminiClient(config.MODEL_SPECS[config.DISTRACTOR_SCORING_MODEL_KEY]["model_id"])
    clean_prompt = build_prompt(example_instance, "clean")
    clean_conf = measure_confidence(example_instance, clean_prompt, client)
    print(f"Clean confidence: {clean_conf:.2f}")

    best, drop = score_candidates(example_instance, candidates, "high", client, clean_conf)
    print(f"Best candidate (drop={drop:.2f}): {best}")
