"""
generator.py

WHAT: Generates NUM_DISTRACTOR_CANDIDATES candidate distractor paragraphs
      for one (instance, intensity) pair, using the three prompt
      templates from the project's research design.

WHY: This is the "propose candidates" half of Wang et al. 2025's adaptive
     tree-search distractor method — distractors/scorer.py then measures
     each candidate's actual effect on the target model and keeps the
     strongest one. Separating generation from scoring means either half
     can be tested/swapped independently (e.g. a future extension trying
     a different generator model only touches this file).

RESEARCH STEP: Week 1, Step 6.

INPUT: A normalized instance dict (data/normalize.py) and an intensity
       level (one of config.DISTRACTOR_INTENSITIES).

OUTPUT: list[str] of NUM_DISTRACTOR_CANDIDATES candidate distractor
        paragraphs.

DEPENDS ON: config.py (generator model, candidate count), models/gemini_client.py
            (to call the generator model). Used by
            distractors/build_benchmark.py (directly) and
            distractors/scorer.py (indirectly, via build_benchmark.py's
            orchestration).

DESIGN DECISIONS:
- The three prompt templates below are reproduced verbatim from the
  project's research design document — not paraphrased — because
  scorer.py's confidence-drop measurement is only meaningful if every
  candidate was generated from the exact wording the design specifies;
  changing the wording would make "high intensity" candidates
  incomparable to what EXPECTED_CONFIDENCE_DROP's ranges (config.py) were
  set relative to.
- Candidate diversity across the 3 calls per intensity comes from the
  generator model's own default sampling temperature (not an explicit
  temperature parameter passed by this code) — the google-genai SDK's
  default generation temperature already produces different phrasing
  across independent calls with the same prompt, which is sufficient for
  Wang et al. 2025's "3 candidates, keep the best" method; adding an
  explicit temperature override would be an untested deviation from the
  research design's own prompt templates for no demonstrated benefit.
- Uses GENERATOR_TOKEN_BUDGET = 0 (thinking disabled) since generating a
  2-3 sentence paragraph from a template does not benefit from an
  extended reasoning phase — see config.py's comment on this constant.
"""

import config
from models.gemini_client import GeminiClient

logger = config.get_logger(__name__)

_LOW_INTENSITY_TEMPLATE = """Write a short paragraph (2-3 sentences) about the general topic \
of this question, but do NOT include any information that would \
help answer it. The paragraph should be clearly off-topic from \
the specific question being asked.
Question: {question}
Write only the paragraph, nothing else."""

_MEDIUM_INTENSITY_TEMPLATE = """Write a short paragraph (2-3 sentences) that is in the same \
domain as this question and uses related vocabulary, but does \
not contain the specific information needed to answer it. \
It should sound relevant but actually be unhelpful.
Question: {question}
Write only the paragraph, nothing else."""

_HIGH_INTENSITY_TEMPLATE = """Write a short paragraph (2-3 sentences) that uses the exact \
numbers, names, and vocabulary from this question but describes \
a DIFFERENT scenario that would lead someone to the WRONG answer. \
Make it sound authoritative and plausible. Do not include the \
correct answer.
Question: {question}
Correct answer (do NOT include this): {answer}
Write only the paragraph, nothing else."""

_TEMPLATES = {
    "low": _LOW_INTENSITY_TEMPLATE,
    "medium": _MEDIUM_INTENSITY_TEMPLATE,
    "high": _HIGH_INTENSITY_TEMPLATE,
}


def _build_generation_prompt(instance: dict, intensity: str) -> str:
    """
    WHAT: Fills in the intensity-specific template with this instance's
          question (and, for "high" intensity, its ground-truth answer).

    WHY: A dedicated function keeps the "which fields does each template
         need" logic in one place — only "high" needs `answer`, and
         forgetting that when adding a new intensity level in a future
         extension would raise a clear KeyError here rather than
         silently formatting garbage into the prompt.

    Args:
        instance (dict): Normalized instance dict; uses `question`
                          always, `answer` only for "high" intensity.
        intensity (str): One of config.DISTRACTOR_INTENSITIES.

    Returns:
        str: The filled-in prompt to send to the generator model.

    Raises:
        ValueError: If intensity is not recognized.
    """
    if intensity not in _TEMPLATES:
        raise ValueError(f"Unknown intensity {intensity!r}; expected one of {config.DISTRACTOR_INTENSITIES}")

    if intensity == "high":
        return _HIGH_INTENSITY_TEMPLATE.format(question=instance["question"], answer=instance["answer"])
    return _TEMPLATES[intensity].format(question=instance["question"])


def generate_candidates(instance: dict, intensity: str, generator_client: GeminiClient) -> list[str]:
    """
    WHAT: Generates config.NUM_DISTRACTOR_CANDIDATES candidate distractor
          paragraphs for one (instance, intensity) pair.

    WHY: A dedicated function (rather than inlining the loop in
         build_benchmark.py) makes this step independently unit-testable
         with a mocked generator_client — see tests/test_generator.py.

    Args:
        instance (dict): Normalized instance dict (data/normalize.py).
        intensity (str): One of config.DISTRACTOR_INTENSITIES.
        generator_client (GeminiClient): A client already constructed
                                          against config.DISTRACTOR_GENERATOR_MODEL
                                          — passed in (rather than
                                          constructed here) so
                                          build_benchmark.py can build one
                                          client and reuse it across all
                                          600 instances instead of paying
                                          the model-list validation call
                                          (see GeminiClient.__init__)
                                          600*3 times.

    Returns:
        list[str]: config.NUM_DISTRACTOR_CANDIDATES candidate paragraphs,
                   stripped of leading/trailing whitespace. May contain
                   fewer than NUM_DISTRACTOR_CANDIDATES entries if the
                   model returned an empty response for some calls (this
                   is logged as a warning rather than raised, since a
                   downstream candidate with fewer options to score from
                   is a degraded-but-survivable outcome, unlike an
                   unhandled exception aborting the whole benchmark
                   build).
    """
    prompt = _build_generation_prompt(instance, intensity)
    candidates = []
    for i in range(config.NUM_DISTRACTOR_CANDIDATES):
        response = generator_client.query(prompt, token_budget=config.GENERATOR_TOKEN_BUDGET)
        text = response.answer_text.strip()
        if not text:
            logger.warning(
                "Empty distractor candidate %d/%d for instance=%s intensity=%s",
                i + 1, config.NUM_DISTRACTOR_CANDIDATES, instance["id"], intensity,
            )
            continue
        candidates.append(text)
    return candidates


if __name__ == "__main__":
    # Manual smoke test: requires GOOGLE_API_KEY to be set in .env.
    # Generates candidates for one real instance at each intensity level
    # so a human can visually confirm the templates produce sensible
    # output before scorer.py and build_benchmark.py depend on this.
    example_instance = {
        "id": "gsm8k_0000",
        "task_type": "math",
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes muffins with 4. She sells the rest for $2 each. How much does she make daily?",
        "answer": "18",
        "choices": None,
        "context": "",
        "split": "test",
    }
    client = GeminiClient(config.DISTRACTOR_GENERATOR_MODEL)
    for intensity in config.DISTRACTOR_INTENSITIES:
        print(f"\n=== {intensity} ===")
        for candidate in generate_candidates(example_instance, intensity, client):
            print(f"- {candidate}")
