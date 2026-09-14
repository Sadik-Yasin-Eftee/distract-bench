"""
gemini_client.py

WHAT: ModelClient implementation for Gemini reasoning models
      (gemini-3.5-flash, gemini-3.6-flash), with explicit thinking-token
      budget control.

WHY: This is the project's primary reasoning-model family — the one
     whose thinking_budget parameter gives direct, numeric control over
     the x-axis of every inverse scaling curve (config.TOKEN_BUDGETS).
     Also used, at a different model id, as the distractor-generator
     model in distractors/generator.py.

RESEARCH STEP: Week 2, Step 10.

INPUT: A prompt string and a token_budget int, via the inherited
       query(prompt, token_budget) interface.

OUTPUT: A models.base_client.ModelResponse with `reasoning_text` set to
        the model's thought summary (if the API returns one) and
        `answer_text` set to the final response text.

DEPENDS ON: models/base_client.py (retry loop, ModelResponse), config.py
            (API key, model ids, retry constants). Used by
            experiments/run_scaling_curves.py and distractors/generator.py.

DESIGN DECISIONS:
- Uses the `google-genai` SDK (not the deprecated `google.generativeai`
  named in the original project spec) — see config.py's MODEL_SPECS
  comments and requirements.txt for the full deviation rationale: the old
  SDK has no thinking_budget field at all, and is deprecated by Google.
- `token_budget` is passed as `types.ThinkingConfig(thinking_budget=...)`,
  which caps thinking tokens specifically, separate from the final
  answer. `max_output_tokens` is deliberately left unset (uncapped) so a
  small thinking_budget can never also truncate the final answer text
  itself — otherwise a "256 budget, wrong answer" result could mean
  either "the model reasoned worse with less budget" (the effect under
  study) or "the model's answer got cut off before it finished" (a
  measurement artifact), and the two must not be conflated.
- Validates `model_id` against the live `client.models.list()` result at
  construction time (not at every query()) and raises immediately with a
  clear message if the configured model is not currently being served.
  WHY: Google has been retiring Gemini model generations roughly every
  few months in 2026 (see config.py) — failing fast at startup, before
  any of the 75,600 calls are attempted, surfaces a dead model id as a
  one-line error instead of a wall of retry-then-fail log spam from
  base_client.py's retry loop treating a 404 as if it might be transient.
"""

import time

from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

import config
from models.base_client import ModelClient, ModelResponse

logger = config.get_logger(__name__)


class GeminiClient(ModelClient):
    """
    WHAT: ModelClient for Gemini reasoning models via the google-genai SDK.
    """

    # ServerError covers Gemini's 5xx responses (transient backend
    # issues); ClientError covers 4xx, which includes 429 (rate limit) —
    # both are retried by base_client.py's loop. A ClientError from a
    # genuinely bad request (e.g. malformed prompt) would also be caught
    # here and retried up to config.MAX_RETRIES times before failing,
    # which is a deliberate tradeoff: the google-genai SDK does not
    # subclass 429 separately from other 4xx codes, so distinguishing
    # "retry this" from "this will never succeed" would require parsing
    # the exception's status code manually. Given retries are capped at 5
    # attempts with bounded backoff (config.MAX_RETRIES,
    # config.MAX_BACKOFF_SECONDS), the cost of retrying a truly
    # non-recoverable 4xx a few extra times is small compared to the risk
    # of misclassifying an actual rate limit as non-recoverable and
    # aborting the run early.
    retryable_exceptions = (ServerError, ClientError)

    def __init__(self, model_id: str):
        """
        Args:
            model_id (str): A Gemini model id, e.g. "gemini-3.5-flash".
                             Validated against the live model list before
                             this constructor returns.

        Raises:
            RuntimeError: If config.GOOGLE_API_KEY is not set, or if
                          `model_id` is not present in the account's
                          currently-servable model list.
        """
        super().__init__(model_id)
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self._validate_model_id()

    def _validate_model_id(self) -> None:
        """
        WHAT: Confirms `self.model_id` is in the live list of models the
              configured API key can call.

        WHY: See module DESIGN DECISIONS — fails fast on a decommissioned
             model id rather than discovering it 5 retries into the first
             query() call.

        Raises:
            RuntimeError: If `self.model_id` is not found in
                          `self._client.models.list()`. The error message
                          lists the actually-available model ids
                          containing "flash" or "pro" to point toward a
                          likely replacement, per Google's own
                          fast-moving 2026 deprecation cadence documented
                          in config.py.
        """
        available = {m.name.removeprefix("models/") for m in self._client.models.list()}
        if self.model_id not in available:
            candidates = sorted(m for m in available if "flash" in m or "pro" in m)
            raise RuntimeError(
                f"Gemini model {self.model_id!r} is not available to this API key "
                f"(it may have been deprecated — Google has retired several Gemini "
                f"generations throughout 2026). Update config.MODEL_SPECS. "
                f"Currently available flash/pro models: {candidates}"
            )

    def _query_once(self, prompt: str, token_budget: int) -> ModelResponse:
        """
        WHAT: Makes one Gemini generate_content call with thinking_budget
              set to `token_budget`.

        WHY: See module DESIGN DECISIONS for why thinking_budget (not
             max_output_tokens) is used to operationalize the project's
             token-budget independent variable.

        Args:
            prompt (str): Full prompt text.
            token_budget (int): Passed straight through as
                                 ThinkingConfig.thinking_budget.

        Returns:
            ModelResponse: `reasoning_text` from the response's thought
                           summary if present (else None), `answer_text`
                           from the response's final text.
        """
        start = time.monotonic()

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=token_budget,
                    # WHY True: without requesting thought summaries, the
                    # API only returns the final answer text, which would
                    # leave ModelResponse.reasoning_text permanently None
                    # for Gemini — defeating Week 4's attention/probe
                    # analysis need to distinguish reasoning content from
                    # final-answer content.
                    include_thoughts=True,
                ),
            ),
        )
        elapsed = time.monotonic() - start

        answer_text = response.text or ""
        reasoning_text = self._extract_reasoning_text(response)

        return ModelResponse(
            answer_text=answer_text,
            reasoning_text=reasoning_text,
            model_id=self.model_id,
            token_budget=token_budget,
            elapsed_seconds=elapsed,
            raw_metadata={
                "usage_metadata": (
                    response.usage_metadata.model_dump()
                    if response.usage_metadata else {}
                ),
            },
        )

    @staticmethod
    def _extract_reasoning_text(response) -> str | None:
        """
        WHAT: Pulls thought-summary text out of a GenerateContentResponse,
              if present.

        WHY: Thought summaries are returned as parts with `thought=True`
             inside the candidate's content, mixed in with the regular
             answer parts — this isolates just the thinking parts, since
             `response.text` (used for answer_text above) already
             concatenates only the non-thought parts.

        Args:
            response: The raw GenerateContentResponse from
                      self._client.models.generate_content().

        Returns:
            str | None: Concatenated thought-summary text, or None if
                        the response has no candidates or no parts marked
                        as thoughts (e.g. include_thoughts had no effect
                        for this particular call).
        """
        if not response.candidates:
            return None
        content = response.candidates[0].content
        if content is None or not content.parts:
            return None
        thought_parts = [p.text for p in content.parts if getattr(p, "thought", False) and p.text]
        return "\n".join(thought_parts) if thought_parts else None


if __name__ == "__main__":
    # Manual smoke test: requires GOOGLE_API_KEY to be set in .env.
    # Confirms the client constructs, validates its model id against the
    # live API, and can complete one real query before
    # run_scaling_curves.py depends on it for thousands of calls.
    client = GeminiClient(config.MODEL_SPECS["gemini_primary"]["model_id"])
    result = client.query("What is 2 + 2? Answer with just the number.", token_budget=256)
    print(result)
