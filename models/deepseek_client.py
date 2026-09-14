"""
deepseek_client.py

WHAT: ModelClient implementation for DeepSeek's reasoning model
      (deepseek-flash, thinking mode enabled), via its OpenAI-compatible
      API.

WHY: This is the project's second reasoning-model *family* — an
     open-weight model trained with RLVR, needed to check whether inverse
     scaling under distractor noise generalizes beyond Gemini's training
     pipeline.

RESEARCH STEP: Week 2, Step 11.

INPUT: A prompt string and a token_budget int, via the inherited
       query(prompt, token_budget) interface.

OUTPUT: A models.base_client.ModelResponse with `reasoning_text` set from
        the API's `reasoning_content` field and `answer_text` from the
        standard `content` field.

DEPENDS ON: models/base_client.py, config.py. Used by
            experiments/run_scaling_curves.py.

DESIGN DECISIONS:
- Uses the `openai` Python SDK pointed at DeepSeek's base URL, per
  config.py's requirements.txt comment — DeepSeek's API is
  OpenAI-compatible, so a second bespoke SDK is unnecessary.
- DEVIATION FROM INITIAL SPEC: the project design named model id
  "deepseek-reasoner" with reasoning presumably controlled by a token
  count. Verified via web search (2026-09-14) that DeepSeek retired the
  "deepseek-reasoner" alias on 2026-07-24; the current model,
  "deepseek-flash", controls its thinking mode via a qualitative
  `reasoning_effort` parameter (low/high/max) rather than a numeric
  budget. This project's token-budget sweep is instead applied as a hard
  `max_tokens` ceiling on the whole completion (reasoning + answer
  combined) — the model is always run with thinking explicitly enabled
  and `reasoning_effort="high"` held constant, so the *only* thing that
  varies across the budget sweep for this model is how many tokens it is
  allowed to spend in total before being cut off, which is the same
  "compute ceiling" semantics config.TOKEN_BUDGETS is meant to sweep.
  This is a coarser proxy than Gemini's dedicated thinking-token cap
  (which never truncates the final answer — see gemini_client.py) and
  should be reported as a limitation in the paper: a DeepSeek response
  cut off by `max_tokens` mid-reasoning has no room left to write a final
  answer at all, which is a stronger and qualitatively different
  truncation effect than Gemini's design.
"""

import time

from openai import OpenAI, APIStatusError, APIConnectionError

import config
from models.base_client import ModelClient, ModelResponse

logger = config.get_logger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient(ModelClient):
    """
    WHAT: ModelClient for DeepSeek's deepseek-flash model in thinking mode.
    """

    # APIStatusError covers DeepSeek's HTTP error responses (4xx/5xx,
    # including 429 rate limits); APIConnectionError covers network-level
    # failures (timeouts, DNS, connection reset) — both are transient
    # classes of failure appropriate to retry via base_client.py's loop.
    retryable_exceptions = (APIStatusError, APIConnectionError)

    def __init__(self, model_id: str):
        """
        Args:
            model_id (str): A DeepSeek model id, e.g. "deepseek-flash".

        Raises:
            RuntimeError: If config.DEEPSEEK_API_KEY is not set, or if
                          `model_id` is not present in the account's
                          currently-servable model list.
        """
        super().__init__(model_id)
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._validate_model_id()

    def _validate_model_id(self) -> None:
        """
        WHAT: Confirms `self.model_id` is in the live model list DeepSeek
              serves to this API key.

        WHY: Mirrors gemini_client.py's fail-fast validation — DeepSeek
             retired its previous model aliases mid-2026 (see module
             docstring), so the same class of risk applies here.

        Raises:
            RuntimeError: If `self.model_id` is not found in
                          `self._client.models.list()`.
        """
        available = {m.id for m in self._client.models.list().data}
        if self.model_id not in available:
            raise RuntimeError(
                f"DeepSeek model {self.model_id!r} is not available to this API "
                f"key (it may have been renamed or retired). Update "
                f"config.MODEL_SPECS. Currently available models: {sorted(available)}"
            )

    def _query_once(self, prompt: str, token_budget: int) -> ModelResponse:
        """
        WHAT: Makes one DeepSeek chat completion call with thinking mode
              enabled and `max_tokens` set to `token_budget`.

        WHY: See module DESIGN DECISIONS for why `max_tokens` (not a
             thinking-specific field, which the current API no longer
             exposes numerically) is used to operationalize the token
             budget for this model.

        Args:
            prompt (str): Full prompt text, sent as a single user message.
            token_budget (int): Passed as `max_tokens` — the hard ceiling
                                 on total completion tokens (reasoning +
                                 answer combined).

        Returns:
            ModelResponse: `reasoning_text` from the response message's
                           `reasoning_content` field (DeepSeek-specific,
                           accessed as a raw attribute since it is not
                           part of the standard OpenAI response schema),
                           `answer_text` from the standard `content` field.
        """
        start = time.monotonic()

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=token_budget,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        elapsed = time.monotonic() - start

        message = response.choices[0].message
        answer_text = message.content or ""
        # `reasoning_content` is a DeepSeek-specific field not modeled by
        # the openai SDK's typed Message class — accessed via getattr
        # with a None default so a future API response missing this
        # field (e.g. thinking mode silently disabled server-side)
        # degrades to "no reasoning captured" rather than an AttributeError.
        reasoning_text = getattr(message, "reasoning_content", None)

        return ModelResponse(
            answer_text=answer_text,
            reasoning_text=reasoning_text,
            model_id=self.model_id,
            token_budget=token_budget,
            elapsed_seconds=elapsed,
            raw_metadata={
                "usage": response.usage.model_dump() if response.usage else {},
                "finish_reason": response.choices[0].finish_reason,
            },
        )


if __name__ == "__main__":
    # Manual smoke test: requires DEEPSEEK_API_KEY to be set in .env.
    client = DeepSeekClient(config.MODEL_SPECS["deepseek_reasoning"]["model_id"])
    result = client.query("What is 2 + 2? Answer with just the number.", token_budget=256)
    print(result)
