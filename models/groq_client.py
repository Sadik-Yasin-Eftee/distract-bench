"""
groq_client.py

WHAT: ModelClient implementation for the project's non-reasoning
      baseline, served via Groq.

WHY: Every reasoning model in this study (Gemini, DeepSeek) spends extra
     compute on an explicit thinking phase before answering. To claim
     that inverse scaling under distractor noise is specific to that
     thinking mechanism — rather than a generic property of "any LLM
     given a larger token allowance" — the study needs a model with no
     thinking phase at all, run through the exact same 7 conditions and
     6 nominal "budgets."

RESEARCH STEP: Week 2, Step 12.

INPUT: A prompt string and a token_budget int, via the inherited
       query(prompt, token_budget) interface. token_budget is accepted
       for interface compatibility but has no effect on this model's
       behavior (see below).

OUTPUT: A models.base_client.ModelResponse with `reasoning_text` always
        None (Groq's instruct models return only a direct answer) and
        `answer_text` set to the full response text.

DEPENDS ON: models/base_client.py, config.py. Used by
            experiments/run_scaling_curves.py.

DESIGN DECISIONS:
- `token_budget` is documented as ignored, not silently accepted and
  forgotten: `query()`'s ModelResponse still records the requested
  budget (inherited from base_client.py), so a saved result row still
  shows which nominal budget the call was made "at," but
  analysis/plot_curves.py must show this baseline as a flat line across
  budgets rather than a real curve — this is why config.MODEL_SPECS
  marks `is_reasoning_model: False`, so run_scaling_curves.py and the
  analysis scripts can treat it differently without guessing from the
  model name.
- DEVIATION FROM INITIAL SPEC — model id resolved at runtime, not
  hardcoded: the project design named "llama-3.3-70b-versatile" as a
  fixed model id. Verified via web search (2026-09-14) that Groq
  decommissioned that exact model on 2026-08-16, and has been retiring
  models on a roughly monthly cadence throughout 2026 (llama-3.1-8b-
  instant announced deprecated the same cycle; qwen/qwen3-32b and
  llama-4-scout-17b-16e-instruct shut down 2026-07-17). A second
  hardcoded guess would face the same risk of being dead by the time
  this project's Week 2-3 experiments actually run. Instead,
  config.MODEL_SPECS["llama_baseline"]["model_id_candidates"] lists
  several plausible non-reasoning instruct models in preference order,
  and this client picks the first one Groq is actually still serving at
  construction time — this is a general defense against Groq's
  decommission cadence, not a one-time fix.
"""

import time

from groq import Groq, APIStatusError, APIConnectionError

import config
from models.base_client import ModelClient, ModelResponse

logger = config.get_logger(__name__)


class GroqClient(ModelClient):
    """
    WHAT: ModelClient for the non-reasoning Llama baseline via Groq.
    """

    retryable_exceptions = (APIStatusError, APIConnectionError)

    def __init__(self, model_id_candidates: tuple[str, ...]):
        """
        Args:
            model_id_candidates (tuple[str, ...]): Candidate model ids in
                                                     preference order (see
                                                     config.MODEL_SPECS).
                                                     The first candidate
                                                     found in Groq's live
                                                     model list is used.

        Raises:
            RuntimeError: If config.GROQ_API_KEY is not set, or if none
                          of `model_id_candidates` are currently being
                          served.
        """
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = Groq(api_key=config.GROQ_API_KEY)
        resolved_model_id = self._resolve_model_id(model_id_candidates)
        super().__init__(resolved_model_id)

    def _resolve_model_id(self, candidates: tuple[str, ...]) -> str:
        """
        WHAT: Picks the first of `candidates` that Groq's live model list
              currently serves.

        WHY: See module DESIGN DECISIONS — Groq's model roster has been
             changing faster than this project's own timeline, so a
             single hardcoded id is not durable across even a 5-week
             project.

        Args:
            candidates (tuple[str, ...]): Candidate model ids, in
                                           preference order.

        Returns:
            str: The first candidate present in the live model list.

        Raises:
            RuntimeError: If none of `candidates` are currently served —
                          lists the live model ids Groq's docs classify
                          as `meta-llama` or `llama` so a human can pick
                          a replacement and update
                          config.MODEL_SPECS["llama_baseline"]
                          ["model_id_candidates"].
        """
        available = {m.id for m in self._client.models.list().data}
        for candidate in candidates:
            if candidate in available:
                logger.info("Resolved Groq baseline model to %s", candidate)
                return candidate

        llama_like = sorted(m for m in available if "llama" in m.lower())
        raise RuntimeError(
            f"None of the candidate Groq models {candidates} are currently "
            f"available (Groq deprecates models frequently — see "
            f"https://console.groq.com/docs/deprecations). Update "
            f"config.MODEL_SPECS['llama_baseline']['model_id_candidates']. "
            f"Currently available llama-family models: {llama_like}"
        )

    def _query_once(self, prompt: str, token_budget: int) -> ModelResponse:
        """
        WHAT: Makes one Groq chat completion call. `token_budget` is
              accepted for interface compatibility but not passed to the
              API as a generation-limiting parameter.

        WHY: See module DESIGN DECISIONS — this model has no thinking
             phase, so there is no "budget" for it to spend; passing
             `token_budget` as `max_tokens` here would conflate "ran out
             of budget to think" (the effect under study for the two
             reasoning models) with "response got truncated," which is
             not the phenomenon this baseline is meant to control for.

        Args:
            prompt (str): Full prompt text, sent as a single user message.
            token_budget (int): Recorded on the returned ModelResponse for
                                 traceability, but otherwise unused.

        Returns:
            ModelResponse: `reasoning_text` always None; `answer_text`
                           the full response text.
        """
        start = time.monotonic()

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.monotonic() - start

        answer_text = response.choices[0].message.content or ""

        return ModelResponse(
            answer_text=answer_text,
            reasoning_text=None,
            model_id=self.model_id,
            token_budget=token_budget,
            elapsed_seconds=elapsed,
            raw_metadata={
                "usage": response.usage.model_dump() if response.usage else {},
                "finish_reason": response.choices[0].finish_reason,
            },
        )


if __name__ == "__main__":
    # Manual smoke test: requires GROQ_API_KEY to be set in .env.
    client = GroqClient(config.MODEL_SPECS["llama_baseline"]["model_id_candidates"])
    result = client.query("What is 2 + 2? Answer with just the number.", token_budget=256)
    print(result)
