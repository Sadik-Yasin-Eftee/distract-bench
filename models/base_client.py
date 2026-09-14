"""
base_client.py

WHAT: Abstract base class defining the shared query(prompt, token_budget)
      interface and retry/backoff logic used by all three model clients.

WHY: experiments/run_scaling_curves.py must treat Gemini, DeepSeek, and
     Groq identically — it loops over models and calls the same method on
     each. Without a shared base class, the retry-on-rate-limit logic
     (5 attempts, exponential backoff, 60s cap — config.py's RATE LIMIT
     HANDLING section) would need to be copy-pasted into all three
     provider-specific clients, and a bug fix in one copy could easily be
     missed in the other two.

RESEARCH STEP: Week 2, Step 9. Subclassed by models/gemini_client.py,
               models/deepseek_client.py, models/groq_client.py.

INPUT: N/A (abstract).

OUTPUT: N/A (abstract). Concrete subclasses' query() returns a
        ModelResponse.

DEPENDS ON: config.py (retry constants, logger). Subclassed by every file
            in models/. Used by experiments/run_scaling_curves.py and
            distractors/scorer.py (which queries a target model to
            measure confidence drop).

DESIGN DECISIONS:
- Retry logic lives in this base class's concrete `query()` method, which
  calls an abstract `_query_once()` that subclasses implement — this
  means the backoff loop itself (and its unit-testable behavior: retries
  N times, waits exponentially, gives up after MAX_RETRIES) is written
  and tested exactly once, while each subclass only needs to implement
  the provider-specific API call and declare which exception types from
  its own SDK mean "rate limited, retry me."
- `retryable_exceptions` is a class attribute (a tuple of exception
  types) rather than a hardcoded set of `except` clauses in the base
  class, because Gemini, DeepSeek (OpenAI-compatible), and Groq each
  raise different SDK-specific exception classes for a rate limit — the
  base class's retry loop stays provider-agnostic by catching whatever
  tuple the subclass declares.
- ModelResponse always carries both `reasoning_text` and `answer_text`
  separately (even though Llama/Groq never populates `reasoning_text`),
  rather than a single `text` field, because Gemini and DeepSeek expose
  their thinking-token content distinctly from the final answer — Week 4's
  attention/probe analysis (analysis/attention_entropy.py,
  analysis/linear_probe.py) needs to know how much of the response was
  "thinking" vs. "answer," which a merged field would lose.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import config

logger = config.get_logger(__name__)


@dataclass
class ModelResponse:
    """
    WHAT: The normalized result of one query() call, regardless of
          provider.

    WHY: experiments/run_scaling_curves.py and experiments/save_results.py
         need one consistent shape to log and save, whether the
         underlying call was to Gemini, DeepSeek, or Groq.

    Attributes:
        answer_text (str): The model's final answer text — for a
                            reasoning model, this is the content *after*
                            its thinking/reasoning block; for Llama, the
                            entire response (Llama has no separate
                            reasoning phase).
        reasoning_text (str | None): The model's thinking/reasoning
                                      content, if the provider exposes it
                                      separately (Gemini, DeepSeek). None
                                      for Llama, and None for Gemini/
                                      DeepSeek responses where the API
                                      did not return a distinct reasoning
                                      field (this is a real possibility —
                                      not every Gemini response includes
                                      thought summaries — and is recorded
                                      as None rather than "" so a caller
                                      can distinguish "no reasoning
                                      exposed" from "reasoning was
                                      exposed and was empty").
        model_id (str): The provider's model identifier actually used
                         (e.g. "gemini-2.0-flash-thinking-exp"), recorded
                         per-response for the REPRODUCIBILITY requirement
                         of saving exact model versions used.
        token_budget (int): The token_budget this query() call was made
                             with, so a saved result is self-describing
                             without needing to be joined back to the
                             experiment loop's iteration variables.
        elapsed_seconds (float): Wall-clock time for the (possibly
                                  retried) call, logged per the
                                  engineering requirements' LOGGING
                                  section.
        raw_metadata (dict): Provider-specific extra fields (e.g. actual
                              output token count, finish reason) kept as
                              a free-form dict rather than named fields,
                              since each provider exposes different
                              metadata and forcing a common schema here
                              would either drop provider-specific detail
                              or require constantly extending this class.
    """

    answer_text: str
    reasoning_text: str | None
    model_id: str
    token_budget: int
    elapsed_seconds: float
    raw_metadata: dict = field(default_factory=dict)


class ModelClient(ABC):
    """
    WHAT: Abstract base class for a model API client with retry/backoff.

    WHY: See module docstring — centralizes the retry loop so it is
         written and tested once.
    """

    # Subclasses must override with a tuple of exception classes from
    # their own SDK that indicate a transient, retryable failure (rate
    # limit, timeout, transient 5xx) as opposed to a non-recoverable one
    # (bad API key, malformed request) that should propagate immediately.
    retryable_exceptions: tuple = ()

    def __init__(self, model_id: str):
        """
        Args:
            model_id (str): The provider's model identifier this client
                             instance targets (e.g. "deepseek-reasoner").
                             Stored so it can be attached to every
                             ModelResponse without the caller repeating it.
        """
        self.model_id = model_id

    @abstractmethod
    def _query_once(self, prompt: str, token_budget: int) -> ModelResponse:
        """
        WHAT: Makes exactly one API call (no retry) and returns a
              ModelResponse.

        WHY: Kept separate from the public query() so the retry loop
             below can wrap it uniformly — a subclass never needs to
             implement retry logic itself.

        Args:
            prompt (str): The full prompt text, as built by
                          experiments/prompt_builder.py.
            token_budget (int): The reasoning/output token budget for
                                 this call. Interpreted differently per
                                 provider — see each subclass's own
                                 docstring (e.g. Gemini's thinking_budget
                                 generation-config field vs. Groq's
                                 documented no-op, since Llama has no
                                 thinking-token mechanism).

        Returns:
            ModelResponse: The parsed result of this single API call.

        Raises:
            Exception: Any exception from the underlying SDK. Exceptions
                       whose type is in `self.retryable_exceptions` are
                       caught and retried by query(); all others
                       propagate immediately as non-recoverable.
        """
        raise NotImplementedError

    def query(self, prompt: str, token_budget: int) -> ModelResponse:
        """
        WHAT: Calls _query_once(), retrying on the subclass's declared
              retryable exceptions with exponential backoff.

        WHY: Implements the project's RATE LIMIT HANDLING requirement
             (config.MAX_RETRIES attempts, backoff doubling from
             config.BACKOFF_BASE_SECONDS up to config.MAX_BACKOFF_SECONDS)
             in one place so every model client behaves identically under
             rate limiting, rather than each provider's client
             reimplementing (and potentially subtly diverging in) the
             same loop.

        Args:
            prompt (str): Passed through to _query_once().
            token_budget (int): Passed through to _query_once().

        Returns:
            ModelResponse: The result of the first successful call.

        Raises:
            Exception: Re-raises the last retryable exception if all
                       config.MAX_RETRIES attempts are exhausted, or
                       immediately re-raises any non-retryable exception
                       from the first attempt — per the engineering
                       requirement to never silently swallow exceptions.
        """
        last_exception = None
        for attempt in range(config.MAX_RETRIES):
            try:
                return self._query_once(prompt, token_budget)
            except self.retryable_exceptions as exc:
                last_exception = exc
                # Exponential backoff starting at BACKOFF_BASE_SECONDS,
                # capped at MAX_BACKOFF_SECONDS. WHY this formula: doubling
                # per attempt spreads retries out automatically without
                # needing to parse provider-specific "retry after N
                # seconds" headers (which not all three SDKs expose
                # uniformly), while the cap keeps a single stuck instance
                # from stalling the 5-week schedule waiting on one retry.
                wait_time = min(
                    config.MAX_BACKOFF_SECONDS,
                    config.BACKOFF_BASE_SECONDS * (2 ** attempt),
                )
                logger.warning(
                    "model=%s attempt=%d/%d retryable error: %s — waiting %ds",
                    self.model_id, attempt + 1, config.MAX_RETRIES, exc, wait_time,
                )
                time.sleep(wait_time)

        logger.error(
            "model=%s exhausted %d retries; last error: %s",
            self.model_id, config.MAX_RETRIES, last_exception,
        )
        raise last_exception
