"""
local_client.py

WHAT: Loads and runs the local, open-weight reasoning model used only for
      Week 4's attention-entropy and linear-probe analysis, exposing the
      raw attentions and hidden states no API-based client can provide.

WHY: See config.py's "LOCAL MODEL FOR ATTENTION / HIDDEN-STATE ANALYSIS"
     section for the full rationale — Gemini, DeepSeek, and Groq's hosted
     APIs return text only, never internal tensors, which the Week 4
     analysis fundamentally requires. This file is the only place in the
     project that loads model weights directly rather than calling an
     API.

RESEARCH STEP: Week 4, Steps 19-20 (prerequisite for both).

INPUT: A prompt string.

OUTPUT: A LocalGenerationResult carrying the generated answer text
        alongside per-layer attention weight tensors and hidden-state
        tensors for the full (prompt + answer) sequence.

DEPENDS ON: config.py (LOCAL_MODEL_ID, LOCAL_MAX_NEW_TOKENS). Used by
            analysis/attention_entropy.py and analysis/linear_probe.py.

DESIGN DECISIONS:
- Does NOT subclass models.base_client.ModelClient. WHY: that base class's
  contract (query(prompt, token_budget) -> ModelResponse, with
  retry/backoff on network exceptions) is specifically about API
  clients — a local model has no network calls to retry and needs to
  return tensors, not just text, so forcing it into that interface would
  mean either bloating ModelResponse with fields every API client leaves
  unused, or subclassing and immediately overriding most of the
  contract's meaning. A separate, purpose-built interface is clearer.
- Generation happens in two passes: first a plain `model.generate()` call
  to get the answer text (cheap, no attention/hidden-state materialization),
  then a single teacher-forced forward pass over the full
  (prompt + generated answer) sequence with `output_attentions=True,
  output_hidden_states=True`. WHY: collecting attentions at every
  autoregressive decoding step would multiply memory use by the number of
  generated tokens; one forward pass over the complete sequence gives the
  exact same attention/hidden-state tensors (this is standard practice
  for post-hoc interpretability analysis) at a fraction of the memory and
  code complexity.
- `attn_implementation="eager"` is required and set explicitly. WHY: the
  default/SDPA and flash-attention backends compute attention without
  ever materializing the full attention weight matrix (that is the whole
  point of those optimizations), so they cannot return `attentions` at
  all — only the unoptimized "eager" implementation does.
- Runs in float32 on the auto-detected device (MPS on Apple Silicon,
  else CPU). WHY float32 over a lower-precision dtype: PyTorch's MPS
  backend has had inconsistent bfloat16 op coverage across versions;
  float32 is the safest choice for correctness on Apple Silicon, and at
  1.5B parameters (~6GB in float32) comfortably fits the project's 24GB
  unified-memory hardware without needing quantization.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

logger = config.get_logger(__name__)


@dataclass
class LocalGenerationResult:
    """
    WHAT: The full output of one local-model generation + analysis pass.

    Attributes:
        prompt (str): The original input prompt.
        answer_text (str): The model's generated continuation (decoded
                            text of only the newly generated tokens).
        input_ids (torch.Tensor): Token ids of the full (prompt + answer)
                                   sequence, shape (1, seq_len). Needed to
                                   map attention/hidden-state positions
                                   back to specific tokens (e.g. "where
                                   does the distractor text start").
        prompt_length (int): Number of tokens in the prompt alone — the
                              boundary index separating prompt tokens from
                              generated-answer tokens within `input_ids`
                              and every tensor below.
        attentions (tuple[torch.Tensor, ...]): One tensor per transformer
                                                layer, each shaped
                                                (1, num_heads, seq_len,
                                                seq_len) — attention
                                                weights from the single
                                                teacher-forced forward
                                                pass over the full
                                                sequence.
        hidden_states (tuple[torch.Tensor, ...]): One tensor per layer
                                                    (plus the input
                                                    embedding layer, so
                                                    len == num_layers + 1,
                                                    matching
                                                    transformers'
                                                    convention), each
                                                    shaped
                                                    (1, seq_len, hidden_dim).
    """

    prompt: str
    answer_text: str
    input_ids: torch.Tensor
    prompt_length: int
    attentions: tuple
    hidden_states: tuple


class LocalReasoningModel:
    """
    WHAT: Wraps config.LOCAL_MODEL_ID for generation with full
          attention/hidden-state access.
    """

    def __init__(self, model_id: str = config.LOCAL_MODEL_ID):
        """
        Args:
            model_id (str): A HuggingFace Hub model id. Defaults to
                             config.LOCAL_MODEL_ID; overridable for
                             testing with a smaller model.
        """
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading local model %s on device=%s", model_id, self.device)

        # cache_dir=config.DATA_CACHE_DIR keeps this ~3GB download inside
        # the project directory, consistent with how data/load_datasets.py
        # caches HF datasets — see config.py's comment on DATA_CACHE_DIR.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(config.DATA_CACHE_DIR))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(config.DATA_CACHE_DIR),
            attn_implementation="eager",  # required to materialize attention weights — see module DESIGN DECISIONS
            torch_dtype=torch.float32,
            output_hidden_states=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_with_internals(self, prompt: str, max_new_tokens: int = config.LOCAL_MAX_NEW_TOKENS) -> LocalGenerationResult:
        """
        WHAT: Generates an answer for `prompt`, then runs one additional
              forward pass over the full (prompt + answer) sequence to
              extract attention weights and hidden states.

        WHY: See module DESIGN DECISIONS for why generation and
             internals-extraction are two separate passes.

        Args:
            prompt (str): Input prompt text.
            max_new_tokens (int): Generation length cap. Defaults to
                                   config.LOCAL_MAX_NEW_TOKENS.

        Returns:
            LocalGenerationResult: See its own docstring for field
                                   meanings.
        """
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_inputs["input_ids"].shape[1]

        generated = self.model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # WHY greedy: Week 4's analysis is about a
            # single representative forward pass's internals, not
            # sampling variance — deterministic generation makes a
            # given instance's attention/hidden-state extraction
            # reproducible across re-runs, matching the project's
            # REPRODUCIBILITY requirement.
            pad_token_id=self.tokenizer.eos_token_id,
        )
        answer_ids = generated[:, prompt_length:]
        answer_text = self.tokenizer.decode(answer_ids[0], skip_special_tokens=True)

        outputs = self.model(
            input_ids=generated,
            output_attentions=True,
            output_hidden_states=True,
        )

        return LocalGenerationResult(
            prompt=prompt,
            answer_text=answer_text,
            input_ids=generated,
            prompt_length=prompt_length,
            attentions=outputs.attentions,
            hidden_states=outputs.hidden_states,
        )


if __name__ == "__main__":
    # Manual smoke test: downloads config.LOCAL_MODEL_ID (~6GB in
    # float32) on first run, then runs one generation + internals
    # extraction to confirm the pipeline works before
    # analysis/attention_entropy.py and analysis/linear_probe.py depend
    # on it for all config.LOCAL_ANALYSIS_SAMPLE_SIZE instances.
    model = LocalReasoningModel()
    result = model.generate_with_internals("Question: What is 2 + 2?\nAnswer:", max_new_tokens=32)
    print("Answer:", result.answer_text)
    print("Num layers (attentions):", len(result.attentions))
    print("Attention shape (layer 0):", result.attentions[0].shape)
    print("Num hidden state tensors:", len(result.hidden_states))
    print("Hidden state shape (layer 0):", result.hidden_states[0].shape)
    print("Prompt length (tokens):", result.prompt_length)
