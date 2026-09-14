"""
attention_entropy.py

WHAT: Computes per-layer/per-head attention entropy, attention rollout,
      and clean-vs-noisy entropy comparisons for the local reasoning
      model, plus the heatmap visualizations the research design
      specifies.

WHY: Lower attention entropy under a distractor (attention concentrating
     on fewer tokens, e.g. the distractor itself) versus a flatter,
     higher-entropy distribution under the clean condition would be
     direct mechanistic evidence for *how* a distractor derails the
     model's reasoning, complementing the black-box accuracy-based
     scaling curves (Weeks 2-3) with a white-box explanation.

RESEARCH STEP: Week 4, Step 19.

INPUT: LocalGenerationResult objects from models/local_client.py, for
       clean and noisy versions of the same sampled instances.

OUTPUT: results/attention/ — entropy comparison data (JSON) and heatmap
        figures (PNG, 300 DPI).

DEPENDS ON: models/local_client.py, config.py (LINEAR_PROBE_LAYERS is
            reused here only for the heatmap's layer labeling, not its
            selection logic — entropy analysis itself covers every layer
            the local model has, not just the 3 probe depths). Read by
            any Week 4 write-up.

DESIGN DECISIONS:
- Entropy is computed exactly as the research design specifies:
  entropy = -sum(p_i * log(p_i)) over the attention weight distribution,
  per head, per query position, then averaged over query positions to
  get one value per (layer, head), then averaged over heads to get one
  value per layer — matching "Computed per head, averaged per layer."
- Attention rollout follows Abnar & Zuidema (2020): at each layer,
  average attention over heads, add the identity matrix scaled by 0.5
  (accounting for the residual connection each transformer layer has, so
  a token's own representation always retains some direct weight
  regardless of what it attends to), row-normalize, then matrix-multiply
  rollout matrices cumulatively across layers — this is the standard
  formulation used in essentially every attention-rollout implementation
  since the original paper, not something this project's own variant of.
- The "x-axis = input tokens, y-axis = layers" heatmap (research design's
  point 5) is built from the *last sequence token's* per-layer,
  head-averaged attention distribution over all earlier tokens — i.e.
  "what did the model attend to, right before finalizing its answer,
  at each depth" — rather than from the rolled-out matrix, because a
  single rolled-out matrix collapses all layers into one row and cannot
  itself produce a layers-by-tokens heatmap. Attention rollout is instead
  used for point 6 ("highlight which tokens receive most attention"): the
  rolled-out matrix's last row is the model's fully-depth-aggregated
  attribution from its final token back to every input token, which is
  the more standard use of rollout for token-attribution than a
  per-layer breakdown.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from models.local_client import LocalGenerationResult

logger = config.get_logger(__name__)


def per_layer_head_entropy(attentions: tuple) -> np.ndarray:
    """
    WHAT: Computes attention entropy for every (layer, head), averaged
          over query positions.

    WHY: This is the base measurement everything else in this module
         (per-layer entropy, clean-vs-noisy comparison) is built from —
         isolated in its own function so it is independently testable
         against a hand-constructed attention tensor with a known,
         hand-computed entropy value.

    Args:
        attentions (tuple[torch.Tensor]): One tensor per layer, each
                                           shaped (1, num_heads, seq_len,
                                           seq_len), as returned by
                                           models.local_client.LocalGenerationResult.attentions.

    Returns:
        np.ndarray: Shape (num_layers, num_heads). Entry [l, h] is the
                    mean, over all query positions, of
                    -sum(p_i * log(p_i)) computed over that position's
                    attention distribution for head h at layer l.

    Note:
        A causal attention distribution's masked (future) positions have
        weight ~0 after softmax, contributing ~0 to the entropy sum
        (since x*log(x) -> 0 as x -> 0) — so no explicit masking is
        needed here; the model's own softmax output already handles it.
        A small epsilon is added before the log to avoid log(0) for
        positions with exactly zero weight.
    """
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    entropy = np.zeros((num_layers, num_heads))

    eps = 1e-12
    for layer_idx, layer_attention in enumerate(attentions):
        # layer_attention: (1, num_heads, seq_len, seq_len)
        weights = layer_attention[0]  # (num_heads, seq_len, seq_len)
        per_position_entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)  # (num_heads, seq_len)
        entropy[layer_idx] = per_position_entropy.mean(dim=-1).numpy()  # (num_heads,)

    return entropy


def per_layer_entropy(attentions: tuple) -> np.ndarray:
    """
    WHAT: Averages per_layer_head_entropy() over heads to get one
          entropy value per layer.

    WHY: This is the exact granularity the research design's clean-vs-
         noisy comparison (point 4) operates at — one number per layer,
         to plot as a single comparison line/bars per condition.

    Args:
        attentions (tuple[torch.Tensor]): See per_layer_head_entropy().

    Returns:
        np.ndarray: Shape (num_layers,).
    """
    return per_layer_head_entropy(attentions).mean(axis=1)


def compute_rollout(attentions: tuple) -> np.ndarray:
    """
    WHAT: Computes attention rollout (Abnar & Zuidema, 2020) across all
          layers.

    WHY: See module DESIGN DECISIONS for the exact formulation used.
         Rollout approximates how information from each input token
         actually propagates to the final layer through the residual
         stream, which raw single-layer attention weights do not capture
         on their own (a token can influence another indirectly, through
         several layers, without ever directly attending to it).

    Args:
        attentions (tuple[torch.Tensor]): See per_layer_head_entropy().

    Returns:
        np.ndarray: Shape (seq_len, seq_len) — the cumulative rolled-out
                    attention matrix. Row i, column j is the
                    depth-aggregated attribution from output position i
                    back to input position j.
    """
    seq_len = attentions[0].shape[-1]
    rollout = np.eye(seq_len)

    for layer_attention in attentions:
        # Average over heads, per Abnar & Zuidema's formulation.
        head_averaged = layer_attention[0].mean(dim=0).numpy()  # (seq_len, seq_len)
        # Add identity scaled by 0.5 to account for the residual
        # connection (each token's representation is always partly
        # itself, independent of attention), then re-normalize each row
        # to sum to 1 so the matrix remains a valid stochastic matrix
        # for the next cumulative multiplication.
        residual_adjusted = 0.5 * head_averaged + 0.5 * np.eye(seq_len)
        residual_adjusted = residual_adjusted / residual_adjusted.sum(axis=-1, keepdims=True)
        rollout = residual_adjusted @ rollout

    return rollout


def compare_clean_vs_noisy(clean_result: LocalGenerationResult, noisy_result: LocalGenerationResult) -> dict:
    """
    WHAT: Computes per-layer entropy for both a clean and a noisy
          generation of the same instance and returns them side by side.

    WHY: This is the exact comparison the research design's point 4
         ("Compare clean vs noisy entropy distributions") asks for, at
         the instance level — analysis across many instances (e.g. mean
         entropy curve over 50 sampled instances) is done by the caller
         aggregating this function's output across instances.

    Args:
        clean_result (LocalGenerationResult): Generation on the clean
                                               prompt.
        noisy_result (LocalGenerationResult): Generation on a
                                               distractor-injected prompt
                                               for the *same* instance.

    Returns:
        dict: {"clean_entropy": list[float], "noisy_entropy": list[float]},
              each of length num_layers, from per_layer_entropy().
    """
    return {
        "clean_entropy": per_layer_entropy(clean_result.attentions).tolist(),
        "noisy_entropy": per_layer_entropy(noisy_result.attentions).tolist(),
    }


def plot_entropy_comparison(clean_entropy: list, noisy_entropy: list, title: str, out_path) -> None:
    """
    WHAT: Plots per-layer entropy for clean vs. noisy conditions as two
          lines.

    WHY: The direct visualization of research design point 4 — a
         downward-shifted noisy line relative to clean at deeper layers
         would be the visual signature of "distractor tokens dominating"
         (config.py's METRICS section description of what lower entropy
         under noise means).

    Args:
        clean_entropy (list[float]): Per-layer entropy, clean condition.
        noisy_entropy (list[float]): Per-layer entropy, noisy condition,
                                      same length as clean_entropy.
        title (str): Figure title (e.g. instance id or "mean over 50
                     instances").
        out_path: Path to save the PNG to.
    """
    layers = list(range(len(clean_entropy)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, clean_entropy, color="black", marker="o", label="clean")
    ax.plot(layers, noisy_entropy, color="black", linestyle="--", marker="s", label="noisy")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean attention entropy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_attention_heatmap(result: LocalGenerationResult, out_path, max_tokens_shown: int = 40) -> None:
    """
    WHAT: Plots a (layers x input tokens) heatmap of the final sequence
          position's attention back to earlier tokens, at every layer.

    WHY: Implements the research design's point 5 exactly ("x-axis =
         input tokens, y-axis = layers, color = attention weight") — see
         module DESIGN DECISIONS for why this is built from the last
         token's per-layer attention rather than the rolled-out matrix.

    Args:
        result (LocalGenerationResult): One generation's full internals.
        out_path: Path to save the PNG to.
        max_tokens_shown (int): Truncates the token axis to the first N
                                 tokens for readability — a full prompt
                                 can be 100+ tokens, which would render
                                 illegibly small column labels; WHY 40:
                                 comfortably covers a typical GSM8K-length
                                 question plus a short injected distractor
                                 while staying readable at standard figure
                                 widths.
    """
    tokens = result.input_ids[0].tolist()
    # (Tokenizer is not passed in here — token IDs are used as axis
    # labels rather than decoded strings, since decoding requires the
    # tokenizer object; callers needing decoded token text should decode
    # `result.input_ids` themselves before calling this function if a
    # more readable axis is needed. This keeps the function's dependency
    # surface to just the result object.)
    num_tokens_to_show = min(max_tokens_shown, len(tokens))

    num_layers = len(result.attentions)
    heatmap = np.zeros((num_layers, num_tokens_to_show))
    for layer_idx, layer_attention in enumerate(result.attentions):
        head_averaged = layer_attention[0].mean(dim=0)  # (seq_len, seq_len)
        last_token_attention = head_averaged[-1, :num_tokens_to_show].numpy()
        heatmap[layer_idx] = last_token_attention

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap="Greys")
    ax.set_xlabel("Input token position")
    ax.set_ylabel("Layer")
    ax.set_title("Attention from final token, by layer")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # Manual smoke test: requires models/local_client.py's model to be
    # downloaded (see that file's own smoke test). Generates a clean and
    # a hand-written "noisy" version of one instance, compares entropy,
    # and saves both plots so a human can visually confirm the analysis
    # before running it across all config.LOCAL_ANALYSIS_SAMPLE_SIZE
    # instances.
    from models.local_client import LocalReasoningModel

    model = LocalReasoningModel()
    clean_prompt = "Question: What is 9 + 9?\nAnswer:"
    noisy_prompt = (
        "Context:\nSome people believe 9 + 9 equals 16 due to a common "
        "arithmetic mistake confusing addition with a different operation.\n\n"
        "Question: What is 9 + 9?\nAnswer:"
    )

    clean_result = model.generate_with_internals(clean_prompt, max_new_tokens=32)
    noisy_result = model.generate_with_internals(noisy_prompt, max_new_tokens=32)

    print("Clean answer:", clean_result.answer_text)
    print("Noisy answer:", noisy_result.answer_text)

    comparison = compare_clean_vs_noisy(clean_result, noisy_result)
    print("Clean entropy per layer:", [f"{v:.3f}" for v in comparison["clean_entropy"]])
    print("Noisy entropy per layer:", [f"{v:.3f}" for v in comparison["noisy_entropy"]])

    out_dir = config.ATTENTION_DIR
    plot_entropy_comparison(
        comparison["clean_entropy"], comparison["noisy_entropy"],
        title="Smoke test instance", out_path=out_dir / "smoke_test_entropy.png",
    )
    plot_attention_heatmap(noisy_result, out_dir / "smoke_test_heatmap.png")
    with open(out_dir / "smoke_test_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved plots and data to {out_dir}")
