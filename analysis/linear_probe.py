"""
linear_probe.py

WHAT: Trains a logistic-regression probe per layer to classify whether a
      hidden-state vector came from a clean or a distractor-injected
      prompt, and visualizes the separation with t-SNE.

WHY: If a distractor's presence is linearly decodable from a model's
     intermediate representations, that is direct evidence the model
     "notices" the corruption internally — regardless of whether that
     awareness translates into a correct final answer — which is a
     mechanistic claim the accuracy-based scaling curves (Weeks 2-3)
     cannot make on their own.

RESEARCH STEP: Week 4, Step 20.

INPUT: LocalGenerationResult objects from models/local_client.py, for
       clean and noisy versions of config.LOCAL_ANALYSIS_SAMPLE_SIZE
       sampled instances.

OUTPUT: results/probe/ — per-layer classification metrics (JSON) and
        t-SNE scatter plots (PNG, 300 DPI).

DEPENDS ON: models/local_client.py, config.py (LINEAR_PROBE_LAYERS,
            RANDOM_SEED), scikit-learn (LogisticRegression, TSNE,
            train_test_split, metrics). Read by any Week 4 write-up.

DESIGN DECISIONS:
- The "hidden state vector" representing one instance at one layer is
  the hidden state at the LAST PROMPT TOKEN position (i.e.
  hidden_states[layer][0, prompt_length - 1, :]) — the model's
  representation immediately after having read the full (clean or noisy)
  input, before generating any answer tokens. This is a deliberate
  choice not spelled out in the research design (which just says "hidden
  state vectors" without specifying a token position): using the last
  prompt token means every instance's clean and noisy vector are taken
  from a comparable point in each sequence (right after reading input,
  before answering) even though the clean and noisy prompts have
  different lengths — averaging over the whole sequence, by contrast,
  would mix in a variable number of generated-answer-token positions
  whose count itself differs between the clean and noisy runs (since a
  distractor can change how many tokens the model generates), which
  would let the probe trivially exploit sequence-length artifacts rather
  than genuine representational shift.
- Uses config.RANDOM_SEED for both the train/test split and
  LogisticRegression's own internal randomness (via `random_state`),
  matching the project's REPRODUCIBILITY requirement that every random
  operation traces back to the same single seed.
- Reports ROC-AUC alongside accuracy/precision/recall/F1 even though the
  research design's step 8 only asks "which layer gives best
  separation" without naming a specific metric — ROC-AUC is the
  standard metric for "how separable are two classes" independent of a
  chosen decision threshold, making it the most defensible metric for
  that specific comparison across layers.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config
from models.local_client import LocalGenerationResult

logger = config.get_logger(__name__)


def extract_layer_vector(result: LocalGenerationResult, layer_index: int) -> np.ndarray:
    """
    WHAT: Extracts the hidden-state vector for one instance at one layer,
          at the last prompt token position.

    WHY: See module DESIGN DECISIONS for why this specific token position
         is used.

    Args:
        result (LocalGenerationResult): One generation's full internals.
        layer_index (int): Index into `result.hidden_states` — note this
                            is the raw index into that tuple (which has
                            num_layers + 1 entries, index 0 being the
                            input embeddings per transformers'
                            convention), not necessarily equal to a
                            "layer 8/16/24" label — see
                            build_probe_dataset() for how
                            config.LINEAR_PROBE_LAYERS values are mapped
                            to this index.

    Returns:
        np.ndarray: Shape (hidden_dim,).
    """
    last_prompt_position = result.prompt_length - 1
    return result.hidden_states[layer_index][0, last_prompt_position, :].numpy()


def build_probe_dataset(
    clean_results: list, noisy_results: list, layer_index: int
) -> tuple:
    """
    WHAT: Builds (X, y) for one layer from paired clean/noisy generation
          results.

    WHY: A dedicated builder keeps the "clean=0, noisy=1" labeling
         convention (matching config.py's linear probe description) in
         one place, and keeps this function trivially testable with
         hand-constructed fake results.

    Args:
        clean_results (list[LocalGenerationResult]): One entry per
                                                       sampled instance,
                                                       clean condition.
        noisy_results (list[LocalGenerationResult]): One entry per
                                                       sampled instance,
                                                       same order as
                                                       clean_results,
                                                       noisy condition.
        layer_index (int): Raw index into each result's `hidden_states`
                            tuple.

    Returns:
        tuple[np.ndarray, np.ndarray]: X of shape (2 * n_instances,
                                       hidden_dim), y of shape
                                       (2 * n_instances,) with 0 for
                                       every clean row and 1 for every
                                       noisy row.
    """
    clean_vectors = [extract_layer_vector(r, layer_index) for r in clean_results]
    noisy_vectors = [extract_layer_vector(r, layer_index) for r in noisy_results]

    X = np.vstack(clean_vectors + noisy_vectors)
    y = np.array([0] * len(clean_vectors) + [1] * len(noisy_vectors))
    return X, y


def train_and_evaluate_probe(X: np.ndarray, y: np.ndarray) -> dict:
    """
    WHAT: Trains a LogisticRegression probe on an 80/20 stratified
          train/test split and reports classification metrics on the
          held-out test set.

    WHY: Implements the research design's exact prescription: "80/20
         train/test split, stratified. Train LogisticRegression (sklearn,
         max_iter=1000, C=1.0)." Stratification matters here because the
         classes are exactly balanced by construction (build_probe_dataset
         always produces equal clean/noisy counts), but stratifying is
         still the correct default to guarantee that balance survives an
         80/20 split with a small n (config.LOCAL_ANALYSIS_SAMPLE_SIZE=50
         instances per class means a 20% test split is only 10 per class
         — losing stratification here could leave a test fold with a
         skewed 7-3 split by chance).

    Args:
        X (np.ndarray): Shape (n_samples, hidden_dim), from
                         build_probe_dataset().
        y (np.ndarray): Shape (n_samples,), 0/1 labels.

    Returns:
        dict: {"accuracy", "precision", "recall", "f1", "roc_auc"} — all
              computed on the held-out 20% test fold only.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=config.RANDOM_SEED)
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    y_proba = probe.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def run_probe_for_all_layers(clean_results: list, noisy_results: list) -> dict:
    """
    WHAT: Runs build_probe_dataset() + train_and_evaluate_probe() for
          every layer in config.LINEAR_PROBE_LAYERS.

    WHY: This is the per-layer sweep the research design's step 6 asks
         for ("Report per-layer: accuracy, precision, recall, F1,
         ROC-AUC").

    Args:
        clean_results (list[LocalGenerationResult]): See
                                                       build_probe_dataset().
        noisy_results (list[LocalGenerationResult]): See
                                                       build_probe_dataset().

    Returns:
        dict: {layer_label (int, e.g. 8/16/24): metrics dict from
              train_and_evaluate_probe()}.
    """
    results = {}
    for layer_label in config.LINEAR_PROBE_LAYERS:
        X, y = build_probe_dataset(clean_results, noisy_results, layer_index=layer_label)
        results[layer_label] = train_and_evaluate_probe(X, y)
        logger.info("Layer %d probe: %s", layer_label, results[layer_label])
    return results


def best_layer(per_layer_metrics: dict, metric: str = "roc_auc") -> int:
    """
    WHAT: Picks the layer with the highest value of `metric`.

    WHY: Implements the research design's step 8 ("Report which layer
         gives best separation") as an explicit, callable function rather
         than something read off a printed table by hand.

    Args:
        per_layer_metrics (dict): Output of run_probe_for_all_layers().
        metric (str): Which metric key to maximize. Defaults to
                       "roc_auc" — see module DESIGN DECISIONS for why
                       ROC-AUC is the default separability metric.

    Returns:
        int: The layer label (e.g. 8, 16, or 24) with the highest value
             of `metric`.
    """
    return max(per_layer_metrics, key=lambda layer: per_layer_metrics[layer][metric])


def plot_tsne(clean_results: list, noisy_results: list, layer_index: int, out_path) -> None:
    """
    WHAT: Reduces one layer's hidden-state vectors to 2D via t-SNE and
          scatter-plots them colored by clean/noisy label.

    WHY: Implements the research design's step 7 exactly — a visual
         complement to the quantitative probe metrics, letting a reader
         see whether clean and noisy points form separated clusters or
         are intermixed.

    Args:
        clean_results (list[LocalGenerationResult]): See
                                                       build_probe_dataset().
        noisy_results (list[LocalGenerationResult]): See
                                                       build_probe_dataset().
        layer_index (int): Which layer to visualize.
        out_path: Path to save the PNG to.

    Note:
        t-SNE's perplexity parameter is capped below the sample count
        (scikit-learn requires perplexity < n_samples) — with
        config.LOCAL_ANALYSIS_SAMPLE_SIZE=50 instances per class (100
        total points), the default perplexity of 30 is safe, but this is
        computed defensively so the function does not crash on a smaller
        ad hoc sample (e.g. during a quick smoke test with 2-3 instances).
    """
    X, y = build_probe_dataset(clean_results, noisy_results, layer_index)
    perplexity = min(30, X.shape[0] - 1)

    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=perplexity)
    embedded = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, color, marker, name in [(0, "0.6", "o", "clean"), (1, "black", "^", "noisy")]:
        mask = y == label
        ax.scatter(embedded[mask, 0], embedded[mask, 1], c=color, marker=marker, label=name, alpha=0.7)
    ax.set_title(f"t-SNE of hidden states, layer {layer_index}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # Manual smoke test: requires models/local_client.py's model to be
    # downloaded. Runs a handful of instances (far fewer than
    # config.LOCAL_ANALYSIS_SAMPLE_SIZE) through the local model to
    # confirm the probe pipeline runs end-to-end before the full Week 4
    # analysis depends on it.
    from models.local_client import LocalReasoningModel

    model = LocalReasoningModel()
    questions = [
        ("What is 9 + 9?", "18 is a common wrong distractor: some claim 9+9=16 due to a base-8 mixup."),
        ("What is 12 + 15?", "A local news article once misreported that 12+15 equals 26 due to a printing error."),
        ("What is 7 + 8?", "In some board games, dice sums of 7 and 8 combine differently, leading players to miscalculate 7+8 as 14."),
    ]

    clean_results, noisy_results = [], []
    for question, distractor in questions:
        clean_results.append(model.generate_with_internals(f"Question: {question}\nAnswer:", max_new_tokens=16))
        noisy_results.append(model.generate_with_internals(
            f"Context:\n{distractor}\n\nQuestion: {question}\nAnswer:", max_new_tokens=16
        ))

    metrics = run_probe_for_all_layers(clean_results, noisy_results)
    print("Per-layer metrics:", json.dumps(metrics, indent=2))
    print("Best layer (by ROC-AUC):", best_layer(metrics))

    plot_tsne(clean_results, noisy_results, layer_index=config.LINEAR_PROBE_LAYERS[0], out_path=config.PROBE_DIR / "smoke_test_tsne.png")
    with open(config.PROBE_DIR / "smoke_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved plot and metrics to {config.PROBE_DIR}")
