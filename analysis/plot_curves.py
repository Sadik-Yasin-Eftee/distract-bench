"""
plot_curves.py

WHAT: Renders one publication-quality inverse-scaling-curve figure per
      task type per model, plotting accuracy vs. token budget for all 7
      conditions on the same axes.

WHY: This is the paper's central figure — the visual evidence for (or
     against) inverse scaling under distractor noise. One figure per
     (task_type, model) rather than one giant combined figure keeps each
     plot readable (7 lines is already a lot) and matches how the
     research design's plotting spec describes them: "One plot per task
     type, all conditions on the same axes."

RESEARCH STEP: Week 3, Step 18.

INPUT: results/scaling_curves/scaling_results.jsonl, via
       analysis/compute_metrics.py's accuracy_table().

OUTPUT: One PNG per (task_type, model_key) under
        results/scaling_curves/plots/, at 300 DPI.

DEPENDS ON: analysis/compute_metrics.py, config.py. Consumes the same
            accuracy table Step 17 already validated.

DESIGN DECISIONS:
- Color coding follows the research design's spec literally: clean=black
  solid, low=light grey solid, medium=dark grey solid, high=black
  dashed. That spec only assigns one style per *intensity*, not per
  (intensity, position) — the 7th and 8th degrees of freedom (early vs.
  late) are distinguished by marker shape instead (early='o' circle,
  late='s' square) rather than by color, since introducing more colors
  than the design specifies would make clean/low/medium/high harder to
  tell apart at a glance, which is the property the original spec's
  grayscale progression (light grey -> dark grey -> black) is clearly
  designed to preserve (printable in black-and-white, a common
  requirement for academic paper figures).
- Y-axis is always accuracy in [0, 100], with a fixed range across all
  figures (rather than auto-scaled per plot) — WHY: readers comparing
  the math vs. factual vs. commonsense figures side by side (as this
  project's 3 task types invite) need the same vertical scale to
  visually compare drop magnitudes; auto-scaling per figure would make a
  20-point drop on one task's plot look identical in height to a
  60-point drop on another's.
- X-axis (token budget) is log2-scaled, since config.TOKEN_BUDGETS
  doubles at each step (256...8192) — a linear axis would visually
  compress the first several points into the left edge of the plot.
"""

import matplotlib
matplotlib.use("Agg")  # WHY: this script runs headlessly (no display),
# matching how it will actually run during the Week 2-3 batch experiment
# schedule — the default interactive backend would either fail or hang
# waiting for a display that does not exist there.
import matplotlib.pyplot as plt

import config
from analysis.compute_metrics import load_results, accuracy_table

logger = config.get_logger(__name__)

PLOTS_DIR = config.SCALING_CURVES_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Per module DESIGN DECISIONS: color keyed by intensity (clean counts as
# its own "intensity" for this purpose), linestyle keyed by intensity
# (high is dashed per the spec; everything else solid), marker keyed by
# injection position (clean has no position, so no marker).
_INTENSITY_COLOR = {"clean": "black", "low": "0.75", "medium": "0.35", "high": "black"}
_INTENSITY_LINESTYLE = {"clean": "-", "low": "-", "medium": "-", "high": "--"}
_POSITION_MARKER = {"early": "o", "late": "s"}


def _condition_style(condition: str) -> dict:
    """
    WHAT: Maps one condition string to its matplotlib color/linestyle/
          marker/label, per the scheme in the module docstring.

    WHY: A single lookup function keeps the "which condition gets which
         visual encoding" rule in one place — plot_task_model() below
         just calls this once per line instead of embedding a chain of
         if/elif branches inline in the plotting loop.

    Args:
        condition (str): One of config.CONDITIONS.

    Returns:
        dict: kwargs ready to splat into ax.plot(..., **style) — keys
              "color", "linestyle", "marker", "label".
    """
    if condition == "clean":
        return {"color": _INTENSITY_COLOR["clean"], "linestyle": _INTENSITY_LINESTYLE["clean"],
                "marker": "^", "label": "clean"}

    intensity, position = condition.rsplit("_", 1)
    return {
        "color": _INTENSITY_COLOR[intensity],
        "linestyle": _INTENSITY_LINESTYLE[intensity],
        "marker": _POSITION_MARKER[position],
        "label": f"{intensity} ({position})",
    }


def plot_task_model(accuracy_df, task_type: str, model_key: str) -> "plt.Figure":
    """
    WHAT: Builds one figure: accuracy (y) vs. token budget (x, log2
          scale), one line per condition, for a single (task_type,
          model_key) pair.

    WHY: A dedicated function (rather than inlining the plotting loop in
         generate_all_plots()) makes this independently callable — useful
         for regenerating a single figure after a partial re-run, or for
         interactive inspection in a notebook.

    Args:
        accuracy_df: Output of analysis.compute_metrics.accuracy_table().
        task_type (str): One of config.TASK_TYPES.
        model_key (str): One of config.MODEL_SPECS's keys.

    Returns:
        matplotlib.figure.Figure: The rendered figure, not yet saved —
                                  the caller (generate_all_plots()) or
                                  the manual smoke test below is
                                  responsible for calling savefig().
    """
    subset = accuracy_df[(accuracy_df["task_type"] == task_type) & (accuracy_df["model_key"] == model_key)]

    fig, ax = plt.subplots(figsize=(7, 5))
    for condition in config.CONDITIONS:
        condition_data = subset[subset["condition"] == condition].sort_values("token_budget")
        if condition_data.empty:
            continue
        style = _condition_style(condition)
        ax.plot(condition_data["token_budget"], condition_data["accuracy"], **style)

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(config.TOKEN_BUDGETS))
    ax.set_xticklabels([str(b) for b in config.TOKEN_BUDGETS])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{task_type} — {model_key}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_all_plots() -> None:
    """
    WHAT: Generates and saves one PNG per (task_type, model_key)
          combination present in the results.

    WHY: This is the Week 3, Step 18 entry point — running this module
         directly produces every scaling-curve figure the paper needs in
         one call.
    """
    raw = load_results()
    accuracy_df = accuracy_table(raw)

    for task_type in accuracy_df["task_type"].unique():
        for model_key in accuracy_df["model_key"].unique():
            fig = plot_task_model(accuracy_df, task_type, model_key)
            out_path = PLOTS_DIR / f"{task_type}_{model_key}.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            logger.info("Saved %s", out_path)


if __name__ == "__main__":
    generate_all_plots()
