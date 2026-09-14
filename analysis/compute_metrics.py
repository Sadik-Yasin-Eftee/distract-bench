"""
compute_metrics.py

WHAT: Turns the raw per-call results (experiments/save_results.py's
      output) into the project's primary and secondary metrics: accuracy,
      Performance Drop Rate (PDR), Relative Performance Drop (RPD), and
      the inversion point per (task_type, model, condition).

WHY: The scaling-curve experiment produces one row per API call — this
     file is where those 75,600 rows become the small set of numbers
     (accuracy at each budget, PDR, RPD, inversion point) that actually
     go into the paper's tables and are then plotted by
     analysis/plot_curves.py.

RESEARCH STEP: Week 3, Step 17.

INPUT: results/scaling_curves/scaling_results.jsonl (experiments/save_results.py).

OUTPUT: pandas DataFrames — accuracy_table(), pdr_table(), rpd_table(),
        inversion_points() — no file is written by this module itself;
        analysis/plot_curves.py and any notebook/script writing the
        paper's tables call these functions directly.

DEPENDS ON: config.py (output path). Used by analysis/plot_curves.py and
            (manually) when writing up Week 3-5 results.

DESIGN DECISIONS:
- Accuracy is computed as a simple group-wise mean of the boolean
  `is_correct` column — matches the project's own PRIMARY METRIC
  definition (accuracy = correct / total x 100) exactly; no smoothing or
  weighting is applied, since every (instance, condition, budget, model)
  cell in the experiment design has exactly one row (assuming the
  experiment ran to completion with no duplicate saves — see
  experiments/save_results.py's dedup-on-resume behavior).
- PDR and RPD are computed relative to the "clean" condition of the
  *same* task_type, model, and token_budget — not relative to a single
  fixed clean baseline across all budgets. WHY: the research question is
  about the noisy-vs-clean gap *at each budget*, since inverse scaling
  means both curves can move as budget increases — anchoring PDR to a
  single budget=256 clean value would conflate "the model's clean
  accuracy also changed with budget" with "the distractor hurt more or
  less at this budget," which are two different phenomena this metric
  needs to keep separate.
- find_inversion_point() implements the project design's exact
  definition: "the token budget at which noisy_accuracy first drops
  BELOW clean_accuracy at budget=256" — note this compares against the
  clean accuracy at the *smallest* budget specifically (not the clean
  accuracy at the same budget as the noisy point being tested), per the
  literal wording of that definition in the research design.
"""

import json

import pandas as pd

import config

logger = config.get_logger(__name__)


def load_results() -> pd.DataFrame:
    """
    WHAT: Loads results/scaling_curves/scaling_results.jsonl into a
          pandas DataFrame.

    WHY: Every function below operates on this same DataFrame shape —
         centralizing the load means a future change to the saved schema
         (experiments/save_results.build_result_row()) only needs a
         corresponding change here, not in every metric function.

    Returns:
        pd.DataFrame: One row per saved API call, columns matching
                      experiments.save_results.build_result_row()'s keys.

    Raises:
        FileNotFoundError: If the experiment (experiments/run_scaling_curves.py)
                           has not been run yet.
    """
    if not config.SCALING_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{config.SCALING_RESULTS_PATH} not found. Run "
            f"`python -m experiments.run_scaling_curves` (Week 2-3, Step 15) first."
        )
    rows = []
    with open(config.SCALING_RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Computes accuracy (%) grouped by task_type, model_key,
          condition, and token_budget.

    WHY: This is the exact table analysis/plot_curves.py needs — one
         accuracy value per (task_type, model, condition, budget) cell,
         which is precisely one point on one line of one scaling-curve
         plot.

    Args:
        df (pd.DataFrame): Raw results from load_results().

    Returns:
        pd.DataFrame: Columns [task_type, model_key, condition,
                      token_budget, accuracy, n] where `accuracy` is in
                      [0, 100] and `n` is the number of rows the mean was
                      computed over (useful for spotting an incomplete
                      experiment cell before trusting its accuracy value).
    """
    grouped = (
        df.groupby(["task_type", "model_key", "condition", "token_budget"])["is_correct"]
        .agg(accuracy="mean", n="count")
        .reset_index()
    )
    grouped["accuracy"] = grouped["accuracy"] * 100
    return grouped


def pdr_table(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Computes Performance Drop Rate (PDR) = clean_accuracy -
          noisy_accuracy, for every non-clean condition, at each
          (task_type, model_key, token_budget).

    WHY: PDR is the project's primary secondary metric (config.py's
         METRICS section) — the absolute accuracy lost to a distractor,
         computed at the *same* token budget as the noisy measurement
         (see module DESIGN DECISIONS for why not a fixed baseline
         budget).

    Args:
        accuracy_df (pd.DataFrame): Output of accuracy_table().

    Returns:
        pd.DataFrame: Columns [task_type, model_key, condition,
                      token_budget, pdr] — one row per non-clean
                      condition at each (task_type, model_key,
                      token_budget), where `pdr` = the clean accuracy at
                      that same (task_type, model_key, token_budget)
                      minus this row's noisy accuracy.
    """
    clean = accuracy_df[accuracy_df["condition"] == "clean"].rename(columns={"accuracy": "clean_accuracy"})
    clean = clean[["task_type", "model_key", "token_budget", "clean_accuracy"]]

    noisy = accuracy_df[accuracy_df["condition"] != "clean"]
    merged = noisy.merge(clean, on=["task_type", "model_key", "token_budget"], how="left")
    merged["pdr"] = merged["clean_accuracy"] - merged["accuracy"]
    return merged[["task_type", "model_key", "condition", "token_budget", "pdr"]]


def rpd_table(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: Computes Relative Performance Drop (RPD) = PDR / clean_accuracy
          x 100, for every non-clean condition.

    WHY: RPD is the project's cross-task-comparable secondary metric
         (config.py's METRICS section) — normalizing by clean accuracy
         lets a 10-point drop on an 80%-accuracy task be compared
         meaningfully against a 10-point drop on a 30%-accuracy task.

    Args:
        accuracy_df (pd.DataFrame): Output of accuracy_table().

    Returns:
        pd.DataFrame: Columns [task_type, model_key, condition,
                      token_budget, rpd]. `rpd` is NaN wherever
                      clean_accuracy is exactly 0 (division by zero is a
                      real possible outcome, not a bug, if a model scores
                      0% clean accuracy on some task/budget combination —
                      left as NaN rather than silently coerced to 0 or
                      inf, so it is visibly excluded from any downstream
                      mean/plot rather than distorting it).
    """
    clean = accuracy_df[accuracy_df["condition"] == "clean"].rename(columns={"accuracy": "clean_accuracy"})
    clean = clean[["task_type", "model_key", "token_budget", "clean_accuracy"]]

    noisy = accuracy_df[accuracy_df["condition"] != "clean"]
    merged = noisy.merge(clean, on=["task_type", "model_key", "token_budget"], how="left")
    pdr = merged["clean_accuracy"] - merged["accuracy"]
    merged["rpd"] = (pdr / merged["clean_accuracy"].replace(0, pd.NA)) * 100
    return merged[["task_type", "model_key", "condition", "token_budget", "rpd"]]


def find_inversion_points(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT: For every (task_type, model_key, condition) with condition !=
          "clean", finds the smallest token_budget at which that
          condition's accuracy first drops below the *clean* accuracy at
          the smallest token budget (config.TOKEN_BUDGETS[0]).

    WHY: Implements the project design's exact stated definition of
         "Inversion Point" (config.py's METRICS section) — described as
         the paper's key novel finding, so this function's behavior must
         match that definition precisely rather than a more "sensible"
         alternative (e.g. comparing against same-budget clean accuracy,
         which is what PDR/RPD already do).

    Args:
        accuracy_df (pd.DataFrame): Output of accuracy_table().

    Returns:
        pd.DataFrame: Columns [task_type, model_key, condition,
                      inversion_point] where `inversion_point` is the
                      smallest token_budget value at which this
                      condition's accuracy < the clean accuracy at
                      config.TOKEN_BUDGETS[0], or None if no such budget
                      exists in the swept range (i.e. this condition
                      never dropped below that reference point — a real,
                      reportable "no inversion observed" outcome, not a
                      missing-data situation, so None is used rather
                      than the row being omitted).
    """
    smallest_budget = min(config.TOKEN_BUDGETS)
    clean_reference = (
        accuracy_df[(accuracy_df["condition"] == "clean") & (accuracy_df["token_budget"] == smallest_budget)]
        .set_index(["task_type", "model_key"])["accuracy"]
    )

    results = []
    noisy = accuracy_df[accuracy_df["condition"] != "clean"]
    for (task_type, model_key, condition), group in noisy.groupby(["task_type", "model_key", "condition"]):
        reference_accuracy = clean_reference.get((task_type, model_key))
        if reference_accuracy is None:
            continue

        below = group[group["accuracy"] < reference_accuracy].sort_values("token_budget")
        inversion_point = below["token_budget"].iloc[0] if not below.empty else None
        results.append({
            "task_type": task_type,
            "model_key": model_key,
            "condition": condition,
            "inversion_point": inversion_point,
        })

    return pd.DataFrame(results)


def summary_table(df: pd.DataFrame) -> dict:
    """
    WHAT: Convenience wrapper computing all four metric tables from one
          call.

    WHY: Most consumers (analysis/plot_curves.py, a Week 5 write-up
         script) want all four tables together rather than calling each
         function separately and re-deriving accuracy_df each time.

    Args:
        df (pd.DataFrame): Raw results from load_results().

    Returns:
        dict: {"accuracy": DataFrame, "pdr": DataFrame, "rpd": DataFrame,
              "inversion_points": DataFrame}.
    """
    accuracy_df = accuracy_table(df)
    return {
        "accuracy": accuracy_df,
        "pdr": pdr_table(accuracy_df),
        "rpd": rpd_table(accuracy_df),
        "inversion_points": find_inversion_points(accuracy_df),
    }


if __name__ == "__main__":
    # Manual smoke test: requires results/scaling_curves/scaling_results.jsonl
    # to already exist (i.e. experiments/run_scaling_curves.py has been run,
    # even partially). Prints each table's shape and head so a human can
    # sanity-check the computed metrics before analysis/plot_curves.py
    # depends on them.
    raw = load_results()
    print(f"Loaded {len(raw)} raw result rows.")
    tables = summary_table(raw)
    for name, table in tables.items():
        print(f"\n=== {name} ({len(table)} rows) ===")
        print(table.head(10))
