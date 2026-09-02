"""Follow-up analysis (not in the original PLAN.md phases — added after the
first full pass found the same pattern repeatedly: delta_gap and
top5_retrieval_acc often move in different directions, e.g. Harmony
on/off, exp A's quantity axis, exp C's single-lineage subsets).

Pools every condition that has the full gap_report() metric suite
(paired=True: delta_gap, alignment, uniformity_a/b, linear_separability,
top5_retrieval_acc) and asks which of these components actually predicts
downstream retrieval performance, rather than assuming delta_gap alone is
the right lens. Pure analysis over existing result tables — no retraining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

METRIC_COLS = ["delta_gap", "alignment", "uniformity_a", "uniformity_b", "linear_separability"]
TARGET_COL = "top5_retrieval_acc"

SOURCES = {
    "phase1_baseline_matchclot_arch": "results/tables/phase1_baseline_matchclot_arch.csv",
    "phase2_expA_dial_swipe": "results/tables/phase2_expA_dial_swipe.csv",
    "phase2_expC_lineage": "results/tables/phase2_expC_lineage.csv",
}


def build_pooled_table() -> pd.DataFrame:
    frames = []
    for source, path in SOURCES.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"WARNING: {path} not found, skipping")
            continue
        missing = [c for c in METRIC_COLS + [TARGET_COL] if c not in df.columns]
        if missing:
            print(f"WARNING: {path} missing columns {missing}, skipping")
            continue
        df = df.copy()
        df["source"] = source
        frames.append(df[METRIC_COLS + [TARGET_COL, "source"]])
    if not frames:
        raise RuntimeError("No result tables with the full gap_report metric suite found")
    return pd.concat(frames, ignore_index=True)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in METRIC_COLS:
        r_pearson, p_pearson = pearsonr(df[col], df[TARGET_COL])
        r_spearman, p_spearman = spearmanr(df[col], df[TARGET_COL])
        rows.append({
            "metric": col,
            "pearson_r": r_pearson, "pearson_p": p_pearson,
            "spearman_r": r_spearman, "spearman_p": p_spearman,
        })
    return pd.DataFrame(rows)


def multivariate_regression(df: pd.DataFrame):
    x = sm.add_constant(df[METRIC_COLS])
    # standardize predictors so coefficients are comparable in magnitude
    x_std = x.copy()
    for col in METRIC_COLS:
        x_std[col] = (x[col] - x[col].mean()) / x[col].std()
    model = sm.OLS(df[TARGET_COL], x_std).fit()
    return model


def main():
    pooled = build_pooled_table()
    out_path = "results/tables/followup_gap_decomposition_pooled.csv"
    pooled.to_csv(out_path, index=False)
    print(f"Pooled {len(pooled)} rows from {pooled['source'].unique().tolist()} -> {out_path}")

    corr_df = correlations(pooled)
    corr_path = "results/tables/followup_gap_decomposition_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print("\n=== Correlation of each metric with top5_retrieval_acc (pooled across all conditions) ===")
    print(corr_df.to_string(index=False))

    print("\n=== Multivariate OLS: top5_retrieval_acc ~ standardized(all 5 metrics) ===")
    model = multivariate_regression(pooled)
    print(model.summary().tables[1])

    print("\n=== Per-source correlations (does the pattern hold within each experiment separately?) ===")
    for source, group in pooled.groupby("source"):
        if len(group) < 5:
            continue
        print(f"\n-- {source} (n={len(group)}) --")
        print(correlations(group).to_string(index=False))


if __name__ == "__main__":
    main()
