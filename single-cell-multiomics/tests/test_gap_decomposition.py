import numpy as np
import pandas as pd

from src.experiments.followup_gap_decomposition import correlations, multivariate_regression, METRIC_COLS, TARGET_COL


def _synthetic_df(seed=0, n=100):
    rng = np.random.default_rng(seed)
    delta_gap = rng.uniform(0, 1, n)
    perf = -0.8 * delta_gap + rng.normal(scale=0.05, size=n)  # strong negative relationship
    df = pd.DataFrame({
        "delta_gap": delta_gap,
        "alignment": delta_gap + rng.normal(scale=0.05, size=n),
        "uniformity_a": -delta_gap + rng.normal(scale=0.05, size=n),
        "uniformity_b": -delta_gap + rng.normal(scale=0.05, size=n),
        "linear_separability": delta_gap + rng.normal(scale=0.05, size=n),
        TARGET_COL: perf,
    })
    return df


def test_correlations_detects_strong_negative_relationship():
    df = _synthetic_df()
    corr = correlations(df)
    row = corr[corr["metric"] == "delta_gap"].iloc[0]
    assert row["pearson_r"] < -0.7
    assert row["pearson_p"] < 0.001


def test_correlations_returns_all_metrics():
    df = _synthetic_df()
    corr = correlations(df)
    assert set(corr["metric"]) == set(METRIC_COLS)


def test_multivariate_regression_runs_and_has_expected_columns():
    df = _synthetic_df(n=200)
    model = multivariate_regression(df)
    assert "delta_gap" in model.params.index
    assert model.rsquared > 0.3
