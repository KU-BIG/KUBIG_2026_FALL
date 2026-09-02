import numpy as np
import pandas as pd

from src.experiments.phase3_integration import ADT_DIM, information_asymmetry_index, run_mediation


def test_information_asymmetry_index_quantity_row():
    row = pd.Series({"axis": "quantity", "n_genes_actual": 2000})
    expected = np.log(2000 / ADT_DIM)
    assert np.isclose(information_asymmetry_index(row), expected)


def test_information_asymmetry_index_nan_for_non_quantity():
    row = pd.Series({"axis": "quality", "n_genes_actual": 134})
    assert np.isnan(information_asymmetry_index(row))


def test_information_asymmetry_index_nan_when_missing():
    row = pd.Series({"axis": "quantity"})
    assert np.isnan(information_asymmetry_index(row))


def test_run_mediation_detects_true_mediation_signature():
    """Synthetic data where asymmetry -> gap -> performance is the true
    causal chain (with no direct asymmetry -> performance edge) — mediation
    regression should find gap significant and the direct effect ~0."""
    rng = np.random.default_rng(0)
    n = 300
    asymmetry = rng.normal(size=n)
    gap = 2.0 * asymmetry + rng.normal(scale=0.3, size=n)
    performance = -1.5 * gap + rng.normal(scale=0.3, size=n)  # only acts through gap
    df = pd.DataFrame({
        "information_asymmetry_index": asymmetry,
        "delta_gap": gap,
        "top5_retrieval_acc": performance,
    })
    result = run_mediation(df, asymmetry_col="information_asymmetry_index")
    assert result is not None
    assert result["step3"].pvalues["delta_gap"] < 0.01
    # direct effect in step3 should be much smaller than the total effect in step2
    assert abs(result["step3"].params["information_asymmetry_index"]) < abs(result["step2"].params["information_asymmetry_index"]) * 0.5


def test_run_mediation_returns_none_with_too_few_rows():
    df = pd.DataFrame({
        "information_asymmetry_index": [1.0, 2.0, np.nan],
        "delta_gap": [0.1, 0.2, 0.3],
        "top5_retrieval_acc": [0.5, 0.6, 0.7],
    })
    assert run_mediation(df, asymmetry_col="information_asymmetry_index") is None
