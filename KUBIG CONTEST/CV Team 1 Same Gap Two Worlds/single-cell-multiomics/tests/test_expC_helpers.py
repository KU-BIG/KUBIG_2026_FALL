import numpy as np

from src.experiments.phase2_expC_lineage import _shannon_entropy


def test_entropy_zero_for_single_type():
    assert _shannon_entropy(["A"] * 100) == 0.0


def test_entropy_higher_for_more_types():
    two_types = ["A"] * 50 + ["B"] * 50
    four_types = ["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25
    assert _shannon_entropy(four_types) > _shannon_entropy(two_types) > 0


def test_entropy_matches_known_uniform_value():
    labels = ["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25
    assert np.isclose(_shannon_entropy(labels), np.log(4), atol=1e-9)


def test_entropy_lower_for_skewed_distribution():
    skewed = ["A"] * 99 + ["B"] * 1
    balanced = ["A"] * 50 + ["B"] * 50
    assert _shannon_entropy(skewed) < _shannon_entropy(balanced)
