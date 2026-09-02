import numpy as np

from src.metrics.variance_partitioning import group_r2, permutation_test_r2


def test_group_r2_zero_when_groups_identical_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 10))
    labels = rng.integers(0, 4, size=400)  # random assignment, unrelated to x
    r2 = group_r2(x, labels)
    assert r2 < 0.05


def test_group_r2_high_when_groups_well_separated():
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1, 2, 3], 100)
    offsets = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
    x = offsets[labels] + rng.normal(scale=0.1, size=(400, 2))
    r2 = group_r2(x, labels)
    assert r2 > 0.95


def test_group_r2_bounded_0_1():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 5))
    labels = rng.integers(0, 3, size=200)
    r2 = group_r2(x, labels)
    assert 0.0 <= r2 <= 1.0


def test_permutation_pvalue_small_for_real_effect():
    rng = np.random.default_rng(3)
    labels = np.repeat([0, 1], 60)
    x = np.zeros((120, 2))
    x[:60] = rng.normal(loc=0, scale=0.1, size=(60, 2))
    x[60:] = rng.normal(loc=5, scale=0.1, size=(60, 2))
    r2, p = permutation_test_r2(x, labels, n_perm=200, seed=0)
    assert r2 > 0.9
    assert p < 0.01


def test_permutation_pvalue_large_for_no_effect():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 2))
    labels = rng.integers(0, 2, size=120)
    r2, p = permutation_test_r2(x, labels, n_perm=200, seed=0)
    assert p > 0.05
