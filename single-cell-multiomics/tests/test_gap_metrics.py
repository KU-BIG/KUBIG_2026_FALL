"""Synthetic-data unit tests for src/metrics/gap_metrics.py.

These validate metric *behavior* (monotonicity, expected extremes) against
controlled synthetic embeddings, independent of any real single-cell data —
per PLAN.md Phase 0, gap-metric code must be verified before it ever touches
GEX/ADT/ATAC data.
"""
import numpy as np
import pytest

from src.metrics.gap_metrics import (
    alignment,
    delta_gap,
    gap_report,
    linear_separability,
    topk_retrieval_accuracy,
    uniformity,
)


def _make_shared_latent(n=500, d=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_delta_gap_zero_when_no_shift():
    rng = np.random.default_rng(1)
    base = _make_shared_latent()
    a = base + rng.normal(scale=0.01, size=base.shape)
    b = base + rng.normal(scale=0.01, size=base.shape)
    assert delta_gap(a, b) < 0.05


@pytest.mark.parametrize("shift", [0.0, 0.5, 1.0, 2.0, 4.0])
def test_delta_gap_monotonic_in_shift(shift):
    base = _make_shared_latent(seed=2)
    direction = np.ones(base.shape[1])
    direction /= np.linalg.norm(direction)
    a = base
    b = base + shift * direction
    gaps = {}
    for s in [0.0, 0.5, 1.0, 2.0, 4.0]:
        gaps[s] = delta_gap(a, base + s * direction)
    values = [gaps[s] for s in sorted(gaps)]
    assert all(x <= y + 1e-9 for x, y in zip(values, values[1:]))


def test_linear_separability_near_chance_when_identical_distribution():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(300, 16))
    b = rng.normal(size=(300, 16))
    score = linear_separability(a, b)
    assert 0.35 <= score <= 0.65


def test_linear_separability_high_when_well_separated():
    rng = np.random.default_rng(4)
    a = rng.normal(loc=0.0, size=(300, 16))
    b = rng.normal(loc=10.0, size=(300, 16))
    score = linear_separability(a, b)
    assert score > 0.95


def test_alignment_zero_for_identical_paired_points():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(200, 8))
    assert alignment(a, a) == pytest.approx(0.0, abs=1e-9)


def test_alignment_requires_matching_row_count():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(200, 8))
    b = rng.normal(size=(150, 8))
    with pytest.raises(ValueError):
        alignment(a, b)


def test_alignment_increases_with_pair_noise():
    rng = np.random.default_rng(7)
    base = _make_shared_latent(n=300, d=16, seed=7)
    low_noise = base + rng.normal(scale=0.1, size=base.shape)
    high_noise = base + rng.normal(scale=2.0, size=base.shape)
    assert alignment(base, low_noise) < alignment(base, high_noise)


def test_uniformity_more_negative_for_spread_than_clustered():
    rng = np.random.default_rng(8)
    spread = rng.normal(size=(300, 16))
    clustered = rng.normal(size=(300, 16)) * 0.01 + np.ones(16)
    assert uniformity(spread) < uniformity(clustered)


def test_topk_retrieval_perfect_for_identical_pairs():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(200, 16))
    assert topk_retrieval_accuracy(a, a, k=5) == pytest.approx(1.0)


def test_topk_retrieval_drops_with_noise():
    rng = np.random.default_rng(10)
    base = _make_shared_latent(n=400, d=16, seed=10)
    low_noise = base + rng.normal(scale=0.05, size=base.shape)
    high_noise = rng.normal(size=base.shape)  # unrelated to base -> ~chance
    acc_low = topk_retrieval_accuracy(base, low_noise, k=5)
    acc_high = topk_retrieval_accuracy(base, high_noise, k=5)
    assert acc_low > acc_high
    assert acc_low > 0.9
    assert acc_high < 0.15  # chance level ~ k/n = 5/400


def test_uniformity_matches_naive_on_small_input():
    """Regression test for the OOM fix in docs/HISTORY.md 2026-08-13: the
    matmul-based pairwise-distance computation must agree with the original
    naive O(n^2*d) broadcast, just without allocating the n*n*d tensor."""
    rng = np.random.default_rng(20)
    x = rng.normal(size=(50, 6))
    from src.metrics.gap_metrics import _unit_normalize

    xn = _unit_normalize(x)
    naive_sq = np.sum((xn[:, None, :] - xn[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(50, k=1)
    naive_uniformity = np.log(np.mean(np.exp(-2.0 * naive_sq[iu])) + 1e-12)
    assert uniformity(x, t=2.0, max_n=50) == pytest.approx(naive_uniformity, abs=1e-6)


def test_uniformity_handles_large_n_without_oom():
    """n=20,000 would allocate ~64GB for the naive n*n*d broadcast at d=32;
    this must run in a couple seconds using only O(max_n^2) memory."""
    rng = np.random.default_rng(21)
    x = rng.normal(size=(20_000, 32))
    val = uniformity(x, max_n=2000)
    assert np.isfinite(val)


def test_topk_retrieval_handles_large_n_without_oom():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(30_000, 16))
    acc = topk_retrieval_accuracy(x, x, k=5, max_n=3000)
    assert acc == pytest.approx(1.0)


def test_gap_report_paired_vs_unpaired_keys():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(100, 8))
    b = rng.normal(size=(100, 8))
    paired_report = gap_report(a, b, paired=True)
    unpaired_report = gap_report(a, b, paired=False)
    assert "alignment" in paired_report and "top5_retrieval_acc" in paired_report
    assert "alignment" not in unpaired_report and "top5_retrieval_acc" not in unpaired_report
    assert "delta_gap" in unpaired_report and "linear_separability" in unpaired_report
