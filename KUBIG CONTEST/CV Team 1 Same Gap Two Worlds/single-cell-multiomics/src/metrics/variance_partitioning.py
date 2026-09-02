"""Variance partitioning for the batch-effect confound analysis (PLAN.md sec 2).

Quantifies what fraction of total variance in an embedding space is
explained by a grouping variable (batch label or modality label), using the
same sum-of-squares decomposition PERMANOVA is built on. We use a direct
ANOVA-style R^2 in Euclidean space rather than a full permutation-based
PERMANOVA (which needs an O(n^2) pairwise distance matrix) because it is
mathematically equivalent for squared-Euclidean distance and scales to the
tens of thousands of cells here in O(n * d) instead.
"""
from __future__ import annotations

import numpy as np


def group_r2(embeddings: np.ndarray, group_labels: np.ndarray) -> float:
    """Fraction of total sum-of-squares explained by `group_labels`
    (ANOVA R^2 / PERMANOVA-equivalent for squared-Euclidean distance).
    0 = grouping explains none of the spread, 1 = all points within a group
    are identical (grouping explains everything).
    """
    x = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(group_labels)
    grand_mean = x.mean(axis=0)
    ss_total = np.sum((x - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(labels):
        mask = labels == g
        group_mean = x[mask].mean(axis=0)
        ss_between += mask.sum() * np.sum((group_mean - grand_mean) ** 2)
    return float(ss_between / ss_total)


def permutation_test_r2(
    embeddings: np.ndarray, group_labels: np.ndarray, n_perm: int = 999, seed: int = 0
) -> tuple[float, float]:
    """Observed R^2 plus a permutation p-value (label-shuffled null),
    for reporting the batch-contribution-% claim with a significance check.
    """
    rng = np.random.default_rng(seed)
    observed = group_r2(embeddings, group_labels)
    labels = np.asarray(group_labels).copy()
    null_dist = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(labels)
        null_dist[i] = group_r2(embeddings, shuffled)
    p_value = float((np.sum(null_dist >= observed) + 1) / (n_perm + 1))
    return observed, p_value
