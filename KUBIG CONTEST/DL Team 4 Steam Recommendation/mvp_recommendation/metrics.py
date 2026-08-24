"""Classification and one-positive sampled ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def one_positive_ranking_metrics(scores: np.ndarray, group_size: int) -> dict[str, float]:
    """Candidate 0 in each group is positive; the remaining candidates are negatives.

    With one positive per group, Recall@K and HitRate@K are mathematically equal.
    Both names are retained to match common recommender reporting conventions.
    """
    assert scores.ndim == 1 and len(scores) % group_size == 0
    grouped = scores.reshape(-1, group_size)
    positive_scores = grouped[:, :1]
    ranks = 1 + (grouped[:, 1:] > positive_scores).sum(axis=1)
    result: dict[str, float] = {}
    for k in (5, 10, 20):
        hit = (ranks <= k).mean()
        result[f"recall@{k}"] = float(hit)
        result[f"hitrate@{k}"] = float(hit)
        result[f"ndcg@{k}"] = float(
            np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).mean()
        )
    result["mrr"] = float((1.0 / ranks).mean())
    result["evaluated_positives"] = int(len(ranks))
    return result


def ranking_diagnostics(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    train_positive_counts: pd.Series,
    valid_catalog_size: int,
    k: int = 10,
) -> dict[str, float]:
    """Catalog coverage and long-tail behavior inside the sampled evaluation.

    Coverage is reported against both the full valid catalog and the union of
    items that actually appeared in sampled candidate sets. Long-tail games are
    the bottom popularity quartile among games with at least one positive train
    interaction; unseen games are included in the long-tail group.
    """
    assert len(candidates) == len(scores)
    work = candidates[["group_id", "app_id", "target"]].copy()
    work["score"] = scores
    topk = (
        work.sort_values(["group_id", "score", "app_id"], ascending=[True, False, True])
        .groupby("group_id", sort=False)
        .head(k)
    )
    recommended = int(topk.app_id.nunique())
    candidate_catalog = int(work.app_id.nunique())
    positives = work.loc[work.target.eq(1), ["group_id", "app_id"]].copy()
    threshold = float(train_positive_counts[train_positive_counts > 0].quantile(0.25))
    positive_popularity = positives.app_id.map(train_positive_counts).fillna(0)
    long_tail_groups = set(positives.loc[positive_popularity.le(threshold), "group_id"])
    hit_groups = set(topk.loc[topk.target.eq(1), "group_id"])
    long_tail_recall = (
        len(long_tail_groups & hit_groups) / len(long_tail_groups) if long_tail_groups else float("nan")
    )
    topk_popularity = topk.app_id.map(train_positive_counts).fillna(0).to_numpy(float)
    return {
        f"catalog_coverage@{k}": recommended / valid_catalog_size,
        f"candidate_coverage@{k}": recommended / candidate_catalog,
        f"unique_recommended@{k}": recommended,
        f"long_tail_recall@{k}": float(long_tail_recall),
        "long_tail_positive_count": int(len(long_tail_groups)),
        f"mean_train_positive_popularity@{k}": float(topk_popularity.mean()),
        "long_tail_popularity_threshold": threshold,
    }
