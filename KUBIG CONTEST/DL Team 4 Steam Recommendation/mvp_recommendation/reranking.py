"""MMR diversity reranking for recommendation candidate lists."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd


GENERIC_TITLE_WORDS = {
    "the", "a", "an", "of", "and", "game", "edition", "complete", "definitive",
    "dlc", "pack", "expansion", "pass", "season", "collection", "bundle", "remastered",
}


def _title_tokens(title: object) -> set[str]:
    normalized = unicodedata.normalize("NFKD", str(title)).casefold()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {token for token in tokens if len(token) > 1 and token not in GENERIC_TITLE_WORDS}


def _title_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _minmax(values: np.ndarray) -> np.ndarray:
    minimum, maximum = float(values.min()), float(values.max())
    if maximum - minimum <= 1e-12:
        return np.ones_like(values, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def _rerank_group(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    app_to_row: dict[int, int],
    top_k: int,
    lambda_relevance: float,
) -> pd.DataFrame:
    work = frame.sort_values(["score", "app_id"], ascending=[False, True]).reset_index(drop=True)
    work["original_rank"] = np.arange(1, len(work) + 1)
    rows = np.array([app_to_row[int(app_id)] for app_id in work.app_id], dtype=np.int64)
    vectors = embeddings[rows].astype(np.float64, copy=False)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    semantic = np.clip(vectors @ vectors.T, 0.0, 1.0)
    tokens = [_title_tokens(title) for title in work.title]
    title_similarity = np.zeros_like(semantic)
    for left in range(len(work)):
        for right in range(left + 1, len(work)):
            value = _title_jaccard(tokens[left], tokens[right])
            title_similarity[left, right] = title_similarity[right, left] = value
    redundancy = np.maximum(semantic, title_similarity)
    relevance = _minmax(work.score.to_numpy(np.float64))

    selected: list[int] = []
    rerank_scores: list[float] = []
    penalties: list[float] = []
    remaining = set(range(len(work)))
    while remaining and len(selected) < min(top_k, len(work)):
        best_index = -1
        best_key: tuple[float, float, float] | None = None
        best_penalty = 0.0
        for candidate in remaining:
            penalty = float(redundancy[candidate, selected].max()) if selected else 0.0
            mmr = lambda_relevance * relevance[candidate] - (1.0 - lambda_relevance) * penalty
            # Deterministic ties: MMR, relevance, then smaller original rank.
            key = (mmr, relevance[candidate], -float(work.at[candidate, "original_rank"]))
            if best_key is None or key > best_key:
                best_index, best_key, best_penalty = candidate, key, penalty
        selected.append(best_index)
        remaining.remove(best_index)
        rerank_scores.append(float(best_key[0]))
        penalties.append(best_penalty)

    result = work.iloc[selected].copy()
    result["rank"] = np.arange(1, len(result) + 1)
    result["rerank_score"] = rerank_scores
    result["redundancy_penalty"] = penalties
    result["diversity_lambda"] = float(lambda_relevance)
    result["diversity_applied"] = True
    return result


def mmr_rerank(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    app_to_row: dict[int, int],
    top_k: int,
    lambda_relevance: float = 0.65,
    group_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Rerank each candidate group while preserving all original score columns."""
    if not 0.0 <= lambda_relevance <= 1.0:
        raise ValueError("lambda_relevance must be between 0 and 1")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {"app_id", "title", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"reranking input missing columns: {sorted(missing)}")
    missing_ids = [int(app_id) for app_id in frame.app_id.unique() if int(app_id) not in app_to_row]
    if missing_ids:
        raise ValueError(f"reranking embedding missing app_ids: {missing_ids[:5]}")

    groups = list(group_columns)
    if groups:
        output = [
            _rerank_group(group, embeddings, app_to_row, top_k, lambda_relevance)
            for _, group in frame.groupby(groups, sort=False, dropna=False)
        ]
        result = pd.concat(output, ignore_index=True)
        assert result.groupby(groups).size().eq(top_k).all()
    else:
        result = _rerank_group(frame, embeddings, app_to_row, top_k, lambda_relevance)
    assert result.app_id.notna().all()
    return result


def mean_intra_list_cosine(
    app_ids: Iterable[int], embeddings: np.ndarray, app_to_row: dict[int, int]
) -> float:
    ids = list(map(int, app_ids))
    if len(ids) < 2:
        return 0.0
    vectors = embeddings[[app_to_row[app_id] for app_id in ids]].astype(np.float64, copy=False)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(ids), k=1)]
    return float(upper.mean())
