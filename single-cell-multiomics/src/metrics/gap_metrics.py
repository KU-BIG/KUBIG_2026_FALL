"""Modality-gap metrics (PLAN.md, section 1).

Primary metric: delta_gap (unit-normalized centroid distance, Liang et al. 2022).
Secondary metrics: alignment/uniformity decomposition, linear separability,
top-k cross-modal retrieval accuracy.

All functions take raw (non-normalized) embeddings and normalize internally
where the metric definition calls for it, so callers never have to remember
which metric expects unit vectors.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _unit_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def delta_gap(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """L2 distance between unit-normalized centroids of two modalities' embeddings.

    This is the primary gap metric (PLAN.md sec 1). Unit-normalizing before
    taking centroids removes scale/norm dependence, so values are comparable
    across datasets (e.g. GEX-ADT vs GEX-ATAC) and across encoders.
    """
    a = _unit_normalize(np.asarray(emb_a, dtype=np.float64))
    b = _unit_normalize(np.asarray(emb_b, dtype=np.float64))
    centroid_diff = a.mean(axis=0) - b.mean(axis=0)
    return float(np.linalg.norm(centroid_diff))


def alignment(emb_a: np.ndarray, emb_b: np.ndarray, alpha: float = 2.0) -> float:
    """Alignment term (Wang & Isola 2020): mean distance between paired positives.

    emb_a[i] and emb_b[i] must be the same cell measured in two modalities.
    Lower = better aligned (paired points sit closer together).
    """
    a = _unit_normalize(np.asarray(emb_a, dtype=np.float64))
    b = _unit_normalize(np.asarray(emb_b, dtype=np.float64))
    if a.shape[0] != b.shape[0]:
        raise ValueError("alignment() requires paired embeddings (same n_cells)")
    dists = np.linalg.norm(a - b, axis=1)
    return float(np.mean(dists ** alpha))


def _subsample(x: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if x.shape[0] <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_n, replace=False)
    return x[idx]


def uniformity(emb: np.ndarray, t: float = 2.0, max_n: int = 5000, seed: int = 0) -> float:
    """Uniformity term (Wang & Isola 2020): log mean pairwise Gaussian potential.

    Computed per-modality (single embedding set). Closer to 0 (from below) =
    more spread out / uniform on the hypersphere; more negative = more clustered.

    Pairwise squared distances are computed via ||a-b||^2 = 2 - 2*a.b (valid
    since inputs are unit-normalized) through a single matmul, i.e. O(n^2)
    memory instead of the O(n^2 * d) a naive broadcast would allocate — with
    tens of thousands of single cells, the naive version exhausts memory
    (see docs/HISTORY.md 2026-08-13, Phase 1 baseline OOM). n is additionally
    capped at `max_n` via random subsampling, since even O(n^2) is too much
    once n is in the tens of thousands.
    """
    x = _unit_normalize(np.asarray(emb, dtype=np.float64))
    x = _subsample(x, max_n, seed)
    n = x.shape[0]
    sq_dists = np.clip(2 - 2 * (x @ x.T), a_min=0, a_max=None)
    iu = np.triu_indices(n, k=1)
    pairwise = sq_dists[iu]
    return float(np.log(np.mean(np.exp(-t * pairwise)) + 1e-12))


def linear_separability(
    emb_a: np.ndarray, emb_b: np.ndarray, n_splits: int = 5, random_state: int = 0
) -> float:
    """Cross-validated accuracy of a linear classifier separating modality A vs B.

    0.5 = embeddings from the two modalities are linearly indistinguishable
    (no gap along any linear direction). 1.0 = perfectly separable (large gap).
    """
    a = _unit_normalize(np.asarray(emb_a, dtype=np.float64))
    b = _unit_normalize(np.asarray(emb_b, dtype=np.float64))
    x = np.concatenate([a, b], axis=0)
    y = np.concatenate([np.zeros(a.shape[0]), np.ones(b.shape[0])])
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores))


def topk_retrieval_accuracy(
    emb_query: np.ndarray, emb_target: np.ndarray, k: int = 5, max_n: int = 20000, seed: int = 0
) -> float:
    """Fraction of query cells whose true paired match is in the top-k nearest
    (by cosine similarity) targets. emb_query[i] and emb_target[i] must be the
    same cell (ground-truth pair index = row index).

    The retrieval task is defined over the *whole* candidate pool, so unlike
    uniformity() we can't subsample query and target independently — that
    would change what "top-k out of n" means. If n exceeds `max_n`, query and
    target are subsampled together with the same indices (preserving row
    correspondence) purely to keep the n x n similarity matrix (float32)
    within memory; this does shrink the candidate pool, which is a real
    (documented) tradeoff, not a free simplification.
    """
    q = _unit_normalize(np.asarray(emb_query, dtype=np.float32))
    t = _unit_normalize(np.asarray(emb_target, dtype=np.float32))
    if q.shape[0] != t.shape[0]:
        raise ValueError("topk_retrieval_accuracy() requires paired embeddings (same n_cells)")
    if q.shape[0] > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(q.shape[0], size=max_n, replace=False)
        q, t = q[idx], t[idx]
    n = q.shape[0]
    sims = q @ t.T
    k = min(k, n)
    topk_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    hits = np.array([i in topk_idx[i] for i in range(n)])
    return float(hits.mean())


def gap_report(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    paired: bool = True,
    k: int = 5,
) -> dict:
    """Compute the full metric suite used throughout PLAN.md sec 1/3/4.

    `paired` controls whether alignment/retrieval (which need true cross-modal
    correspondence) are computed; set False for e.g. cross-cell-type mismatch
    experiments where emb_a/emb_b are deliberately not aligned by row index.
    """
    report = {
        "delta_gap": delta_gap(emb_a, emb_b),
        "linear_separability": linear_separability(emb_a, emb_b),
        "uniformity_a": uniformity(emb_a),
        "uniformity_b": uniformity(emb_b),
    }
    if paired:
        report["alignment"] = alignment(emb_a, emb_b)
        report["top5_retrieval_acc"] = topk_retrieval_accuracy(emb_a, emb_b, k=k)
    return report
