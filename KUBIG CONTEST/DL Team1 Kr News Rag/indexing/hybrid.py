"""Hybrid retrieval: dense + BM25 combined with Reciprocal Rank Fusion.

Dense retrieval matches meaning, BM25 matches exact tokens. RRF combines the two
by rank rather than by score, which matters because a cosine similarity (~0.6)
and a BM25 score (~12) live on scales that cannot be compared or averaged
directly. Each chunk scores `sum(1 / (k + rank))` over the lists it appears in,
so a chunk both retrievers rank highly beats a chunk only one of them loves.

`HybridRetriever` exposes the same `retrieve(question, top_k)` interface as
`DenseRetriever`, so `RAGPipeline` accepts it with no changes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Protocol

from indexing.build_index import DEFAULT_COLLECTION, DEFAULT_DB, DEFAULT_INPUT, DEFAULT_MODEL

DEFAULT_TOP_K = 5
# Number of candidates pulled from each retriever before fusing. Fusing two
# top-5 lists leaves almost nothing to fuse; the pool has to be wider than the
# final cut for rank agreement to mean anything.
DEFAULT_CANDIDATE_K = 20
# The conventional RRF constant. Larger values flatten the contribution of top
# ranks, making the fusion less sensitive to either retriever's exact ordering.
DEFAULT_RRF_K = 60


class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int = ...) -> list[dict]: ...


def rrf_fuse_many(
    rankings: Sequence[tuple[str, list[dict]]],
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Fuse any number of ranked chunk lists, ordered by descending RRF score.

    Each fused chunk gets `rrf_score` plus a `{label}_rank` for every list, set
    to None where the chunk is absent. Only those keys are initialised, so
    fields an inner fusion already wrote — `dense_rank` on hits coming out of
    `HybridRetriever`, for instance — survive a second pass on top.
    """
    labels = [label for label, _ in rankings]
    fused: dict[Any, dict] = {}
    order: list[Any] = []

    for label, hits in rankings:
        for rank, hit in enumerate(hits, 1):
            key = hit["chunk_id"]
            if key not in fused:
                fused[key] = {**hit, "rrf_score": 0.0}
                for other in labels:
                    fused[key].setdefault(f"{other}_rank", None)
                order.append(key)
            else:
                # Same chunk from another list: keep what is already there and
                # take whichever score fields this one contributes.
                for field, value in hit.items():
                    if fused[key].get(field) is None:
                        fused[key][field] = value
            fused[key]["rrf_score"] += 1.0 / (k + rank)
            fused[key][f"{label}_rank"] = rank

    return sorted(
        (fused[key] for key in order),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def rrf_fuse(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Fuse the dense and BM25 result lists into one."""
    return rrf_fuse_many([("dense", dense_hits), ("bm25", bm25_hits)], k=k)


class HybridRetriever:
    """Question -> top-k chunks fused from dense and BM25 retrieval."""

    def __init__(
        self,
        dense: Retriever | None = None,
        bm25: Retriever | None = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
        db_path: Path | str = DEFAULT_DB,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        chunks_path: Path | str = DEFAULT_INPUT,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._dense = dense
        self._bm25 = bm25
        self._dense_config = (db_path, collection_name, model_name, device)
        self._chunks_path = chunks_path
        self._alignment_checked = False

    def _ensure_dense(self) -> Retriever:
        if self._dense is None:
            from indexing.retriever import get_retriever

            self._dense = get_retriever(*self._dense_config)
        return self._dense

    def _ensure_bm25(self) -> Retriever:
        if self._bm25 is None:
            from indexing.bm25 import get_bm25_retriever

            self._bm25 = get_bm25_retriever(self._chunks_path)
        return self._bm25

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidate_k = max(self.candidate_k, top_k)
        dense_hits = self._ensure_dense().retrieve(question, top_k=candidate_k)
        bm25_hits = self._ensure_bm25().retrieve(question, top_k=candidate_k)

        self._warn_if_indexes_disagree(dense_hits)
        return rrf_fuse(dense_hits, bm25_hits, k=self.rrf_k)[:top_k]

    def _warn_if_indexes_disagree(self, dense_hits: list[dict]) -> None:
        """Catch a Chroma index built from a different chunk run than the JSONL.

        `chunk_id` is a hash of the chunk's content, so editing clean.py or the
        chunking parameters without rebuilding Chroma leaves the two sides joined
        on keys that can never match — RRF then silently degrades to two
        disjoint lists instead of erroring.
        """
        if self._alignment_checked or not dense_hits:
            return
        known = getattr(self._bm25, "chunk_ids", None)
        if known is None:
            return
        self._alignment_checked = True
        if not any(hit["chunk_id"] in known for hit in dense_hits):
            warnings.warn(
                "no dense chunk_id appears in the BM25 chunk file — the Chroma index "
                "and news_chunks.jsonl look like different chunk runs; re-run "
                "indexing/chunk.py and indexing/build_index.py --rebuild",
                RuntimeWarning,
                stacklevel=3,
            )


_RETRIEVERS: dict[tuple, HybridRetriever] = {}


def get_hybrid_retriever(
    db_path: Path | str = DEFAULT_DB,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    chunks_path: Path | str = DEFAULT_INPUT,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> HybridRetriever:
    """Reuse one hybrid retriever per configuration (BGE-M3 and BM25 load once)."""
    key = (str(Path(db_path)), collection_name, model_name, device, str(Path(chunks_path)), candidate_k, rrf_k)
    if key not in _RETRIEVERS:
        _RETRIEVERS[key] = HybridRetriever(
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            db_path=db_path,
            collection_name=collection_name,
            model_name=model_name,
            device=device,
            chunks_path=chunks_path,
        )
    return _RETRIEVERS[key]
