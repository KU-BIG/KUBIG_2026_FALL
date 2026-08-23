"""Retrieval over an expanded set of queries.

Wraps any retriever: the expander turns one question into several search
queries, each is searched independently, and the result lists are fused with the
same RRF used for dense+BM25. A chunk several phrasings agree on rises to the
top.

This is deliberately the same shape as the other query transformations we may
want later — rewriting a follow-up question into a standalone one is also
"question in, search queries out", so it slots in here as another expander
rather than as new plumbing.

Only the *retriever* sees the expanded queries. The answer prompt is still built
from the user's original question, which is what keeps a fabricated HyDE passage
out of the model's source material.
"""

from __future__ import annotations

from typing import Protocol

from indexing.hybrid import DEFAULT_CANDIDATE_K, DEFAULT_RRF_K, DEFAULT_TOP_K, rrf_fuse_many


class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int = ...) -> list[dict]: ...


class Expander(Protocol):
    def expand(self, question: str) -> list[str]: ...


class ExpandingRetriever:
    """Question -> expanded queries -> per-query retrieval -> fused top-k."""

    def __init__(
        self,
        base: Retriever,
        expander: Expander,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self.base = base
        self.expander = expander
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        queries = self.expander.expand(question)
        candidate_k = max(self.candidate_k, top_k)

        rankings = [
            (f"q{i}", self.base.retrieve(query, top_k=candidate_k))
            for i, query in enumerate(queries)
        ]
        fused = rrf_fuse_many(rankings, k=self.rrf_k)[:top_k]

        # Collapse the per-query rank keys into something a UI can render, and
        # carry the queries themselves so the expansion is inspectable.
        for item in fused:
            item["matched_queries"] = [
                i for i in range(len(queries)) if item.pop(f"q{i}_rank", None) is not None
            ]
            item["expanded_queries"] = queries
        return fused
