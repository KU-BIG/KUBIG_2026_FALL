"""BM25 keyword retrieval over the same chunks the dense index embeds.

Dense retrieval matches meaning and misses exact tokens — a stock code, a model
name like HBM, a rarely-used term the embedding smooths over. BM25 covers that
gap, and `HybridRetriever` fuses the two.

The corpus is 1,377 chunks (~4.3MB), so the index lives in memory and is rebuilt
on first use rather than persisted: tokenizing the whole corpus takes about half
a second, which is cheaper than keeping another artifact in sync with the chunk
file.
"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from indexing.build_index import DEFAULT_INPUT
from indexing.query_preprocess import normalize_query
from indexing.tokenize_ko import tokenize

DEFAULT_TOP_K = 5

# Fields the dense path returns; BM25 results carry the same shape so the two are
# interchangeable behind `RAGPipeline`.
_PASSTHROUGH_FIELDS = (
    "chunk_id",
    "article_id",
    "chunk_index",
    "title",
    "content",
    "date",
    "url",
    "stock_names",
    "stock_codes",
    "source_ids",
    "doc_type",
)


class BM25Retriever:
    """Question -> top-k chunks ranked by BM25 over Kiwi tokens."""

    def __init__(self, chunks_path: Path | str = DEFAULT_INPUT) -> None:
        self.chunks_path = Path(chunks_path)
        self._chunks: list[dict] | None = None
        self._index: BM25Okapi | None = None

    def _ensure_index(self) -> BM25Okapi:
        if self._index is None:
            if not self.chunks_path.is_file():
                raise FileNotFoundError(
                    f"chunk file not found: {self.chunks_path} (run indexing/chunk.py first)"
                )
            chunks = []
            with self.chunks_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSON on line {line_number} of {self.chunks_path}"
                        ) from exc
            if not chunks:
                raise ValueError(f"no chunks in {self.chunks_path}")
            # Index title + body, matching what the dense index embeds, so a title
            # keyword is findable by both retrievers.
            self._chunks = chunks
            self._index = BM25Okapi([tokenize(_indexed_text(c)) for c in chunks])
        return self._index

    @property
    def chunk_ids(self) -> set[str]:
        """Every chunk_id in the corpus — used to detect a stale Chroma index."""
        self._ensure_index()
        assert self._chunks is not None
        return {chunk["chunk_id"] for chunk in self._chunks}

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        index = self._ensure_index()
        assert self._chunks is not None  # set together with the index

        # Same normalization the dense path applies, so "삼전" reaches both
        # retrievers as "삼성전자". `normalize_query` is idempotent.
        query = normalize_query(question)
        scores = index.get_scores(tokenize(query))

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for position in ranked[:top_k]:
            chunk = self._chunks[position]
            item = {field: chunk.get(field) for field in _PASSTHROUGH_FIELDS}
            item["bm25_score"] = float(scores[position])
            results.append(item)
        return results


def _indexed_text(chunk: dict) -> str:
    embedding_text = chunk.get("embedding_text")
    if embedding_text:
        return embedding_text
    return f"{chunk.get('title', '')}\n{chunk.get('content', '')}".strip()


_RETRIEVERS: dict[str, BM25Retriever] = {}


def get_bm25_retriever(chunks_path: Path | str = DEFAULT_INPUT) -> BM25Retriever:
    """Reuse one retriever per chunk file so the index is built at most once."""
    key = str(Path(chunks_path))
    if key not in _RETRIEVERS:
        _RETRIEVERS[key] = BM25Retriever(chunks_path)
    return _RETRIEVERS[key]
