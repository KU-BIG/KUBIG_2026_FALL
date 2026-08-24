"""Article-level ranking transforms and retrieval metrics."""

from __future__ import annotations


def dedupe_articles(hits: list[dict], top_k: int = 5) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    grouped: dict[int, dict] = {}
    order: list[int] = []
    for rank, hit in enumerate(hits, 1):
        article_id = hit.get("article_id")
        if type(article_id) is not int:
            raise ValueError(f"chunk at rank {rank} has no integer article_id")
        if article_id not in grouped:
            grouped[article_id] = {
                "article_id": article_id,
                "first_chunk_rank": rank,
                "chunk_ids": [],
                "title": hit.get("title", ""),
                "date": hit.get("date", ""),
                "content": hit.get("content", ""),
                "url": hit.get("url", ""),
                "doc_type": hit.get("doc_type", ""),
            }
            order.append(article_id)
        grouped[article_id]["chunk_ids"].append(hit.get("chunk_id"))
    return [grouped[article_id] for article_id in order[:top_k]]


def score_ranking(article_ids: list[int], gold_ids: set[int]) -> dict:
    first = next((rank for rank, article_id in enumerate(article_ids[:5], 1) if article_id in gold_ids), None)
    return {
        "first_relevant_rank": first,
        "hit_at_1": int(first is not None and first <= 1),
        "hit_at_3": int(first is not None and first <= 3),
        "hit_at_5": int(first is not None and first <= 5),
        "mrr_at_5": 1.0 / first if first is not None else 0.0,
    }


def top_k_overlap(first: list[int], second: list[int], k: int = 5) -> int:
    return len(set(first[:k]) & set(second[:k]))
