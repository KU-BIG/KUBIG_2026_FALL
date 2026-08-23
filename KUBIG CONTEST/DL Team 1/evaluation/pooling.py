"""Pool article candidates for human relevance judgments."""

from __future__ import annotations

from evaluation.metrics import dedupe_articles


def build_pool(dense_hits: list[dict], hybrid_hits: list[dict], *, depth: int = 20) -> list[dict]:
    rankings = {
        "dense": dedupe_articles(dense_hits, depth),
        "hybrid": dedupe_articles(hybrid_hits, depth),
    }
    pooled: dict[int, dict] = {}
    order: list[int] = []
    for system, articles in rankings.items():
        for rank, article in enumerate(articles, 1):
            article_id = article["article_id"]
            if article_id not in pooled:
                pooled[article_id] = {
                    "article_id": article_id,
                    "title": article["title"],
                    "date": article["date"],
                    "content": article["content"],
                    "url": article["url"],
                    "doc_type": article["doc_type"],
                    "dense_rank": None,
                    "hybrid_rank": None,
                }
                order.append(article_id)
            pooled[article_id][f"{system}_rank"] = rank
    result = [pooled[article_id] for article_id in order]
    visible = ("article_id", "title", "date", "content", "url", "doc_type")
    return [
        {"candidate_id": f"candidate_{index:03d}", **{field: item[field] for field in visible}}
        for index, item in enumerate(result, 1)
    ]
