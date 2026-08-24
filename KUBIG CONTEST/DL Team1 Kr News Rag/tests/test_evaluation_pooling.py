from evaluation.pooling import build_pool


def hit(chunk_id, article_id, title):
    return {
        "chunk_id": chunk_id, "article_id": article_id, "title": title, "date": "2026.08.01",
        "content": "본문", "url": "https://example.com", "doc_type": "article",
    }


def test_pool_deduplicates_by_article_and_always_hides_system_ranks():
    pool = build_pool(
        [hit("a1", 1, "A"), hit("b", 2, "B")],
        [hit("a2", 1, "A"), hit("c", 3, "C")],
        depth=20,
    )
    assert [item["article_id"] for item in pool] == [1, 2, 3]
    assert all("dense_rank" not in item and "hybrid_rank" not in item for item in pool)


def test_blind_pool_keeps_article_content():
    pool = build_pool([hit("a", 1, "A")], [], depth=20)
    assert set(pool[0]) == {"candidate_id", "article_id", "title", "date", "content", "url", "doc_type"}
    assert pool[0]["candidate_id"] == "candidate_001"
