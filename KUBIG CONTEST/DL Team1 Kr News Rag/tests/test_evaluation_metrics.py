import pytest

from evaluation.metrics import dedupe_articles, score_ranking, top_k_overlap


def hit(chunk_id, article_id):
    return {"chunk_id": chunk_id, "article_id": article_id, "title": f"기사 {article_id}", "content": "본문"}


def test_dedupe_articles_preserves_first_occurrence_and_reaches_five_unique_articles():
    hits = [hit("1a", 1), hit("1b", 1), hit("2a", 2), hit("3a", 3), hit("4a", 4), hit("5a", 5)]
    result = dedupe_articles(hits, top_k=5)
    assert [item["article_id"] for item in result] == [1, 2, 3, 4, 5]
    assert result[0]["first_chunk_rank"] == 1
    assert result[0]["chunk_ids"] == ["1a", "1b"]


def test_score_ranking_supports_multiple_golds_and_all_requested_metrics():
    scores = score_ranking([8, 2, 7, 3, 9], {3, 4})
    assert scores == {
        "first_relevant_rank": 4,
        "hit_at_1": 0,
        "hit_at_3": 0,
        "hit_at_5": 1,
        "mrr_at_5": pytest.approx(0.25),
    }


def test_score_ranking_reports_zero_when_gold_is_absent():
    assert score_ranking([1, 2, 3], {9}) == {
        "first_relevant_rank": None, "hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0, "mrr_at_5": 0.0
    }


def test_top_five_overlap_counts_shared_articles():
    assert top_k_overlap([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == 3
