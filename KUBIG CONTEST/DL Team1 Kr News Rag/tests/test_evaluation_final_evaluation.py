import pytest

from evaluation.final_evaluation import build_final_evaluation


def test_build_final_evaluation_merges_relevant_only_and_scores_paired_rankings():
    records = [{
        "query_id": "q1", "question": "question", "category": "factoid",
        "date_stratum": "s1", "gold_article_ids": [10], "seed_article_id": 10,
    }]
    mappings = [
        {"query_id": "q1", "candidate_key": "a", "article_id": 10,
         "dense_rank": 2, "hybrid_rank": 1, "calibration_source": True},
        {"query_id": "q1", "candidate_key": "b", "article_id": 20,
         "dense_rank": 1, "hybrid_rank": 2, "calibration_source": False},
        {"query_id": "q1", "candidate_key": "c", "article_id": 30,
         "dense_rank": 3, "hybrid_rank": 3, "calibration_source": False},
    ]
    judgments = [
        {"query_id": "q1", "candidate_key": "a", "final_label": "relevant"},
        {"query_id": "q1", "candidate_key": "b", "final_label": "relevant"},
        {"query_id": "q1", "candidate_key": "c", "final_label": "uncertain"},
    ]

    result = build_final_evaluation(records, mappings, judgments)

    merged = result["merged_rows"]
    assert [row["article_id"] for row in merged] == [10, 20, 30]
    assert [row["included_in_final_gold"] for row in merged] == [True, True, False]
    assert result["query_rows"][0]["final_gold_article_ids"] == [10, 20]
    assert result["query_rows"][0]["dense_mrr_at_5"] == pytest.approx(1.0)
    assert result["query_rows"][0]["hybrid_mrr_at_5"] == pytest.approx(1.0)
    assert result["query_rows"][0]["mrr_at_5_outcome"] == "tie"
    assert result["metrics"]["overall"]["dense"]["hit_at_1"] == 1.0
    assert result["diagnostics"]["uncertain_count"] == 1


def test_build_final_evaluation_rejects_missing_or_duplicate_keys():
    records = [{"query_id": "q1", "question": "q", "category": "abstract",
                "date_stratum": "s", "gold_article_ids": [1], "seed_article_id": 1}]
    mapping = [{"query_id": "q1", "candidate_key": "a", "article_id": 1,
                "dense_rank": 1, "hybrid_rank": 1, "calibration_source": True}]
    judgment = [{"query_id": "q1", "candidate_key": "a", "final_label": "relevant"}]

    with pytest.raises(ValueError, match="duplicate mapping key"):
        build_final_evaluation(records, mapping * 2, judgment)
    with pytest.raises(ValueError, match="key sets differ"):
        build_final_evaluation(records, mapping, [])
