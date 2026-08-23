from collections import Counter

import pytest

from evaluation.assignment import create_assignment, replace_source, swap_categories


def articles():
    dates = ["2026.08.04"] * 65 + ["2026.08.06"] * 67 + ["2026.08.07"] * 195 + ["2026.08.08"] * 105
    return [{"id": index, "date": date} for index, date in enumerate(dates, 1)]


def test_assignment_stratifies_50_unique_sources_and_balances_authors_and_categories():
    assignment = create_assignment(articles(), seed=42)
    items = assignment["assignments"]
    assert len(items) == 50
    assert len({item["source_article_id"] for item in items}) == 50
    assert Counter(item["date_stratum"] for item in items) == {
        "2026-07-31_to_2026-08-05": 10, "2026-08-06": 10, "2026-08-07": 18, "2026-08-08": 12,
    }
    assert Counter(item["author"] for item in items) == {"kahyun": 25, "ryeowon": 25}
    for stratum, per_author in {"2026-07-31_to_2026-08-05": 5, "2026-08-06": 5, "2026-08-07": 9, "2026-08-08": 6}.items():
        assert Counter(item["author"] for item in items if item["date_stratum"] == stratum) == {"kahyun": per_author, "ryeowon": per_author}
    assert Counter(item["category"] for item in items) == {"exact_token": 13, "abstract": 13, "multi_aspect": 12, "factoid": 12}
    for stratum in assignment["strata"]:
        counts = Counter(item["category"] for item in items if item["date_stratum"] == stratum)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_assignment_is_reproducible_and_keeps_more_reserves_than_quota_per_stratum():
    first = create_assignment(articles(), seed=7)
    assert first == create_assignment(articles(), seed=7)
    for info in first["strata"].values():
        assert len(info["reserves"]) > info["sample_quota"]


def test_replacement_uses_next_reserve_from_same_stratum_and_records_predefined_reason():
    assignment = create_assignment(articles(), seed=42)
    old = assignment["assignments"][0]
    reserve = assignment["strata"][old["date_stratum"]]["reserves"][0]
    replaced = replace_source(assignment, position=1, reason_code="incomplete_article")
    item = replaced["assignments"][0]
    assert item["source_article_id"] == reserve
    assert item["date_stratum"] == old["date_stratum"]
    assert item["replacement"] == {"replaced_article_id": old["source_article_id"], "reason_code": "incomplete_article"}
    assert len({row["source_article_id"] for row in replaced["assignments"]}) == 50


def test_replacement_rejects_unapproved_reason():
    with pytest.raises(ValueError, match="reason"):
        replace_source(create_assignment(articles(), seed=42), position=1, reason_code="low_retrieval_score")


def test_category_swap_requires_same_stratum_and_preserves_global_quota():
    assignment = create_assignment(articles(), seed=42)
    before = Counter(item["category"] for item in assignment["assignments"])
    swapped = swap_categories(assignment, first_position=1, second_position=4)
    assert Counter(item["category"] for item in swapped["assignments"]) == before
    with pytest.raises(ValueError, match="stratum"):
        swap_categories(assignment, first_position=1, second_position=11)
