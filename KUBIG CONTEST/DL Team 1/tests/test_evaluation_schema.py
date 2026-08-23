import json

import pytest

from evaluation.schema import load_records, validate_records


def record(**changes):
    value = {
        "query_id": "q1",
        "question": "반도체 투자 전망은?",
        "construction_method": "source_seeded",
        "category": "abstract",
        "date_stratum": "2026-08-07",
        "seed_article_id": 1,
        "gold_article_ids": [1],
        "evidence": [{"article_id": 1, "title": "기사", "support": "근거", "relevance": "relevant"}],
        "author": "annotator_a",
        "reviewer": None,
        "review_mode": "ai_assisted_self_check",
        "self_check": {
            "answer_supported_by_source": True,
            "natural_question": True,
            "not_title_copy": True,
            "not_duplicate": True,
            "source_article_id_verified": True,
        },
        "review_status": "approved",
        "annotation_minutes": 8.5,
        "naturalness": 5,
        "gold_clarity": 4,
        "notes": "",
    }
    value.update(changes)
    return value


def test_load_records_reads_nonblank_jsonl_lines(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("\n" + json.dumps(record(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_records(path)[0]["query_id"] == "q1"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"question": " "}, "question"),
        ({"construction_method": "unsupported"}, "construction_method"),
        ({"category": ""}, "category"),
        ({"date_stratum": "2026-08-09"}, "date_stratum"),
        ({"seed_article_id": None}, "seed_article_id"),
        ({"gold_article_ids": [1, 1]}, "duplicate"),
        ({"gold_article_ids": [999]}, "corpus"),
        ({"evidence": [{"article_id": 2, "title": "x", "support": "x", "relevance": "relevant"}]}, "gold_article_ids"),
        ({"review_mode": "cross_review"}, "review_mode"),
    ],
)
def test_validate_records_rejects_invalid_records(changes, message):
    with pytest.raises(ValueError, match=message):
        validate_records([record(**changes)], {1, 2}, allow_draft=False)


def test_source_seeded_requires_seed_article_id():
    with pytest.raises(ValueError, match="seed_article_id"):
        validate_records([record(seed_article_id=None)], {1}, allow_draft=False)


def test_legacy_query_first_record_remains_supported():
    legacy = record(construction_method="query_first", category="entity", seed_article_id=None)
    for field in ("date_stratum", "review_mode", "self_check"):
        legacy.pop(field)
    validate_records([legacy], {1}, allow_draft=False)


def test_duplicate_query_ids_are_rejected():
    with pytest.raises(ValueError, match="query_id"):
        validate_records([record(), record()], {1}, allow_draft=False)


def test_duplicate_questions_are_rejected():
    with pytest.raises(ValueError, match="duplicate question"):
        validate_records([record(), record(query_id="q2", question="  반도체   투자 전망은?  ")], {1}, allow_draft=False)


def test_draft_requires_explicit_allow_draft():
    draft = record(review_status="draft", evidence=[])
    with pytest.raises(ValueError, match="draft"):
        validate_records([draft], {1}, allow_draft=False)
    validate_records([draft], {1}, allow_draft=True)


def test_final_rejects_uncertain_evidence():
    uncertain = record(evidence=[{"article_id": 1, "title": "기사", "support": "검토 중", "relevance": "uncertain"}])
    with pytest.raises(ValueError, match="uncertain"):
        validate_records([uncertain], {1}, allow_draft=False)


def test_approved_requires_at_least_one_gold_article():
    with pytest.raises(ValueError, match="gold_article_ids"):
        validate_records([record(gold_article_ids=[], evidence=[])], {1}, allow_draft=False)


def test_approved_allows_null_reviewer_when_self_check_is_complete():
    validate_records([record(reviewer=None)], {1}, allow_draft=False)


@pytest.mark.parametrize("failed_check", [
    "answer_supported_by_source", "natural_question", "not_title_copy",
    "not_duplicate", "source_article_id_verified",
])
def test_approved_requires_every_self_check(failed_check):
    checks = record()["self_check"] | {failed_check: False}
    with pytest.raises(ValueError, match="self_check"):
        validate_records([record(self_check=checks)], {1}, allow_draft=False)


def test_approved_requires_seed_article_in_initial_gold():
    with pytest.raises(ValueError, match="initial gold"):
        validate_records([record(seed_article_id=2, gold_article_ids=[1])], {1, 2}, allow_draft=False)


def test_additional_relevant_pooled_article_can_be_gold():
    expanded = record(
        gold_article_ids=[1, 2],
        evidence=[
            {"article_id": 1, "title": "seed", "support": "근거", "relevance": "relevant"},
            {"article_id": 2, "title": "other", "support": "추가 근거", "relevance": "relevant"},
        ],
    )
    validate_records([expanded], {1, 2}, allow_draft=False)


def test_evidence_requires_article_title_support_and_relevance_fields():
    with pytest.raises(ValueError, match="evidence"):
        validate_records([record(evidence=[{"article_id": 1, "relevance": "relevant"}])], {1}, allow_draft=False)
