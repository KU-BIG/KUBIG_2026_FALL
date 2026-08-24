import json

import pytest

from evaluation.combine import combine_annotations


CHECKS = {
    "answer_supported_by_source": True, "natural_question": True, "not_title_copy": True,
    "not_duplicate": True, "source_article_id_verified": True,
}


def record(query_id, author, article_id):
    return {
        "query_id": query_id, "question": f"question {query_id}",
        "construction_method": "source_seeded", "category": "abstract",
        "date_stratum": "2026-08-07", "seed_article_id": article_id,
        "gold_article_ids": [article_id],
        "evidence": [{"article_id": article_id, "title": "title", "support": "support", "relevance": "relevant"}],
        "author": author, "reviewer": None, "review_mode": "ai_assisted_self_check",
        "self_check": CHECKS, "review_status": "approved", "annotation_minutes": None,
        "naturalness": None, "gold_clarity": None, "notes": "",
    }


def write(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def valid_files(tmp_path):
    kahyun = tmp_path / "kahyun_25.jsonl"
    ryeowon = tmp_path / "ryeowon_25.jsonl"
    write(kahyun, [record(f"K{i:03d}", "kahyun", i) for i in range(1, 26)])
    write(ryeowon, [record(f"R{i:03d}", "ryeowon", i + 25) for i in range(1, 26)])
    return kahyun, ryeowon


def test_combine_validates_and_writes_deterministic_author_order(tmp_path):
    kahyun, ryeowon = valid_files(tmp_path)
    before = (kahyun.read_bytes(), ryeowon.read_bytes())
    output = tmp_path / "combined.jsonl"
    records = combine_annotations([ryeowon, kahyun], output, set(range(1, 51)))
    assert [row["query_id"] for row in records] == [
        *(f"K{i:03d}" for i in range(1, 26)), *(f"R{i:03d}" for i in range(1, 26))]
    assert (kahyun.read_bytes(), ryeowon.read_bytes()) == before
    assert [json.loads(line)["query_id"] for line in output.read_text().splitlines()] == [
        row["query_id"] for row in records]


@pytest.mark.parametrize("mutation", ["wrong_author", "duplicate_source", "draft", "bad_query_id"])
def test_combine_validation_failure_never_overwrites_output(tmp_path, mutation):
    kahyun, ryeowon = valid_files(tmp_path)
    rows = [json.loads(line) for line in ryeowon.read_text().splitlines()]
    if mutation == "wrong_author":
        rows[0]["author"] = "kahyun"
    elif mutation == "duplicate_source":
        rows[0]["seed_article_id"] = 1
        rows[0]["gold_article_ids"] = [1]
        rows[0]["evidence"][0]["article_id"] = 1
    elif mutation == "draft":
        rows[0]["review_status"] = "draft"
    else:
        rows[0]["query_id"] = "R999"
    write(ryeowon, rows)
    output = tmp_path / "combined.jsonl"
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError):
        combine_annotations([kahyun, ryeowon], output, set(range(1, 51)))
    assert output.read_text(encoding="utf-8") == "keep me"


def test_combine_rejects_missing_records(tmp_path):
    kahyun, ryeowon = valid_files(tmp_path)
    write(kahyun, [record(f"K{i:03d}", "kahyun", i) for i in range(1, 25)])
    with pytest.raises(ValueError, match="25"):
        combine_annotations([kahyun, ryeowon], tmp_path / "out.jsonl", set(range(1, 51)))
