import csv
import json

from evaluation.reporting import write_csv, write_json


def report():
    return {
        "metadata": {
            "executed_at": "2026-08-20T00:00:00+00:00", "git_commit": "abc",
            "question_sha256": "deadbeef", "corpus": {"article_count": 432, "chunk_count": 1377},
            "source_article_count": 50,
            "retrieval_settings": {"candidate_k": 20, "rrf_k": 60},
        },
        "results": [{
            "query_id": "q1", "construction_method": "source_seeded", "question": "질문",
            "category": "abstract", "date_stratum": "2026-08-06", "system": "dense",
            "raw_chunk_ranking": [{"rank": 1, "chunk_id": "c1", "article_id": 1}],
            "unique_article_ranking": [{"article_id": 1}], "first_relevant_rank": 1,
            "hit_at_1": 1, "hit_at_3": 1, "hit_at_5": 1, "mrr_at_5": 1.0,
            "top_5_overlap": 1, "latency_seconds": 0.01, "error": None,
        }],
        "summaries": [],
    }


def test_reporting_writes_reproducible_json_and_flat_csv(tmp_path):
    json_path, csv_path = tmp_path / "result.json", tmp_path / "result.csv"
    write_json(report(), json_path)
    write_csv(report(), csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["metadata"]["git_commit"] == "abc"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert json.loads(row["raw_chunk_ranking"])[0]["chunk_id"] == "c1"
    assert json.loads(row["unique_article_ranking"])[0]["article_id"] == 1
    assert row["executed_at"] == "2026-08-20T00:00:00+00:00"
    assert row["git_commit"] == "abc"
    assert row["question_sha256"] == "deadbeef"
    assert row["corpus_article_count"] == "432"
    assert row["source_article_count"] == "50"
    assert row["date_stratum"] == "2026-08-06"
    assert json.loads(row["retrieval_settings"])["candidate_k"] == 20
