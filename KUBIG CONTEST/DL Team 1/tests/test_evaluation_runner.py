import hashlib

from evaluation.runner import run_evaluation


def hit(chunk_id, article_id):
    return {"chunk_id": chunk_id, "article_id": article_id, "title": f"article {article_id}",
            "date": "2026.08.01", "content": "body", "url": f"https://example.com/{article_id}",
            "doc_type": "article"}


class FakeRetriever:
    def __init__(self, hits=None, error=None):
        self.hits, self.error, self.calls = hits or [], error, []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        if self.error:
            raise RuntimeError(self.error)
        return self.hits[:top_k]


def record(query_id="q1", *, category="abstract", date_stratum="2026-07-31_to_2026-08-05"):
    return {
        "query_id": query_id, "question": "question", "construction_method": "source_seeded",
        "category": category, "date_stratum": date_stratum, "seed_article_id": 2,
        "gold_article_ids": [2, 9], "evidence": [], "author": "annotator_a", "reviewer": None,
        "review_status": "approved", "annotation_minutes": 10, "naturalness": 4,
        "gold_clarity": 5, "notes": "",
    }


def run(records, retrievers, path, **kwargs):
    return run_evaluation(
        records, retrievers, question_path=path, corpus_article_count=432,
        corpus_chunk_count=1377, source_article_count=50,
        stratum_corpus_counts={"2026-07-31_to_2026-08-05": 65, "2026-08-06": 67,
                               "2026-08-07": 195, "2026-08-08": 105},
        settings={"candidate_k": 20, "rrf_k": 60, "model": "BAAI/bge-m3"},
        git_commit="abc123", **kwargs)


def test_runner_creates_paired_raw_and_unique_article_results_and_hash(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_bytes(b"question bytes\n")
    dense = FakeRetriever([hit("1a", 1), hit("1b", 1), hit("2", 2), hit("3", 3), hit("4", 4), hit("5", 5)])
    hybrid = FakeRetriever([hit("2", 2), hit("6", 6)])
    report = run([record()], {"dense": dense, "hybrid": hybrid}, questions,
                 chunk_depth=100, now=lambda: "2026-08-20T00:00:00+00:00")
    assert dense.calls == [("question", 100)] and hybrid.calls == [("question", 100)]
    assert report["metadata"]["question_sha256"] == hashlib.sha256(b"question bytes\n").hexdigest()
    assert report["metadata"]["corpus"]["article_count"] == 432
    assert report["metadata"]["source_article_count"] == 50
    by_system = {result["system"]: result for result in report["results"]}
    assert by_system["dense"]["date_stratum"] == "2026-07-31_to_2026-08-05"
    assert by_system["dense"]["raw_chunk_ranking"][1] == {"rank": 2, "chunk_id": "1b", "article_id": 1}
    assert [item["article_id"] for item in by_system["dense"]["unique_article_ranking"]] == [1, 2, 3, 4, 5]
    assert by_system["dense"]["first_relevant_rank"] == 2
    assert by_system["hybrid"]["hit_at_1"] == 1


def test_runner_records_one_system_error_without_losing_paired_result(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("", encoding="utf-8")
    report = run([record()], {"dense": FakeRetriever(error="boom"), "hybrid": FakeRetriever([hit("2", 2)])}, path)
    by_system = {result["system"]: result for result in report["results"]}
    assert by_system["dense"]["error"] == "RuntimeError: boom"
    assert by_system["hybrid"]["hit_at_1"] == 1


def test_runner_summarizes_overall_date_strata_and_question_categories(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("", encoding="utf-8")
    records = [record(), record("q2", category="factoid", date_stratum="2026-08-06")]
    report = run(records, {"dense": FakeRetriever([hit("2", 2)]), "hybrid": FakeRetriever([])}, path)
    assert report["summaries"]["overall"]["question_count"] == 2
    assert report["summaries"]["overall"]["systems"]["dense"]["hit_at_1"] == 1.0
    early = report["summaries"]["by_date_stratum"]["2026-07-31_to_2026-08-05"]
    assert early["corpus_article_count"] == 65 and early["sample_count"] == 1
    assert report["summaries"]["by_category"]["factoid"]["question_count"] == 1
    assert "exploratory" in report["summaries"]["date_stratum_interpretation"].lower()
    assert "single-annotator" in report["summaries"]["limitations"][0]
