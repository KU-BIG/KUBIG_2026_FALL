import json
import subprocess
import sys

import pytest

from evaluation.cli import main
from tests.test_evaluation_combine import record


def article_file(path):
    path.write_text(json.dumps([{
        "id": 1, "title": "반도체", "content": "HBM", "date": "2026.08.01",
        "url": "https://example.com/1", "doc_type": "article",
    }], ensure_ascii=False), encoding="utf-8")


def draft_record():
    return {
        "query_id": "q1", "question": "질문", "construction_method": "source_seeded",
        "category": "abstract", "date_stratum": "2026-08-07", "seed_article_id": 1,
        "gold_article_ids": [1], "evidence": [], "author": "a", "reviewer": None,
        "review_mode": "ai_assisted_self_check", "self_check": {
            "answer_supported_by_source": False, "natural_question": False, "not_title_copy": False,
            "not_duplicate": False, "source_article_id_verified": False,
        }, "review_status": "draft", "annotation_minutes": None,
        "naturalness": None, "gold_clarity": None, "notes": "",
    }


def test_validate_requires_allow_draft_for_draft_records(tmp_path):
    corpus, questions = tmp_path / "articles.json", tmp_path / "questions.jsonl"
    article_file(corpus)
    questions.write_text(json.dumps(draft_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="draft"):
        main(["validate", str(questions), "--corpus", str(corpus)])
    assert main(["validate", str(questions), "--corpus", str(corpus), "--allow-draft"]) == 0


def test_empty_run_does_not_load_retrievers_or_write_results(tmp_path, capsys):
    corpus, questions = tmp_path / "articles.json", tmp_path / "questions.jsonl"
    article_file(corpus)
    questions.write_text("", encoding="utf-8")
    called = False
    def factory():
        nonlocal called
        called = True
        raise AssertionError("must not load")
    output = tmp_path / "result.json"
    assert main(["run", str(questions), "--corpus", str(corpus), "--json-output", str(output)], retriever_factory=factory) == 0
    assert not called and not output.exists()
    assert "no evaluation records" in capsys.readouterr().out.lower()


def test_pool_always_hides_system_provenance(tmp_path, capsys):
    class Fake:
        def __init__(self, article_id): self.article_id = article_id
        def retrieve(self, question, top_k=5):
            return [{"chunk_id": str(self.article_id), "article_id": self.article_id, "title": "A", "date": "d", "content": "c", "url": "u", "doc_type": "article"}]
    factory = lambda: {"dense": Fake(1), "hybrid": Fake(2)}
    assert main(["pool", "질문"], retriever_factory=factory) == 0
    output = json.loads(capsys.readouterr().out)
    assert "dense_rank" not in output[0]
    assert output[0]["candidate_id"] == "candidate_001"


def test_module_help_does_not_load_models():
    result = subprocess.run([sys.executable, "-m", "evaluation.cli", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "validate" in result.stdout and "pool" in result.stdout


def test_assign_cli_writes_stratified_assignment_without_retrievers(tmp_path):
    dates = ["2026.08.04"] * 65 + ["2026.08.06"] * 67 + ["2026.08.07"] * 195 + ["2026.08.08"] * 105
    corpus = tmp_path / "articles.json"
    corpus.write_text(json.dumps([{"id": i, "date": date} for i, date in enumerate(dates, 1)]), encoding="utf-8")
    output = tmp_path / "assignment.json"
    assert main(["assign", "--corpus", str(corpus), "--seed", "42", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text(encoding="utf-8"))["assignments"]) == 50


def test_combine_cli_writes_50_records_in_query_id_order(tmp_path):
    corpus = tmp_path / "articles.json"
    corpus.write_text(json.dumps([
        {"id": i, "title": f"title {i}", "content": "body", "date": "2026.08.07",
         "url": f"https://example.com/{i}", "doc_type": "article"}
        for i in range(1, 51)
    ]), encoding="utf-8")
    kahyun, ryeowon = tmp_path / "kahyun_25.jsonl", tmp_path / "ryeowon_25.jsonl"
    kahyun.write_text("".join(json.dumps(record(f"K{i:03d}", "kahyun", i)) + "\n"
                              for i in range(1, 26)), encoding="utf-8")
    ryeowon.write_text("".join(json.dumps(record(f"R{i:03d}", "ryeowon", i + 25)) + "\n"
                               for i in range(1, 26)), encoding="utf-8")
    output = tmp_path / "retrieval_eval_50.jsonl"
    assert main(["combine", str(kahyun), str(ryeowon), "--output", str(output),
                 "--corpus", str(corpus)]) == 0
    assert [json.loads(line)["query_id"] for line in output.read_text().splitlines()] == [
        *(f"K{i:03d}" for i in range(1, 26)), *(f"R{i:03d}" for i in range(1, 26))]
