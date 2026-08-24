import json

import pytest

from evaluation.batch_pooling import build_batch_pool, write_pool_artifacts
from evaluation.cli import main
from tests.test_evaluation_combine import record as full_record


def article(article_id):
    return {
        "article_id": article_id,
        "title": f"기사 {article_id}",
        "date": "2026.08.01",
        "content": f"전체 본문 {article_id}",
        "url": f"https://example.com/{article_id}",
        "doc_type": "article",
    }


def hit(chunk_id, article_id):
    return {"chunk_id": chunk_id, **article(article_id)}


def record(query_id, source_article_id):
    return {
        "query_id": query_id,
        "question": f"질문 {query_id}",
        "category": "factoid",
        "seed_article_id": source_article_id,
    }


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error

    def retrieve(self, question, top_k):
        if self.error:
            raise self.error
        return self.results[question][:top_k]


def test_batch_pool_is_blind_deterministic_and_preserves_mapping_ranks():
    records = [record("K001", 1), record("K002", 4)]
    catalog = {article_id: article(article_id) for article_id in range(1, 7)}
    dense = FakeRetriever({
        "질문 K001": [hit("1a", 1), hit("1b", 1), hit("2a", 2)],
        "질문 K002": [hit("5a", 5)],
    })
    hybrid = FakeRetriever({
        "질문 K001": [hit("3a", 3), hit("2b", 2)],
        "질문 K002": [hit("6a", 6)],
    })

    first = build_batch_pool(records, {"dense": dense, "hybrid": hybrid}, catalog, seed=42)
    second = build_batch_pool(records, {"dense": dense, "hybrid": hybrid}, catalog, seed=42)

    assert first == second
    packet, mapping, stats = first
    candidates = [candidate for query in packet for candidate in query["candidates"]]
    assert len({candidate["candidate_key"] for candidate in candidates}) == len(candidates)
    assert all(set(candidate) == {
        "candidate_key", "title", "date", "content", "url", "doc_type"
    } for candidate in candidates)
    assert {(item["query_id"], item["candidate_key"]) for item in mapping} == {
        (query["query_id"], item["candidate_key"])
        for query in packet for item in query["candidates"]
    }
    source_one = next(item for item in mapping if item["query_id"] == "K001" and item["article_id"] == 1)
    assert source_one == {
        "query_id": "K001", "candidate_key": source_one["candidate_key"], "article_id": 1,
        "dense_rank": 1, "hybrid_rank": None, "calibration_source": True,
    }
    source_four = next(item for item in mapping if item["query_id"] == "K002" and item["article_id"] == 4)
    assert source_four["dense_rank"] is None and source_four["hybrid_rank"] is None
    assert source_four["calibration_source"] is True
    assert stats[0]["dense_unique_count"] == 2
    assert stats[0]["hybrid_unique_count"] == 2


def test_batch_pool_caps_each_system_at_twenty_unique_articles():
    records = [record("K001", 1)]
    catalog = {article_id: article(article_id) for article_id in range(1, 31)}
    hits = [hit(f"{article_id}a", article_id) for article_id in range(1, 26)]
    packet, mapping, stats = build_batch_pool(
        records,
        {"dense": FakeRetriever({"질문 K001": hits}), "hybrid": FakeRetriever({"질문 K001": hits})},
        catalog,
        pool_depth=20,
    )
    assert len(packet[0]["candidates"]) == 20
    assert len(mapping) == 20
    assert stats[0]["dense_unique_count"] == stats[0]["hybrid_unique_count"] == 20


def test_retrieval_error_writes_no_artifacts(tmp_path):
    records = [record("K001", 1)]
    with pytest.raises(RuntimeError, match="K001 dense retrieval failed"):
        build_batch_pool(
            records,
            {"dense": FakeRetriever(error=ValueError("boom")), "hybrid": FakeRetriever()},
            {1: article(1)},
        )
    outputs = [tmp_path / "blind.jsonl", tmp_path / "mapping.json", tmp_path / "manifest.json"]
    assert not any(path.exists() for path in outputs)


def test_artifact_writer_rejects_invalid_packet_without_partial_outputs(tmp_path):
    outputs = [tmp_path / "blind.jsonl", tmp_path / "mapping.json", tmp_path / "manifest.json"]
    packet = [{"query_id": "K001", "question": "q", "category": "factoid", "candidates": [
        {"candidate_key": "opaque", "article_id": 1}
    ]}]
    with pytest.raises(ValueError, match="forbidden blind field"):
        write_pool_artifacts(packet, [], {}, *outputs)
    assert not any(path.exists() for path in outputs)


def test_pool_batch_cli_writes_three_valid_artifacts(tmp_path):
    corpus = tmp_path / "articles.json"
    corpus.write_text(json.dumps([
        {"id": article_id, "title": f"기사 {article_id}", "date": "2026.08.07",
         "content": f"전체 본문 {article_id}", "url": f"u{article_id}", "doc_type": "article"}
        for article_id in range(1, 51)
    ], ensure_ascii=False), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text("".join(
        json.dumps(full_record(f"K{i:03d}", "kahyun", i), ensure_ascii=False) + "\n"
        for i in range(1, 26)
    ) + "".join(
        json.dumps(full_record(f"R{i:03d}", "ryeowon", i + 25), ensure_ascii=False) + "\n"
        for i in range(1, 26)
    ), encoding="utf-8")
    frozen_hash = __import__("hashlib").sha256(questions.read_bytes()).hexdigest()

    class EchoRetriever:
        def retrieve(self, question, top_k):
            return []

    outputs = [tmp_path / "blind.jsonl", tmp_path / "mapping.json", tmp_path / "manifest.json"]
    assert main([
        "pool-batch", str(questions), "--corpus", str(corpus),
        "--expected-freeze-sha256", frozen_hash,
        "--freeze-commit", "test-freeze",
        "--blind-output", str(outputs[0]), "--mapping-output", str(outputs[1]),
        "--manifest-output", str(outputs[2]),
    ], retriever_factory=lambda: {"dense": EchoRetriever(), "hybrid": EchoRetriever()}) == 0
    assert all(path.is_file() for path in outputs)
    packets = [json.loads(line) for line in outputs[0].read_text(encoding="utf-8").splitlines()]
    mapping = json.loads(outputs[1].read_text(encoding="utf-8"))
    manifest = json.loads(outputs[2].read_text(encoding="utf-8"))
    assert len(packets) == 50 and len(mapping) == 50
    assert manifest["freeze_sha256"] == frozen_hash
    assert manifest["random_seed"] == 42
    assert manifest["candidate_k"] == 20 and manifest["rrf_k"] == 60
    assert "Do not open the mapping" in manifest["judgment_instruction"]
