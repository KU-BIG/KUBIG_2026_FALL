import json

import pytest

from indexing.bm25 import BM25Retriever


def chunk(chunk_id, content, title="제목", names=None):
    names = names or ["삼성전자"]
    return {
        "chunk_id": chunk_id,
        "article_id": 1,
        "chunk_index": 0,
        "title": title,
        "content": content,
        "embedding_text": f"{title}\n{content}",
        "date": "2026.08.08",
        "url": f"https://example.com/{chunk_id}",
        "stock_names": names,
        "stock_codes": ["005930"],
        "source_ids": [1],
        "doc_type": "article",
    }


def write_chunks(path, records):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return path


CORPUS = [
    chunk("semiconductor", "삼성전자의 반도체 실적이 크게 개선됐다."),
    chunk("battery", "LG에너지솔루션의 배터리 수주가 늘었다.", names=["LG에너지솔루션"]),
    chunk("shipbuilding", "HD현대중공업이 카타르에서 LNG선을 수주했다.", names=["HD현대중공업"]),
]


@pytest.fixture
def corpus_path(tmp_path):
    return write_chunks(tmp_path / "chunks.jsonl", CORPUS)


def test_keyword_match_ranks_its_chunk_first(corpus_path):
    results = BM25Retriever(corpus_path).retrieve("배터리 수주")

    assert results[0]["chunk_id"] == "battery"


def test_query_matches_a_noun_that_carries_a_particle_in_the_document(corpus_path):
    # The document says "삼성전자의"; without morphological splitting BM25 would
    # never match the bare "삼성전자" in the question. This is the whole reason
    # the Kiwi tokenizer is here.
    results = BM25Retriever(corpus_path).retrieve("삼성전자 반도체")

    assert results[0]["chunk_id"] == "semiconductor"


def test_stock_alias_in_the_question_is_normalized_before_matching(corpus_path):
    results = BM25Retriever(corpus_path).retrieve("삼전 반도체")

    assert results[0]["chunk_id"] == "semiconductor"


def test_results_carry_the_same_metadata_dense_retrieval_returns(corpus_path):
    result = BM25Retriever(corpus_path).retrieve("반도체 실적")[0]

    assert result["title"] == "제목"
    assert result["date"] == "2026.08.08"
    assert result["url"] == "https://example.com/semiconductor"
    assert result["content"] == "삼성전자의 반도체 실적이 크게 개선됐다."
    assert result["stock_names"] == ["삼성전자"]
    assert result["article_id"] == 1


def test_results_are_scored_and_ordered_by_descending_score(corpus_path):
    results = BM25Retriever(corpus_path).retrieve("반도체 실적", top_k=3)

    scores = [r["bm25_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_caps_results_at_top_k(corpus_path):
    assert len(BM25Retriever(corpus_path).retrieve("수주", top_k=2)) == 2


def test_retrieve_returns_everything_when_top_k_exceeds_the_corpus(corpus_path):
    assert len(BM25Retriever(corpus_path).retrieve("수주", top_k=99)) == len(CORPUS)


def test_construction_is_lazy_and_a_missing_file_fails_on_retrieve(tmp_path):
    retriever = BM25Retriever(tmp_path / "absent.jsonl")

    with pytest.raises(FileNotFoundError, match="absent.jsonl"):
        retriever.retrieve("반도체")


def test_blank_question_is_rejected(corpus_path):
    with pytest.raises(ValueError):
        BM25Retriever(corpus_path).retrieve("   ")


def test_index_is_built_once_and_reused(corpus_path):
    retriever = BM25Retriever(corpus_path)
    retriever.retrieve("반도체")
    first_index = retriever._index

    retriever.retrieve("배터리")

    assert retriever._index is first_index
