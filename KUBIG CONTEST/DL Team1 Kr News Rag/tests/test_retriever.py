import pytest

from indexing.build_index import build_index
from indexing.retriever import DenseRetriever, get_retriever
from tests.test_index import DeterministicEmbedder, chunk, write_jsonl


def build_test_db(tmp_path, records, collection_name="test_retriever"):
    source = tmp_path / "chunks.jsonl"
    write_jsonl(source, records)
    build_index(source, tmp_path / "chroma", collection_name, DeterministicEmbedder())
    return tmp_path / "chroma"


def test_retrieve_returns_five_chunks_by_default(tmp_path):
    records = [chunk(f"c{i}", "반도체 실적", ["삼성전자"]) for i in range(7)]
    db_path = build_test_db(tmp_path, records)
    retriever = DenseRetriever(db_path=db_path, collection_name="test_retriever", embedder=DeterministicEmbedder())

    results = retriever.retrieve("반도체 전망")

    assert len(results) == 5


def test_retrieve_reports_similarity_derived_from_distance(tmp_path):
    db_path = build_test_db(tmp_path, [chunk("semiconductor", "반도체 실적", ["삼성전자"])])
    retriever = DenseRetriever(db_path=db_path, collection_name="test_retriever", embedder=DeterministicEmbedder())

    result = retriever.retrieve("반도체 전망")[0]

    assert result["distance"] == pytest.approx(0.0)
    assert result["similarity"] == pytest.approx(1.0)


def test_retrieve_carries_source_fields_for_citation(tmp_path):
    db_path = build_test_db(tmp_path, [chunk("semiconductor", "반도체 실적", ["삼성전자", "SK하이닉스"])])
    retriever = DenseRetriever(db_path=db_path, collection_name="test_retriever", embedder=DeterministicEmbedder())

    result = retriever.retrieve("반도체 전망")[0]

    assert result["title"] == "제목"
    assert result["date"] == "2026.08.08"
    assert result["url"] == "https://example.com/semiconductor"
    assert result["content"] == "반도체 실적"
    assert result["stock_names"] == ["삼성전자", "SK하이닉스"]


def test_retrieve_normalizes_stock_alias_before_embedding(tmp_path):
    db_path = build_test_db(tmp_path, [chunk("semiconductor", "반도체 실적", ["삼성전자"])])
    embedder = DeterministicEmbedder()
    retriever = DenseRetriever(db_path=db_path, collection_name="test_retriever", embedder=embedder)

    retriever.retrieve("  삼전   반도체 전망  ")

    assert embedder.encoded_texts == ["삼성전자 반도체 전망"]


def test_results_are_ordered_by_descending_similarity(tmp_path):
    records = [chunk("semiconductor", "반도체 실적", ["삼성전자"]), chunk("battery", "배터리 수요", ["LG에너지솔루션"])]
    db_path = build_test_db(tmp_path, records)
    retriever = DenseRetriever(db_path=db_path, collection_name="test_retriever", embedder=DeterministicEmbedder())

    results = retriever.retrieve("반도체 전망", top_k=2)

    assert [r["chunk_id"] for r in results] == ["semiconductor", "battery"]
    assert results[0]["similarity"] > results[1]["similarity"]


def test_construction_is_lazy_and_missing_db_fails_on_retrieve(tmp_path):
    retriever = DenseRetriever(db_path=tmp_path / "absent", collection_name="test_retriever")

    with pytest.raises(FileNotFoundError, match="absent"):
        retriever.retrieve("반도체 전망")


def test_missing_collection_names_itself_in_the_error(tmp_path):
    db_path = build_test_db(tmp_path, [chunk("semiconductor", "반도체 실적", ["삼성전자"])])
    retriever = DenseRetriever(db_path=db_path, collection_name="absent_collection", embedder=DeterministicEmbedder())

    with pytest.raises(RuntimeError, match="absent_collection"):
        retriever.retrieve("반도체 전망")


def test_blank_question_is_rejected(tmp_path):
    db_path = build_test_db(tmp_path, [chunk("semiconductor", "반도체 실적", ["삼성전자"])])
    retriever = DenseRetriever(db_path=db_path, collection_name="test_retriever", embedder=DeterministicEmbedder())

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_get_retriever_reuses_one_instance_per_configuration(tmp_path):
    first = get_retriever(db_path=tmp_path / "chroma", collection_name="test_retriever")
    second = get_retriever(db_path=tmp_path / "chroma", collection_name="test_retriever")
    other = get_retriever(db_path=tmp_path / "chroma", collection_name="another")

    assert first is second
    assert first is not other


def test_a_cuda_machine_uses_the_gpu():
    from indexing.retriever import choose_device

    assert choose_device(has_cuda=True, has_mps=False) == "cuda"


def test_apple_silicon_falls_back_to_mps():
    from indexing.retriever import choose_device

    assert choose_device(has_cuda=False, has_mps=True) == "mps"


def test_a_machine_with_no_accelerator_uses_the_cpu():
    from indexing.retriever import choose_device

    assert choose_device(has_cuda=False, has_mps=False) == "cpu"


def test_cuda_wins_when_both_are_somehow_available():
    from indexing.retriever import choose_device

    assert choose_device(has_cuda=True, has_mps=True) == "cuda"
