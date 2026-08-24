import pytest

from indexing.hybrid import DEFAULT_RRF_K, HybridRetriever, rrf_fuse
from rag import RAGPipeline


def hit(chunk_id, **extra):
    base = {
        "chunk_id": chunk_id,
        "article_id": 1,
        "chunk_index": 0,
        "title": f"제목-{chunk_id}",
        "content": f"본문-{chunk_id}",
        "date": "2026.08.08",
        "url": f"https://example.com/{chunk_id}",
        "stock_names": ["삼성전자"],
        "stock_codes": ["005930"],
        "source_ids": [1],
        "doc_type": "article",
    }
    base.update(extra)
    return base


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.hits[:top_k]


class FakeBM25(FakeRetriever):
    def __init__(self, hits, chunk_ids=None):
        super().__init__(hits)
        self.chunk_ids = set(chunk_ids) if chunk_ids is not None else {h["chunk_id"] for h in hits}


# --- RRF math -------------------------------------------------------------


def test_a_chunk_found_by_both_retrievers_scores_the_sum_of_both_ranks():
    fused = rrf_fuse([hit("a", similarity=0.7)], [hit("a", bm25_score=3.0)])

    assert len(fused) == 1
    assert fused[0]["rrf_score"] == pytest.approx(2 / (DEFAULT_RRF_K + 1))


def test_a_chunk_found_by_one_retriever_scores_that_rank_alone():
    fused = rrf_fuse([hit("a", similarity=0.7)], [])

    assert fused[0]["rrf_score"] == pytest.approx(1 / (DEFAULT_RRF_K + 1))


def test_agreement_between_retrievers_outranks_a_single_top_hit():
    dense = [hit("only_dense", similarity=0.9), hit("both", similarity=0.5)]
    bm25 = [hit("both", bm25_score=4.0)]

    fused = rrf_fuse(dense, bm25)

    # "both" is rank 2 in dense and rank 1 in bm25; "only_dense" is rank 1 in one
    # list and absent from the other. Fusion should prefer the agreed-on chunk.
    assert [f["chunk_id"] for f in fused] == ["both", "only_dense"]


def test_fused_results_record_where_each_chunk_came_from():
    fused = rrf_fuse(
        [hit("x", similarity=0.5), hit("a", similarity=0.9)],
        [hit("a", bm25_score=4.0)],
    )
    by_id = {f["chunk_id"]: f for f in fused}

    assert by_id["a"]["dense_rank"] == 2
    assert by_id["a"]["bm25_rank"] == 1
    assert by_id["x"]["dense_rank"] == 1
    assert by_id["x"]["bm25_rank"] is None


def test_fusion_keeps_scores_from_both_retrievers():
    fused = rrf_fuse([hit("a", similarity=0.7)], [hit("a", bm25_score=3.0)])

    assert fused[0]["similarity"] == 0.7
    assert fused[0]["bm25_score"] == 3.0


def test_fusion_preserves_chunk_metadata_for_a_bm25_only_hit():
    fused = rrf_fuse([], [hit("b", bm25_score=1.0)])

    assert fused[0]["title"] == "제목-b"
    assert fused[0]["url"] == "https://example.com/b"
    assert fused[0]["stock_names"] == ["삼성전자"]


def test_fusing_nothing_yields_nothing():
    assert rrf_fuse([], []) == []


def test_results_are_ordered_by_descending_rrf_score():
    fused = rrf_fuse(
        [hit("a"), hit("b"), hit("c")],
        [hit("c"), hit("a")],
    )
    scores = [f["rrf_score"] for f in fused]

    assert scores == sorted(scores, reverse=True)


# --- HybridRetriever ------------------------------------------------------


def test_hybrid_asks_both_retrievers_for_a_wider_candidate_pool_than_top_k():
    dense, bm25 = FakeRetriever([hit("a")]), FakeBM25([hit("a")])
    retriever = HybridRetriever(dense=dense, bm25=bm25, candidate_k=20)

    retriever.retrieve("반도체 전망", top_k=5)

    # Fusing two top-5 lists leaves almost nothing to fuse; both sides must be
    # asked for the candidate pool, not the final cut.
    assert dense.calls == [("반도체 전망", 20)]
    assert bm25.calls == [("반도체 전망", 20)]


def test_hybrid_returns_at_most_top_k():
    hits = [hit(c) for c in "abcdefgh"]
    retriever = HybridRetriever(dense=FakeRetriever(hits), bm25=FakeBM25(hits))

    assert len(retriever.retrieve("반도체", top_k=3)) == 3


def test_hybrid_joins_the_two_result_lists_by_chunk_id():
    dense, bm25 = FakeRetriever([hit("same")]), FakeBM25([hit("same")])
    retriever = HybridRetriever(dense=dense, bm25=bm25)

    results = retriever.retrieve("반도체")

    assert len(results) == 1
    assert results[0]["dense_rank"] == 1 and results[0]["bm25_rank"] == 1


def test_hybrid_rejects_a_blank_question():
    retriever = HybridRetriever(dense=FakeRetriever([]), bm25=FakeBM25([]))

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_hybrid_warns_when_the_two_indexes_hold_different_chunks(recwarn):
    # chunk_id is a content hash, so re-running clean.py or chunk.py without
    # rebuilding Chroma leaves the two indexes joined on keys that never match.
    dense = FakeRetriever([hit("from_old_chunk_run")])
    bm25 = FakeBM25([hit("from_new_chunk_run")])

    HybridRetriever(dense=dense, bm25=bm25).retrieve("반도체")

    assert any("chunk" in str(w.message).lower() for w in recwarn)


def test_hybrid_does_not_warn_when_the_indexes_agree(recwarn):
    dense = FakeRetriever([hit("shared")])
    bm25 = FakeBM25([hit("shared")])

    HybridRetriever(dense=dense, bm25=bm25).retrieve("반도체")

    assert not [w for w in recwarn if "chunk" in str(w.message).lower()]


def test_hybrid_drops_into_the_rag_pipeline_unchanged():
    # The whole point of matching the dense retriever's interface: swapping in
    # hybrid search must not require touching rag.py.
    hits = [hit("a", similarity=0.8)]
    retriever = HybridRetriever(dense=FakeRetriever(hits), bm25=FakeBM25(hits))

    class FakeLLM:
        def generate(self, system_prompt, user_prompt, history=None):
            return "답변 [뉴스1]"

    response = RAGPipeline(retriever=retriever, llm=FakeLLM()).ask("반도체 전망")

    assert response.answer == "답변 [뉴스1]"
    assert response.sources[0]["url"] == "https://example.com/a"


# --- N-way fusion ---------------------------------------------------------


def test_fusing_many_lists_sums_one_term_per_list_the_chunk_appears_in():
    from indexing.hybrid import rrf_fuse_many

    fused = rrf_fuse_many(
        [("q0", [hit("a")]), ("q1", [hit("a")]), ("q2", [hit("a")])]
    )

    assert fused[0]["rrf_score"] == pytest.approx(3 / (DEFAULT_RRF_K + 1))


def test_fusing_many_lists_records_a_rank_per_label():
    from indexing.hybrid import rrf_fuse_many

    fused = rrf_fuse_many([("q0", [hit("x"), hit("a")]), ("q1", [hit("a")])])
    by_id = {f["chunk_id"]: f for f in fused}

    assert by_id["a"]["q0_rank"] == 2 and by_id["a"]["q1_rank"] == 1
    assert by_id["x"]["q0_rank"] == 1 and by_id["x"]["q1_rank"] is None


def test_fusing_many_lists_keeps_fields_the_inner_retriever_already_set():
    from indexing.hybrid import rrf_fuse_many

    # Chunks coming out of HybridRetriever already carry dense_rank/bm25_rank.
    # A second fusion pass on top must not blank them out.
    inner = hit("a", similarity=0.7, bm25_score=3.0, dense_rank=2, bm25_rank=1)

    fused = rrf_fuse_many([("q0", [inner])])

    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert fused[0]["similarity"] == 0.7
