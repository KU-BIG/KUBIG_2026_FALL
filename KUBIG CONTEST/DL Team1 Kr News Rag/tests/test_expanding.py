import pytest

from indexing.expanding import ExpandingRetriever
from rag import RAGPipeline
from tests.test_hybrid import FakeRetriever, hit


class FakeExpander:
    def __init__(self, queries):
        self.queries = queries
        self.calls = []

    def expand(self, question):
        self.calls.append(question)
        return self.queries


class PerQueryRetriever:
    """Returns a different ranked list per query, so fusion has work to do."""

    def __init__(self, by_query):
        self.by_query = by_query
        self.calls = []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.by_query.get(question, [])[:top_k]


def test_every_expanded_query_is_searched():
    base = FakeRetriever([hit("a")])
    expander = FakeExpander(["원본", "재작성1", "재작성2"])

    ExpandingRetriever(base=base, expander=expander, candidate_k=20).retrieve("원본")

    assert [q for q, _ in base.calls] == ["원본", "재작성1", "재작성2"]
    assert {k for _, k in base.calls} == {20}


def test_results_from_different_queries_are_fused_by_chunk_id():
    base = PerQueryRetriever({
        "원본": [hit("shared"), hit("only_original")],
        "재작성": [hit("shared"), hit("only_rewrite")],
    })
    retriever = ExpandingRetriever(base=base, expander=FakeExpander(["원본", "재작성"]))

    results = retriever.retrieve("원본", top_k=5)

    ids = [r["chunk_id"] for r in results]
    assert ids.count("shared") == 1
    # Found by both phrasings, so it should outrank chunks only one query saw.
    assert ids[0] == "shared"
    assert set(ids) == {"shared", "only_original", "only_rewrite"}


def test_results_record_which_queries_found_them():
    base = PerQueryRetriever({
        "원본": [hit("shared")],
        "재작성": [hit("shared"), hit("extra")],
    })
    retriever = ExpandingRetriever(base=base, expander=FakeExpander(["원본", "재작성"]))

    by_id = {r["chunk_id"]: r for r in retriever.retrieve("원본")}

    assert by_id["shared"]["matched_queries"] == [0, 1]
    assert by_id["extra"]["matched_queries"] == [1]


def test_results_carry_the_queries_that_were_searched():
    # The UI shows what the expander generated — a HyDE passage is worth seeing.
    base = FakeRetriever([hit("a")])
    retriever = ExpandingRetriever(base=base, expander=FakeExpander(["원본", "가상 본문"]))

    result = retriever.retrieve("원본")[0]

    assert result["expanded_queries"] == ["원본", "가상 본문"]


def test_fields_the_base_retriever_set_survive_the_second_fusion():
    base = FakeRetriever([hit("a", similarity=0.7, dense_rank=3, bm25_rank=1)])
    retriever = ExpandingRetriever(base=base, expander=FakeExpander(["원본"]))

    result = retriever.retrieve("원본")[0]

    assert result["dense_rank"] == 3
    assert result["bm25_rank"] == 1
    assert result["similarity"] == 0.7


def test_retrieve_caps_results_at_top_k():
    base = FakeRetriever([hit(c) for c in "abcdef"])
    retriever = ExpandingRetriever(base=base, expander=FakeExpander(["원본"]))

    assert len(retriever.retrieve("원본", top_k=2)) == 2


def test_blank_question_is_rejected():
    retriever = ExpandingRetriever(
        base=FakeRetriever([]), expander=FakeExpander(["원본"])
    )

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_the_hypothetical_passage_never_reaches_the_answer_prompt():
    # HyDE invents a passage to embed. If it leaked into the prompt, Claude would
    # be answering from fabricated text presented as source material.
    hypothetical = "삼성전자는 영업이익 20조원을 기록했다고 밝혔다."
    base = FakeRetriever([hit("a", similarity=0.8)])
    retriever = ExpandingRetriever(
        base=base, expander=FakeExpander(["삼성전자 실적", hypothetical])
    )

    class RecordingLLM:
        def __init__(self):
            self.prompts = []

        def generate(self, system_prompt, user_prompt, history=None):
            self.prompts.append(user_prompt)
            return "답변 [뉴스1]"

    llm = RecordingLLM()
    RAGPipeline(retriever=retriever, llm=llm).ask("삼성전자 실적")

    assert hypothetical not in llm.prompts[0]
    assert "삼성전자 실적" in llm.prompts[0]


def test_expanding_retriever_drops_into_the_rag_pipeline_unchanged():
    base = FakeRetriever([hit("a", similarity=0.8)])
    retriever = ExpandingRetriever(base=base, expander=FakeExpander(["원본"]))

    class FakeLLM:
        def generate(self, system_prompt, user_prompt, history=None):
            return "답변 [뉴스1]"

    response = RAGPipeline(retriever=retriever, llm=FakeLLM()).ask("원본 질문")

    assert response.answer == "답변 [뉴스1]"
    assert response.sources[0]["url"] == "https://example.com/a"
