from types import SimpleNamespace

import pytest

from generation.llm import ClaudeClient, LLMError
from generation.prompt import SYSTEM_PROMPT, build_messages, build_user_prompt
from rag import RAGPipeline


RESULTS = [
    {
        "chunk_id": "news-1",
        "title": "삼성전자, 반도체 실적 개선 기대",
        "date": "2026.08.08",
        "url": "https://example.com/1",
        "stock_names": ["삼성전자"],
        "content": "삼성전자의 반도체 사업 실적 개선이 기대된다는 분석이 나왔다.",
        "similarity": 0.91,
    },
    {
        "chunk_id": "news-2",
        "title": "반도체 업황 회복 전망",
        "date": "2026.08.07",
        "url": "https://example.com/2",
        "stock_names": ["삼성전자", "SK하이닉스"],
        "content": "메모리 업황이 회복될 가능성이 있다는 전망이 제기됐다.",
        "similarity": 0.84,
    },
]


def test_prompt_contains_question_and_retrieved_news():
    prompt = build_user_prompt("삼성전자 반도체 실적 전망은?", RESULTS)
    assert "삼성전자 반도체 실적 전망은?" in prompt
    assert "[뉴스1]" in prompt
    assert "[뉴스2]" in prompt
    assert RESULTS[0]["content"] in prompt
    assert RESULTS[0]["url"] in prompt


def test_prompt_requires_grounded_answer():
    system, user = build_messages("질문", RESULTS)
    assert "제공된 자료에서 확인할 수 없습니다." in system
    assert "뉴스 자료에 직접 근거가 있는 내용만" in user


def test_prompt_rejects_empty_question_or_results():
    with pytest.raises(ValueError):
        build_user_prompt("   ", RESULTS)
    with pytest.raises(ValueError):
        build_user_prompt("질문", [])


class FakeMessages:
    def __init__(self, answer="테스트 답변 [뉴스1]"):
        self.answer = answer
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.answer)]
        )


class FakeClient:
    def __init__(self, answer="테스트 답변 [뉴스1]"):
        self.messages = FakeMessages(answer)


def test_claude_client_sends_expected_messages():
    fake = FakeClient()
    client = ClaudeClient(client=fake, model="test-model", max_tokens=123)

    answer = client.generate("system", "user")

    assert answer == "테스트 답변 [뉴스1]"
    call = fake.messages.calls[0]
    assert call["model"] == "test-model"
    assert call["max_tokens"] == 123
    assert call["system"] == "system"
    assert call["messages"] == [{"role": "user", "content": "user"}]


def test_claude_client_optionally_requests_structured_output_and_metadata():
    fake = FakeClient('{"ok":true}')
    fake.messages.create = lambda **kwargs: SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"ok":true}')],
        model="claude-actual",
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
        _request=kwargs,
    )
    calls = []
    original = fake.messages.create
    fake.messages.create = lambda **kwargs: (calls.append(kwargs), original(**kwargs))[1]

    result = ClaudeClient(client=fake, model="configured").generate(
        "system", "user", temperature=0,
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        return_metadata=True,
    )

    assert result.text == '{"ok":true}'
    assert result.model == "claude-actual"
    assert result.input_tokens == 12 and result.output_tokens == 3
    assert calls[0]["temperature"] == 0
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_claude_client_raises_on_empty_response():
    fake = FakeClient(answer="")
    client = ClaudeClient(client=fake)
    with pytest.raises(LLMError, match="empty"):
        client.generate("system", "user")


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.results


class FakeLLM:
    def __init__(self, answer="삼성전자의 반도체 실적 개선이 기대됩니다. [뉴스1]"):
        self.answer = answer
        self.calls = []

    def generate(self, system_prompt, user_prompt, history=None):
        self.calls.append((system_prompt, user_prompt, history))
        return self.answer

    def stream(self, system_prompt, user_prompt, history=None):
        self.calls.append((system_prompt, user_prompt, history))
        for word in self.answer.split(" "):
            yield word + " "


def test_rag_pipeline_connects_retrieval_prompt_and_generation():
    retriever = FakeRetriever(RESULTS)
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    response = pipeline.ask("삼전 반도체 실적 전망", top_k=5)

    assert response.answer.startswith("삼성전자의")
    assert retriever.calls == [("삼성전자 반도체 실적 전망", 5)]
    assert len(llm.calls) == 1
    assert "삼성전자 반도체 실적 전망" in llm.calls[0][1]
    assert response.sources[0]["news_number"] == 1
    assert response.sources[0]["title"] == RESULTS[0]["title"]
    assert response.sources[0]["url"] == RESULTS[0]["url"]


def test_rag_pipeline_returns_no_answer_when_retrieval_is_empty():
    retriever = FakeRetriever([])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    response = pipeline.ask("삼성전자 전망")

    assert response.answer == "제공된 자료에서 확인할 수 없습니다."
    assert response.sources == []
    assert llm.calls == []


def test_sources_carry_the_chunk_text_and_its_retrieval_provenance():
    # The UI shows each source card with the passage the answer was grounded in
    # and why that chunk surfaced, so the citation payload has to carry both.
    results = [
        {**RESULTS[0], "rrf_score": 0.032, "dense_rank": 2, "bm25_rank": 1, "bm25_score": 7.5}
    ]
    pipeline = RAGPipeline(retriever=FakeRetriever(results), llm=FakeLLM())

    source = pipeline.ask("삼성전자 실적").sources[0]

    assert source["content"] == RESULTS[0]["content"]
    assert source["similarity"] == RESULTS[0]["similarity"]
    assert source["rrf_score"] == 0.032
    assert source["dense_rank"] == 2
    assert source["bm25_rank"] == 1
    assert source["bm25_score"] == 7.5


def test_dense_only_sources_omit_the_fusion_fields():
    pipeline = RAGPipeline(retriever=FakeRetriever(RESULTS), llm=FakeLLM())

    source = pipeline.ask("삼성전자 실적").sources[0]

    assert source["rrf_score"] is None
    assert source["dense_rank"] is None


def test_sources_carry_the_expanded_queries_that_found_them():
    results = [{**RESULTS[0], "expanded_queries": ["원본", "가상 본문"], "matched_queries": [0, 1]}]
    pipeline = RAGPipeline(retriever=FakeRetriever(results), llm=FakeLLM())

    source = pipeline.ask("삼성전자 실적").sources[0]

    assert source["expanded_queries"] == ["원본", "가상 본문"]
    assert source["matched_queries"] == [0, 1]


# --- multi-turn -----------------------------------------------------------


def test_claude_client_sends_prior_turns_before_the_current_question():
    fake = FakeClient()
    client = ClaudeClient(client=fake)
    history = [
        {"role": "user", "content": "삼성전자 실적은?"},
        {"role": "assistant", "content": "개선 전망입니다 [뉴스1]"},
    ]

    client.generate("system", "그럼 SK하이닉스는?", history=history)

    assert fake.messages.calls[0]["messages"] == [
        {"role": "user", "content": "삼성전자 실적은?"},
        {"role": "assistant", "content": "개선 전망입니다 [뉴스1]"},
        {"role": "user", "content": "그럼 SK하이닉스는?"},
    ]


def test_claude_client_without_history_sends_a_single_turn():
    fake = FakeClient()

    ClaudeClient(client=fake).generate("system", "질문")

    assert fake.messages.calls[0]["messages"] == [{"role": "user", "content": "질문"}]


def test_chat_prompt_answers_from_the_conversation_without_news_material():
    from generation.prompt import build_chat_messages

    system, user = build_chat_messages("더 쉽게 설명해줘")

    assert "뉴스 자료" not in user
    assert user.strip() == "더 쉽게 설명해줘"
    # No sources this turn, so the model must not invent facts to fill the gap.
    assert "지어내" in system or "추측" in system


def test_chat_prompt_rejects_a_blank_question():
    from generation.prompt import build_chat_messages

    with pytest.raises(ValueError):
        build_chat_messages("   ")


def test_grounded_prompt_warns_that_old_citation_numbers_do_not_carry_over():
    # In a multi-turn chat the previous answer's [뉴스2] referred to a different
    # set of chunks; the model must renumber against this turn's material.
    system, _ = build_messages("질문", RESULTS)

    assert "이전" in system


# --- search gate ----------------------------------------------------------


class FakeGate:
    def __init__(self, decision=True):
        self.decision = decision
        self.calls = []

    def needs_search(self, question, history=None):
        self.calls.append((question, history))
        return self.decision


def test_a_turn_the_gate_sends_to_chat_never_touches_the_retriever():
    retriever, llm = FakeRetriever(RESULTS), FakeLLM("천만에요")
    pipeline = RAGPipeline(retriever=retriever, llm=llm, gate=FakeGate(False))

    response = pipeline.ask("고마워")

    assert retriever.calls == []
    assert response.searched is False
    assert response.sources == []
    assert response.answer == "천만에요"


def test_a_chat_turn_is_answered_without_news_material_in_the_prompt():
    llm = FakeLLM("천만에요")
    pipeline = RAGPipeline(retriever=FakeRetriever(RESULTS), llm=llm, gate=FakeGate(False))

    pipeline.ask("고마워")

    _, user_prompt, _ = llm.calls[0]
    assert "[뉴스1]" not in user_prompt


def test_a_turn_the_gate_sends_to_search_runs_retrieval():
    retriever = FakeRetriever(RESULTS)
    pipeline = RAGPipeline(retriever=retriever, llm=FakeLLM(), gate=FakeGate(True))

    response = pipeline.ask("삼성전자 실적")

    assert retriever.calls
    assert response.searched is True
    assert response.sources


def test_without_a_gate_every_turn_still_searches():
    retriever = FakeRetriever(RESULTS)
    pipeline = RAGPipeline(retriever=retriever, llm=FakeLLM())

    assert pipeline.ask("고마워").searched is True
    assert retriever.calls


def test_the_gate_sees_the_conversation_so_far():
    gate = FakeGate(False)
    history = [{"role": "user", "content": "삼성전자 실적은?"}]
    pipeline = RAGPipeline(retriever=FakeRetriever(RESULTS), llm=FakeLLM(), gate=gate)

    pipeline.ask("더 쉽게 설명해줘", history=history)

    assert gate.calls == [("더 쉽게 설명해줘", history)]


def test_prior_turns_are_passed_to_the_answering_model():
    llm = FakeLLM()
    history = [
        {"role": "user", "content": "삼성전자 실적은?"},
        {"role": "assistant", "content": "개선 전망입니다"},
    ]
    pipeline = RAGPipeline(retriever=FakeRetriever(RESULTS), llm=llm)

    pipeline.ask("그럼 SK하이닉스는?", history=history)

    assert llm.calls[0][2] == history


def test_the_response_dict_reports_whether_the_turn_searched():
    pipeline = RAGPipeline(retriever=FakeRetriever(RESULTS), llm=FakeLLM(), gate=FakeGate(False))

    assert pipeline.ask("고마워").to_dict()["searched"] is False


# --- streaming ------------------------------------------------------------


class FakeStreamContext:
    def __init__(self, parts):
        self.parts = parts
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    @property
    def text_stream(self):
        yield from self.parts


class FakeStreamingMessages:
    def __init__(self, parts):
        self.parts = parts
        self.calls = []
        self.context = None

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        self.context = FakeStreamContext(self.parts)
        return self.context


class FakeStreamingClient:
    def __init__(self, parts):
        self.messages = FakeStreamingMessages(parts)


def test_streaming_yields_the_answer_in_pieces():
    fake = FakeStreamingClient(["삼성전자는 ", "반도체 실적이 ", "개선될 전망입니다 [뉴스1]"])
    client = ClaudeClient(client=fake)

    assert list(client.stream("system", "질문")) == [
        "삼성전자는 ",
        "반도체 실적이 ",
        "개선될 전망입니다 [뉴스1]",
    ]


def test_streaming_sends_the_same_request_shape_as_a_plain_call():
    fake = FakeStreamingClient(["답변"])
    history = [{"role": "user", "content": "이전 질문"}]

    list(ClaudeClient(client=fake, model="test-model").stream("system", "질문", history=history))

    call = fake.messages.calls[0]
    assert call["model"] == "test-model"
    assert call["system"] == "system"
    assert call["messages"] == [
        {"role": "user", "content": "이전 질문"},
        {"role": "user", "content": "질문"},
    ]


def test_the_stream_is_closed_once_it_is_consumed():
    fake = FakeStreamingClient(["답변"])

    list(ClaudeClient(client=fake).stream("system", "질문"))

    assert fake.messages.context.closed is True


def test_a_stream_that_produced_nothing_is_an_error():
    fake = FakeStreamingClient([])

    with pytest.raises(LLMError, match="empty"):
        list(ClaudeClient(client=fake).stream("system", "질문"))


def test_streaming_rejects_empty_prompts():
    with pytest.raises(ValueError):
        list(ClaudeClient(client=FakeStreamingClient(["x"])).stream("", "질문"))


def test_streaming_ask_exposes_sources_before_the_answer_is_written():
    # Retrieval finishes first, so the UI can show what it is about to cite while
    # the text is still arriving.
    pipeline = RAGPipeline(retriever=FakeRetriever(RESULTS), llm=FakeLLM())

    response = pipeline.ask_stream("삼성전자 실적")

    assert response.sources[0]["title"] == RESULTS[0]["title"]
    assert response.searched is True


def test_streaming_ask_yields_the_answer_in_pieces():
    pipeline = RAGPipeline(retriever=FakeRetriever(RESULTS), llm=FakeLLM("가 나 다"))

    pieces = list(pipeline.ask_stream("삼성전자 실적").stream)

    assert len(pieces) > 1
    assert "".join(pieces).strip() == "가 나 다"


def test_a_gated_chat_turn_streams_without_retrieving():
    retriever = FakeRetriever(RESULTS)
    pipeline = RAGPipeline(retriever=retriever, llm=FakeLLM("천만에요"), gate=FakeGate(False))

    response = pipeline.ask_stream("고마워")
    text = "".join(response.stream)

    assert retriever.calls == []
    assert response.searched is False
    assert response.sources == []
    assert text.strip() == "천만에요"


def test_streaming_ask_passes_prior_turns_to_the_model():
    llm = FakeLLM()
    history = [{"role": "user", "content": "이전 질문"}]
    pipeline = RAGPipeline(retriever=FakeRetriever(RESULTS), llm=llm)

    list(pipeline.ask_stream("후속 질문", history=history).stream)

    assert llm.calls[0][2] == history


def test_streaming_ask_reports_an_empty_index_without_calling_the_model():
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=FakeRetriever([]), llm=llm)

    response = pipeline.ask_stream("삼성전자 실적")
    text = "".join(response.stream)

    assert text == "제공된 자료에서 확인할 수 없습니다."
    assert llm.calls == []
