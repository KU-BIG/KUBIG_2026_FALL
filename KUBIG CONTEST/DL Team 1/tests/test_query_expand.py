import pytest

from indexing.query_expand import HyDEExpander, MultiQueryExpander


class FakeLLM:
    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return self.reply


# --- Multi-Query ----------------------------------------------------------


def test_multi_query_returns_the_original_question_first():
    llm = FakeLLM("삼성전자 메모리 사업 전망\n삼성전자 3분기 반도체 매출")

    queries = MultiQueryExpander(llm=llm).expand("삼성전자 반도체 실적 전망")

    # The original is always searched. A rephrasing can drift off-topic, and
    # fusion needs a query that is definitely on target to fall back on.
    assert queries[0] == "삼성전자 반도체 실적 전망"


def test_multi_query_adds_the_generated_rephrasings():
    llm = FakeLLM("삼성전자 메모리 사업 전망\n삼성전자 3분기 반도체 매출")

    queries = MultiQueryExpander(llm=llm).expand("삼성전자 반도체 실적 전망")

    assert "삼성전자 메모리 사업 전망" in queries
    assert "삼성전자 3분기 반도체 매출" in queries


def test_multi_query_strips_list_markers_the_model_adds():
    llm = FakeLLM("1. 첫 번째 질의\n- 두 번째 질의\n* 세 번째 질의")

    queries = MultiQueryExpander(llm=llm, n=3).expand("원본")

    assert queries[1:] == ["첫 번째 질의", "두 번째 질의", "세 번째 질의"]


def test_multi_query_drops_blank_lines_and_duplicates():
    llm = FakeLLM("질의 하나\n\n질의 하나\n   \n질의 둘")

    queries = MultiQueryExpander(llm=llm).expand("원본")

    assert queries == ["원본", "질의 하나", "질의 둘"]


def test_multi_query_caps_the_number_of_rephrasings():
    llm = FakeLLM("\n".join(f"질의{i}" for i in range(10)))

    queries = MultiQueryExpander(llm=llm, n=3).expand("원본")

    assert len(queries) == 4  # 원본 + 3


def test_multi_query_falls_back_to_the_original_when_the_llm_fails():
    # Expansion is an optimisation. If it breaks, retrieval must still run.
    llm = FakeLLM(error=RuntimeError("API down"))

    assert MultiQueryExpander(llm=llm).expand("원본 질문") == ["원본 질문"]


def test_multi_query_falls_back_when_the_llm_returns_nothing_usable():
    assert MultiQueryExpander(llm=FakeLLM("   ")).expand("원본") == ["원본"]


# --- HyDE -----------------------------------------------------------------


def test_hyde_searches_the_hypothetical_passage_alongside_the_question():
    llm = FakeLLM("삼성전자는 3분기 반도체 부문에서 영업이익이 개선됐다고 밝혔다.")

    queries = HyDEExpander(llm=llm).expand("삼성전자 반도체 실적 전망")

    # A question and a news article are written differently, so embedding a
    # passage that looks like the answer matches real articles more closely.
    assert queries[0] == "삼성전자 반도체 실적 전망"
    assert "영업이익이 개선됐다" in queries[1]


def test_hyde_asks_for_a_passage_not_an_answer():
    llm = FakeLLM("가상 본문")

    HyDEExpander(llm=llm).expand("삼성전자 실적")

    system_prompt, user_prompt = llm.calls[0]
    assert "삼성전자 실적" in user_prompt


def test_hyde_falls_back_to_the_original_when_the_llm_fails():
    llm = FakeLLM(error=RuntimeError("API down"))

    assert HyDEExpander(llm=llm).expand("원본 질문") == ["원본 질문"]


def test_hyde_falls_back_when_the_passage_is_empty():
    assert HyDEExpander(llm=FakeLLM("  ")).expand("원본") == ["원본"]


# --- shared ---------------------------------------------------------------


@pytest.mark.parametrize("expander_cls", [MultiQueryExpander, HyDEExpander])
def test_expanders_reject_a_blank_question(expander_cls):
    with pytest.raises(ValueError):
        expander_cls(llm=FakeLLM("x")).expand("   ")
