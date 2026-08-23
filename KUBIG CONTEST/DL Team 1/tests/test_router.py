import pytest

from generation.router import SearchGate


class FakeLLM:
    def __init__(self, reply="RETRIEVE", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def generate(self, system_prompt, user_prompt, history=None):
        self.calls.append((system_prompt, user_prompt, history))
        if self.error:
            raise self.error
        return self.reply


def test_a_question_about_the_news_is_routed_to_search():
    gate = SearchGate(llm=FakeLLM("RETRIEVE"))

    assert gate.needs_search("삼성전자 반도체 실적 전망은?") is True


def test_small_talk_skips_search():
    gate = SearchGate(llm=FakeLLM("CHAT"))

    assert gate.needs_search("고마워") is False


def test_the_decision_tolerates_surrounding_text():
    gate = SearchGate(llm=FakeLLM("  chat\n"))

    assert gate.needs_search("표로 정리해줘") is False


def test_the_conversation_so_far_is_given_to_the_router():
    # "더 쉽게 설명해줘" is only classifiable against what came before.
    llm = FakeLLM("CHAT")
    history = [
        {"role": "user", "content": "삼성전자 실적은?"},
        {"role": "assistant", "content": "개선 전망입니다"},
    ]

    gate = SearchGate(llm=llm)
    gate.needs_search("더 쉽게 설명해줘", history=history)

    _, user_prompt, _ = llm.calls[0]
    assert "삼성전자 실적은?" in user_prompt
    assert "더 쉽게 설명해줘" in user_prompt


def test_an_unreadable_decision_falls_back_to_searching():
    # Answering a factual question with no sources is worse than retrieving
    # when it was not needed, so ambiguity resolves towards search.
    gate = SearchGate(llm=FakeLLM("음... 글쎄요"))

    assert gate.needs_search("삼성전자 실적") is True


def test_an_llm_failure_falls_back_to_searching():
    gate = SearchGate(llm=FakeLLM(error=RuntimeError("API down")))

    assert gate.needs_search("삼성전자 실적") is True


def test_blank_question_is_rejected():
    with pytest.raises(ValueError):
        SearchGate(llm=FakeLLM()).needs_search("   ")


def test_the_router_asks_for_a_one_word_decision():
    llm = FakeLLM("RETRIEVE")

    SearchGate(llm=llm).needs_search("삼성전자 실적")

    system_prompt, _, _ = llm.calls[0]
    assert "RETRIEVE" in system_prompt and "CHAT" in system_prompt
