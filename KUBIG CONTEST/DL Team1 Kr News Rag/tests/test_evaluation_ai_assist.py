import json

import pytest

from evaluation.ai_assist import generate_question, judge_candidates


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, system_prompt, user_prompt, history=None):
        self.calls.append((system_prompt, user_prompt))
        return self.response


def article(article_id=7):
    return {"article_id": article_id, "title": "반도체 투자 확대", "content": "기업이 HBM 투자를 늘렸다.", "date": "2026.08.01", "url": "u", "doc_type": "article"}


def test_question_generation_returns_draft_with_initial_gold_and_unchecked_self_check():
    llm = FakeLLM(json.dumps({"question": "HBM 투자가 늘어난 배경은 무엇인가?", "rationale": "기사로 답할 수 있음"}, ensure_ascii=False))
    result = generate_question(article(), "abstract", llm)

    assert result["question"] == "HBM 투자가 늘어난 배경은 무엇인가?"
    assert result["review_status"] == "draft"
    assert result["seed_article_id"] == 7
    assert result["gold_article_ids"] == [7]
    assert not any(result["self_check"].values())


def test_blind_judgment_never_sends_system_names_or_original_ranks_and_preserves_question():
    question = "HBM 투자 확대 이유는?"
    candidates = [{"candidate_id": "candidate_001", **article(8)}]
    llm = FakeLLM(json.dumps({"judgments": [{"article_id": 8, "relevance": "relevant", "support": "투자 배경 설명"}]}, ensure_ascii=False))

    result = judge_candidates(question, candidates, llm)

    assert result["question"] == question
    assert result["evidence"][0]["relevance"] == "relevant"
    sent = " ".join(llm.calls[0])
    assert "dense_rank" not in sent and "hybrid_rank" not in sent and "system" not in sent.lower()


@pytest.mark.parametrize("label", ["relevant", "not_relevant", "uncertain"])
def test_judgment_accepts_only_supported_relevance_labels(label):
    llm = FakeLLM(json.dumps({"judgments": [{"article_id": 8, "relevance": label, "support": "근거"}]}))
    assert judge_candidates("고정 질문", [{"candidate_id": "candidate_001", **article(8)}], llm)["evidence"][0]["relevance"] == label


def test_judgment_rejects_unknown_candidates():
    llm = FakeLLM(json.dumps({"judgments": [{"article_id": 999, "relevance": "relevant", "support": "근거"}]}))
    with pytest.raises(ValueError, match="candidate"):
        judge_candidates("고정 질문", [{"candidate_id": "candidate_001", **article(8)}], llm)
