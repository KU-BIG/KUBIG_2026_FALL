"""Decide whether a turn needs the news index at all.

Without this, every message runs retrieval — "고마워" embeds, queries Chroma, and
stuffs five unrelated articles into the prompt. In a multi-turn conversation that
gets worse: "더 쉽게 설명해줘" refers to the previous answer, not to anything in
the corpus.

A similarity threshold was tried first and does not separate the two classes on
this corpus: "오늘 서울 날씨 어때" scores 0.558, higher than four legitimate
questions, while the ticker query "005930" scores 0.428 — the lowest of any real
question, because embeddings do not understand stock codes. No scalar cut
survives both, so the decision is made by the model instead.

Ambiguity resolves towards searching: answering a factual question with no
sources is a worse failure than retrieving when it was not needed.
"""

from __future__ import annotations

from typing import Protocol

SYSTEM_PROMPT = """당신은 한국 금융 뉴스 QA 시스템의 라우터입니다.

사용자의 마지막 메시지에 답하기 위해 뉴스 검색이 필요한지 판단하세요.

RETRIEVE — 뉴스 자료를 찾아봐야 답할 수 있는 경우
- 종목·시황·실적·업황에 대한 질문
- 새로운 사실이나 수치를 묻는 경우
- 이전 답변에서 다루지 않은 다른 종목이나 주제로 넘어가는 경우

CHAT — 검색 없이 대화 맥락만으로 답할 수 있는 경우
- 인사, 감사, 잡담
- 직전 답변을 요약·재설명·번역·서식 변경해달라는 요청
- 시스템 자체에 대한 질문
- 금융 뉴스와 무관한 요청

RETRIEVE 또는 CHAT 중 한 단어만 출력하세요. 다른 말은 쓰지 마세요.
"""

MAX_HISTORY_TURNS = 6


class LLM(Protocol):
    def generate(
        self, system_prompt: str, user_prompt: str, history: list[dict] | None = ...
    ) -> str: ...


class SearchGate:
    """Question (+ conversation) -> whether to run retrieval this turn."""

    def __init__(self, llm: LLM | None = None, model: str | None = None) -> None:
        self._llm = llm
        self._model = model

    def _ensure_llm(self) -> LLM:
        if self._llm is None:
            from generation.llm import get_llm

            self._llm = get_llm(model=self._model)
        return self._llm

    def needs_search(self, question: str, history: list[dict] | None = None) -> bool:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        try:
            reply = self._ensure_llm().generate(
                SYSTEM_PROMPT, _build_prompt(question.strip(), history)
            )
        except Exception:
            return True  # routing failed — retrieve rather than answer unsourced

        decision = reply.strip().upper()
        if decision.startswith("CHAT"):
            return False
        if decision.startswith("RETRIEVE"):
            return True
        return True


def _build_prompt(question: str, history: list[dict] | None) -> str:
    """Render the recent conversation plus the new message for the router."""
    lines = []
    if history:
        lines.append("[지금까지의 대화]")
        for turn in history[-MAX_HISTORY_TURNS:]:
            speaker = "사용자" if turn.get("role") == "user" else "assistant"
            lines.append(f"{speaker}: {turn.get('content', '')}")
        lines.append("")
    lines.append(f"[사용자의 마지막 메시지]\n{question}")
    lines.append("\nRETRIEVE 또는 CHAT:")
    return "\n".join(lines)
