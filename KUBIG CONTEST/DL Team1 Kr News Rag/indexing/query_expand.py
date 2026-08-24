"""LLM query expansion: one question in, several search queries out.

Two strategies, both aimed at the same weakness — the user's phrasing is only
one of many ways the answer could be worded in a news article.

- **Multi-Query** rephrases the question a few different ways.
- **HyDE** writes a short hypothetical passage that *would* answer the question
  and searches with that. A question and a news article are written very
  differently; embedding something article-shaped lands closer to real articles.

Both return the original question first. A rephrasing can drift off-topic and a
hypothetical passage is invented outright, so fusion always has one query that is
definitely on target to fall back on. For the same reason, expansion never raises
on an LLM failure — it degrades to the original question and retrieval carries on.

> ⚠️ The HyDE passage is **fabricated** and exists only to be embedded. It must
> never reach the answer prompt or the user. `ExpandingRetriever` passes expanded
> queries to the retriever and nothing else.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

DEFAULT_QUERY_COUNT = 3

# Leading "1.", "1)", "-", "*", "•" the model adds when it formats a list.
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")

MULTI_QUERY_SYSTEM = """당신은 한국 금융 뉴스 검색 시스템의 질의 확장기입니다.

사용자 질문을 검색엔진이 찾기 좋은 다른 표현으로 바꿔 쓰세요.
- 질문의 의도는 유지하되, 뉴스 기사에서 쓰일 법한 어휘로 바꾸세요.
- 종목명은 그대로 두세요. 다른 종목으로 바꾸지 마세요.
- 한 줄에 하나씩, 검색어만 출력하세요. 번호나 설명을 붙이지 마세요.
"""

HYDE_SYSTEM = """당신은 한국 금융 뉴스 검색 시스템의 질의 확장기입니다.

사용자 질문에 답이 될 만한 짧은 뉴스 기사 본문을 지어내세요.
- 실제 기사처럼 2~3문장으로, 평서문으로 쓰세요.
- 이 글은 검색용으로만 쓰이고 사용자에게 보여주지 않습니다. 사실 여부보다
  실제 기사와 비슷한 어휘와 문체를 쓰는 것이 중요합니다.
- 질문을 다시 쓰거나 "~에 대한 기사입니다" 같은 설명을 붙이지 마세요. 본문만 쓰세요.
"""


class LLM(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class _Expander:
    """Shared plumbing: lazy default LLM, blank-question guard, safe fallback."""

    def __init__(self, llm: LLM | None = None, model: str | None = None) -> None:
        self._llm = llm
        self._model = model

    def _ensure_llm(self) -> LLM:
        if self._llm is None:
            from generation.llm import get_llm

            self._llm = get_llm(model=self._model)
        return self._llm

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._ensure_llm().generate(system_prompt, user_prompt)
        except Exception:
            # Expansion is an optimisation, not a requirement. A failure here
            # must not take retrieval down with it.
            return ""

    @staticmethod
    def _require_question(question: str) -> str:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")
        return question.strip()

    def expand(self, question: str) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError


class MultiQueryExpander(_Expander):
    """Original question plus a few rephrasings of it."""

    def __init__(
        self,
        llm: LLM | None = None,
        n: int = DEFAULT_QUERY_COUNT,
        model: str | None = None,
    ) -> None:
        super().__init__(llm=llm, model=model)
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n

    def expand(self, question: str) -> list[str]:
        question = self._require_question(question)
        reply = self._generate(MULTI_QUERY_SYSTEM, f"질문: {question}\n\n검색어 {self.n}개:")

        queries = [question]
        for line in reply.splitlines():
            candidate = _LIST_MARKER.sub("", line).strip()
            if candidate and candidate not in queries:
                queries.append(candidate)
            if len(queries) > self.n:
                break
        return queries[: self.n + 1]


class HyDEExpander(_Expander):
    """Original question plus a hypothetical passage that would answer it."""

    def expand(self, question: str) -> list[str]:
        question = self._require_question(question)
        passage = self._generate(HYDE_SYSTEM, f"질문: {question}\n\n기사 본문:").strip()
        return [question, passage] if passage else [question]


def get_expander(name: str, **kwargs: Any) -> _Expander | None:
    """Build an expander by name; `None` means no expansion."""
    if name in (None, "", "none"):
        return None
    if name == "multi_query":
        return MultiQueryExpander(**kwargs)
    if name == "hyde":
        return HyDEExpander(**kwargs)
    raise ValueError(f"unknown expander: {name}")
