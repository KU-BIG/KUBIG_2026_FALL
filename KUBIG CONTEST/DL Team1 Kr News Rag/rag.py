
"""End-to-end dense-retrieval + Claude generation pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from generation.llm import get_llm
from generation.prompt import build_chat_messages, build_messages
from indexing.query_preprocess import normalize_query

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    sources: list[dict]
    # False when the search gate routed this turn to the conversation instead of
    # the news index, so the UI can say why there are no sources.
    searched: bool = True

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "searched": self.searched,
        }


NO_CONTEXT_ANSWER = "제공된 자료에서 확인할 수 없습니다."


@dataclass(frozen=True)
class StreamingRAGResponse:
    """Sources up front, answer text as it is written.

    Retrieval has already finished by the time this is returned, so a UI can show
    what the answer will cite while the text is still arriving.
    """

    question: str
    sources: list[dict]
    searched: bool
    stream: Iterator[str] = field(repr=False)


class RAGPipeline:
    """Question -> retrieval -> grounded Claude answer + retrieved sources."""

    def __init__(
        self,
        retriever: Any | None = None,
        llm: Any | None = None,
        gate: Any | None = None,
    ) -> None:
        if retriever is None:
            from indexing.retriever import get_retriever
            retriever = get_retriever()
        self.retriever = retriever
        self.llm = llm or get_llm()
        # Optional. Without it every turn searches, which is the single-question
        # behaviour; with it, small talk and follow-ups skip retrieval.
        self.gate = gate

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        history: list[dict] | None = None,
    ) -> RAGResponse:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if self.gate is not None and not self.gate.needs_search(question, history):
            system_prompt, user_prompt = build_chat_messages(question)
            return RAGResponse(
                question=question,
                answer=self.llm.generate(system_prompt, user_prompt, history=history),
                sources=[],
                searched=False,
            )

        question = normalize_query(question)
        results = self.retriever.retrieve(question, top_k=top_k)
        if not results:
            return RAGResponse(
                question=question,
                answer=NO_CONTEXT_ANSWER,
                sources=[],
            )

        # 프롬프트는 청크 단위(results) 그대로 [뉴스1]..[뉴스N]으로 번호를 매겨
        # Claude에게 전달한다. 답변 안의 인용 번호가 이 번호를 그대로 가리키므로,
        # 여기서 청크 순서나 개수를 바꾸면 인용과 출처가 어긋난다.
        system_prompt, user_prompt = build_messages(question, results)
        answer = self.llm.generate(system_prompt, user_prompt, history=history)

        raw_sources = [
            self._source_from_result(rank, result)
            for rank, result in enumerate(results, 1)
        ]
        sources = self._dedupe_sources_by_article(raw_sources)

        return RAGResponse(question=question, answer=answer, sources=sources)

    def ask_stream(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        history: list[dict] | None = None,
    ) -> StreamingRAGResponse:
        """Same as `ask`, but the answer text arrives incrementally."""
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if self.gate is not None and not self.gate.needs_search(question, history):
            system_prompt, user_prompt = build_chat_messages(question)
            return StreamingRAGResponse(
                question=question,
                sources=[],
                searched=False,
                stream=self.llm.stream(system_prompt, user_prompt, history=history),
            )

        question = normalize_query(question)
        results = self.retriever.retrieve(question, top_k=top_k)
        if not results:
            return StreamingRAGResponse(
                question=question,
                sources=[],
                searched=True,
                stream=iter([NO_CONTEXT_ANSWER]),
            )

        system_prompt, user_prompt = build_messages(question, results)
        sources = self._dedupe_sources_by_article(
            [self._source_from_result(rank, r) for rank, r in enumerate(results, 1)]
        )
        return StreamingRAGResponse(
            question=question,
            sources=sources,
            searched=True,
            stream=self.llm.stream(system_prompt, user_prompt, history=history),
        )

    @staticmethod
    def _source_from_result(rank: int, result: dict) -> dict:
        return {
            "news_number": rank,
            "article_id": result.get("article_id"),
            "title": result.get("title", ""),
            "date": result.get("date", ""),
            "url": result.get("url", ""),
            "stock_names": result.get("stock_names", []),
            "similarity": result.get("similarity"),
            # The UI shows the passage the answer was grounded in, and — for
            # hybrid retrieval — which retriever surfaced it. Absent on
            # dense-only results, so these stay None rather than missing.
            "content": result.get("content", ""),
            "doc_type": result.get("doc_type"),
            "rrf_score": result.get("rrf_score"),
            "dense_rank": result.get("dense_rank"),
            "bm25_rank": result.get("bm25_rank"),
            "bm25_score": result.get("bm25_score"),
            # Query expansion: which phrasings were searched, and which of them
            # surfaced this chunk. None when expansion is off.
            "expanded_queries": result.get("expanded_queries"),
            "matched_queries": result.get("matched_queries"),
        }

    @staticmethod
    def _dedupe_sources_by_article(sources: list[dict]) -> list[dict]:
        """Collapse sources that come from the same article into one entry.

        Chunking overlaps the last sentence of an article across adjacent
        chunks, so top-k retrieval can return two chunks of the same article
        (e.g. news_number 1 and 3 both belong to article_id 135). The answer
        text still cites the original chunk numbers as [뉴스1], [뉴스3], so we
        don't drop the later duplicate outright -- we merge it into the first
        (higher-similarity) occurrence and keep every news_number that pointed
        to that article under "cited_as", so downstream UIs can show one card
        per article without breaking the citation trail back to the answer.
        """
        merged: dict[Any, dict] = {}
        order: list[Any] = []
        for source in sources:
            key = source.get("article_id")
            if key is None:
                # article_id가 없는 경우(예: 테스트용 mock 데이터)는
                # 병합 대상이 아니라고 보고 각각 별개로 취급한다.
                key = ("no_article_id", source["news_number"])
            if key not in merged:
                merged[key] = {**source, "cited_as": [source["news_number"]]}
                order.append(key)
            else:
                merged[key]["cited_as"].append(source["news_number"])
        return [merged[key] for key in order]


def ask(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Convenience function for one-shot use from another script."""
    return RAGPipeline().ask(question, top_k=top_k).to_dict()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Korean financial-news question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    response = ask(args.question, top_k=args.top_k)
    print(json.dumps(response, ensure_ascii=False, indent=2))
