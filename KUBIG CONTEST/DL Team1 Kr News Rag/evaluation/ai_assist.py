"""Draft-only Claude helpers for source questions and blind relevance labels."""

from __future__ import annotations

import json

from evaluation.schema import CATEGORIES, RELEVANCE, SELF_CHECK_FIELDS


def _json_object(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("AI response must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("AI response must be a JSON object")
    return value


def generate_question(article: dict, category: str, llm) -> dict:
    if category not in CATEGORIES:
        raise ValueError("invalid category")
    article_id = article.get("article_id")
    if type(article_id) is not int:
        raise ValueError("article requires an integer article_id")
    instruction = (
        "Create one natural Korean retrieval-evaluation question answerable from the source. "
        "Do not copy the title or a sentence verbatim, and do not optimize for any retriever. "
        "Return only JSON with question and rationale."
    )
    source = json.dumps(
        {"category": category, "title": article.get("title", ""), "content": article.get("content", "")},
        ensure_ascii=False,
    )
    value = _json_object(llm.generate(instruction, source))
    question = value.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("AI question cannot be empty")
    return {
        "question": question.strip(),
        "category": category,
        "seed_article_id": article_id,
        "gold_article_ids": [article_id],
        "review_status": "draft",
        "review_mode": "ai_assisted_self_check",
        "self_check": {field: False for field in sorted(SELF_CHECK_FIELDS)},
        "ai_rationale": str(value.get("rationale", "")),
    }


def judge_candidates(question: str, candidates: list[dict], llm) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    visible_fields = ("candidate_id", "article_id", "title", "date", "content", "url", "doc_type")
    blind = [{field: candidate.get(field) for field in visible_fields} for candidate in candidates]
    instruction = (
        "Judge whether each anonymous article can sufficiently answer the frozen question. "
        "Do not rewrite the question. Use relevant, not_relevant, or uncertain. "
        "Return only JSON: {\"judgments\":[{\"article_id\":1,\"relevance\":\"relevant\",\"support\":\"...\"}]}"
    )
    payload = json.dumps({"frozen_question": question, "candidates": blind}, ensure_ascii=False)
    value = _json_object(llm.generate(instruction, payload))
    judgments = value.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("AI response requires a judgments list")
    by_id = {candidate.get("article_id"): candidate for candidate in blind}
    evidence = []
    for judgment in judgments:
        article_id = judgment.get("article_id")
        if article_id not in by_id:
            raise ValueError(f"judgment references an unknown candidate: {article_id}")
        relevance = judgment.get("relevance")
        support = judgment.get("support")
        if relevance not in RELEVANCE:
            raise ValueError("invalid relevance label")
        if not isinstance(support, str) or not support.strip():
            raise ValueError("judgment support cannot be empty")
        evidence.append({
            "article_id": article_id,
            "title": by_id[article_id].get("title", ""),
            "support": support.strip(),
            "relevance": relevance,
        })
    return {"question": question, "evidence": evidence}
