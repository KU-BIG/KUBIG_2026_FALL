"""Load and validate human-authored retrieval evaluation records."""

from __future__ import annotations

import json
from pathlib import Path

METHODS = {"source_seeded", "query_first"}
CATEGORIES = {"exact_token", "abstract", "multi_aspect", "factoid"}
LEGACY_QUERY_FIRST_CATEGORIES = {"exact_token", "entity", "semantic_paraphrase", "multi_entity", "temporal_numeric"}
STATUSES = {"draft", "approved"}
RELEVANCE = {"relevant", "not_relevant", "uncertain"}
REVIEW_MODES = {"ai_assisted_self_check"}
DATE_STRATA = {"2026-07-31_to_2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08"}
SELF_CHECK_FIELDS = {
    "answer_supported_by_source", "natural_question", "not_title_copy",
    "not_duplicate", "source_article_id_verified",
}
REQUIRED_FIELDS = {
    "query_id", "question", "construction_method", "category", "seed_article_id",
    "gold_article_ids", "evidence", "author", "reviewer", "review_status",
    "annotation_minutes", "naturalness", "gold_clarity", "notes",
}
SOURCE_SEEDED_FIELDS = {"date_stratum", "review_mode", "self_check"}


def load_records(path: Path | str) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(value)
    return records


def validate_records(records: list[dict], article_ids: set[int], *, allow_draft: bool) -> None:
    seen = set()
    seen_questions = set()
    for index, record in enumerate(records, 1):
        method = record.get("construction_method")
        required = REQUIRED_FIELDS | (SOURCE_SEEDED_FIELDS if method == "source_seeded" else set())
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record {index}: missing fields: {', '.join(sorted(missing))}")
        query_id = record["query_id"]
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen:
            raise ValueError(f"record {index}: query_id must be non-empty and unique")
        seen.add(query_id)
        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError(f"{query_id}: question cannot be empty")
        normalized_question = " ".join(record["question"].split()).casefold()
        if normalized_question in seen_questions:
            raise ValueError(f"{query_id}: duplicate question")
        seen_questions.add(normalized_question)
        if method not in METHODS:
            raise ValueError(f"{query_id}: invalid construction_method")
        allowed_categories = CATEGORIES if method == "source_seeded" else LEGACY_QUERY_FIRST_CATEGORIES
        if record["category"] not in allowed_categories:
            raise ValueError(f"{query_id}: invalid category")
        if method == "source_seeded" and record["date_stratum"] not in DATE_STRATA:
            raise ValueError(f"{query_id}: invalid date_stratum")
        seed_id = record["seed_article_id"]
        if method == "source_seeded" and seed_id is None:
            raise ValueError(f"{query_id}: source_seeded requires seed_article_id")
        if seed_id is not None and seed_id not in article_ids:
            raise ValueError(f"{query_id}: seed_article_id is not in corpus")
        gold = record["gold_article_ids"]
        if not isinstance(gold, list) or any(type(item) is not int for item in gold):
            raise ValueError(f"{query_id}: gold_article_ids must be an integer list")
        if len(gold) != len(set(gold)):
            raise ValueError(f"{query_id}: duplicate gold_article_ids")
        if any(item not in article_ids for item in gold):
            raise ValueError(f"{query_id}: gold_article_ids contain an ID outside the corpus")
        if method == "source_seeded" and seed_id not in gold:
            raise ValueError(f"{query_id}: gold_article_ids must include seed_article_id as initial gold")
        reviewer = record["reviewer"]
        if reviewer is not None and not isinstance(reviewer, str):
            raise ValueError(f"{query_id}: reviewer must be a string or null")
        self_check = record.get("self_check")
        if method == "source_seeded":
            if record["review_mode"] not in REVIEW_MODES:
                raise ValueError(f"{query_id}: invalid review_mode")
            if not isinstance(self_check, dict) or set(self_check) != SELF_CHECK_FIELDS:
                raise ValueError(f"{query_id}: self_check must contain all required fields")
            if any(type(value) is not bool for value in self_check.values()):
                raise ValueError(f"{query_id}: self_check values must be boolean")
        status = record["review_status"]
        if status not in STATUSES:
            raise ValueError(f"{query_id}: invalid review_status")
        if status == "draft" and not allow_draft:
            raise ValueError(f"{query_id}: draft is not allowed in final evaluation")
        if status == "approved" and not gold:
            raise ValueError(f"{query_id}: approved records require gold_article_ids")
        if method == "source_seeded" and status == "approved" and not all(self_check.values()):
            raise ValueError(f"{query_id}: approved records require every self_check item to be true")
        for evidence in record["evidence"]:
            if not {"article_id", "title", "support", "relevance"} <= evidence.keys():
                raise ValueError(f"{query_id}: evidence requires article_id, title, support, and relevance")
            if evidence.get("relevance") not in RELEVANCE:
                raise ValueError(f"{query_id}: invalid evidence relevance")
            article_id = evidence.get("article_id")
            if article_id not in article_ids:
                raise ValueError(f"{query_id}: evidence article_id is not in corpus")
            if evidence["relevance"] == "relevant" and article_id not in gold:
                raise ValueError(f"{query_id}: relevant evidence must appear in gold_article_ids")
            if evidence["relevance"] == "uncertain" and not allow_draft:
                raise ValueError(f"{query_id}: uncertain evidence is not allowed in final evaluation")
        for field in ("naturalness", "gold_clarity"):
            value = record[field]
            if value is not None and (type(value) is not int or not 1 <= value <= 5):
                raise ValueError(f"{query_id}: {field} must be 1..5 or null")
        minutes = record["annotation_minutes"]
        if minutes is not None and (not isinstance(minutes, (int, float)) or minutes < 0):
            raise ValueError(f"{query_id}: annotation_minutes must be non-negative or null")
