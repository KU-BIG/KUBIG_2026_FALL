"""Deterministic date-stratified source assignment without retrieval signals."""

from __future__ import annotations

import copy
import random
from datetime import date

STRATA = {
    "2026-07-31_to_2026-08-05": {"quota": 10, "author_quota": 5, "categories": (3, 3, 2, 2)},
    "2026-08-06": {"quota": 10, "author_quota": 5, "categories": (3, 2, 3, 2)},
    "2026-08-07": {"quota": 18, "author_quota": 9, "categories": (4, 5, 4, 5)},
    "2026-08-08": {"quota": 12, "author_quota": 6, "categories": (3, 3, 3, 3)},
}
CATEGORIES = ("exact_token", "abstract", "multi_aspect", "factoid")
REPLACEMENT_REASONS = {"incomplete_article", "question_not_possible", "near_duplicate_event"}


def stratum_for_date(raw: str) -> str:
    value = date.fromisoformat(raw[:10].replace(".", "-"))
    if date(2026, 7, 31) <= value <= date(2026, 8, 5):
        return "2026-07-31_to_2026-08-05"
    label = value.isoformat()
    if label in STRATA:
        return label
    raise ValueError(f"article date is outside evaluation strata: {raw}")


def create_assignment(articles: list[dict], *, seed: int = 42) -> dict:
    grouped = {stratum: [] for stratum in STRATA}
    seen = set()
    for article in articles:
        article_id = article.get("id")
        if type(article_id) is not int or article_id in seen:
            raise ValueError("articles require unique integer ids")
        seen.add(article_id)
        grouped[stratum_for_date(article.get("date", ""))].append(article_id)

    assignments = []
    strata_output = {}
    position = 1
    rng = random.Random(seed)
    for stratum, config in STRATA.items():
        candidates = sorted(grouped[stratum])
        rng.shuffle(candidates)
        quota = config["quota"]
        if len(candidates) - quota <= quota:
            raise ValueError(f"{stratum} does not have more reserves than its sample quota")
        primary = candidates[:quota]
        category_counts = config["categories"]
        categories = [category for category, count in zip(CATEGORIES, category_counts) for _ in range(count)]
        for index, (article_id, category) in enumerate(zip(primary, categories)):
            assignments.append({
                "position": position,
                "source_article_id": article_id,
                "date_stratum": stratum,
                "author": "kahyun" if index < config["author_quota"] else "ryeowon",
                "category": category,
                "replacement": None,
            })
            position += 1
        strata_output[stratum] = {
            "corpus_count": len(candidates),
            "sample_quota": quota,
            "reserves": candidates[quota:],
        }
    return {
        "seed": seed,
        "corpus_article_count": len(articles),
        "source_article_count": 50,
        "assignments": assignments,
        "strata": strata_output,
    }


def replace_source(assignment: dict, *, position: int, reason_code: str) -> dict:
    if reason_code not in REPLACEMENT_REASONS:
        raise ValueError("replacement reason must be one of the predefined reason codes")
    result = copy.deepcopy(assignment)
    if not 1 <= position <= len(result["assignments"]):
        raise ValueError("position is outside the assignment")
    item = result["assignments"][position - 1]
    reserves = result["strata"][item["date_stratum"]]["reserves"]
    if not reserves:
        raise ValueError("no reserve candidates remain in this stratum")
    old_id = item["source_article_id"]
    item["source_article_id"] = reserves.pop(0)
    item["replacement"] = {"replaced_article_id": old_id, "reason_code": reason_code}
    return result


def swap_categories(assignment: dict, *, first_position: int, second_position: int) -> dict:
    result = copy.deepcopy(assignment)
    try:
        first = result["assignments"][first_position - 1]
        second = result["assignments"][second_position - 1]
    except IndexError as exc:
        raise ValueError("position is outside the assignment") from exc
    if first_position < 1 or second_position < 1:
        raise ValueError("position is outside the assignment")
    if first["date_stratum"] != second["date_stratum"]:
        raise ValueError("categories may only be swapped within the same stratum")
    first["category"], second["category"] = second["category"], first["category"]
    return result
