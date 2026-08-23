"""Write portable retrieval evaluation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

CSV_FIELDS = (
    "executed_at", "git_commit", "question_sha256", "corpus_article_count",
    "corpus_chunk_count", "source_article_count", "retrieval_settings",
    "query_id", "construction_method", "question", "category", "date_stratum", "system",
    "raw_chunk_ranking", "unique_article_ranking", "first_relevant_rank",
    "hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5", "top_5_overlap",
    "latency_seconds", "error",
)


def write_json(report: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(report: dict, path: Path | str) -> None:
    metadata = report["metadata"]
    common = {
        "executed_at": metadata["executed_at"],
        "git_commit": metadata["git_commit"],
        "question_sha256": metadata["question_sha256"],
        "corpus_article_count": metadata["corpus"]["article_count"],
        "corpus_chunk_count": metadata["corpus"]["chunk_count"],
        "source_article_count": metadata.get("source_article_count"),
        "retrieval_settings": json.dumps(metadata["retrieval_settings"], ensure_ascii=False, separators=(",", ":")),
    }
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in report["results"]:
            row = {**common, **{field: result.get(field) for field in CSV_FIELDS if field not in common}}
            for field in ("raw_chunk_ranking", "unique_article_ranking"):
                row[field] = json.dumps(row[field], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)
