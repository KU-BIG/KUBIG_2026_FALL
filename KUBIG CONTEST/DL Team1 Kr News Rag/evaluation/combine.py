"""Validate and combine the two author-owned annotation files."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.schema import load_records, validate_records

AUTHOR_FILES = {"kahyun_25.jsonl": ("kahyun", "K"), "ryeowon_25.jsonl": ("ryeowon", "R")}


def combine_annotations(input_paths, output_path, article_ids):
    paths = [Path(path) for path in input_paths]
    if len(paths) != 2 or {path.name for path in paths} != set(AUTHOR_FILES):
        raise ValueError("inputs must be kahyun_25.jsonl and ryeowon_25.jsonl")
    combined = []
    for path in paths:
        author, prefix = AUTHOR_FILES[path.name]
        records = load_records(path)
        if len(records) != 25:
            raise ValueError(f"{path.name} must contain exactly 25 records")
        expected_ids = {f"{prefix}{index:03d}" for index in range(1, 26)}
        if {row.get("query_id") for row in records} != expected_ids:
            raise ValueError(f"{path.name} must use query IDs {prefix}001-{prefix}025")
        if any(row.get("author") != author for row in records):
            raise ValueError(f"{path.name} records must have author {author}")
        if any(row.get("construction_method") != "source_seeded" for row in records):
            raise ValueError("combined evaluation accepts only source_seeded records")
        combined.extend(records)
    validate_records(combined, article_ids, allow_draft=False)
    sources = [row["seed_article_id"] for row in combined]
    if len(set(sources)) != 50:
        raise ValueError("source_article_id values must be unique across both files")
    combined.sort(key=lambda row: row["query_id"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined), encoding="utf-8")
    return combined
