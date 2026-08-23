"""Run paired article-level Dense and Hybrid retrieval evaluation."""

from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics import dedupe_articles, score_ranking, top_k_overlap

DEFAULT_CHUNK_DEPTH = 100


def _mean(values):
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _system_result(record, system, retriever, chunk_depth):
    started, error, hits = time.perf_counter(), None, []
    try:
        hits = retriever.retrieve(record["question"], top_k=chunk_depth)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    raw = [{"rank": rank, "chunk_id": hit.get("chunk_id"), "article_id": hit.get("article_id")}
           for rank, hit in enumerate(hits, 1)]
    unique = dedupe_articles(hits, top_k=5) if not error else []
    scores = score_ranking([item["article_id"] for item in unique], set(record["gold_article_ids"]))
    return {
        "query_id": record["query_id"], "construction_method": record["construction_method"],
        "question": record["question"], "category": record["category"],
        "date_stratum": record.get("date_stratum", "legacy_query_first"), "system": system,
        "raw_chunk_ranking": raw, "unique_article_ranking": unique, **scores,
        "top_5_overlap": 0, "latency_seconds": time.perf_counter() - started, "error": error,
    }


def _group_summary(selected, results):
    ids = {record["query_id"] for record in selected}
    systems = {}
    for system in ("dense", "hybrid"):
        rows = [row for row in results if row["query_id"] in ids and row["system"] == system]
        systems[system] = {metric: _mean([row[metric] for row in rows])
                           for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5")}
    return {"question_count": len(selected), "systems": systems}


def _summaries(records, results, stratum_corpus_counts):
    dates = {}
    for stratum, corpus_count in stratum_corpus_counts.items():
        selected = [record for record in records if record["date_stratum"] == stratum]
        dates[stratum] = {**_group_summary(selected, results), "corpus_article_count": corpus_count,
                          "sample_count": len(selected)}
    categories = {
        category: _group_summary([record for record in records if record["category"] == category], results)
        for category in ("exact_token", "abstract", "multi_aspect", "factoid")
    }
    return {
        "overall": _group_summary(records, results),
        "by_date_stratum": dates,
        "by_category": categories,
        "date_stratum_interpretation": (
            "Date-stratum results are exploratory because each stratum has a small sample; "
            "do not claim statistical superiority from these differences."
        ),
        "limitations": [
            "AI-assisted question generation with single-annotator self-check and no cross-review may introduce bias.",
            "The 50 source articles seed question construction; retrieval still searches all 432 corpus articles.",
        ],
    }


def run_evaluation(
    records, retrievers, *, question_path, corpus_article_count, corpus_chunk_count,
    settings, git_commit, source_article_count=50, stratum_corpus_counts=None,
    chunk_depth=DEFAULT_CHUNK_DEPTH, now: Callable[[], str] | None = None,
):
    if set(retrievers) != {"dense", "hybrid"}:
        raise ValueError("retrievers must contain exactly dense and hybrid")
    path, results = Path(question_path), []
    for record in records:
        paired = {system: _system_result(record, system, retrievers[system], chunk_depth)
                  for system in ("dense", "hybrid")}
        overlap = top_k_overlap(
            [item["article_id"] for item in paired["dense"]["unique_article_ranking"]],
            [item["article_id"] for item in paired["hybrid"]["unique_article_ranking"]])
        paired["dense"]["top_5_overlap"] = overlap
        paired["hybrid"]["top_5_overlap"] = overlap
        results.extend((paired["dense"], paired["hybrid"]))
    timestamp = now() if now else datetime.now(timezone.utc).isoformat()
    return {
        "metadata": {
            "executed_at": timestamp, "git_commit": git_commit,
            "question_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "corpus": {"article_count": corpus_article_count, "chunk_count": corpus_chunk_count},
            "source_article_count": source_article_count,
            "retrieval_settings": {**settings, "evaluation_chunk_depth": chunk_depth, "article_top_k": 5},
        },
        "results": results,
        "summaries": _summaries(records, results, stratum_corpus_counts or {}),
    }
