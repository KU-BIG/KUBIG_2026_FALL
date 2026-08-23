"""Build and atomically write blind candidate pools for frozen questions."""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from pathlib import Path

from evaluation.metrics import dedupe_articles

BLIND_CANDIDATE_FIELDS = {"candidate_key", "title", "date", "content", "url", "doc_type"}
FORBIDDEN_BLIND_FIELDS = {
    "article_id", "source_article_id", "seed_article_id", "gold_article_ids",
    "calibration_source", "dense_rank", "hybrid_rank", "rank", "score", "rrf_score",
    "system", "systems", "dense", "hybrid",
}


def _catalog_article(catalog, article_id: int) -> dict:
    article = catalog.get(article_id)
    if article is None:
        raise KeyError(f"article_id not found in corpus: {article_id}")
    return article


def _unique_ranking(hits: list[dict], depth: int) -> list[dict]:
    return dedupe_articles(hits, top_k=depth)


def _candidate_key(seed: int, query_id: str, position: int) -> str:
    value = f"blind-candidate:{seed}:{query_id}:{position}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def build_batch_pool(
    records: list[dict], retrievers: dict, catalog, *, seed: int = 42,
    pool_depth: int = 20, chunk_depth: int = 100,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Retrieve all questions and return blind packets, private mapping, and counts."""
    if set(retrievers) != {"dense", "hybrid"}:
        raise ValueError("retrievers must contain exactly dense and hybrid")
    if pool_depth < 1 or chunk_depth < 1:
        raise ValueError("pool_depth and chunk_depth must be at least 1")

    packets, mappings, stats = [], [], []
    seen_keys: set[str] = set()
    for record in records:
        query_id, question = record["query_id"], record["question"]
        rankings = {}
        for system in ("dense", "hybrid"):
            try:
                hits = retrievers[system].retrieve(question, top_k=chunk_depth)
            except Exception as exc:
                raise RuntimeError(f"{query_id} {system} retrieval failed: {exc}") from exc
            rankings[system] = _unique_ranking(hits, pool_depth)

        pooled: dict[int, dict] = {}
        for system in ("dense", "hybrid"):
            for rank, item in enumerate(rankings[system], 1):
                article_id = item["article_id"]
                pooled.setdefault(article_id, {"dense_rank": None, "hybrid_rank": None})
                pooled[article_id][f"{system}_rank"] = rank

        source_id = record["seed_article_id"]
        pooled.setdefault(source_id, {"dense_rank": None, "hybrid_rank": None})
        article_ids = list(pooled)
        random.Random(f"{seed}:{query_id}").shuffle(article_ids)

        candidates = []
        for position, article_id in enumerate(article_ids, 1):
            key = _candidate_key(seed, query_id, position)
            if key in seen_keys:
                raise ValueError(f"duplicate candidate_key: {key}")
            seen_keys.add(key)
            article = _catalog_article(catalog, article_id)
            candidates.append({
                "candidate_key": key,
                **{field: article[field] for field in ("title", "date", "content", "url", "doc_type")},
            })
            mappings.append({
                "query_id": query_id,
                "candidate_key": key,
                "article_id": article_id,
                "dense_rank": pooled[article_id]["dense_rank"],
                "hybrid_rank": pooled[article_id]["hybrid_rank"],
                "calibration_source": article_id == source_id,
            })

        packets.append({
            "query_id": query_id,
            "question": question,
            "category": record["category"],
            "candidates": candidates,
        })
        stats.append({
            "query_id": query_id,
            "dense_unique_count": len(rankings["dense"]),
            "hybrid_unique_count": len(rankings["hybrid"]),
            "pooled_candidate_count": len(candidates),
            "calibration_source_retrieved": source_id in {
                item["article_id"] for system in rankings.values() for item in system
            },
            "chunk_retrieval_depth": chunk_depth,
        })
    return packets, mappings, stats


def validate_pool_artifacts(packets: list[dict], mappings: list[dict]) -> None:
    packet_pairs, mapping_pairs, keys = set(), set(), set()
    articles_by_query: dict[str, set[int]] = {}
    calibration_counts: dict[str, int] = {}
    for packet in packets:
        query_id = packet["query_id"]
        for candidate in packet["candidates"]:
            forbidden = set(candidate) & FORBIDDEN_BLIND_FIELDS
            if forbidden:
                raise ValueError(f"forbidden blind field: {sorted(forbidden)[0]}")
            if set(candidate) != BLIND_CANDIDATE_FIELDS:
                raise ValueError("blind candidate fields do not match the schema")
            key = candidate["candidate_key"]
            if key in keys:
                raise ValueError(f"duplicate candidate_key: {key}")
            keys.add(key)
            packet_pairs.add((query_id, key))

    for item in mappings:
        query_id, key, article_id = item["query_id"], item["candidate_key"], item["article_id"]
        mapping_pairs.add((query_id, key))
        articles = articles_by_query.setdefault(query_id, set())
        if article_id in articles:
            raise ValueError(f"duplicate article_id for {query_id}: {article_id}")
        articles.add(article_id)
        calibration_counts[query_id] = calibration_counts.get(query_id, 0) + int(item["calibration_source"])
    if packet_pairs != mapping_pairs:
        raise ValueError("blind packet and mapping candidate sets differ")
    if any(calibration_counts.get(packet["query_id"], 0) != 1 for packet in packets):
        raise ValueError("each query must have exactly one calibration source")


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def _jsonl_bytes(values: list[dict]) -> bytes:
    return "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in values).encode()


def write_pool_artifacts(
    packets: list[dict], mappings: list[dict], manifest: dict,
    blind_path: Path | str, mapping_path: Path | str, manifest_path: Path | str,
) -> dict[str, str]:
    """Validate all data, then write three new files with rollback on failure."""
    validate_pool_artifacts(packets, mappings)
    paths = [Path(blind_path), Path(mapping_path), Path(manifest_path)]
    if any(path.exists() for path in paths):
        raise FileExistsError("pool output already exists; refusing to overwrite")
    blind_bytes, mapping_bytes = _jsonl_bytes(packets), _json_bytes(mappings)
    hashes = {
        "blind_pool_sha256": hashlib.sha256(blind_bytes).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
    }
    manifest_bytes = _json_bytes({**manifest, "generated_file_sha256": hashes})
    payloads = [blind_bytes, mapping_bytes, manifest_bytes]
    temp_paths, committed = [], []
    try:
        for path, payload in zip(paths, payloads):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(payload)
            temp_paths.append(temporary)
        for temporary, path in zip(temp_paths, paths):
            os.replace(temporary, path)
            committed.append(path)
    except Exception:
        for path in temp_paths + committed:
            path.unlink(missing_ok=True)
        raise
    return {**hashes, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
