"""Deterministically split cleaned news articles into retrieval chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "news_data_clean.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "news_chunks.jsonl"
SPEAKER_RE = re.compile(r"(?=\[(?:앵커|기자|진행자|리포터|출연자|인터뷰)\])")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class ChunkConfig:
    target_min: int = 400
    target_max: int = 700
    max_length: int = 800
    overlap: int = 80
    min_tail: int = 120

    def validate(self) -> None:
        if not (0 < self.target_min <= self.target_max <= self.max_length):
            raise ValueError("lengths must satisfy 0 < target_min <= target_max <= max_length")
        if not (0 <= self.overlap < self.max_length):
            raise ValueError("overlap must be between 0 and max_length")
        if self.min_tail < 0:
            raise ValueError("min_tail cannot be negative")


def split_units(text: str, doc_type: str = "article") -> list[str]:
    """Split at speaker and sentence boundaries without deleting source text."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    sections = SPEAKER_RE.split(normalized) if doc_type == "broadcast" else [normalized]
    units: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        units.extend(part.strip() for part in SENTENCE_RE.split(section) if part.strip())
    return units


def _safe_split(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        window = remaining[: limit + 1]
        candidates = [window.rfind(mark) + 1 for mark in (", ", "，", "; ", " ")]
        cut = max((pos for pos in candidates if pos >= max(1, limit // 2)), default=limit)
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _expanded_units(text: str, doc_type: str, maximum: int) -> list[str]:
    result: list[str] = []
    for unit in split_units(text, doc_type):
        result.extend(_safe_split(unit, maximum))
    return result


def _tail_overlap(text: str, size: int) -> str:
    if size <= 0 or not text:
        return ""
    tail = text[-size:]
    if len(text) > size:
        first_space = tail.find(" ")
        if first_space >= 0:
            tail = tail[first_space + 1 :]
    return tail.strip()


def _pack(units: list[str], config: ChunkConfig) -> list[str]:
    if not units:
        return [""]
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > config.target_max:
            chunks.append(current)
            overlap = _tail_overlap(current, config.overlap)
            candidate = f"{overlap} {unit}".strip()
            if len(candidate) > config.max_length:
                candidate = unit
            current = candidate
        else:
            current = candidate
    if current:
        chunks.append(current)

    if len(chunks) > 1 and len(chunks[-1]) < config.min_tail:
        merged = f"{chunks[-2]} {chunks[-1]}".strip()
        if len(merged) <= config.max_length:
            chunks[-2:] = [merged]
        else:
            tail = chunks.pop()
            overlap = _tail_overlap(chunks[-1], config.overlap)
            combined = f"{overlap} {tail}".strip()
            chunks.append(combined if len(combined) <= config.max_length else tail)
    return chunks


def chunk_article(article: dict, config: ChunkConfig | None = None) -> list[dict]:
    config = config or ChunkConfig()
    config.validate()
    body_chunks = _pack(
        _expanded_units(article.get("content", ""), article.get("doc_type", "article"), config.max_length),
        config,
    )
    title = str(article.get("title", "")).strip()
    identity = json.dumps(
        [article.get("id"), article.get("url", ""), article.get("source_ids", [])],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    output = []
    for index, content in enumerate(body_chunks):
        digest = hashlib.sha256(f"{identity}\0{index}\0{content}".encode("utf-8")).hexdigest()[:24]
        output.append(
            {
                "chunk_id": f"news-{digest}",
                "article_id": article.get("id"),
                "chunk_index": index,
                "title": title,
                "content": content,
                "embedding_text": f"{title}\n{content}".strip(),
                "date": article.get("date", ""),
                "url": article.get("url", ""),
                "stock_names": list(article.get("stock_names", [])),
                "stock_codes": list(article.get("stock_codes", [])),
                "source_ids": list(article.get("source_ids", [])),
                "doc_type": article.get("doc_type", "article"),
            }
        )
    return output


def chunk_articles(articles: Iterable[dict], config: ChunkConfig | None = None) -> list[dict]:
    return [chunk for article in articles for chunk in chunk_article(article, config)]


def statistics_for(articles: list[dict], chunks: list[dict], maximum: int) -> dict:
    lengths = [len(chunk["content"]) for chunk in chunks]
    article_ids = {article.get("id") for article in articles}
    chunked_ids = {chunk["article_id"] for chunk in chunks}
    counts = Counter(chunk["doc_type"] for chunk in chunks)
    return {
        "articles": len(articles),
        "chunks": len(chunks),
        "chunks_per_article": len(chunks) / len(articles) if articles else 0,
        "length_min": min(lengths, default=0),
        "length_median": statistics.median(lengths) if lengths else 0,
        "length_mean": statistics.mean(lengths) if lengths else 0,
        "length_max": max(lengths, default=0),
        "max_violations": sum(length > maximum for length in lengths),
        "duplicate_ids": len(chunks) - len({chunk["chunk_id"] for chunk in chunks}),
        "unchunked_articles": len(article_ids - chunked_ids),
        "by_doc_type": dict(sorted(counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-min", type=int, default=400)
    parser.add_argument("--target-max", type=int, default=700)
    parser.add_argument("--max-length", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--min-tail", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ChunkConfig(args.target_min, args.target_max, args.max_length, args.overlap, args.min_tail)
    config.validate()
    articles = json.loads(args.input.read_text(encoding="utf-8"))
    chunks = chunk_articles(articles, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(statistics_for(articles, chunks, config.max_length), ensure_ascii=False, indent=2))
    print(f"wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
