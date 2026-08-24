"""Rebuild the retrieval corpus, chunks, and explicit gold labels.

The script keeps the 13 processed web documents unchanged, re-extracts the
eight PDF documents from ``data/raw`` with the currently installed
``pdfplumber``, creates the three agreed chunk variants, and stores gold chunk
alternatives for every evidence item and question.

Usage:
    python prepare_retrieval_data.py
    python prepare_retrieval_data.py --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
KUBIG_DIR = SCRIPT_DIR.parent
PROJECT_DIR = KUBIG_DIR.parent
DATASET_DIR = KUBIG_DIR / "retriever_dataset"
DOCUMENTS_PATH = DATASET_DIR / "documents" / "documents.jsonl"
CHUNKS_DIR = DATASET_DIR / "chunks"
METADATA_DIR = DATASET_DIR / "metadata"
EVAL_PATH = KUBIG_DIR / "rag_evaluation_dataset.jsonl"
ROOT_EVAL_PATH = PROJECT_DIR / "rag_evaluation_dataset.jsonl"
RAW_PDF_DIR = PROJECT_DIR / "data" / "raw" / "pdf"

CHUNK_VARIANTS = {
    "300_50": (300, 50),
    "400_60": (400, 60),
    "500_80": (500, 80),
}
SPLIT_SEED = "finagent-80-40-v1"
VALIDATION_PER_STRATUM = {"easy": 6, "medium": 10, "hard": 4}

# PDF002 ("Guide to Leading a Safe Life in Seoul") is a 43-page general safety
# guide; only the financial-fraud/phishing section (pages 17-19, 1-indexed) was
# originally curated for this corpus. Without this filter, extract_pdf_pages()
# would pull in unrelated content (traffic safety, etc.) from all 43 pages.
PDF_PAGE_FILTERS: dict[str, set[int]] = {
    "PDF002": {17, 18, 19},
}

COMPACT_RE = re.compile(r"[^0-9A-Za-z가-힣]+")
PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]\n")
WORD_RE = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass(frozen=True)
class SourceSpan:
    text: str
    start: int
    end: int
    method: str
    score: float


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def compact(value: str) -> str:
    return COMPACT_RE.sub("", unicodedata.normalize("NFKC", value)).lower()


def _compact_with_offsets(value: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(value):
        folded = unicodedata.normalize("NFKC", character).lower()
        for output_character in folded:
            if output_character.isalnum() or "가" <= output_character <= "힣":
                normalized.append(output_character)
                offsets.append(index)
    return "".join(normalized), offsets


def _ngram_counter(value: str, size: int = 3) -> Counter[str]:
    normalized = compact(value)
    if len(normalized) < size:
        return Counter([normalized]) if normalized else Counter()
    return Counter(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def ngram_recall(reference: str, candidate: str) -> float:
    reference_grams = _ngram_counter(reference)
    candidate_grams = _ngram_counter(candidate)
    if not reference_grams or not candidate_grams:
        return 0.0
    return sum((reference_grams & candidate_grams).values()) / sum(reference_grams.values())


def overlap_score(reference: str, candidate: str) -> float:
    reference_grams = _ngram_counter(reference)
    candidate_grams = _ngram_counter(candidate)
    if not reference_grams or not candidate_grams:
        return 0.0
    common = sum((reference_grams & candidate_grams).values())
    recall = common / sum(reference_grams.values())
    precision = common / sum(candidate_grams.values())
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0

    reference_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", reference))
    candidate_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", candidate))
    number_recall = (
        len(reference_numbers & candidate_numbers) / len(reference_numbers)
        if reference_numbers
        else 1.0
    )
    return 0.8 * f1 + 0.2 * number_recall


def exact_normalized_span(source: str, quote: str) -> SourceSpan | None:
    normalized_source, offsets = _compact_with_offsets(source)
    normalized_quote = compact(quote)
    if not normalized_quote:
        return None
    position = normalized_source.find(normalized_quote)
    if position < 0:
        return None
    start = offsets[position]
    end = offsets[position + len(normalized_quote) - 1] + 1
    return SourceSpan(source[start:end], start, end, "normalized_exact", 1.0)


def line_window_span(source: str, quote: str, section: str = "") -> SourceSpan:
    line_matches = [match for match in re.finditer(r"[^\n]+", source) if match.group().strip()]
    if not line_matches:
        return SourceSpan(source, 0, len(source), "whole_scope", overlap_score(quote, source))

    maximum_lines = min(80, len(line_matches))
    best: SourceSpan | None = None
    best_rank = -1.0
    best_length = len(source) + 1
    normalized_section = compact(section)
    reference_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", quote))

    for start_index in range(len(line_matches)):
        for end_index in range(start_index, min(len(line_matches), start_index + maximum_lines)):
            start = line_matches[start_index].start()
            end = line_matches[end_index].end()
            candidate = source[start:end]
            normalized_candidate = compact(candidate)
            recall = ngram_recall(quote, candidate)
            candidate_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", candidate))
            number_recall = (
                len(reference_numbers & candidate_numbers) / len(reference_numbers)
                if reference_numbers
                else 1.0
            )
            section_bonus = 0.002 if normalized_section and normalized_section in normalized_candidate else 0.0
            rank = recall + 0.02 * number_recall + section_bonus
            candidate_length = len(normalized_candidate)
            if rank > best_rank + 1e-9 or (
                abs(rank - best_rank) <= 1e-9 and candidate_length < best_length
            ):
                best_rank = rank
                best_length = candidate_length
                best = SourceSpan(candidate, start, end, "line_window", recall)

    whole_recall = ngram_recall(quote, source)
    whole_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", source))
    whole_number_recall = (
        len(reference_numbers & whole_numbers) / len(reference_numbers)
        if reference_numbers
        else 1.0
    )
    whole_rank = whole_recall + 0.02 * whole_number_recall
    if whole_rank > best_rank + 1e-9:
        best = SourceSpan(source, 0, len(source), "whole_scope", whole_recall)

    assert best is not None
    return best


def locate_source_span(source: str, quote: str, section: str = "") -> SourceSpan:
    exact = exact_normalized_span(source, quote)
    if exact is not None:
        return exact
    return line_window_span(source, quote, section)


def normalized_filename(path: Path | str) -> str:
    return unicodedata.normalize("NFC", Path(path).name)


def resolve_pdf_path(source_file: str) -> Path:
    target = normalized_filename(source_file)
    matches = [path for path in RAW_PDF_DIR.glob("*.pdf") if normalized_filename(path) == target]
    if len(matches) != 1:
        raise ValueError(f"Expected one local PDF for {source_file!r}, found {matches}")
    return matches[0]


def extract_pdf_pages(path: Path, page_filter: set[int] | None = None) -> list[tuple[int, str]]:
    """Returns a list of (1-indexed page number, page text) pairs.

    ``page_filter``, when given, is a set of 1-indexed page numbers to keep
    (pdfplumber's own ``pdf.pages`` list is 0-indexed, so page N corresponds
    to ``pdf.pages[N - 1]``). Original page numbers are preserved as-is (not
    renumbered) so that evidence ``page`` fields keep referring to the same
    physical PDF page.
    """
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(path) as pdf:
        for zero_indexed, page in enumerate(pdf.pages):
            page_number = zero_indexed + 1
            if page_filter is not None and page_number not in page_filter:
                continue
            text = page.extract_text() or ""
            text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
            pages.append((page_number, text))
    return pages


def build_pdf_document_text(
    pages: Sequence[tuple[int, str]]
) -> tuple[str, dict[int, tuple[int, int]]]:
    parts: list[str] = []
    page_spans: dict[int, tuple[int, int]] = {}
    cursor = 0
    for page_number, page_text in pages:
        if parts:
            parts.append("\n\n")
            cursor += 2
        marker = f"[Page {page_number}]\n"
        parts.append(marker)
        cursor += len(marker)
        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        page_spans[page_number] = (start, cursor)
    return "".join(parts), page_spans


def page_spans_from_document(text: str) -> dict[int, tuple[int, int]]:
    matches = list(PAGE_MARKER_RE.finditer(text))
    spans: dict[int, tuple[int, int]] = {}
    for index, match in enumerate(matches):
        page = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans[page] = (start, end)
    return spans


def rebuild_documents(documents: list[dict]) -> tuple[list[dict], dict[str, dict[int, tuple[int, int]]]]:
    rebuilt: list[dict] = []
    page_spans: dict[str, dict[int, tuple[int, int]]] = {}
    for document in documents:
        updated = dict(document)
        if document["source_type"] == "pdf":
            source_path = resolve_pdf_path(document["source_file"])
            page_filter = PDF_PAGE_FILTERS.get(document["id"])
            pages = extract_pdf_pages(source_path, page_filter=page_filter)
            updated["text"], page_spans[document["id"]] = build_pdf_document_text(pages)
        else:
            page_spans[document["id"]] = {}
        rebuilt.append(updated)
    return rebuilt, page_spans


def load_tokenizer():
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tokenizer.model_max_length = 1_000_000_000
    return tokenizer


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def token_windows(
    document_text: str,
    span_start: int,
    span_end: int,
    tokenizer,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int, int]]:
    source = document_text[span_start:span_end]
    encoded = tokenizer(
        source,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoded["offset_mapping"]
    if not offsets:
        return []
    step = target_tokens - overlap_tokens
    windows: list[tuple[int, int, int]] = []
    token_start = 0
    while token_start < len(offsets):
        token_end = min(token_start + target_tokens, len(offsets))
        local_start = offsets[token_start][0]
        local_end = offsets[token_end - 1][1]
        start, end = _trim_span(document_text, span_start + local_start, span_start + local_end)
        if start < end:
            windows.append((start, end, token_end - token_start))
        if token_end == len(offsets):
            break
        token_start += step
    return windows


def build_chunks(
    documents: Sequence[dict],
    page_spans: dict[str, dict[int, tuple[int, int]]],
    tokenizer,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    chunks: list[dict] = []
    for document in documents:
        text = document["text"]
        document_windows: list[tuple[int, int, int, int | None]] = []
        if document["source_type"] == "pdf":
            for page, (start, end) in sorted(page_spans[document["id"]].items()):
                for chunk_start, chunk_end, token_count in token_windows(
                    text,
                    start,
                    end,
                    tokenizer,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                ):
                    document_windows.append((chunk_start, chunk_end, token_count, page))
        else:
            for chunk_start, chunk_end, token_count in token_windows(
                text,
                0,
                len(text),
                tokenizer,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            ):
                document_windows.append((chunk_start, chunk_end, token_count, None))

        for index, (start, end, token_count, page) in enumerate(document_windows, start=1):
            chunks.append(
                {
                    "chunk_id": f"{document['id']}_{index:04d}",
                    "doc_id": document["id"],
                    "title": document["title"],
                    "organization": document["organization"],
                    "language": document["language"],
                    "category": document["category"],
                    "source_type": document["source_type"],
                    "source_url": document.get("source_url"),
                    "page_start": page,
                    "page_end": page,
                    "start_char": start,
                    "end_char": end,
                    "token_count": token_count,
                    "text": text[start:end],
                }
            )
    return chunks


def build_document_index(documents: Sequence[dict]) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for document in documents:
        source_type = document["source_type"]
        key = document.get("source_file") if source_type == "pdf" else document.get("source_url")
        if key:
            index[(source_type, normalized_filename(key) if source_type == "pdf" else key.strip())] = document
    return index


def source_scope(
    document: dict,
    evidence: dict,
    page_spans: dict[str, dict[int, tuple[int, int]]],
) -> tuple[str, int]:
    page = evidence.get("page")
    if document["source_type"] == "pdf" and page in page_spans[document["id"]]:
        start, end = page_spans[document["id"]][page]
        return document["text"][start:end], start
    return document["text"], 0


def best_gold_chunks(
    quote: str,
    raw_start: int,
    raw_end: int,
    chunks: Sequence[dict],
    page: int | None,
) -> list[str]:
    candidates = [
        chunk
        for chunk in chunks
        if page is None or chunk.get("page_start") == page
    ]
    if not candidates:
        return []

    scored: list[tuple[float, str]] = []
    for chunk in candidates:
        interval_overlap = max(
            0,
            min(raw_end, chunk["end_char"]) - max(raw_start, chunk["start_char"]),
        )
        raw_length = max(raw_end - raw_start, 1)
        coverage = interval_overlap / raw_length
        semantic_recall = ngram_recall(quote, chunk["text"])
        semantic_overlap = overlap_score(quote, chunk["text"])
        score = 0.75 * semantic_recall + 0.2 * semantic_overlap + 0.05 * coverage
        if interval_overlap or semantic_recall >= 0.2:
            scored.append((score, chunk["chunk_id"]))
    if not scored:
        return []

    scored.sort(reverse=True)
    best_score = scored[0][0]
    threshold = max(best_score * 0.95, best_score - 0.03)
    return sorted(chunk_id for score, chunk_id in scored if score >= threshold)[:3]


def stratified_split(records: list[dict]) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["category"], record["difficulty"])].append(record)

    for (category, difficulty), group in groups.items():
        expected = {"easy": 9, "medium": 15, "hard": 6}[difficulty]
        if len(group) != expected:
            raise ValueError(f"Unexpected stratum size for {(category, difficulty)}: {len(group)}")
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"{SPLIT_SEED}:{item['id']}".encode("utf-8")
            ).hexdigest(),
        )
        validation_count = VALIDATION_PER_STRATUM[difficulty]
        validation_ids = {item["id"] for item in ordered[:validation_count]}
        for item in group:
            item["split"] = "validation" if item["id"] in validation_ids else "test"


def attach_source_quotes_and_gold(
    records: list[dict],
    documents: Sequence[dict],
    page_spans: dict[str, dict[int, tuple[int, int]]],
    chunks_by_variant: dict[str, list[dict]],
) -> list[dict]:
    document_index = build_document_index(documents)
    chunks_by_variant_and_doc = {
        variant: defaultdict(list) for variant in CHUNK_VARIANTS
    }
    for variant, chunks in chunks_by_variant.items():
        for chunk in chunks:
            chunks_by_variant_and_doc[variant][chunk["doc_id"]].append(chunk)

    audit: list[dict] = []
    for record in records:
        aggregate_gold = {variant: set() for variant in CHUNK_VARIANTS}
        for evidence_index, evidence in enumerate(record.get("evidence", [])):
            source_type = evidence["source_type"]
            key = evidence.get("source_file") if source_type == "pdf" else evidence.get("source_url")
            lookup_key = normalized_filename(key) if source_type == "pdf" else (key or "").strip()
            document = document_index.get((source_type, lookup_key))
            if document is None:
                raise ValueError(f"Document not found for {record['id']} evidence {evidence_index}: {key}")

            scope, scope_offset = source_scope(document, evidence, page_spans)
            located = locate_source_span(scope, evidence["quote"], evidence.get("section") or "")
            raw_start = scope_offset + located.start
            raw_end = scope_offset + located.end
            raw_text = document["text"][raw_start:raw_end]
            if raw_text != located.text:
                raise AssertionError(f"Raw span mismatch for {record['id']} evidence {evidence_index}")

            evidence["source_quote_raw"] = raw_text
            evidence["source_quote_match_method"] = located.method
            evidence["source_quote_match_score"] = round(located.score, 4)
            evidence["source_quote_match_recall"] = round(ngram_recall(evidence["quote"], raw_text), 4)
            evidence_gold: dict[str, list[str]] = {}
            for variant in CHUNK_VARIANTS:
                gold_ids = best_gold_chunks(
                    evidence["quote"],
                    raw_start,
                    raw_end,
                    chunks_by_variant_and_doc[variant][document["id"]],
                    evidence.get("page") if source_type == "pdf" else None,
                )
                if not gold_ids:
                    raise ValueError(
                        f"No gold chunk for {record['id']} evidence {evidence_index} ({variant})"
                    )
                evidence_gold[variant] = gold_ids
                aggregate_gold[variant].update(gold_ids)
            evidence["gold_chunk_ids"] = evidence_gold
            audit.append(
                {
                    "question_id": record["id"],
                    "evidence_index": evidence_index,
                    "source_type": source_type,
                    "document_id": document["id"],
                    "page": evidence.get("page"),
                    "method": located.method,
                    "score": round(located.score, 4),
                    "recall": round(ngram_recall(evidence["quote"], raw_text), 4),
                }
            )
        record["gold_chunk_ids"] = {
            variant: sorted(chunk_ids) for variant, chunk_ids in aggregate_gold.items()
        }
    return audit


def validate_outputs(
    documents: Sequence[dict],
    records: Sequence[dict],
    chunks_by_variant: dict[str, list[dict]],
) -> dict:
    if len(documents) != 21:
        raise ValueError(f"Expected 21 documents, found {len(documents)}")
    source_counts = Counter(document["source_type"] for document in documents)
    if source_counts != {"pdf": 8, "web": 13}:
        raise ValueError(f"Unexpected document sources: {dict(source_counts)}")
    if len(records) != 120:
        raise ValueError(f"Expected 120 evaluation records, found {len(records)}")

    split_counts = Counter(record["split"] for record in records)
    if split_counts != {"validation": 80, "test": 40}:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")

    split_strata = Counter(
        (record["split"], record["category"], record["difficulty"])
        for record in records
    )
    for category in sorted({record["category"] for record in records}):
        for difficulty, validation_count in VALIDATION_PER_STRATUM.items():
            expected = {
                "validation": validation_count,
                "test": {"easy": 3, "medium": 5, "hard": 2}[difficulty],
            }
            for split, count in expected.items():
                actual = split_strata[(split, category, difficulty)]
                if actual != count:
                    raise ValueError(
                        f"Unexpected split stratum {(split, category, difficulty)}: {actual}"
                    )

    document_text = {document["id"]: document["text"] for document in documents}
    document_index = build_document_index(documents)
    chunk_counts: dict[str, int] = {}
    minimum_gold_quote_recall: dict[str, float] = {}
    for variant, chunks in chunks_by_variant.items():
        chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        gold_quote_recalls: list[float] = []
        if len(chunk_ids) != len(chunks):
            raise ValueError(f"Duplicate chunk IDs in {variant}")
        for chunk in chunks:
            if chunk["text"] != document_text[chunk["doc_id"]][chunk["start_char"] : chunk["end_char"]]:
                raise ValueError(f"Chunk offset mismatch: {variant}/{chunk['chunk_id']}")
        for record in records:
            gold = record["gold_chunk_ids"][variant]
            if not gold or not set(gold).issubset(chunk_ids):
                raise ValueError(f"Invalid question gold labels: {variant}/{record['id']}")
            for evidence in record["evidence"]:
                evidence_gold = evidence["gold_chunk_ids"][variant]
                if not evidence_gold or not set(evidence_gold).issubset(chunk_ids):
                    raise ValueError(f"Invalid evidence gold labels: {variant}/{record['id']}")
                best_quote_recall = max(
                    ngram_recall(evidence["quote"], chunk_by_id[chunk_id]["text"])
                    for chunk_id in evidence_gold
                )
                gold_quote_recalls.append(best_quote_recall)
                if best_quote_recall < 0.4:
                    raise ValueError(
                        f"Weak evidence-to-gold link: {variant}/{record['id']} "
                        f"recall={best_quote_recall:.3f}"
                    )
            expected_gold = sorted(
                {
                    chunk_id
                    for evidence in record["evidence"]
                    for chunk_id in evidence["gold_chunk_ids"][variant]
                }
            )
            if gold != expected_gold:
                raise ValueError(f"Question/evidence gold mismatch: {variant}/{record['id']}")
        chunk_counts[variant] = len(chunks)
        minimum_gold_quote_recall[variant] = round(min(gold_quote_recalls), 4)

    pdf_numeric_evidence_checked = 0
    for record in records:
        for evidence_index, evidence in enumerate(record["evidence"]):
            source_type = evidence["source_type"]
            key = evidence.get("source_file") if source_type == "pdf" else evidence.get("source_url")
            lookup_key = normalized_filename(key) if source_type == "pdf" else (key or "").strip()
            document = document_index.get((source_type, lookup_key))
            if document is None:
                raise ValueError(f"Missing evidence document: {record['id']}/{evidence_index}")
            if evidence.get("source_quote_raw") not in document["text"]:
                raise ValueError(f"Raw quote not found in document: {record['id']}/{evidence_index}")
            if not 0.0 <= evidence.get("source_quote_match_recall", -1.0) <= 1.0:
                raise ValueError(f"Invalid quote recall: {record['id']}/{evidence_index}")
            if source_type == "pdf":
                quote_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", evidence["quote"]))
                raw_numbers = set(
                    re.findall(r"\d+(?:[.,]\d+)*", evidence["source_quote_raw"])
                )
                if quote_numbers:
                    pdf_numeric_evidence_checked += 1
                    if not quote_numbers.issubset(raw_numbers):
                        missing = sorted(quote_numbers - raw_numbers)
                        raise ValueError(
                            f"Possible PDF table-cell omission: {record['id']}/{evidence_index} "
                            f"missing numbers={missing}"
                        )

    return {
        "documents": len(documents),
        "source_types": dict(source_counts),
        "questions": len(records),
        "splits": dict(split_counts),
        "chunks": chunk_counts,
        "minimum_gold_quote_recall": minimum_gold_quote_recall,
        "pdf_numeric_evidence_checked": pdf_numeric_evidence_checked,
    }


def build_corpus_statistics(documents: Sequence[dict]) -> dict:
    lengths = {document["id"]: len(document["text"]) for document in documents}
    longest_id = max(lengths, key=lengths.get)
    shortest_id = min(lengths, key=lengths.get)
    return {
        "documents": len(documents),
        "web_documents": sum(document["source_type"] == "web" for document in documents),
        "pdf_documents": sum(document["source_type"] == "pdf" for document in documents),
        "languages": dict(Counter(document["language"] for document in documents)),
        "categories": dict(Counter(document["category"] for document in documents)),
        "average_text_length": round(sum(lengths.values()) / len(lengths), 2),
        "total_text_length": sum(lengths.values()),
        "longest_document": {"id": longest_id, "length": lengths[longest_id]},
        "shortest_document": {"id": shortest_id, "length": lengths[shortest_id]},
    }


def build_chunk_statistics(
    variant: str,
    chunks: Sequence[dict],
    target_tokens: int,
    overlap_tokens: int,
) -> dict:
    character_lengths = [len(chunk["text"]) for chunk in chunks]
    token_lengths = [chunk["token_count"] for chunk in chunks]
    return {
        "chunk_version": f"chunk_{variant}",
        "chunk_size_tokens": target_tokens,
        "chunk_overlap_tokens": overlap_tokens,
        "document_count": len({chunk["doc_id"] for chunk in chunks}),
        "chunk_count": len(chunks),
        "avg_chunk_length_chars": round(sum(character_lengths) / len(chunks), 2),
        "max_chunk_length_chars": max(character_lengths),
        "min_chunk_length_chars": min(character_lengths),
        "avg_chunk_length_tokens": round(sum(token_lengths) / len(chunks), 2),
        "max_chunk_length_tokens": max(token_lengths),
        "min_chunk_length_tokens": min(token_lengths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate already-generated files without rebuilding them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        documents = read_jsonl(DOCUMENTS_PATH)
        records = read_jsonl(EVAL_PATH)
        chunks_by_variant = {
            variant: read_jsonl(CHUNKS_DIR / f"chunk_{variant}" / "chunks.jsonl")
            for variant in CHUNK_VARIANTS
        }
        print(json.dumps(validate_outputs(documents, records, chunks_by_variant), ensure_ascii=False, indent=2))
        return

    original_documents = read_jsonl(DOCUMENTS_PATH)
    records = read_jsonl(EVAL_PATH)
    documents, page_spans = rebuild_documents(original_documents)
    tokenizer = load_tokenizer()
    chunks_by_variant = {
        variant: build_chunks(
            documents,
            page_spans,
            tokenizer,
            target_tokens=target,
            overlap_tokens=overlap,
        )
        for variant, (target, overlap) in CHUNK_VARIANTS.items()
    }
    stratified_split(records)
    audit = attach_source_quotes_and_gold(records, documents, page_spans, chunks_by_variant)
    report = validate_outputs(documents, records, chunks_by_variant)
    low_confidence = [item for item in audit if item["recall"] < 0.7]
    report["source_quote_matches"] = dict(Counter(item["method"] for item in audit))
    report["low_confidence_source_quotes"] = low_confidence

    write_jsonl(DOCUMENTS_PATH, documents)
    for variant, chunks in chunks_by_variant.items():
        write_jsonl(CHUNKS_DIR / f"chunk_{variant}" / "chunks.jsonl", chunks)
        target_tokens, overlap_tokens = CHUNK_VARIANTS[variant]
        write_json(
            METADATA_DIR / f"chunk_statistics_{variant}.json",
            build_chunk_statistics(
                variant,
                chunks,
                target_tokens,
                overlap_tokens,
            ),
        )
    write_json(METADATA_DIR / "corpus_statistics.json", build_corpus_statistics(documents))
    write_jsonl(EVAL_PATH, records)
    write_jsonl(ROOT_EVAL_PATH, records)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
