"""One-off driver: recompute source_quote_raw + gold_chunk_ids for the
400/60 chunk variant ONLY, from scratch, against the newly rebuilt documents
and chunks. Does not touch 300/50 or 500/80 gold labels.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_retrieval_data import (  # noqa: E402
    DOCUMENTS_PATH,
    CHUNKS_DIR,
    EVAL_PATH,
    read_jsonl,
    build_document_index,
    source_scope,
    locate_source_span,
    best_gold_chunks,
    ngram_recall,
    normalized_filename,
)

documents = read_jsonl(DOCUMENTS_PATH)
chunks = read_jsonl(CHUNKS_DIR / "chunk_400_60" / "chunks.jsonl")
records = read_jsonl(EVAL_PATH)

document_index = build_document_index(documents)
doc_by_id = {d["id"]: d for d in documents}

# page_spans reconstructed the same way rebuild_documents() would have derived
# them, but here we just need them for source_scope(); recompute from the
# regenerated documents' own [Page N] markers.
import re
PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]\n")

def page_spans_from_document(text: str) -> dict[int, tuple[int, int]]:
    matches = list(PAGE_MARKER_RE.finditer(text))
    spans: dict[int, tuple[int, int]] = {}
    for index, match in enumerate(matches):
        page = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans[page] = (start, end)
    return spans

page_spans = {d["id"]: (page_spans_from_document(d["text"]) if d["source_type"] == "pdf" else {}) for d in documents}

chunks_by_doc: dict[str, list[dict]] = {}
for c in chunks:
    chunks_by_doc.setdefault(c["doc_id"], []).append(c)

results = []
unmatched = []
for record in records:
    for evidence_index, evidence in enumerate(record.get("evidence", [])):
        source_type = evidence["source_type"]
        key = evidence.get("source_file") if source_type == "pdf" else evidence.get("source_url")
        lookup_key = normalized_filename(key) if source_type == "pdf" else (key or "").strip()
        document = document_index.get((source_type, lookup_key))
        if document is None:
            unmatched.append((record["id"], evidence_index, "document_not_found", key))
            continue

        scope, scope_offset = source_scope(document, evidence, page_spans)
        located = locate_source_span(scope, evidence["quote"], evidence.get("section") or "")
        raw_start = scope_offset + located.start
        raw_end = scope_offset + located.end
        raw_text = document["text"][raw_start:raw_end]

        gold_ids = best_gold_chunks(
            evidence["quote"],
            raw_start,
            raw_end,
            chunks_by_doc.get(document["id"], []),
            evidence.get("page") if source_type == "pdf" else None,
        )

        # overwrite (in-memory only) the 400_60 slice of this evidence, leave
        # 300_50/500_80 slices exactly as they were
        evidence["source_quote_raw"] = raw_text
        evidence["source_quote_match_method"] = located.method
        evidence["source_quote_match_score"] = round(located.score, 4)
        evidence["source_quote_match_recall"] = round(ngram_recall(evidence["quote"], raw_text), 4)
        evidence.setdefault("gold_chunk_ids", {})["400_60"] = gold_ids

        if not gold_ids:
            unmatched.append((record["id"], evidence_index, "no_gold_chunk", document["id"]))

        results.append({
            "question_id": record["id"],
            "evidence_index": evidence_index,
            "doc_id": document["id"],
            "method": located.method,
            "score": round(located.score, 4),
            "gold_ids": gold_ids,
        })

    # recompute the question-level aggregate gold for 400_60 only
    agg = sorted({cid for ev in record.get("evidence", []) for cid in ev.get("gold_chunk_ids", {}).get("400_60", [])})
    record.setdefault("gold_chunk_ids", {})["400_60"] = agg

print(f"total evidence processed: {len(results)}")
print(f"unmatched: {len(unmatched)}")
for u in unmatched:
    print("  UNMATCHED:", u)

n_single = sum(1 for r in results if len(r["gold_ids"]) == 1)
n_multi = sum(1 for r in results if len(r["gold_ids"]) > 1)
n_zero = sum(1 for r in results if len(r["gold_ids"]) == 0)
print(f"single gold: {n_single}  multi gold: {n_multi}  zero gold: {n_zero}")

from collections import Counter
print("match methods:", Counter(r["method"] for r in results))

# Save the FULL records (with fresh 400_60 fields + original untouched 300_50/500_80)
# to a separate working file for now; STEP 6/7/8 will inspect/adjust before this
# becomes the final eval dataset.
out_path = Path(__file__).resolve().parent / "gold" / "gold_400_60_recomputed.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"wrote working copy -> {out_path}")
