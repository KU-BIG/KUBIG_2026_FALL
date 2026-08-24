"""STEP 6: boundary-aware multi-gold correction for 400/60, re-derived from
scratch against the freshly regenerated documents/chunks (does not blindly
copy the previous audit's chunk IDs -- those are used only to identify which
question/evidence pairs to re-examine).
"""
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent
RECOMPUTED_PATH = BASE / "gold" / "gold_400_60_recomputed.jsonl"
CHUNKS_PATH = BASE.parent / "retriever_dataset" / "chunks" / "chunk_400_60" / "chunks.jsonl"
AUDIT_CSV = BASE / "gold" / "gold_quality_risk_audit_400_60.csv"

records = [json.loads(l) for l in open(RECOMPUTED_PATH, encoding="utf-8")]
chunks = [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]
chunk_by_id = {c["chunk_id"]: c for c in chunks}
chunks_by_doc = {}
for c in chunks:
    chunks_by_doc.setdefault(c["doc_id"], []).append(c)
for d in chunks_by_doc:
    chunks_by_doc[d].sort(key=lambda c: c["start_char"])

import csv
audit_rows = list(csv.DictReader(open(AUDIT_CSV, encoding="utf-8")))
target_pairs = {
    (r["question_id"], int(r["evidence_index"]))
    for r in audit_rows
    if r["classification"] == "PARTIAL_MULTI_GOLD_FIXABLE"
}
print(f"Re-examining {len(target_pairs)} evidence pairs flagged by the prior audit")


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


NUM_RE = re.compile(r"\d+(?:[.,]\d+)*\s*(?:원|%|만원|건|일|개월|년|분|달러|USD|장|개|회|번)?")


def extract_facts(text):
    facts = set()
    for m in NUM_RE.finditer(text):
        t = m.group().strip()
        if t and any(ch.isdigit() for ch in t):
            facts.add(t)
    for m in re.finditer(r"[‘’'\"“”]([^‘’'\"“”]{2,30})[‘’'\"“”]", text):
        facts.add(m.group(1))
    for c in re.split(r"[.\n]", text):
        c = c.strip()
        if len(re.sub(r"\s+", "", c)) >= 6:
            facts.add(c)
    for line in text.split("\n"):
        line = line.strip()
        if len(re.sub(r"\s+", "", line)) >= 4:
            facts.add(line)
    return facts


results = []
record_by_qid = {r["id"]: r for r in records}

for qid, ev_idx in sorted(target_pairs):
    record = record_by_qid[qid]
    evidence = record["evidence"][ev_idx]
    doc_id = None
    for c in chunks_by_doc:
        pass
    gold_ids = evidence["gold_chunk_ids"]["400_60"]
    gold_chunks = [chunk_by_id[g] for g in gold_ids if g in chunk_by_id]
    if not gold_chunks:
        results.append((qid, ev_idx, "ERROR_NO_GOLD_CHUNK", None, None))
        continue
    doc_id = gold_chunks[0]["doc_id"]
    doc_chunks = chunks_by_doc[doc_id]

    source_for_facts = evidence.get("source_quote_raw") or evidence["quote"]
    facts = extract_facts(source_for_facts)

    gold_text_norm = norm(" ".join(c["text"] for c in gold_chunks))
    missing_now = [f for f in facts if norm(f) and len(norm(f)) >= 2 and norm(f) not in gold_text_norm]

    if not missing_now:
        results.append((qid, ev_idx, "CASE_A_single_ok", gold_ids, []))
        continue

    ids_in_order = [c["chunk_id"] for c in doc_chunks]
    first_idx = ids_in_order.index(gold_chunks[0]["chunk_id"])
    last_idx = ids_in_order.index(gold_chunks[-1]["chunk_id"])
    adjacent = []
    if first_idx > 0:
        adjacent.append(("prev", doc_chunks[first_idx - 1]))
    if last_idx + 1 < len(doc_chunks):
        adjacent.append(("next", doc_chunks[last_idx + 1]))

    resolved_by = []
    still_missing = list(missing_now)
    for tag, c in adjacent:
        cn = norm(c["text"])
        newly_resolved = [f for f in still_missing if norm(f) in cn]
        if newly_resolved:
            resolved_by.append(c["chunk_id"])
            still_missing = [f for f in still_missing if f not in newly_resolved]

    if not still_missing:
        new_gold = sorted(set(gold_ids) | set(resolved_by))
        evidence["gold_chunk_ids"]["400_60"] = new_gold
        results.append((qid, ev_idx, "CASE_B_fixed_multi_gold", new_gold, []))
    else:
        results.append((qid, ev_idx, "CASE_C_not_fixable", gold_ids, still_missing[:5]))

# recompute question-level aggregates for the touched questions only
touched_qids = {qid for qid, _ in target_pairs}
for qid in touched_qids:
    record = record_by_qid[qid]
    agg = sorted({cid for ev in record["evidence"] for cid in ev.get("gold_chunk_ids", {}).get("400_60", [])})
    record["gold_chunk_ids"]["400_60"] = agg

print()
print(f"{'question_id':12s} {'idx':4s} {'case':28s} gold_ids")
counts = Counter()
for qid, ev_idx, case, gold_ids, missing in results:
    counts[case] += 1
    print(f"{qid:12s} {ev_idx:<4d} {case:28s} {gold_ids} missing={missing}")

print()
print("case counts:", dict(counts))

with open(BASE / "gold" / "gold_400_60_final.jsonl", "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"wrote -> {BASE / 'gold' / 'gold_400_60_final.jsonl'}")
