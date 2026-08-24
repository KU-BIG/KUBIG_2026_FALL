"""Reuse eval_retrieval.py's UNMODIFIED retriever classes + metric functions to
compare metrics computed against the pre-boundary-fix gold (gold_400_60_recomputed.jsonl,
STEP5 output: single-gold only, before the 30-item multi-gold correction) vs the
final post-fix gold (gold_400_60_final.jsonl / rag_evaluation_dataset.jsonl).

Only ONE set of retrieval rankings is generated (BM25/Dense/Hybrid/Reranker are
never modified or re-implemented here); the same rankings are scored against both
gold versions using eval_retrieval.py's own evidence_recall_at_k / hit_at_k /
mrr_at_k / ndcg_at_k functions, imported unchanged.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import eval_retrieval as er  # noqa: E402

ROOT_DIR = SCRIPT_DIR.parent
CHUNKS_PATH = ROOT_DIR / "retriever_dataset" / "chunks" / "chunk_400_60" / "chunks.jsonl"
CACHE_DIR = SCRIPT_DIR / "cache"

BEFORE_GOLD_PATH = SCRIPT_DIR / "gold" / "gold_400_60_recomputed.jsonl"  # pre boundary-fix
AFTER_GOLD_PATH = ROOT_DIR / "rag_evaluation_dataset.jsonl"  # final (post boundary-fix)

SPLIT = "test"
LANG = "ko"
CANDIDATE_K = 20
FINAL_K = 5
RRF_SOURCE_K = 50
RRF_K = 60


def load_split(path):
    all_q = er.read_jsonl(path)
    return [q for q in all_q if q["split"] == SPLIT]


chunks = er.read_jsonl(CHUNKS_PATH)
chunk_by_id = {c["chunk_id"]: c for c in chunks}
valid_ids = set(chunk_by_id)

before_questions = load_split(BEFORE_GOLD_PATH)
after_questions = load_split(AFTER_GOLD_PATH)
# evidence/question content (query text, etc.) is identical between the two
# files for the test split; only gold_chunk_ids["400_60"] differs.
questions = after_questions

gold_before = er.load_gold_groups(before_questions, chunk_size="400_60", valid_chunk_ids=valid_ids)
gold_after = er.load_gold_groups(after_questions, chunk_size="400_60", valid_chunk_ids=valid_ids)

print(f"[data] {len(questions)} questions (split={SPLIT})")

print("[build] BM25")
bm25 = er.BM25Retriever(chunks)
print("[build] Dense (reusing cache if present)")
dense = er.DenseRetriever(chunks, cache_path=CACHE_DIR / "dense_emb_400_60.npy")
print("[build] Reranker")
reranker = er.CandidateReranker(chunk_by_id)

retrieval_depth = max(50, RRF_SOURCE_K, CANDIDATE_K)
hybrid_rankings = {}
reranked_rankings = {}

for i, q in enumerate(questions, start=1):
    query = q["question"][LANG]
    bm25_results = bm25.search(query, top_k=retrieval_depth)
    dense_results = dense.search(query, top_k=retrieval_depth)
    hybrid_results = er.rrf_fuse([bm25_results[:RRF_SOURCE_K], dense_results[:RRF_SOURCE_K]], k=RRF_K, top_k=retrieval_depth)
    reranked_results = reranker.rerank(query, hybrid_results[:CANDIDATE_K])
    hybrid_rankings[q["id"]] = [cid for cid, _ in hybrid_results]
    reranked_rankings[q["id"]] = [cid for cid, _ in reranked_results]
    if i % 10 == 0 or i == len(questions):
        print(f"[eval] {i}/{len(questions)}")

for label, gold in [("BEFORE (pre boundary-fix, single-gold only)", gold_before), ("AFTER (final, post boundary-fix)", gold_after)]:
    print(f"\n=== {label} ===")
    hybrid_res = er.evaluate_final_rankings("Hybrid Retrieval (RRF)", hybrid_rankings, questions, gold, final_k=FINAL_K)
    rerank_res = er.evaluate_final_rankings(er.CandidateReranker.name, reranked_rankings, questions, gold, final_k=FINAL_K)
    print(json.dumps([hybrid_res, rerank_res], ensure_ascii=False, indent=2))

out = {
    "before": {
        "hybrid": er.evaluate_final_rankings("Hybrid Retrieval (RRF)", hybrid_rankings, questions, gold_before, final_k=FINAL_K),
        "reranker": er.evaluate_final_rankings(er.CandidateReranker.name, reranked_rankings, questions, gold_before, final_k=FINAL_K),
    },
    "after": {
        "hybrid": er.evaluate_final_rankings("Hybrid Retrieval (RRF)", hybrid_rankings, questions, gold_after, final_k=FINAL_K),
        "reranker": er.evaluate_final_rankings(er.CandidateReranker.name, reranked_rankings, questions, gold_after, final_k=FINAL_K),
    },
}
out_path = SCRIPT_DIR / "results" / "results_400_60_gold_before_after_comparison.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"\n[done] -> {out_path}")
