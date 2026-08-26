"""Query each document's per-DPI ColPali index with the pilot questions and write top-5 results."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent.parent
from colpali_backend import embed_queries, score_multi_vector  # noqa: E402

SAMPLES_PATH = ROOT / "data" / "full_samples.json"
EMB_ROOT = ROOT / "embeddings"
OUTPUT_ROOT = ROOT / "output"
DPIS = [144, 72, 36]
TOP_K = 5


def main():
    samples = json.loads(SAMPLES_PATH.read_text())

    by_doc = defaultdict(list)
    for s in samples:
        by_doc[s["doc_id"]].append(s)

    # Query embeddings are text-only and independent of DPI, so compute once and reuse.
    t0 = time.time()
    questions = [s["question"] for s in samples]
    query_embeddings = embed_queries(questions, batch_size=8)
    query_by_qid = {s["question_id"]: query_embeddings[i] for i, s in enumerate(samples)}
    print(f"Embedded {len(questions)} queries once in {time.time() - t0:.1f}s (reused across all DPIs)")

    doc_page_counts = {}

    for dpi in DPIS:
        t_dpi = time.time()
        results = []
        for doc_id, doc_samples in by_doc.items():
            index_path = EMB_ROOT / str(dpi) / f"{doc_id}.pt"
            cached = torch.load(index_path)
            page_numbers = cached["page_numbers"]
            doc_embeddings = list(cached["embeddings"])
            doc_page_counts[doc_id] = len(page_numbers)

            qids = [s["question_id"] for s in doc_samples]
            q_embeds = [query_by_qid[qid] for qid in qids]

            scores = score_multi_vector(q_embeds, doc_embeddings)  # (n_questions, n_pages)

            for row_idx, s in enumerate(doc_samples):
                row_scores = scores[row_idx]
                top_idx = torch.topk(row_scores, k=min(TOP_K, len(page_numbers))).indices.tolist()
                top_pages = [page_numbers[i] for i in top_idx]
                top_scores = [round(float(row_scores[i]), 4) for i in top_idx]
                results.append(
                    {
                        "question_id": s["question_id"],
                        "doc_id": s["doc_id"],
                        "question": s["question"],
                        "top_pages": top_pages,
                        "scores": top_scores,
                    }
                )

        results.sort(key=lambda r: r["question_id"])

        assert len(results) == len(samples)
        for r in results:
            assert len(r["top_pages"]) == TOP_K
            n_pages = doc_page_counts[r["doc_id"]]
            assert all(1 <= p <= n_pages for p in r["top_pages"]), (
                f"page out of range for {r['doc_id']}: {r['top_pages']} (n_pages={n_pages})"
            )

        out_path = OUTPUT_ROOT / f"retrieval_vision_{dpi}.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"dpi={dpi}: wrote {len(results)} results to {out_path} ({time.time() - t_dpi:.1f}s)")

    # Self sanity check: recall@5 against evidence_pages (informal, official scoring is done by C).
    recall_hits = {dpi: 0 for dpi in DPIS}
    for dpi in DPIS:
        result_by_qid = {r["question_id"]: r for r in json.loads((OUTPUT_ROOT / f"retrieval_vision_{dpi}.json").read_text())}
        for s in samples:
            evidence = set(s["evidence_pages"])
            top_pages = set(result_by_qid[s["question_id"]]["top_pages"])
            if evidence & top_pages:
                recall_hits[dpi] += 1
    for dpi in DPIS:
        print(
            f"[sanity check] dpi={dpi}: recall@5 (any evidence page hit) = "
            f"{recall_hits[dpi]}/{len(samples)} = {recall_hits[dpi]/len(samples):.2%}"
        )


if __name__ == "__main__":
    main()
