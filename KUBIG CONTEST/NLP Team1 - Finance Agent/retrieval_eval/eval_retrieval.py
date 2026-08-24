"""Evaluate retrieval candidate recall separately from reranking quality.

Retrieval stage (BM25, Dense, Hybrid): Recall@20 is the primary metric.
Reranking stage (Hybrid vs Hybrid+Reranker): MRR@5 and nDCG@5 are the
primary metrics. All comparisons use explicit gold labels stored in the
evaluation dataset.

Usage:
    python eval_retrieval.py --all-chunk-sizes --split validation
    python eval_retrieval.py --chunk-size 400_60 --split validation
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CHUNKS_DIR = ROOT_DIR / "retriever_dataset" / "chunks"
EVAL_DATASET_PATH = ROOT_DIR / "rag_evaluation_dataset.jsonl"
CACHE_DIR = SCRIPT_DIR / "cache"
DEFAULT_REPORT_PATH = SCRIPT_DIR / "reports" / "initial_method_chunk_comparison.md"

CHUNK_VARIANTS = ("300_50", "400_60", "500_80")
CHUNK_LABELS = {
    "300_50": "300/50 (기본값)",
    "400_60": "400/60",
    "500_80": "500/80",
}
HANGUL_RE = re.compile(r"[가-힣]+")
LATIN_NUM_RE = re.compile(r"[A-Za-z0-9]+")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for match in HANGUL_RE.finditer(text):
        word = match.group()
        tokens.append(word)
        if len(word) > 2:
            tokens.extend(word[index : index + 2] for index in range(len(word) - 1))
    tokens.extend(match.group() for match in LATIN_NUM_RE.finditer(text))
    return tokens


class BM25Retriever:
    name = "BM25"

    def __init__(self, chunks: Sequence[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.index = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        return [(self.chunk_ids[index], float(scores[index])) for index in ranked]


class DenseRetriever:
    name = "Dense Retrieval (BGE-M3)"

    def __init__(
        self,
        chunks: Sequence[dict],
        *,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        cache_path: Path | None = None,
        model=None,
    ) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.model = model or SentenceTransformer(model_name)
        if cache_path is not None and cache_path.exists():
            embeddings = np.load(cache_path)
            if embeddings.shape[0] != len(chunks):
                raise ValueError(
                    f"Dense cache row mismatch: {cache_path} has {embeddings.shape[0]}, "
                    f"expected {len(chunks)}. Delete the stale cache and rerun."
                )
            self.embeddings = embeddings
        else:
            texts = [chunk["text"] for chunk in chunks]
            self.embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.embeddings)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        import numpy as np

        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding
        ranked = np.argsort(-scores)[:top_k]
        return [(self.chunk_ids[index], float(scores[index])) for index in ranked]


def rrf_fuse(
    rank_lists: Sequence[Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for items in rank_lists:
        for rank, (chunk_id, _) in enumerate(items, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


class CandidateReranker:
    name = "Hybrid + Reranker (bge-reranker-v2-m3)"

    def __init__(
        self,
        chunk_by_id: dict[str, dict],
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model=None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.chunk_by_id = chunk_by_id
        self.model = model or CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, self.chunk_by_id[chunk_id]["text"]) for chunk_id, _ in candidates]
        scores = self.model.predict(pairs)
        return sorted(
            ((chunk_id, float(score)) for (chunk_id, _), score in zip(candidates, scores, strict=True)),
            key=lambda item: item[1],
            reverse=True,
        )


def load_gold_groups(
    questions: Sequence[dict],
    *,
    chunk_size: str,
    valid_chunk_ids: set[str],
) -> dict[str, list[set[str]]]:
    gold: dict[str, list[set[str]]] = {}
    for question in questions:
        groups: list[set[str]] = []
        for evidence in question.get("evidence", []):
            ids = set(evidence.get("gold_chunk_ids", {}).get(chunk_size, []))
            if not ids:
                raise ValueError(
                    f"Missing evidence gold labels: {question['id']} ({chunk_size})"
                )
            missing = ids - valid_chunk_ids
            if missing:
                raise ValueError(
                    f"Unknown gold chunk IDs for {question['id']} ({chunk_size}): {sorted(missing)}"
                )
            groups.append(ids)
        if not groups:
            raise ValueError(f"Question has no evidence groups: {question['id']}")
        gold[question["id"]] = groups
    return gold


def evidence_recall_at_k(
    retrieved_ids: Sequence[str],
    gold_groups: Sequence[set[str]],
    k: int,
) -> float:
    top = set(retrieved_ids[:k])
    return sum(bool(top & group) for group in gold_groups) / len(gold_groups)


def hit_at_k(
    retrieved_ids: Sequence[str],
    gold_groups: Sequence[set[str]],
    k: int,
) -> float:
    top = set(retrieved_ids[:k])
    return float(any(top & group for group in gold_groups))


def mrr_at_k(
    retrieved_ids: Sequence[str],
    gold_groups: Sequence[set[str]],
    k: int,
) -> float:
    union = set().union(*gold_groups)
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in union:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    gold_groups: Sequence[set[str]],
    k: int,
) -> float:
    # IDs inside one evidence group are overlapping alternatives, not separate
    # relevance items. Dynamic programming finds the best one-to-one assignment
    # between ranked chunks and evidence groups, so alternatives cannot inflate
    # either DCG or the ideal ranking.
    scores_by_mask = {0: 0.0}
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        discount = 1.0 / math.log2(rank + 1)
        next_scores = dict(scores_by_mask)
        matching_groups = [
            index for index, group in enumerate(gold_groups) if chunk_id in group
        ]
        for mask, score in scores_by_mask.items():
            for group_index in matching_groups:
                bit = 1 << group_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_scores[next_mask] = max(
                    next_scores.get(next_mask, 0.0),
                    score + discount,
                )
        scores_by_mask = next_scores
    dcg = max(scores_by_mask.values(), default=0.0)
    ideal_hits = min(len(gold_groups), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval_rankings(
    method: str,
    rankings: dict[str, list[str]],
    questions: Sequence[dict],
    gold: dict[str, list[set[str]]],
    *,
    candidate_k: int,
    diagnostic_k: int,
) -> dict:
    recall_candidate: list[float] = []
    hit_candidate: list[float] = []
    mrr_candidate: list[float] = []
    recall_diagnostic: list[float] = []
    for question in questions:
        retrieved = rankings[question["id"]]
        groups = gold[question["id"]]
        recall_candidate.append(evidence_recall_at_k(retrieved, groups, candidate_k))
        hit_candidate.append(hit_at_k(retrieved, groups, candidate_k))
        mrr_candidate.append(mrr_at_k(retrieved, groups, candidate_k))
        recall_diagnostic.append(evidence_recall_at_k(retrieved, groups, diagnostic_k))
    return {
        "method": method,
        "n_questions": len(questions),
        f"Recall@{candidate_k}": mean(recall_candidate),
        f"Hit@{candidate_k}": mean(hit_candidate),
        f"MRR@{candidate_k}": mean(mrr_candidate),
        f"Recall@{diagnostic_k}": mean(recall_diagnostic),
    }


def evaluate_final_rankings(
    method: str,
    rankings: dict[str, list[str]],
    questions: Sequence[dict],
    gold: dict[str, list[set[str]]],
    *,
    final_k: int,
) -> dict:
    recalls: list[float] = []
    hits: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    for question in questions:
        retrieved = rankings[question["id"]]
        groups = gold[question["id"]]
        recalls.append(evidence_recall_at_k(retrieved, groups, final_k))
        hits.append(hit_at_k(retrieved, groups, final_k))
        mrrs.append(mrr_at_k(retrieved, groups, final_k))
        ndcgs.append(ndcg_at_k(retrieved, groups, final_k))
    return {
        "method": method,
        "n_questions": len(questions),
        f"Recall@{final_k}": mean(recalls),
        f"Hit@{final_k}": mean(hits),
        f"MRR@{final_k}": mean(mrrs),
        f"nDCG@{final_k}": mean(ndcgs),
    }


def evaluate_chunk_size(
    chunk_size: str,
    questions: Sequence[dict],
    *,
    language: str,
    candidate_k: int,
    diagnostic_k: int,
    final_k: int,
    rrf_source_k: int,
    rrf_k: int,
    no_cache: bool,
    dense_model=None,
    reranker_model=None,
) -> tuple[dict, object, object]:
    chunks_path = CHUNKS_DIR / f"chunk_{chunk_size}" / "chunks.jsonl"
    chunks = read_jsonl(chunks_path)
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    gold = load_gold_groups(
        questions,
        chunk_size=chunk_size,
        valid_chunk_ids=set(chunk_by_id),
    )

    print(f"[build] {chunk_size}: BM25 index ({len(chunks)} chunks)")
    bm25 = BM25Retriever(chunks)
    cache_path = None if no_cache else CACHE_DIR / f"dense_emb_{chunk_size}.npy"
    print(f"[build] {chunk_size}: Dense index")
    dense = DenseRetriever(chunks, cache_path=cache_path, model=dense_model)
    print(f"[build] {chunk_size}: Reranker")
    reranker = CandidateReranker(chunk_by_id, model=reranker_model)

    retrieval_depth = max(diagnostic_k, rrf_source_k, candidate_k)
    bm25_rankings: dict[str, list[str]] = {}
    dense_rankings: dict[str, list[str]] = {}
    hybrid_rankings: dict[str, list[str]] = {}
    reranked_rankings: dict[str, list[str]] = {}

    print(f"[eval] {chunk_size}: {len(questions)} questions")
    for index, question in enumerate(questions, start=1):
        query = question["question"][language]
        bm25_results = bm25.search(query, top_k=retrieval_depth)
        dense_results = dense.search(query, top_k=retrieval_depth)
        hybrid_results = rrf_fuse(
            [bm25_results[:rrf_source_k], dense_results[:rrf_source_k]],
            k=rrf_k,
            top_k=retrieval_depth,
        )
        reranked_results = reranker.rerank(query, hybrid_results[:candidate_k])

        question_id = question["id"]
        bm25_rankings[question_id] = [chunk_id for chunk_id, _ in bm25_results]
        dense_rankings[question_id] = [chunk_id for chunk_id, _ in dense_results]
        hybrid_rankings[question_id] = [chunk_id for chunk_id, _ in hybrid_results]
        reranked_rankings[question_id] = [chunk_id for chunk_id, _ in reranked_results]
        if index % 10 == 0 or index == len(questions):
            print(f"[eval] {chunk_size}: {index}/{len(questions)}")

    retrieval_results = [
        evaluate_retrieval_rankings(
            "BM25",
            bm25_rankings,
            questions,
            gold,
            candidate_k=candidate_k,
            diagnostic_k=diagnostic_k,
        ),
        evaluate_retrieval_rankings(
            "Dense Retrieval (BGE-M3)",
            dense_rankings,
            questions,
            gold,
            candidate_k=candidate_k,
            diagnostic_k=diagnostic_k,
        ),
        evaluate_retrieval_rankings(
            "Hybrid Retrieval (RRF)",
            hybrid_rankings,
            questions,
            gold,
            candidate_k=candidate_k,
            diagnostic_k=diagnostic_k,
        ),
    ]
    final_results = [
        evaluate_final_rankings(
            "Hybrid Retrieval (RRF)",
            hybrid_rankings,
            questions,
            gold,
            final_k=final_k,
        ),
        evaluate_final_rankings(
            CandidateReranker.name,
            reranked_rankings,
            questions,
            gold,
            final_k=final_k,
        ),
    ]
    result = {
        "chunk_size": chunk_size,
        "n_chunks": len(chunks),
        "n_questions": len(questions),
        "candidate_k": candidate_k,
        "diagnostic_k": diagnostic_k,
        "final_k": final_k,
        "retrieval_results": retrieval_results,
        "final_ranking_results": final_results,
    }
    return result, dense.model, reranker.model


def _best_values(rows: Sequence[dict], keys: Sequence[str]) -> dict[str, float]:
    return {key: max(row[key] for row in rows) for key in keys}


def _format_metric(value: float, *, best: float) -> str:
    formatted = f"{value:.3f}"
    return f"**{formatted}**" if math.isclose(value, best, abs_tol=1e-12) else formatted


def markdown_chunk_section(result: dict, section_number: int) -> str:
    candidate_k = result["candidate_k"]
    diagnostic_k = result["diagnostic_k"]
    final_k = result["final_k"]
    retrieval_rows = result["retrieval_results"]
    final_rows = result["final_ranking_results"]

    retrieval_keys = [
        f"Recall@{candidate_k}",
        f"Hit@{candidate_k}",
        f"MRR@{candidate_k}",
        f"Recall@{diagnostic_k}",
    ]
    final_keys = [f"Recall@{final_k}", f"Hit@{final_k}", f"MRR@{final_k}", f"nDCG@{final_k}"]
    retrieval_best = _best_values(retrieval_rows, retrieval_keys)
    final_best = _best_values(final_rows, final_keys)

    lines = [
        f"### 5.{section_number} Chunk {CHUNK_LABELS[result['chunk_size']]}",
        "",
        f"Validation **{result['n_questions']}개 전부**에서 gold chunk를 확보했습니다 "
        f"(chunk {result['n_chunks']}개).",
        "",
        "**Rerank 전 후보 확보 성능**",
        "",
        f"| Method | Recall@{candidate_k} | Hit@{candidate_k} | MRR@{candidate_k} | Recall@{diagnostic_k} | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in retrieval_rows:
        metrics = [
            _format_metric(row[key], best=retrieval_best[key]) for key in retrieval_keys
        ]
        lines.append(
            f"| {row['method']} | {' | '.join(metrics)} | {row['n_questions']} |"
        )

    lines.extend(
        [
            "",
            "**Rerank 후 최종 정렬 성능**",
            "",
            f"| Method | Recall@{final_k} | Hit@{final_k} | MRR@{final_k} | nDCG@{final_k} | N |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        metrics = [_format_metric(row[key], best=final_best[key]) for key in final_keys]
        lines.append(
            f"| {row['method']} | {' | '.join(metrics)} | {row['n_questions']} |"
        )
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    results: Sequence[dict],
    *,
    split: str,
    language: str,
) -> None:
    sections = [
        "# Retrieval 및 Reranker 평가",
        "",
        f"- 평가 split: `{split}`",
        f"- 질문 언어: `{language}`",
        "- 후보 확보: BM25 / Dense / Hybrid의 Recall@20 중심 비교",
        "- 최종 정렬: Hybrid와 Hybrid+Reranker의 MRR@5·nDCG@5 비교",
        "- Recall은 evidence별 대체 gold chunk 중 하나 이상을 회수한 비율",
        "",
        "## 5. 결과",
        "",
    ]
    for index, result in enumerate(results, start=1):
        sections.append(markdown_chunk_section(result, index))
        sections.append("")
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--chunk-size", choices=CHUNK_VARIANTS)
    group.add_argument(
        "--all-chunk-sizes",
        action="store_true",
        help="Evaluate 300/50, 400/60, and 500/80 and write one report.",
    )
    parser.add_argument("--lang", choices=["ko", "en"], default="ko")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--diagnostic-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--rrf-source-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_k <= 0 or args.diagnostic_k < args.candidate_k or args.final_k <= 0:
        raise SystemExit("Require 0 < final_k, 0 < candidate_k <= diagnostic_k")
    chunk_sizes = CHUNK_VARIANTS if args.all_chunk_sizes or args.chunk_size is None else (args.chunk_size,)

    all_questions = read_jsonl(EVAL_DATASET_PATH)
    questions = [question for question in all_questions if question["split"] == args.split]
    if not questions:
        raise SystemExit(f"No questions found for split={args.split}")
    print(f"[data] split={args.split}, questions={len(questions)}, language={args.lang}")

    results: list[dict] = []
    dense_model = None
    reranker_model = None
    for chunk_size in chunk_sizes:
        result, dense_model, reranker_model = evaluate_chunk_size(
            chunk_size,
            questions,
            language=args.lang,
            candidate_k=args.candidate_k,
            diagnostic_k=args.diagnostic_k,
            final_k=args.final_k,
            rrf_source_k=args.rrf_source_k,
            rrf_k=args.rrf_k,
            no_cache=args.no_cache,
            dense_model=dense_model,
            reranker_model=reranker_model,
        )
        result["split"] = args.split
        result["language"] = args.lang
        out_path = SCRIPT_DIR / "results" / f"results_{chunk_size}_{args.lang}_{args.split}.json"
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[done] result -> {out_path}")
        results.append(result)

    write_markdown_report(args.report, results, split=args.split, language=args.lang)
    print(f"[done] markdown report -> {args.report}")


if __name__ == "__main__":
    main()
