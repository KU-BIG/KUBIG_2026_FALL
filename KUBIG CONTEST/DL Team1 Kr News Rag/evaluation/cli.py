"""CLI for human-authored Dense versus Hybrid retrieval evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from evaluation.ai_assist import generate_question, judge_candidates
from evaluation.adjudication import run_adjudication
from evaluation.articles import ArticleCatalog
from evaluation.assignment import create_assignment, stratum_for_date
from evaluation.batch_pooling import build_batch_pool, write_pool_artifacts
from evaluation.combine import combine_annotations
from evaluation.pooling import build_pool
from evaluation.reporting import write_csv, write_json
from evaluation.runner import DEFAULT_CHUNK_DEPTH, run_evaluation
from evaluation.schema import load_records, validate_records
from indexing.build_index import DEFAULT_COLLECTION, DEFAULT_INPUT, DEFAULT_MODEL
from indexing.hybrid import DEFAULT_CANDIDATE_K, DEFAULT_RRF_K

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "processed" / "news_data_clean.json"
DEFAULT_RESULT = PROJECT_ROOT / "evaluation" / "results" / "retrieval_eval.json"
DEFAULT_BLIND_POOL = PROJECT_ROOT / "evaluation" / "pools" / "blind_pool_50.jsonl"
DEFAULT_POOL_MAPPING = PROJECT_ROOT / "evaluation" / "pools" / "blind_pool_50_mapping.json"
DEFAULT_POOL_MANIFEST = PROJECT_ROOT / "evaluation" / "pools" / "blind_pool_50_manifest.json"
DEFAULT_JUDGMENTS = PROJECT_ROOT / "evaluation" / "judgments"
DEFAULT_JUDGE_CHECKPOINT = PROJECT_ROOT / "evaluation" / ".checkpoints" / "blind_adjudication"


def _retrievers() -> dict:
    from indexing.hybrid import HybridRetriever
    from indexing.retriever import detect_device, get_retriever

    device = detect_device()
    dense = get_retriever(device=device)
    return {"dense": dense, "hybrid": HybridRetriever(dense=dense, device=device)}


def _llm():
    from generation.llm import ClaudeClient

    return ClaudeClient()


def _judge_llm():
    from generation.openai_llm import OpenAIClient

    return OpenAIClient(model="gpt-5.6-luna", max_tokens=512)


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _git_file_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(path)], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise ValueError(f"no Git commit found for {path}")
    return commit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    articles = subparsers.add_parser("articles", help="browse cleaned articles without retrieval scores")
    article_commands = articles.add_subparsers(dest="article_command", required=True)
    get = article_commands.add_parser("get")
    get.add_argument("article_id", type=int)
    search = article_commands.add_parser("search")
    search.add_argument("keyword")
    search.add_argument("--limit", type=int, default=20)
    sample = article_commands.add_parser("sample")
    sample.add_argument("--count", type=int, default=5)
    sample.add_argument("--seed", type=int, default=42)
    for command in (get, search, sample):
        command.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    validate = subparsers.add_parser("validate", help="validate an explicitly named evaluation JSONL")
    validate.add_argument("evaluation_file", type=Path)
    validate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    validate.add_argument("--allow-draft", action="store_true")

    assign = subparsers.add_parser("assign", help="create the fixed date-stratified 50-source assignment")
    assign.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    assign.add_argument("--seed", type=int, default=42)
    assign.add_argument("--output", type=Path, required=True)

    generate = subparsers.add_parser("generate-question", help="ask Claude for one draft source-seeded question")
    generate.add_argument("article_id", type=int)
    generate.add_argument("--category", required=True)
    generate.add_argument("--query-id", required=True)
    generate.add_argument("--author", required=True, choices=("kahyun", "ryeowon"))
    generate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    pool = subparsers.add_parser("pool", help="pool blind Dense and Hybrid candidates after freezing a question")
    pool.add_argument("question")
    pool.add_argument("--pool-depth", type=int, default=20)
    pool.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH)

    pool_batch = subparsers.add_parser("pool-batch", help="build atomic blind pools for a frozen evaluation JSONL")
    pool_batch.add_argument("evaluation_file", type=Path)
    pool_batch.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    pool_batch.add_argument("--chunks", type=Path, default=DEFAULT_INPUT)
    pool_batch.add_argument("--expected-freeze-sha256", required=True)
    pool_batch.add_argument("--freeze-commit")
    pool_batch.add_argument("--blind-output", type=Path, default=DEFAULT_BLIND_POOL)
    pool_batch.add_argument("--mapping-output", type=Path, default=DEFAULT_POOL_MAPPING)
    pool_batch.add_argument("--manifest-output", type=Path, default=DEFAULT_POOL_MANIFEST)
    pool_batch.add_argument("--pool-depth", type=int, default=20)
    pool_batch.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH)
    pool_batch.add_argument("--seed", type=int, default=42)

    judge = subparsers.add_parser("judge", help="ask Claude to judge a blind full-corpus candidate pool")
    judge.add_argument("question")
    judge.add_argument("--pool-depth", type=int, default=20)
    judge.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH)

    judge_blind = subparsers.add_parser("judge-blind", help="adjudicate a frozen blind packet")
    judge_blind.add_argument("blind_packet", type=Path)
    judge_blind.add_argument("--expected-sha256", required=True)
    judge_blind.add_argument("--output-dir", type=Path, default=DEFAULT_JUDGMENTS)
    judge_blind.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_JUDGE_CHECKPOINT)
    judge_blind.add_argument("--seed", type=int, default=42)

    combine = subparsers.add_parser("combine", help="validate and combine the two author annotation files")
    combine.add_argument("annotation_files", nargs=2, type=Path)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    run = subparsers.add_parser("run", help="run paired article-level evaluation")
    run.add_argument("evaluation_file", type=Path)
    run.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    run.add_argument("--chunks", type=Path, default=DEFAULT_INPUT)
    run.add_argument("--allow-draft", action="store_true")
    run.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH)
    run.add_argument("--json-output", type=Path, default=DEFAULT_RESULT)
    run.add_argument("--csv-output", type=Path)
    return parser


def main(argv: list[str] | None = None, *, retriever_factory=None, llm_factory=None) -> int:
    args = build_parser().parse_args(argv)
    factory = retriever_factory or _retrievers
    make_llm = llm_factory or _llm
    if args.command == "articles":
        catalog = ArticleCatalog.from_json(args.corpus)
        if args.article_command == "get":
            value = catalog.get(args.article_id)
        elif args.article_command == "search":
            if args.limit < 1:
                raise ValueError("limit must be at least 1")
            value = catalog.search(args.keyword)[: args.limit]
        else:
            value = catalog.sample(args.count, seed=args.seed)
        _print(value)
        return 0

    if args.command == "assign":
        with args.corpus.open(encoding="utf-8") as stream:
            assignment = create_assignment(json.load(stream), seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(assignment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
        return 0

    if args.command == "generate-question":
        source = ArticleCatalog.from_json(args.corpus).get(args.article_id)
        draft = generate_question(source, args.category, make_llm())
        draft.update({
            "query_id": args.query_id,
            "construction_method": "source_seeded",
            "date_stratum": stratum_for_date(source["date"]),
            "evidence": [{
                "article_id": source["article_id"], "title": source["title"],
                "support": draft.pop("ai_rationale"), "relevance": "relevant",
            }],
            "author": args.author, "reviewer": None,
            "annotation_minutes": None, "naturalness": None, "gold_clarity": None,
            "notes": "AI-generated draft; author must freeze question and complete self_check.",
        })
        _print(draft)
        return 0

    if args.command == "combine":
        catalog = ArticleCatalog.from_json(args.corpus)
        records = combine_annotations(args.annotation_files, args.output, catalog.article_ids)
        print(f"wrote {len(records)} records to {args.output}")
        return 0

    if args.command == "pool-batch":
        question_sha = hashlib.sha256(args.evaluation_file.read_bytes()).hexdigest()
        if question_sha.casefold() != args.expected_freeze_sha256.casefold():
            raise ValueError(f"freeze SHA-256 mismatch: expected {args.expected_freeze_sha256}, got {question_sha}")
        catalog = ArticleCatalog.from_json(args.corpus)
        records = load_records(args.evaluation_file)
        validate_records(records, catalog.article_ids, allow_draft=False)
        if len(records) != 50:
            raise ValueError(f"pool-batch requires exactly 50 records, got {len(records)}")
        packets, mapping, stats = build_batch_pool(
            records, factory(), catalog, seed=args.seed,
            pool_depth=args.pool_depth, chunk_depth=args.chunk_depth,
        )
        manifest = {
            "freeze_sha256": question_sha,
            "freeze_commit": args.freeze_commit or _git_file_commit(args.evaluation_file),
            "corpus_article_count": len(catalog.article_ids),
            "corpus_chunk_count": _count_jsonl(args.chunks),
            "collection": DEFAULT_COLLECTION,
            "embedding_model": DEFAULT_MODEL,
            "systems": ["dense", "hybrid"],
            "candidate_k": DEFAULT_CANDIDATE_K,
            "rrf_k": DEFAULT_RRF_K,
            "pool_depth": args.pool_depth,
            "chunk_retrieval_depth": args.chunk_depth,
            "query_candidate_counts": stats,
            "random_seed": args.seed,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "execution_git_commit": _git_commit(),
            "retrieval_success": {"dense": len(records), "hybrid": len(records)},
            "retrieval_error_count": 0,
            "judgment_instruction": "Do not open the mapping file during blind AI relevance judgment.",
        }
        hashes = write_pool_artifacts(
            packets, mapping, manifest,
            args.blind_output, args.mapping_output, args.manifest_output,
        )
        print(f"wrote blind pool for {len(records)} queries ({len(mapping)} candidates)")
        print(json.dumps(hashes, sort_keys=True))
        return 0

    if args.command == "judge-blind":
        judge = llm_factory() if llm_factory else _judge_llm()
        manifest, hashes = run_adjudication(
            args.blind_packet, args.output_dir, args.checkpoint_dir, judge,
            expected_sha256=args.expected_sha256, git_commit=_git_commit(), seed=args.seed,
        )
        print(
            "adjudicated 1299 blind query-candidate pairs in "
            f"{manifest['api_calls_this_invocation']} API calls this invocation"
        )
        print(json.dumps(hashes, sort_keys=True))
        return 0

    if args.command in {"pool", "judge"}:
        retrievers = factory()
        hits = {
            system: retriever.retrieve(args.question, top_k=args.chunk_depth)
            for system, retriever in retrievers.items()
        }
        candidates = build_pool(hits["dense"], hits["hybrid"], depth=args.pool_depth)
        _print(candidates if args.command == "pool" else judge_candidates(args.question, candidates, make_llm()))
        return 0

    catalog = ArticleCatalog.from_json(args.corpus)
    records = load_records(args.evaluation_file)
    validate_records(records, catalog.article_ids, allow_draft=args.allow_draft)
    if args.command == "validate":
        print(f"valid: {len(records)} records")
        return 0
    if not records:
        print("No evaluation records; no result artifacts were written.")
        return 0

    if not args.chunks.is_file():
        raise FileNotFoundError(f"chunk file not found: {args.chunks}")
    with args.corpus.open(encoding="utf-8") as stream:
        corpus_articles = json.load(stream)
    stratum_counts = {}
    for article in corpus_articles:
        label = stratum_for_date(article["date"])
        stratum_counts[label] = stratum_counts.get(label, 0) + 1
    report = run_evaluation(
        records, factory(), question_path=args.evaluation_file,
        corpus_article_count=len(catalog.article_ids), corpus_chunk_count=_count_jsonl(args.chunks),
        settings={
            "systems": ["dense", "hybrid"], "model": DEFAULT_MODEL,
            "collection": DEFAULT_COLLECTION, "candidate_k": DEFAULT_CANDIDATE_K, "rrf_k": DEFAULT_RRF_K,
        },
        git_commit=_git_commit(), chunk_depth=args.chunk_depth,
        source_article_count=50, stratum_corpus_counts=stratum_counts,
    )
    json_output = args.json_output
    csv_output = args.csv_output or json_output.with_suffix(".csv")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, json_output)
    write_csv(report, csv_output)
    print(f"wrote {json_output} and {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
