"""Deterministically unblind frozen judgments and compute paired retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from evaluation.batch_pooling import validate_pool_artifacts
from evaluation.metrics import score_ranking

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5")
LABELS = ("relevant", "not_relevant", "uncertain")


def _mean(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _key(row: dict) -> tuple[str, str]:
    return row["query_id"], row["candidate_key"]


def _unique_index(rows: list[dict], name: str) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate {name} key: {key}")
        indexed[key] = row
    return indexed


def _ranking(rows: list[dict], system: str) -> list[int]:
    rank_field = f"{system}_rank"
    ranked = sorted((row for row in rows if row[rank_field] is not None), key=lambda row: row[rank_field])
    ranks = [row[rank_field] for row in ranked]
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError(f"non-contiguous {system} ranks")
    article_ids = [row["article_id"] for row in ranked]
    if len(article_ids) != len(set(article_ids)):
        raise ValueError(f"duplicate {system} article ranking")
    return article_ids


def _summary(query_rows: list[dict]) -> dict:
    systems = {}
    for system in ("dense", "hybrid"):
        systems[system] = {metric: _mean([row[f"{system}_{metric}"] for row in query_rows]) for metric in METRICS}
    differences = {}
    for metric in METRICS:
        dense, hybrid = systems["dense"][metric], systems["hybrid"][metric]
        differences[metric] = {
            "absolute_hybrid_minus_dense": hybrid - dense,
            "relative_hybrid_vs_dense": ((hybrid - dense) / dense) if dense else None,
        }
    return {"question_count": len(query_rows), "dense": systems["dense"], "hybrid": systems["hybrid"],
            "differences": differences}


def _outcome(dense: float | int, hybrid: float | int) -> str:
    return "hybrid_win" if hybrid > dense else "dense_win" if dense > hybrid else "tie"


def build_final_evaluation(records: list[dict], mappings: list[dict], judgments: list[dict]) -> dict:
    """Join frozen keys, expand human gold with LLM-relevant labels, and score both systems."""
    mapping_by_key = _unique_index(mappings, "mapping")
    judgment_by_key = _unique_index(judgments, "judgment")
    if set(mapping_by_key) != set(judgment_by_key):
        raise ValueError("mapping and judgment key sets differ")
    records_by_id = {row["query_id"]: row for row in records}
    if len(records_by_id) != len(records):
        raise ValueError("duplicate query_id")
    if {key[0] for key in mapping_by_key} != set(records_by_id):
        raise ValueError("mapping and frozen query sets differ")

    merged_rows, query_rows = [], []
    for query_id in sorted(records_by_id):
        record = records_by_id[query_id]
        human_gold = set(record["gold_article_ids"])
        query_mapping = [row for row in mappings if row["query_id"] == query_id]
        article_ids = [row["article_id"] for row in query_mapping]
        if len(article_ids) != len(set(article_ids)):
            raise ValueError(f"duplicate mapped article for {query_id}")
        llm_relevant = {
            row["article_id"] for row in query_mapping
            if judgment_by_key[_key(row)]["final_label"] == "relevant"
        }
        final_gold = human_gold | llm_relevant
        for mapping in query_mapping:
            judgment = judgment_by_key[_key(mapping)]
            label = judgment["final_label"]
            if label not in LABELS:
                raise ValueError(f"invalid final label: {label}")
            merged_rows.append({
                **mapping,
                "final_label": label,
                "human_gold": mapping["article_id"] in human_gold,
                "included_in_final_gold": mapping["article_id"] in final_gold,
            })

        rankings = {system: _ranking(query_mapping, system) for system in ("dense", "hybrid")}
        scores = {system: score_ranking(ranking, final_gold) for system, ranking in rankings.items()}
        query_row = {
            "query_id": query_id, "question": record["question"], "category": record["category"],
            "date_stratum": record["date_stratum"], "human_gold_article_ids": sorted(human_gold),
            "final_gold_article_ids": sorted(final_gold),
            "dense_candidate_count": len(rankings["dense"]),
            "hybrid_candidate_count": len(rankings["hybrid"]),
            "pooled_candidate_count": len(query_mapping),
            "dense_article_ranking": rankings["dense"], "hybrid_article_ranking": rankings["hybrid"],
        }
        for system in ("dense", "hybrid"):
            for metric, value in scores[system].items():
                query_row[f"{system}_{metric}"] = value
        for metric in METRICS:
            query_row[f"{metric}_outcome"] = _outcome(
                query_row[f"dense_{metric}"], query_row[f"hybrid_{metric}"])
        query_rows.append(query_row)

    overall = _summary(query_rows)
    by_date = {value: _summary([row for row in query_rows if row["date_stratum"] == value])
               for value in sorted({row["date_stratum"] for row in query_rows})}
    by_category = {value: _summary([row for row in query_rows if row["category"] == value])
                   for value in sorted({row["category"] for row in query_rows})}
    wins = {metric: dict(Counter(row[f"{metric}_outcome"] for row in query_rows)) for metric in METRICS}
    labels = Counter(row["final_label"] for row in merged_rows)
    return {
        "merged_rows": merged_rows,
        "query_rows": query_rows,
        "metrics": {"overall": overall, "by_date_stratum": by_date, "by_category": by_category,
                    "paired_outcomes": wins},
        "diagnostics": {"uncertain_count": labels["uncertain"],
                        "final_label_counts": {label: labels[label] for label in LABELS}},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def _label_counts(rows: list[dict], field: str) -> dict[str, int]:
    counts = Counter(row[field] for row in rows)
    return {label: counts[label] for label in LABELS}


def _transition_table(pass1: list[dict], review: list[dict]) -> tuple[dict, int]:
    first = {_key(row): row["label"] for row in pass1}
    table = {label: {target: 0 for target in LABELS} for label in LABELS}
    changed = 0
    for row in review:
        before, after = first[_key(row)], row["label"]
        table[before][after] += 1
        changed += before != after
    return table, changed


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate_artifacts(output_dir: Path) -> dict[str, Path]:
    freeze = ROOT / "evaluation/retrieval_eval_50.jsonl"
    blind = ROOT / "evaluation/pools/blind_pool_50.jsonl"
    mapping_path = ROOT / "evaluation/pools/blind_pool_50_mapping.json"
    pool_manifest_path = ROOT / "evaluation/pools/blind_pool_50_manifest.json"
    pass1_path = ROOT / "evaluation/judgments/blind_judgments_pass1.jsonl"
    review_path = ROOT / "evaluation/judgments/blind_judgments_review.jsonl"
    final_path = ROOT / "evaluation/judgments/blind_judgments_final.jsonl"
    judgment_manifest_path = ROOT / "evaluation/judgments/blind_judgments_manifest.json"
    annotations = [ROOT / "evaluation/annotations/kahyun_25.jsonl",
                   ROOT / "evaluation/annotations/ryeowon_25.jsonl"]

    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    judgment_manifest = json.loads(judgment_manifest_path.read_text(encoding="utf-8"))
    expected = {
        freeze: pool_manifest["freeze_sha256"],
        blind: pool_manifest["generated_file_sha256"]["blind_pool_sha256"],
        mapping_path: pool_manifest["generated_file_sha256"]["mapping_sha256"],
        final_path: judgment_manifest["generated_file_sha256"]["final_sha256"],
    }
    for path, digest in expected.items():
        if _sha256(path) != digest:
            raise ValueError(f"SHA-256 mismatch: {path}")
    if judgment_manifest["blind_packet_sha256"] != _sha256(blind):
        raise ValueError("judgment manifest blind packet mismatch")

    records = _jsonl(freeze)
    packets = _jsonl(blind)
    mappings = json.loads(mapping_path.read_text(encoding="utf-8"))
    validate_pool_artifacts(packets, mappings)
    final = _jsonl(final_path)
    pass1, review = _jsonl(pass1_path), _jsonl(review_path)
    result = build_final_evaluation(records, mappings, final)
    merged, queries = result["merged_rows"], result["query_rows"]

    human_overlap = [row for row in merged if row["human_gold"]]
    agreement_counts = Counter(row["final_label"] for row in human_overlap)
    agreement = agreement_counts["relevant"] / len(human_overlap) if human_overlap else None
    review_keys = {_key(row) for row in review}
    review_pass1 = [row for row in pass1 if _key(row) in review_keys]
    transition, changed = _transition_table(pass1, review)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "unblinded": output_dir / "retrieval_eval_50_unblinded.jsonl",
        "metrics": output_dir / "retrieval_eval_50_metrics.json",
        "queries": output_dir / "retrieval_eval_50_query_comparison.csv",
        "manifest": output_dir / "retrieval_eval_50_manifest.json",
    }
    paths["unblinded"].write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in merged),
        encoding="utf-8",
    )
    _write_json(paths["metrics"], result["metrics"])
    fields = list(queries[0])
    with paths["queries"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in queries:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(value, list) else value for key, value in row.items()})

    source_hashes = {
        "freeze_sha256": _sha256(freeze), "blind_pool_sha256": _sha256(blind),
        "mapping_sha256": _sha256(mapping_path), "pool_manifest_sha256": _sha256(pool_manifest_path),
        "annotation_kahyun_sha256": _sha256(annotations[0]),
        "annotation_ryeowon_sha256": _sha256(annotations[1]),
        "pass1_sha256": _sha256(pass1_path), "review_sha256": _sha256(review_path),
        "final_judgment_sha256": _sha256(final_path),
        "judgment_manifest_sha256": _sha256(judgment_manifest_path),
    }
    manifest = {
        "execution_git_commit": _git_head(),
        "source_sha256": source_hashes,
        "rules": {
            "metrics": list(METRICS), "article_cutoff": 5, "candidate_pool_depth": 20,
            "aggregation": "unweighted mean across the same 50 paired queries",
            "tie_handling": "equal per-query metric values are ties",
            "gold_merge": "human gold union final LLM relevant judgments",
            "uncertain_handling": "retained in diagnostics and excluded from final gold",
            "confidence_interval": None, "statistical_test": None,
            "reason_no_inference": "No confidence interval, test, bootstrap seed, or iteration count was predefined.",
        },
        "integrity": {
            "query_count": len(queries), "mapping_row_count": len(mappings),
            "judgment_row_count": len(final), "merged_row_count": len(merged),
            "unique_candidate_key_count": len({_key(row) for row in mappings}),
            "missing_judgment_count": 0, "missing_mapping_count": 0,
            "duplicate_mapping_key_count": 0, "duplicate_judgment_key_count": 0,
            "duplicate_article_within_query_count": 0,
            "system_candidate_counts": {
                system: sum(row[f"{system}_candidate_count"] for row in queries)
                for system in ("dense", "hybrid")
            },
            "query_candidate_count_min_max": {
                system: [min(row[f"{system}_candidate_count"] for row in queries),
                         max(row[f"{system}_candidate_count"] for row in queries)]
                for system in ("dense", "hybrid")
            },
            "metric_input_key_set_matches_final_judgment": True,
            "forbidden_blind_leakage_fields_present": False,
            "freeze_unchanged": True,
        },
        "judgment_diagnostics": {
            "pass1_label_counts": _label_counts(pass1, "label"),
            "review_target_pass1_label_counts": _label_counts(review_pass1, "label"),
            "review_label_counts": _label_counts(review, "label"),
            "review_changed_count": changed, "review_transition_table": transition,
            "final_label_counts": _label_counts(final, "final_label"),
            "human_gold_and_llm_overlap_count": len(human_overlap),
            "human_gold_llm_label_counts": {label: agreement_counts[label] for label in LABELS},
            "human_gold_llm_relevant_agreement_rate": agreement,
            "agreement_statistic": None,
            "agreement_statistic_note": "No additional agreement statistic was predefined; same-model review is not human validation.",
        },
        "metrics": result["metrics"],
        "major_failure_queries": {
            "both_miss_at_5": [row["query_id"] for row in queries
                               if not row["dense_hit_at_5"] and not row["hybrid_hit_at_5"]],
            "dense_only_hit_at_5": [row["query_id"] for row in queries
                                    if row["dense_hit_at_5"] and not row["hybrid_hit_at_5"]],
            "hybrid_only_hit_at_5": [row["query_id"] for row in queries
                                     if row["hybrid_hit_at_5"] and not row["dense_hit_at_5"]],
        },
        "limitations": [
            "Questions were AI-assisted and self-checked by one annotator without independent cross-review.",
            "Pass1 and review used the same model, so review is not independent human validation.",
            "Uncertain judgments are not treated as relevant and remain unresolved diagnostic cases.",
            "The sample contains 50 source-seeded queries; subgroup results are exploratory.",
            "No inferential procedure was predefined, so confidence intervals and p-values are intentionally omitted.",
        ],
    }
    artifact_hashes = {f"{name}_sha256": _sha256(path) for name, path in paths.items() if name != "manifest"}
    manifest["generated_file_sha256"] = artifact_hashes
    _write_json(paths["manifest"], manifest)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results")
    args = parser.parse_args(argv)
    paths = generate_artifacts(args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
