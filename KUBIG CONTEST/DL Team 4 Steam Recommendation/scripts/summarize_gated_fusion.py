"""Combine fixed-candidate BPR results and summarize gated fusion ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-results",
        type=Path,
        default=Path("outputs/mvp_50k/bpr_seed_42/final_fixed_candidate_comparison.csv"),
    )
    parser.add_argument(
        "--gated-results",
        type=Path,
        default=Path("outputs/mvp_50k/gated_seed_42/bpr_model_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/mvp_50k/gated_seed_42")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_results)
    gated = pd.read_csv(args.gated_results)
    assert gated["model"].is_unique
    assert set(gated["evaluated_positives"]) == {9984}, "candidate population changed"

    combined = pd.concat([base, gated], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("model", keep="last")
    combined = combined.sort_values("recall@10", ascending=False).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "all_model_fixed_candidate_comparison.csv"
    combined.to_csv(comparison_path, index=False)

    lookup = combined.set_index("model")
    reference = lookup.loc["text_bpr"]
    rows = []
    for name in ["text_anchored_gated_bpr", "user_modality_gated_bpr"]:
        row = lookup.loc[name]
        rows.append(
            {
                "model": name,
                "delta_recall@10_vs_text_bpr": row["recall@10"] - reference["recall@10"],
                "delta_ndcg@10_vs_text_bpr": row["ndcg@10"] - reference["ndcg@10"],
                "delta_mrr_vs_text_bpr": row["mrr"] - reference["mrr"],
                "delta_long_tail_recall@10_vs_text_bpr": (
                    row["long_tail_recall@10"] - reference["long_tail_recall@10"]
                ),
                "delta_catalog_coverage@10_vs_text_bpr": (
                    row["catalog_coverage@10"] - reference["catalog_coverage@10"]
                ),
                "text_gate_mean": row["text_gate_mean"],
                "text_gate_p10": row["text_gate_p10"],
                "text_gate_p50": row["text_gate_p50"],
                "text_gate_p90": row["text_gate_p90"],
            }
        )
    deltas = pd.DataFrame(rows)
    deltas.to_csv(args.output_dir / "gated_fusion_deltas_vs_text.csv", index=False)

    plot_models = [
        "popularity_train_positive",
        "mf_bpr",
        "text_bpr",
        "tabular_text_fusion_bpr",
        "text_anchored_gated_bpr",
        "user_modality_gated_bpr",
    ]
    plot = lookup.loc[plot_models].reset_index()
    labels = [
        "Popularity",
        "MF-BPR",
        "Text-BPR",
        "Concat fusion",
        "Item gate",
        "User gate",
    ]
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(labels, plot["recall@10"], color=["#8c8c8c", "#4c78a8", "#59a14f", "#e15759", "#f28e2b", "#b07aa1"])
    plt.ylabel("Recall@10")
    plt.ylim(0.72, 0.82)
    plt.xticks(rotation=18, ha="right")
    for bar, value in zip(bars, plot["recall@10"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", fontsize=8)
    plt.title("Fixed-candidate BPR fusion comparison (9,984 queries)")
    plt.tight_layout()
    plt.savefig(figure_dir / "gated_fusion_recall_at_10.png", dpi=160)
    plt.close()

    gate_plot = gated.set_index("model").loc[
        ["text_anchored_gated_bpr", "user_modality_gated_bpr"]
    ]
    plt.figure(figsize=(7, 4.5))
    for name, label, color in [
        ("text_anchored_gated_bpr", "Item-level vector gate", "#f28e2b"),
        ("user_modality_gated_bpr", "User-level scalar gate", "#b07aa1"),
    ]:
        row = gate_plot.loc[name]
        plt.errorbar(
            [label],
            [row["text_gate_p50"]],
            yerr=[[row["text_gate_p50"] - row["text_gate_p10"]], [row["text_gate_p90"] - row["text_gate_p50"]]],
            fmt="o",
            capsize=7,
            color=color,
        )
    plt.ylabel("Learned text weight")
    plt.ylim(0, 1)
    plt.title("Text gate median and p10-p90 range")
    plt.tight_layout()
    plt.savefig(figure_dir / "learned_text_gate_summary.png", dpi=160)
    plt.close()

    concat = lookup.loc["tabular_text_fusion_bpr"]
    user_gate = lookup.loc["user_modality_gated_bpr"]
    summary = {
        "evaluation": {"queries": 9984, "candidates_per_query": 100, "seed": 42},
        "selected_mvp_model": "text_bpr",
        "selection_reason": "highest Recall@10 among tested content/fusion BPR variants",
        "text_bpr": reference[["recall@10", "ndcg@10", "mrr", "long_tail_recall@10", "catalog_coverage@10"]].to_dict(),
        "best_gated_variant": "user_modality_gated_bpr",
        "user_gate_vs_concat": {
            "delta_recall@10": user_gate["recall@10"] - concat["recall@10"],
            "delta_ndcg@10": user_gate["ndcg@10"] - concat["ndcg@10"],
        },
        "conclusion": "Gating mitigates naive concat degradation, but does not beat Text-BPR on this MVP split.",
    }
    (args.output_dir / "gated_fusion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(combined[["model", "recall@10", "ndcg@10", "mrr", "long_tail_recall@10", "catalog_coverage@10"]].to_string(index=False))
    print("GATED_FUSION_SUMMARY_OK")


if __name__ == "__main__":
    main()
