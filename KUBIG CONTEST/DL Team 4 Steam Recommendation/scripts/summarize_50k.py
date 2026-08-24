"""Combine 50k neural and popularity results and visualize the trade-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", type=Path, default=REPO_ROOT / "outputs" / "mvp_50k" / "models_seed_42"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    neural = pd.read_csv(args.result_dir / "mvp_model_comparison.csv")
    popularity = pd.read_csv(args.result_dir / "popularity_baseline.csv")
    for column in neural.columns:
        if column not in popularity.columns:
            popularity[column] = np.nan
    for column in popularity.columns:
        if column not in neural.columns:
            neural[column] = np.nan
    combined = pd.concat([popularity[neural.columns], neural], ignore_index=True)
    combined.to_csv(args.result_dir / "model_comparison_with_popularity.csv", index=False)

    order = [
        "popularity_train_positive",
        "tabular_only",
        "text_only",
        "tabular_text_fusion",
    ]
    combined = combined.set_index("model").reindex(order).reset_index()
    labels = ["Popularity", "Tabular", "Text", "Fusion"]
    figure_dir = args.result_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    for metric in ["recall@10", "ndcg@10", "long_tail_recall@10", "catalog_coverage@10"]:
        plt.figure(figsize=(7, 4))
        plt.bar(labels, combined[metric])
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(figure_dir / f"comparison_{metric.replace('@', '_at_')}.png", dpi=150)
        plt.close()

    popularity_row = combined.iloc[0]
    fusion_row = combined.loc[combined.model.eq("tabular_text_fusion")].iloc[0]
    summary = {
        "users": 50_000,
        "ranking_positives": int(fusion_row.evaluated_positives),
        "candidates_per_positive": 100,
        "popularity_vs_fusion": {
            "recall_at_10_difference": float(fusion_row["recall@10"] - popularity_row["recall@10"]),
            "ndcg_at_10_difference": float(fusion_row["ndcg@10"] - popularity_row["ndcg@10"]),
            "long_tail_recall_at_10_difference": float(
                fusion_row["long_tail_recall@10"] - popularity_row["long_tail_recall@10"]
            ),
        },
        "conclusion": (
            "Popularity dominates uniformly sampled-negative ranking, while Fusion recovers some "
            "long-tail positives. The next model must train against unobserved negatives with a "
            "personalized ranking objective such as BPR."
        ),
    }
    (args.result_dir / "50k_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(combined[[
        "model", "test_auc", "recall@10", "ndcg@10", "mrr",
        "long_tail_recall@10", "catalog_coverage@10", "unique_recommended@10",
    ]].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("SUMMARIZE_50K_OK")


if __name__ == "__main__":
    main()

