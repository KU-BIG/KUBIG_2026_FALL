"""Create the final fixed-candidate comparison across baseline/objectives."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "outputs" / "mvp_50k" / "bpr_seed_42"


def align(frames: list[pd.DataFrame]) -> pd.DataFrame:
    columns: list[str] = []
    for frame in frames:
        for column in frame.columns:
            if column not in columns:
                columns.append(column)
    return pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)


def main() -> None:
    popularity = pd.read_csv(RESULT_DIR / "popularity_bpr_users.csv")
    pointwise = pd.read_csv(RESULT_DIR / "pointwise_bpr_users.csv")
    bpr = pd.read_csv(RESULT_DIR / "bpr_model_comparison.csv")
    combined = align([popularity, pointwise, bpr])
    combined.to_csv(RESULT_DIR / "final_fixed_candidate_comparison.csv", index=False)

    label_map = {
        "popularity_train_positive": "Popularity",
        "tabular_only_pointwise": "Tabular-PW",
        "text_only_pointwise": "Text-PW",
        "tabular_text_fusion_pointwise": "Fusion-PW",
        "mf_bpr": "MF-BPR",
        "tabular_bpr": "Tabular-BPR",
        "text_bpr": "Text-BPR",
        "tabular_text_fusion_bpr": "Fusion-BPR",
    }
    order = list(label_map)
    plot_frame = combined.set_index("model").reindex(order).reset_index()
    labels = [label_map[model] for model in order]
    figure_dir = RESULT_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)

    colors = ["#6B7280"] + ["#93C5FD"] * 3 + ["#2563EB"] * 4
    for metric in ["recall@10", "ndcg@10", "mrr", "long_tail_recall@10", "catalog_coverage@10"]:
        plt.figure(figsize=(10, 4.5))
        plt.bar(labels, plot_frame[metric], color=colors)
        plt.ylabel(metric)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(figure_dir / f"final_{metric.replace('@', '_at_')}.png", dpi=150)
        plt.close()

    indexed = combined.set_index("model")
    pop = indexed.loc["popularity_train_positive"]
    mf = indexed.loc["mf_bpr"]
    text = indexed.loc["text_bpr"]
    fusion_bpr = indexed.loc["tabular_text_fusion_bpr"]
    fusion_pw = indexed.loc["tabular_text_fusion_pointwise"]
    summary = {
        "evaluation": {
            "users_with_train_positive": 49_742,
            "fixed_queries": int(mf.evaluated_positives),
            "candidates_per_query": 100,
            "long_tail_train_positive_threshold": float(mf.long_tail_popularity_threshold),
            "long_tail_queries": int(mf.long_tail_positive_count),
        },
        "mf_bpr_vs_popularity": {
            "recall_at_10_absolute": float(mf["recall@10"] - pop["recall@10"]),
            "ndcg_at_10_absolute": float(mf["ndcg@10"] - pop["ndcg@10"]),
            "mrr_absolute": float(mf["mrr"] - pop["mrr"]),
        },
        "text_bpr_vs_popularity": {
            "recall_at_10_absolute": float(text["recall@10"] - pop["recall@10"]),
            "ndcg_at_10_absolute": float(text["ndcg@10"] - pop["ndcg@10"]),
            "long_tail_recall_at_10": float(text["long_tail_recall@10"]),
        },
        "fusion_findings": {
            "fusion_bpr_minus_text_bpr_recall_at_10": float(
                fusion_bpr["recall@10"] - text["recall@10"]
            ),
            "fusion_pointwise_long_tail_recall_at_10": float(fusion_pw["long_tail_recall@10"]),
            "fusion_bpr_long_tail_recall_at_10": float(fusion_bpr["long_tail_recall@10"]),
        },
        "conclusion": [
            "Pairwise training closes the large gap to Popularity.",
            "MF-BPR slightly exceeds Popularity on overall sampled ranking metrics but has zero long-tail recall.",
            "Text-BPR nearly matches overall Popularity and recovers long-tail positives.",
            "Simple concatenation Fusion-BPR underperforms Text-BPR; fusion requires redesign or regularization.",
            "Pointwise Fusion retains the highest long-tail recall but is weak on overall ranking.",
        ],
        "caveat": (
            "All BPR validation losses were still improving at epoch 10, so these are fixed-budget MVP results, "
            "not fully converged hyperparameter optima."
        ),
    }
    (RESULT_DIR / "bpr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(combined[[
        "model", "test_auc", "recall@10", "ndcg@10", "mrr",
        "long_tail_recall@10", "catalog_coverage@10", "unique_recommended@10",
    ]].sort_values("recall@10", ascending=False).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("SUMMARIZE_BPR_OK")


if __name__ == "__main__":
    main()

