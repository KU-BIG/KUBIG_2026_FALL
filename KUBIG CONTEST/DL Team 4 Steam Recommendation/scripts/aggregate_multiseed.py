"""Aggregate leakage-safe DEBUG recommendation results across user/model seeds."""

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
METRICS = ["test_auc", "recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "mvp_safe")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for seed in args.seeds:
        path = args.root / f"seed_{seed}" / "mvp_model_comparison.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    all_results = pd.concat(frames, ignore_index=True)
    assert all_results.groupby("seed").model.nunique().eq(3).all()
    all_results.to_csv(args.root / "multiseed_all_results.csv", index=False)

    grouped = all_results.groupby("model")[METRICS].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(args.root / "multiseed_summary.csv", index=False)

    paired_rows = []
    for seed, frame in all_results.groupby("seed"):
        indexed = frame.set_index("model")
        fusion = indexed.loc["tabular_text_fusion"]
        for metric in ["recall@10", "ndcg@10"]:
            unimodal = indexed.loc[["tabular_only", "text_only"], metric]
            baseline_model = str(unimodal.idxmax())
            baseline = float(unimodal.max())
            fused = float(fusion[metric])
            paired_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "best_unimodal_model": baseline_model,
                    "best_unimodal": baseline,
                    "fusion": fused,
                    "absolute_difference": fused - baseline,
                    "relative_improvement": (fused - baseline) / baseline if baseline else np.nan,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.root / "multiseed_fusion_improvements.csv", index=False)
    improvement_summary = paired.groupby("metric")[["absolute_difference", "relative_improvement"]].agg(
        ["mean", "std"]
    )
    improvement_summary.columns = [f"{metric}_{stat}" for metric, stat in improvement_summary.columns]
    improvement_summary.reset_index().to_csv(
        args.root / "multiseed_fusion_improvement_summary.csv", index=False
    )

    figure_dir = args.root / "figures"
    figure_dir.mkdir(exist_ok=True)
    for metric in ["recall@10", "ndcg@10"]:
        order = ["tabular_only", "text_only", "tabular_text_fusion"]
        means = all_results.groupby("model")[metric].mean().reindex(order)
        stds = all_results.groupby("model")[metric].std().reindex(order)
        plt.figure(figsize=(7, 4))
        plt.bar(order, means, yerr=stds, capsize=5)
        plt.ylabel(f"{metric} (mean ± SD)")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(figure_dir / f"multiseed_{metric.replace('@', '_at_')}.png", dpi=150)
        plt.close()

    report = {
        "seeds": args.seeds,
        "models": grouped.to_dict(orient="records"),
        "fusion_improvements": improvement_summary.reset_index().to_dict(orient="records"),
        "interpretation": (
            "Report absolute and relative paired improvements together; relative gains can look large "
            "when the unimodal baseline is small."
        ),
    }
    (args.root / "multiseed_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(grouped.to_string(index=False))
    print("\nPaired fusion improvements")
    print(paired.to_string(index=False))
    print("\nImprovement summary")
    print(improvement_summary.to_string())
    print("AGGREGATE_MULTISEED_OK")


if __name__ == "__main__":
    main()

