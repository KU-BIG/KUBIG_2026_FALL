"""Aggregate three-seed MF/Text/Balanced-Hybrid reproducibility results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/mvp_50k")
SEEDS = [42, 7, 2026]
OUTPUT_DIR = ROOT / "hybrid_reproducibility"
MODELS = ["mf_bpr_rescored", "text_bpr_rescored", "mf_text_balanced_hybrid"]
METRICS = ["recall@10", "ndcg@10", "mrr", "long_tail_recall@10", "catalog_coverage@10"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    alpha_rows = []
    for seed in SEEDS:
        directory = ROOT / f"repro_seed_{seed}" / "hybrid"
        frame = pd.read_csv(directory / "hybrid_test_results.csv")
        assert set(MODELS).issubset(frame.model)
        subset = frame[frame.model.isin(MODELS)].copy()
        subset.insert(0, "seed", seed)
        rows.append(subset)
        summary = json.loads((directory / "hybrid_summary.json").read_text(encoding="utf-8"))
        alpha_rows.append(
            {
                "seed": seed,
                "accuracy_alpha_mf": summary["selected_alpha_mf"],
                "balanced_alpha_mf": summary["balanced_alpha_mf"],
                "balanced_alpha_text": summary["balanced_alpha_text"],
            }
        )
    raw = pd.concat(rows, ignore_index=True)
    assert raw.groupby("seed").model.nunique().eq(3).all()
    raw.to_csv(OUTPUT_DIR / "three_seed_raw_results.csv", index=False)
    alphas = pd.DataFrame(alpha_rows)
    alphas.to_csv(OUTPUT_DIR / "selected_hybrid_weights.csv", index=False)

    aggregate_rows = []
    for model in MODELS:
        values = raw[raw.model.eq(model)]
        row: dict[str, float | str | int] = {"model": model, "n_seeds": len(values)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(values[metric].mean())
            row[f"{metric}_std"] = float(values[metric].std(ddof=1))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUTPUT_DIR / "three_seed_mean_std.csv", index=False)

    wide = raw.pivot(index="seed", columns="model", values=METRICS)
    delta_rows = []
    for seed in SEEDS:
        row: dict[str, float | int] = {"seed": seed}
        for metric in METRICS:
            row[f"hybrid_minus_mf_{metric}"] = float(
                wide.loc[seed, (metric, "mf_text_balanced_hybrid")]
                - wide.loc[seed, (metric, "mf_bpr_rescored")]
            )
            row[f"hybrid_minus_text_{metric}"] = float(
                wide.loc[seed, (metric, "mf_text_balanced_hybrid")]
                - wide.loc[seed, (metric, "text_bpr_rescored")]
            )
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(OUTPUT_DIR / "paired_seed_deltas.csv", index=False)

    labels = {"mf_bpr_rescored": "MF-BPR", "text_bpr_rescored": "Text-BPR", "mf_text_balanced_hybrid": "Balanced hybrid"}
    colors = {"mf_bpr_rescored": "#4c78a8", "text_bpr_rescored": "#59a14f", "mf_text_balanced_hybrid": "#b07aa1"}
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for axis, metric, title in zip(axes, ["recall@10", "ndcg@10", "long_tail_recall@10"], ["Recall@10", "NDCG@10", "Long-tail Recall@10"]):
        for position, model in enumerate(MODELS):
            vals = raw.loc[raw.model.eq(model), metric]
            axis.errorbar(position, vals.mean(), yerr=vals.std(ddof=1), fmt="o", capsize=6, color=colors[model])
            axis.scatter(np.full(len(vals), position), vals, color=colors[model], alpha=0.45, s=18)
        axis.set_xticks(range(len(MODELS)), [labels[x] for x in MODELS], rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Three-seed reproducibility (mean ± sample SD; fixed test candidates)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "three_seed_comparison.png", dpi=160)
    plt.close(figure)

    summary_lookup = aggregate.set_index("model")
    ndcg_delta = deltas["hybrid_minus_mf_ndcg@10"]
    summary = {
        "seeds": SEEDS,
        "same_split_and_test_candidates": True,
        "fair_negative_sampling": "sampler reset per model so ablations share epoch-wise negatives",
        "accuracy_hybrid_selection": "alpha_mf=1.0 for all three seeds",
        "balanced_hybrid_rule": "maximize validation long-tail Recall@10 subject to >=99.5% of best validation NDCG@10",
        "balanced_alpha_mf_by_seed": dict(zip(map(str, alphas.seed), alphas.balanced_alpha_mf)),
        "mean_metrics": aggregate.set_index("model").to_dict(orient="index"),
        "hybrid_vs_mf_ndcg_delta_all_positive": bool((ndcg_delta > 0).all()),
        "hybrid_vs_mf_ndcg_delta_mean": float(ndcg_delta.mean()),
        "recommended_accuracy_model": "mf_bpr",
        "recommended_balanced_model": "mf_text_balanced_hybrid",
        "caution": "n=3 measures training-seed stability but is too small for a strong significance claim",
    }
    (OUTPUT_DIR / "three_seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    show = aggregate[["model"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")]]
    print(show.to_string(index=False))
    print("HYBRID_REPRODUCIBILITY_OK")


if __name__ == "__main__":
    main()
