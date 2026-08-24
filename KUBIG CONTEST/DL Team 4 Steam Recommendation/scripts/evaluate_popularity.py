"""Evaluate a train-only game-popularity baseline on fixed ranking candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.metrics import ranking_diagnostics, one_positive_ranking_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-catalog-size", type=int, default=50_872)
    return parser.parse_args()


def deterministic_tie_break(app_ids: np.ndarray) -> np.ndarray:
    """Tiny deterministic app_id hash to avoid optimistic ties for unseen games."""
    values = app_ids.astype(np.uint64)
    hashed = (values * np.uint64(11400714819323198485)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return hashed.astype(np.float64) / np.float64(2**64)


def main() -> None:
    args = parse_args()
    train = pd.read_parquet(args.data_dir / "debug_train.parquet")
    candidates = pd.read_parquet(args.candidates)
    assert candidates.groupby("group_id").size().nunique() == 1
    group_size = int(candidates.groupby("group_id").size().iloc[0])
    assert candidates.groupby("group_id").target.sum().eq(1).all()

    positive_counts = train.loc[train.is_recommended].groupby("app_id").size()
    popularity = candidates.app_id.map(positive_counts).fillna(0).to_numpy(np.float64)
    # log1p preserves ordering; epsilon only breaks equal-count ties reproducibly.
    scores = np.log1p(popularity) + deterministic_tie_break(
        candidates.app_id.to_numpy(np.int64)
    ) * 1e-9
    metrics = one_positive_ranking_metrics(scores, group_size)
    metrics.update(
        ranking_diagnostics(
            candidates,
            scores,
            positive_counts,
            valid_catalog_size=args.valid_catalog_size,
            k=10,
        )
    )
    result = {"model": "popularity_train_positive", **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("EVALUATE_POPULARITY_OK")


if __name__ == "__main__":
    main()
