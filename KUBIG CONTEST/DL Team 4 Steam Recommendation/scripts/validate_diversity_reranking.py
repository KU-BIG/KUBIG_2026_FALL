"""Deterministic integrity tests for MMR recommendation reranking."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.cold_start import ColdStartRecommendationPipeline  # noqa: E402
from mvp_recommendation.reranking import mmr_rerank  # noqa: E402
from scripts.evaluate_diversity_reranking import duplicate_title_pairs  # noqa: E402


def main() -> None:
    pipeline = ColdStartRecommendationPipeline(
        text_prefix=REPO_ROOT / "text_data" / "emb_text_minilm",
        catalog_path=REPO_ROOT / "text_data" / "games_text_ready.csv",
        train_path=REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42" / "debug_train.parquet",
    )
    candidates = pipeline.recommend(
        "validation_profile",
        top_k=100,
        preferred_tags=["RPG", "Open World", "Story Rich"],
        liked_app_ids=[292030],
    )
    original = candidates.head(10)
    identity = mmr_rerank(
        candidates, pipeline.text_items, pipeline.app_to_row, top_k=10, lambda_relevance=1.0
    )
    assert identity.app_id.tolist() == original.app_id.tolist()

    diverse_one = mmr_rerank(
        candidates, pipeline.text_items, pipeline.app_to_row, top_k=10, lambda_relevance=0.65
    )
    diverse_two = mmr_rerank(
        candidates, pipeline.text_items, pipeline.app_to_row, top_k=10, lambda_relevance=0.65
    )
    assert diverse_one.app_id.tolist() == diverse_two.app_id.tolist(), "MMR is not deterministic"
    assert len(diverse_one) == diverse_one.app_id.nunique() == 10
    assert diverse_one["rank"].tolist() == list(range(1, 11))
    assert diverse_one.diversity_applied.all() and diverse_one.diversity_lambda.eq(0.65).all()
    assert not diverse_one.app_id.eq(292030).any()
    assert duplicate_title_pairs(diverse_one) < duplicate_title_pairs(original)
    reconstructed_columns = {"original_rank", "rerank_score", "redundancy_penalty"}
    assert reconstructed_columns.issubset(diverse_one.columns)
    assert np.isfinite(diverse_one[list(reconstructed_columns)].to_numpy()).all()
    print(
        f"candidate pool={len(candidates)}; top_k={len(diverse_one)}; "
        f"title-overlap pairs {duplicate_title_pairs(original)}->{duplicate_title_pairs(diverse_one)}"
    )
    print("DIVERSITY_RERANKING_VALIDATION_OK")


if __name__ == "__main__":
    main()
