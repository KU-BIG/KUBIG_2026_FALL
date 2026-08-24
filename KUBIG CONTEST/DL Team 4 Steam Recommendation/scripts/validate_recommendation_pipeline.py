"""Integrity checks for the final known-user recommendation artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.inference import KnownUserRecommendationPipeline  # noqa: E402


def main() -> None:
    sample_path = REPO_ROOT / "recommendation_mvp" / "sample_recommendations.csv"
    manifest_path = sample_path.with_suffix(".manifest.json")
    sample = pd.read_csv(sample_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert tuple(sample.shape) == tuple(manifest["shape"]) == (90, sample.shape[1])
    assert sample.user_id.nunique() == 3 and sample.model.nunique() == 3
    assert sample.groupby(["user_id", "model"]).size().eq(10).all()
    assert sample.groupby(["user_id", "model"]).app_id.nunique().eq(10).all()
    assert sample.groupby(["user_id", "model"]).apply(
        lambda x: x.sort_values("rank").score.is_monotonic_decreasing,
        include_groups=False,
    ).all()
    assert sample.title.notna().all()
    numeric = sample.select_dtypes(include="number")
    assert np.isfinite(numeric.to_numpy()).all()

    history = pd.concat(
        [
            pd.read_parquet(
                REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42" / name,
                columns=["user_id", "app_id"],
            )
            for name in ["debug_train.parquet", "debug_validation.parquet", "debug_test.parquet"]
        ],
        ignore_index=True,
    ).drop_duplicates()
    overlap = sample[["user_id", "app_id"]].merge(history, on=["user_id", "app_id"], how="inner")
    assert overlap.empty, f"recommendations include {len(overlap)} observed user-game pairs"

    hybrid = sample[sample.model.eq("balanced_hybrid")]
    reconstructed = hybrid.alpha_mf * hybrid.mf_score_z + hybrid.alpha_text * hybrid.text_score_z
    assert np.allclose(hybrid.score, reconstructed, rtol=0, atol=1e-10)
    assert hybrid.alpha_mf.eq(0.2).all() and hybrid.alpha_text.eq(0.8).all()

    pipeline = KnownUserRecommendationPipeline(
        checkpoint_dir=REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "checkpoints",
        hybrid_summary_path=REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "hybrid" / "hybrid_summary.json",
        text_prefix=REPO_ROOT / "text_data" / "emb_text_minilm",
        tabular_prefix=REPO_ROOT / "tabular_embedding" / "leakage_safe" / "emb_tabular_safe_svd64",
        catalog_path=REPO_ROOT / "text_data" / "games_text_ready.csv",
        data_dir=REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42",
        device="cpu",
        history_scope="all",
    )
    assert len(pipeline.catalog) == 50_872
    assert len(pipeline.user_to_idx) == 49_742
    try:
        pipeline.recommend([-1], top_k=1, models=["balanced_hybrid"])
    except KeyError as error:
        assert "unknown user_id" in str(error)
    else:
        raise AssertionError("unknown user must not silently receive known-user recommendations")

    print(
        f"sample={sample.shape}; users={sample.user_id.nunique()}; "
        f"observed overlap={len(overlap)}; catalog={len(pipeline.catalog):,}"
    )
    print("RECOMMENDATION_PIPELINE_VALIDATION_OK")


if __name__ == "__main__":
    main()
