"""Validate explicit-preference and no-preference cold-start artifacts."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.cold_start import ColdStartRecommendationPipeline  # noqa: E402


def main() -> None:
    preference_path = REPO_ROOT / "recommendation_mvp" / "sample_new_user_preferences.csv"
    fallback_path = REPO_ROOT / "recommendation_mvp" / "sample_new_user_popularity.csv"
    preference = pd.read_csv(preference_path)
    fallback = pd.read_csv(fallback_path)
    for path, frame in [(preference_path, preference), (fallback_path, fallback)]:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert tuple(frame.shape) == tuple(manifest["shape"])
        assert len(frame) == frame.app_id.nunique() == 10
        assert frame.score.is_monotonic_decreasing
        assert frame.title.notna().all()

    assert preference.cold_start_method.eq("minilm_preferences_plus_train_popularity").all()
    assert preference.content_weight.eq(0.85).all()
    reconstructed = (
        preference.content_weight * preference.content_score_z
        + preference.popularity_weight * preference.popularity_score_z
    )
    assert np.allclose(preference.score, reconstructed, rtol=0, atol=1e-10)
    assert not preference.app_id.eq(292030).any(), "liked seed game leaked into recommendations"

    catalog = pd.read_csv(
        REPO_ROOT / "text_data" / "games_text_ready.csv", usecols=["app_id", "tags_kaggle"]
    ).set_index("app_id")
    preferred = {"RPG", "Open World", "Story Rich"}
    for app_id in preference.app_id:
        tags = set(ast.literal_eval(catalog.at[int(app_id), "tags_kaggle"]))
        assert tags & preferred, f"app_id={app_id} violates explicit tag eligibility"

    assert fallback.cold_start_method.eq("train_positive_popularity_fallback").all()
    assert fallback.content_weight.eq(0).all() and fallback.popularity_weight.eq(1).all()
    assert fallback.content_score_z.isna().all()
    assert fallback.train_positive_count.is_monotonic_decreasing
    train = pd.read_parquet(
        REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42" / "debug_train.parquet",
        columns=["app_id", "is_recommended"],
    )
    counts = train.loc[train.is_recommended].groupby("app_id").size()
    expected_counts = fallback.app_id.map(counts).fillna(0).to_numpy(np.int64)
    assert np.array_equal(expected_counts, fallback.train_positive_count.to_numpy(np.int64))

    pipeline = ColdStartRecommendationPipeline(
        text_prefix=REPO_ROOT / "text_data" / "emb_text_minilm",
        catalog_path=REPO_ROOT / "text_data" / "games_text_ready.csv",
        train_path=REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42" / "debug_train.parquet",
    )
    tags = pipeline.available_tags()
    assert len(tags) == 441 and tags.tag.is_unique
    try:
        pipeline.recommend("typo", preferred_tags=["Open Wrold"], top_k=1)
    except ValueError as error:
        assert "Open World" in str(error)
    else:
        raise AssertionError("invalid tag must not be silently ignored")

    print(
        f"preference sample={preference.shape}; fallback sample={fallback.shape}; "
        f"available tags={len(tags)}; seed leakage=0"
    )
    print("COLD_START_PIPELINE_VALIDATION_OK")


if __name__ == "__main__":
    main()
