"""Build compact, inference-only data files for Streamlit deployment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CATALOG = REPO_ROOT / "text_data" / "games_text_ready.csv"
DATA_DIR = REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42"
OUTPUT_DIR = REPO_ROOT / "recommendation_mvp" / "deploy_data"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "app_id",
        "title",
        "tags_kaggle",
        "tags_text",
        "rating",
        "positive_ratio",
        "user_reviews",
        "price_final",
    ]
    catalog = pd.read_csv(SOURCE_CATALOG, usecols=columns)
    assert len(catalog) == 50_872 and catalog.app_id.is_unique and catalog.title.notna().all()
    catalog_path = OUTPUT_DIR / "catalog_ui.parquet"
    catalog.to_parquet(catalog_path, index=False, compression="zstd")

    split_files = ["debug_train.parquet", "debug_validation.parquet", "debug_test.parquet"]
    histories = [
        pd.read_parquet(DATA_DIR / name, columns=["user_id", "app_id"])
        for name in split_files
    ]
    history = pd.concat(histories, ignore_index=True).drop_duplicates()
    history_path = OUTPUT_DIR / "seen_history_all.parquet"
    history.to_parquet(history_path, index=False, compression="zstd")

    train = pd.read_parquet(DATA_DIR / "debug_train.parquet", columns=["app_id", "is_recommended"])
    popularity = (
        train.loc[train.is_recommended]
        .groupby("app_id")
        .size()
        .rename("train_positive_count")
        .reset_index()
    )
    popularity_path = OUTPUT_DIR / "train_positive_counts.csv"
    popularity.to_csv(popularity_path, index=False)

    manifest = {
        "catalog_rows": len(catalog),
        "history_rows": len(history),
        "history_users": int(history.user_id.nunique()),
        "popularity_games": len(popularity),
        "source_splits": split_files,
        "files": {
            path.name: path.stat().st_size
            for path in [catalog_path, history_path, popularity_path]
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("BUILD_DEPLOYMENT_ARTIFACTS_OK")


if __name__ == "__main__":
    main()
