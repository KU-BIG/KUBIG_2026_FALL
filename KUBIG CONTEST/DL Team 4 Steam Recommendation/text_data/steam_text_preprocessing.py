# %% [markdown]
# # Steam text preprocessing
#
# Kaggle `games.csv`와 `games_metadata.json`을 기준으로 텍스트 모델에 넣을
# game-level text field를 만듭니다.
# 이미지 전처리는 다른 파트에서 진행하므로 여기서는 `title`, `description`, `tags`
# 세 컬럼만 다룹니다.

# %% 1. Imports
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

try:
    from transformers import AutoTokenizer
except ImportError:  # tokenizer가 없어도 기본 전처리는 실행되게 둡니다.
    AutoTokenizer = None


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = BASE_DIR.parent

RAW_CANDIDATES = [
    BASE_DIR,
    PROJECT_ROOT / "data" / "raw" / "steam_recommendations",
]

OUTPUT_CSV = BASE_DIR / "games_text_ready.csv"
TOKEN_SUMMARY_CSV = BASE_DIR / "text_token_length_summary.csv"
TOKEN_SUMMARY_JSON = BASE_DIR / "text_token_length_summary.json"
OVER_256_CSV = BASE_DIR / "text_over_256_samples.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_MAX_SEQ_LENGTH = 256
TEXT_COLUMNS = [
    "text_v1_title",
    "text_v2_title_tags",
    "text_v3_title_tags_description",
]

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 120)


def section(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def find_raw_dir() -> Path:
    for path in RAW_CANDIDATES:
        if (path / "games.csv").exists() and (path / "games_metadata.json").exists():
            return path
    raise FileNotFoundError(
        "games.csv와 games_metadata.json을 찾지 못했습니다. "
        "Data_process 폴더나 data/raw/steam_recommendations 폴더에 원본 파일을 두세요."
    )


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def tags_to_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(clean_text(tag) for tag in value if clean_text(tag))
    return ""


def count_tags(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def percentile(values: list[int], q: float) -> float:
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = find_raw_dir()
    section("Load raw data")
    print("raw_dir:", raw_dir)

    games = pd.read_csv(raw_dir / "games.csv")
    metadata = pd.read_json(raw_dir / "games_metadata.json", lines=True)

    print("games:", games.shape)
    print("metadata:", metadata.shape)
    return games, metadata


def make_text_frame(games: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    section("Merge text columns")

    text_df = games.merge(
        metadata[["app_id", "description", "tags"]],
        on="app_id",
        how="left",
        validate="one_to_one",
    )

    text_df["title_clean"] = text_df["title"].map(clean_text)
    text_df["description_original"] = text_df["description"].map(clean_text)
    text_df["tags_kaggle"] = text_df["tags"]
    text_df["tags_text"] = text_df["tags"].map(tags_to_text)

    text_df["title_word_len"] = text_df["title_clean"].map(word_count)
    text_df["description_word_len"] = text_df["description_original"].map(word_count)
    text_df["tags_count"] = text_df["tags"].map(count_tags)

    text_df["text_v1_title"] = text_df["title_clean"]
    text_df["text_v2_title_tags"] = (
        text_df["title_clean"] + " " + text_df["tags_text"]
    ).map(clean_text)
    text_df["text_v3_title_tags_description"] = (
        text_df["title_clean"]
        + " "
        + text_df["tags_text"]
        + " "
        + text_df["description_original"]
    ).map(clean_text)

    text_df["text_for_embedding"] = text_df["text_v3_title_tags_description"]
    text_df["text_source"] = "title+tags+description"
    text_df.loc[text_df["description_original"].eq(""), "text_source"] = "title+tags"
    text_df.loc[text_df["tags_text"].eq("") & text_df["description_original"].eq(""), "text_source"] = "title"

    print("merged:", text_df.shape)
    print("description empty:", int(text_df["description_original"].eq("").sum()))
    print("tags empty:", int(text_df["tags_text"].eq("").sum()))
    return text_df


def token_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=True, truncation=False))


def summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    return {
        "count": len(lengths),
        "min": min(lengths),
        "mean": mean(lengths),
        "p50": percentile(lengths, 0.50),
        "p75": percentile(lengths, 0.75),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "over_256_count": sum(length > SBERT_MAX_SEQ_LENGTH for length in lengths),
        "over_256_ratio": sum(length > SBERT_MAX_SEQ_LENGTH for length in lengths) / len(lengths),
    }


def add_token_lengths(text_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    if AutoTokenizer is None:
        print("transformers가 없어 token length는 계산하지 않습니다.")
        text_df["token_len_minilm"] = pd.NA
        text_df["needs_truncation_256"] = pd.NA
        return text_df, {}

    section("Token length check")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    summary = {}
    for column in TEXT_COLUMNS:
        lengths = [token_length(tokenizer, text) for text in text_df[column].tolist()]
        summary[column] = summarize_lengths(lengths)

        if column == "text_v3_title_tags_description":
            text_df["token_len_minilm"] = lengths
            text_df["needs_truncation_256"] = text_df["token_len_minilm"].gt(SBERT_MAX_SEQ_LENGTH)

    print(pd.DataFrame(summary).T[["mean", "p95", "p99", "max", "over_256_count", "over_256_ratio"]])
    return text_df, summary


def save_outputs(text_df: pd.DataFrame, summary: dict[str, dict[str, float | int]]) -> None:
    section("Save outputs")

    keep_columns = [
        "app_id",
        "title",
        "date_release",
        "win",
        "mac",
        "linux",
        "rating",
        "positive_ratio",
        "user_reviews",
        "price_final",
        "price_original",
        "discount",
        "steam_deck",
        "title_clean",
        "description_original",
        "tags_kaggle",
        "tags_text",
        "title_word_len",
        "description_word_len",
        "tags_count",
        "text_source",
        "text_v1_title",
        "text_v2_title_tags",
        "text_v3_title_tags_description",
        "text_for_embedding",
        "token_len_minilm",
        "needs_truncation_256",
    ]

    text_df[keep_columns].to_csv(OUTPUT_CSV, index=False)
    print("saved:", OUTPUT_CSV)

    if summary:
        rows = []
        for column, stats in summary.items():
            row = {"text_column": column}
            row.update(stats)
            rows.append(row)
        pd.DataFrame(rows).to_csv(TOKEN_SUMMARY_CSV, index=False)
        TOKEN_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        over_256 = text_df.loc[
            text_df["needs_truncation_256"],
            ["app_id", "title_clean", "token_len_minilm", "text_for_embedding"],
        ]
        over_256.head(50).to_csv(OVER_256_CSV, index=False)
        print("token summary:", TOKEN_SUMMARY_CSV)
        print("over 256:", len(over_256))


def main() -> None:
    games, metadata = load_data()
    text_df = make_text_frame(games, metadata)
    text_df, summary = add_token_lengths(text_df)
    save_outputs(text_df, summary)


if __name__ == "__main__":
    main()
