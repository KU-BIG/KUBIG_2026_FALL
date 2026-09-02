"""Step 1: Stage A/B 텍스트 구성 + constants.yaml 대조 검증.

Stage A/B 정의와 acceptance criteria는 IMPLEMENTATION_PLAN.md 5절 참고.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSTANTS_PATH = PROJECT_ROOT / "config" / "constants.yaml"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    n_expected = load_config(str(CONSTANTS_PATH))["dataset"]["n_rows_expected"]
    assert len(df) == n_expected, f"expected {n_expected} rows, got {len(df)}"
    return df


def build_stage_a(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "filename": df["filename"],
        "visual_text": df["visual_caption"],
        "contextual_text": df["contextual_caption"],
    }).reset_index(drop=True)


def build_stage_b(df: pd.DataFrame) -> pd.DataFrame:
    visual_text = df["visual_sentences_clean"].apply(lambda s: s[0])
    treated_mask = df["visual_caption"].str.strip() != visual_text.str.strip()
    return pd.DataFrame({
        "filename": df["filename"],
        "visual_text": visual_text,
        "contextual_text": df["contextual_caption"],
        "treated_mask": treated_mask,
    }).reset_index(drop=True)


def compute_stage_b_token_stats(df: pd.DataFrame, tokenizer) -> dict:
    encoded = tokenizer(df["visual_text"].tolist(), truncation=True, max_length=77)
    lengths = np.array([len(ids) for ids in encoded["input_ids"]])
    return {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
    }


def validate_against_constants(stats: dict, constants: dict) -> None:
    sb = constants["stage_b"]
    tol = constants["tolerance"]
    for stat_key, const_key in [
        ("mean", "visual_first_sentence_token_mean"),
        ("median", "visual_first_sentence_token_median"),
        ("std", "visual_first_sentence_token_std"),
    ]:
        actual, expected = stats[stat_key], sb[const_key]
        assert abs(actual - expected) <= tol["mean_abs"], (
            f"{stat_key}: expected {expected} (+/-{tol['mean_abs']}), got {actual}"
        )
    assert stats["min"] == sb["visual_first_sentence_token_min"], (
        f"min: expected {sb['visual_first_sentence_token_min']}, got {stats['min']}"
    )
    assert stats["max"] <= sb["visual_first_sentence_token_max"], (
        f"max: expected <= {sb['visual_first_sentence_token_max']}, got {stats['max']}"
    )
    if "treated_fraction" in stats:
        actual, expected = stats["treated_fraction"], sb["treated_fraction_expected"]
        assert abs(actual - expected) <= tol["fraction_abs"], (
            f"treated_fraction: expected {expected} (+/-{tol['fraction_abs']}), got {actual}"
        )


def save_validation_summary(stage_b_df: pd.DataFrame, stage_b_stats: dict, path: str) -> dict:
    """실제 계산값을 기준값과 비교하지 않고 그대로 기록만 한다 (검증은 validate_against_constants 책임)."""
    treated_count = int(stage_b_df["treated_mask"].sum())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(stage_b_df),
        "treated_count": treated_count,
        "treated_fraction": treated_count / len(stage_b_df),
        "stage_b_token_stats": {
            "mean": stage_b_stats["mean"],
            "median": stage_b_stats["median"],
            "std": stage_b_stats["std"],
            "min": stage_b_stats["min"],
            "max": stage_b_stats["max"],
        },
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
