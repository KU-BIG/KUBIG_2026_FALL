from pathlib import Path

import pytest
from transformers import CLIPTokenizer

from src.data.dataset import (
    build_stage_a,
    build_stage_b,
    compute_stage_b_token_stats,
    load_config,
    load_dataset,
    save_validation_summary,
    validate_against_constants,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "semart_v1_modality_gap_dataset.parquet"
CONSTANTS_PATH = PROJECT_ROOT / "config" / "constants.yaml"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "results" / "step1_validation_summary.json"

BROKEN_CSV_FILENAMES = [
    "32796-31doni2.jpg",
    "07139-flowers1.jpg",
    "37694-portrai2.jpg",
    "17041-flower14.jpg",
    "04023-1hunyadi.jpg",
]


@pytest.fixture(scope="module")
def constants():
    return load_config(str(CONSTANTS_PATH))


@pytest.fixture(scope="module")
def df():
    return load_dataset(str(PARQUET_PATH))


@pytest.fixture(scope="module")
def stage_a_df(df):
    return build_stage_a(df)


@pytest.fixture(scope="module")
def stage_b_df(df):
    return build_stage_b(df)


@pytest.fixture(scope="module")
def tokenizer():
    return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


@pytest.fixture(scope="module")
def stage_b_stats(stage_b_df, tokenizer):
    stats = compute_stage_b_token_stats(stage_b_df, tokenizer)
    stats["treated_fraction"] = float(stage_b_df["treated_mask"].mean())
    save_validation_summary(stage_b_df, stats, str(SUMMARY_PATH))
    return stats


def test_row_count(df, constants):
    assert len(df) == constants["dataset"]["n_rows_expected"]


def test_broken_csv_filenames_spot_check(df):
    found = df[df["filename"].isin(BROKEN_CSV_FILENAMES)]
    assert len(found) == len(BROKEN_CSV_FILENAMES)
    for fn in BROKEN_CSV_FILENAMES:
        row = found[found["filename"] == fn].iloc[0]
        assert isinstance(row["visual_caption"], str) and len(row["visual_caption"]) > 0
        assert isinstance(row["contextual_caption"], str) and len(row["contextual_caption"]) > 0


def test_stage_a_token_stats(df, constants):
    tol = constants["tolerance"]["mean_abs"]
    sa = constants["stage_a"]
    assert abs(df["visual_token_len"].mean() - sa["visual_token_len_mean"]) <= tol
    assert abs(df["visual_token_len"].median() - sa["visual_token_len_median"]) <= tol
    assert abs(df["contextual_token_len"].mean() - sa["contextual_token_len_mean"]) <= tol
    assert abs(df["contextual_token_len"].median() - sa["contextual_token_len_median"]) <= tol


def test_stage_b_token_stats_within_tolerance(stage_b_stats, constants):
    validate_against_constants(stage_b_stats, constants)


def test_stage_b_token_bounds(stage_b_stats, constants):
    sb = constants["stage_b"]
    assert stage_b_stats["min"] == sb["visual_first_sentence_token_min"]
    assert stage_b_stats["max"] <= sb["visual_first_sentence_token_max"]


def test_treated_fraction(stage_b_df, constants):
    tol = constants["tolerance"]["fraction_abs"]
    expected = constants["stage_b"]["treated_fraction_expected"]
    actual = stage_b_df["treated_mask"].mean()
    assert abs(actual - expected) <= tol


def test_stage_a_b_contextual_text_identical(stage_a_df, stage_b_df):
    assert stage_a_df["contextual_text"].tolist() == stage_b_df["contextual_text"].tolist()
