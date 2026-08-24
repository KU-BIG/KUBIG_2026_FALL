"""Build a conservative leakage-safe 64D tabular representation.

Only relatively static catalog metadata is retained. Review/popularity/playtime,
price/discount, Metacritic, owner estimates, achievement and DLC counts are
excluded because their current snapshot can contain information from after a
historical recommendation interaction.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize


REPO_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
BOOLEAN_COLUMNS = ["win", "mac", "linux", "steam_deck"]
MULTI_VALUE_CONFIG = {
    "genres": {"min_frequency": 10, "max_features": 100},
    "categories": {"min_frequency": 10, "max_features": 150},
    "supported_languages_hf": {"min_frequency": 10, "max_features": 100},
    "full_audio_languages_hf": {"min_frequency": 10, "max_features": 100},
    "developers_hf": {"min_frequency": 5, "max_features": 500},
    "publishers_hf": {"min_frequency": 5, "max_features": 500},
}
EXCLUDED_TEMPORAL_COLUMNS = [
    "rating",
    "positive_ratio",
    "user_reviews",
    "price_final",
    "price_original",
    "discount",
    "positive_hf",
    "negative_hf",
    "recommendations_hf",
    "average_playtime_forever_hf",
    "median_playtime_forever_hf",
    "metacritic_score_hf",
    "estimated_owners_hf",
    "peak_ccu_hf",
    "achievements_hf",
    "dlc_count_hf",
    "has_playtime_hf",
    "has_metacritic_hf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "Data_process" / "games_metadata_enriched.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tabular_embedding" / "leakage_safe",
    )
    parser.add_argument("--svd-dim", type=int, default=64)
    return parser.parse_args()


def parse_multi_value(value: Any) -> list[str]:
    if value is None or value is pd.NA:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        text = str(value).strip()
        if text.casefold() in {"", "nan", "none", "null", "[]"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                raw_items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [text]
            except (ValueError, SyntaxError, TypeError):
                raw_items = text.split(",")
        else:
            raw_items = text.split(",")
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        token = re.sub(r"\s+", " ", str(item)).strip()
        key = token.casefold()
        if token and key not in {"nan", "none", "null"} and key not in seen:
            seen.add(key)
            result.append(token)
    return result


def make_vocabulary(token_lists: list[list[str]], min_frequency: int, max_features: int) -> list[str]:
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for tokens in token_lists:
        for token in tokens:
            key = token.casefold()
            counts[key] += 1
            display.setdefault(key, token)
    eligible = [(key, count) for key, count in counts.items() if count >= min_frequency]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [display[key] for key, _ in eligible[:max_features]]


def build_numeric(frame: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str], dict[str, object]]:
    date = pd.to_datetime(frame["date_release"], errors="coerce")
    month = date.dt.month.astype(float)
    numeric = pd.DataFrame(
        {
            "required_age_hf": pd.to_numeric(frame["required_age_hf"], errors="coerce"),
            "release_year": date.dt.year.astype(float),
            "release_month_sin": np.sin(2 * np.pi * month / 12.0),
            "release_month_cos": np.cos(2 * np.pi * month / 12.0),
        },
        index=frame.index,
    )
    missing = numeric.isna().astype(np.float32)
    medians = numeric.median().fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.fillna(medians)).astype(np.float32)
    matrix = np.hstack([scaled, missing.to_numpy(np.float32)])
    names = [f"numeric__{c}" for c in numeric.columns] + [f"missing__{c}" for c in numeric.columns]
    state = {
        "medians": {key: float(value) for key, value in medians.items()},
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
    }
    return sparse.csr_matrix(matrix), names, state


def build_boolean(frame: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str]]:
    values = np.column_stack(
        [frame[column].fillna(False).astype(bool).astype(np.float32) for column in BOOLEAN_COLUMNS]
    )
    return sparse.csr_matrix(values), [f"bool__{column}" for column in BOOLEAN_COLUMNS]


def build_multi_value(
    frame: pd.DataFrame,
) -> tuple[sparse.csr_matrix, list[str], dict[str, list[str]], dict[str, dict[str, int]]]:
    matrices: list[sparse.csr_matrix] = []
    names: list[str] = []
    vocabularies: dict[str, list[str]] = {}
    coverage: dict[str, dict[str, int]] = {}
    for column, config in MULTI_VALUE_CONFIG.items():
        token_lists = [parse_multi_value(value) for value in frame[column]]
        vocabulary = make_vocabulary(token_lists, **config)
        lookup = {token.casefold(): idx for idx, token in enumerate(vocabulary)}
        unknown_idx, other_idx = len(vocabulary), len(vocabulary) + 1
        rows: list[int] = []
        cols: list[int] = []
        for row_idx, tokens in enumerate(token_lists):
            if not tokens:
                rows.append(row_idx)
                cols.append(unknown_idx)
                continue
            selected = {lookup[t.casefold()] for t in tokens if t.casefold() in lookup}
            if any(t.casefold() not in lookup for t in tokens):
                selected.add(other_idx)
            for col_idx in selected:
                rows.append(row_idx)
                cols.append(col_idx)
        matrix = sparse.csr_matrix(
            (np.ones(len(rows), np.float32), (np.asarray(rows), np.asarray(cols))),
            shape=(len(frame), len(vocabulary) + 2),
        )
        matrices.append(matrix)
        names.extend(
            [f"{column}__{token}" for token in vocabulary]
            + [f"{column}____UNKNOWN__", f"{column}____OTHER__"]
        )
        vocabularies[column] = vocabulary
        coverage[column] = {
            "rows_with_value": int(sum(bool(tokens) for tokens in token_lists)),
            "unknown_rows": int(matrix[:, unknown_idx].sum()),
            "other_rows": int(matrix[:, other_idx].sum()),
            "vocabulary_size": len(vocabulary),
        }
    return sparse.hstack(matrices, format="csr"), names, vocabularies, coverage


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input)
    required = ["app_id", "date_release", "required_age_hf"] + BOOLEAN_COLUMNS + list(MULTI_VALUE_CONFIG)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"required columns missing: {missing}")
    assert len(frame) == 50_872
    assert frame.app_id.notna().all() and frame.app_id.is_unique

    numeric, numeric_names, numeric_state = build_numeric(frame)
    boolean, boolean_names = build_boolean(frame)
    multi, multi_names, vocabularies, coverage = build_multi_value(frame)
    features = sparse.hstack([numeric, boolean, multi], format="csr", dtype=np.float32)
    feature_names = numeric_names + boolean_names + multi_names
    assert features.shape == (len(frame), len(feature_names))
    assert np.isfinite(features.data).all()

    svd = TruncatedSVD(n_components=args.svd_dim, random_state=RANDOM_STATE)
    embedding = svd.fit_transform(features).astype(np.float32)
    embedding = normalize(embedding, norm="l2", axis=1).astype(np.float32)
    norms = np.linalg.norm(embedding, axis=1)
    assert embedding.shape == (50_872, args.svd_dim)
    assert np.isfinite(embedding).all()
    assert np.allclose(norms, 1.0, atol=1e-5)

    prefix = args.output_dir / "emb_tabular_safe_svd64"
    sparse.save_npz(args.output_dir / "features_safe.npz", features, compressed=True)
    np.save(prefix.with_suffix(".npy"), embedding, allow_pickle=False)
    pd.DataFrame({"app_id": frame.app_id.astype("int64")}).to_csv(prefix.with_suffix(".csv"), index=False)
    schema = {
        "profile": "leakage_safe_v1",
        "source_file": args.input.name,
        "row_count": len(frame),
        "input_dimension": features.shape[1],
        "svd_dimension": args.svd_dim,
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "matrix_dtype": str(features.dtype),
        "matrix_nnz": int(features.nnz),
        "retained": {
            "numeric": ["required_age_hf", "release_year", "release_month_sin", "release_month_cos"],
            "boolean": BOOLEAN_COLUMNS,
            "multi_value": list(MULTI_VALUE_CONFIG),
        },
        "excluded_temporal_or_outcome_proxy_columns": EXCLUDED_TEMPORAL_COLUMNS,
        "feature_names": feature_names,
        "numeric_state": numeric_state,
        "vocabularies": vocabularies,
        "coverage": coverage,
        "notes": [
            "title, description and tags remain excluded to avoid overlap with the text modality",
            "current price and discount are excluded because they are time-varying snapshots",
            "achievement and DLC counts are excluded because they can grow after release",
        ],
    }
    (args.output_dir / "feature_schema_safe.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reloaded = np.load(prefix.with_suffix(".npy"), allow_pickle=False)
    ids = pd.read_csv(prefix.with_suffix(".csv"))
    assert reloaded.shape == embedding.shape and reloaded.dtype == np.float32
    assert np.array_equal(ids.app_id.to_numpy(np.int64), frame.app_id.to_numpy(np.int64))
    print(f"safe features: {features.shape}, nnz={features.nnz:,}")
    print(f"safe embedding: {embedding.shape}, dtype={embedding.dtype}")
    print(f"explained variance: {svd.explained_variance_ratio_.sum():.6f}")
    print(f"norm range: {norms.min():.6f} ~ {norms.max():.6f}")
    print(f"saved prefix: {prefix}")
    print("BUILD_LEAKAGE_SAFE_TABULAR_OK")


if __name__ == "__main__":
    main()

