"""Memory-conscious data preparation for the recommendation MVP."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "user_id",
    "app_id",
    "is_recommended",
    "date",
    "hours",
    "review_id",
]


@dataclass
class InteractionSummary:
    interactions: int
    unique_users: int
    unique_games: int
    positive_count: int
    negative_count: int
    positive_ratio: float
    negative_ratio: float
    date_min: str | None
    date_max: str | None
    date_missing_or_invalid: int
    user_interactions_describe: dict[str, float]
    game_interactions_describe: dict[str, float]


def _grow(array: np.ndarray, needed: int) -> np.ndarray:
    if needed <= len(array):
        return array
    new_size = max(needed, max(1024, len(array) * 2))
    grown = np.zeros(new_size, dtype=array.dtype)
    grown[: len(array)] = array
    return grown


def _describe_nonzero(counts: np.ndarray) -> dict[str, float]:
    values = counts[counts > 0]
    if values.size == 0:
        return {}
    summary = pd.Series(values).describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    return {key: float(value) for key, value in summary.items()}


def scan_interactions(
    recommendations_csv: Path,
    valid_game_ids: np.ndarray,
    chunksize: int = 1_000_000,
) -> tuple[InteractionSummary, InteractionSummary, np.ndarray]:
    """Scan the large CSV once and return total/valid summaries and valid user counts."""
    valid_game_ids = np.sort(np.asarray(valid_game_ids, dtype=np.int64))
    total_user_counts = np.zeros(1024, dtype=np.int32)
    valid_user_counts = np.zeros(1024, dtype=np.int32)
    total_game_counts = np.zeros(1024, dtype=np.int32)
    valid_game_counts = np.zeros(max(int(valid_game_ids.max()) + 1, 1024), dtype=np.int32)

    total_rows = valid_rows = 0
    total_pos = valid_pos = 0
    total_bad_dates = valid_bad_dates = 0
    total_date_min = total_date_max = None
    valid_date_min = valid_date_max = None

    dtype = {
        "user_id": "int64",
        "app_id": "int64",
        "is_recommended": "boolean",
        "hours": "float32",
        "review_id": "int64",
    }
    reader = pd.read_csv(
        recommendations_csv,
        usecols=REQUIRED_COLUMNS,
        dtype=dtype,
        chunksize=chunksize,
    )

    for chunk_no, chunk in enumerate(reader, start=1):
        assert chunk["user_id"].ge(0).all(), "negative user_id found"
        assert chunk["app_id"].ge(0).all(), "negative app_id found"
        assert chunk["is_recommended"].notna().all(), "target NaN found"

        user_ids = chunk["user_id"].to_numpy(np.int64)
        app_ids = chunk["app_id"].to_numpy(np.int64)
        targets = chunk["is_recommended"].to_numpy(bool)
        dates = pd.to_datetime(chunk["date"], errors="coerce")
        valid_mask = np.isin(app_ids, valid_game_ids, assume_unique=False)

        total_user_counts = _grow(total_user_counts, int(user_ids.max()) + 1)
        total_game_counts = _grow(total_game_counts, int(app_ids.max()) + 1)
        total_user_counts[: np.bincount(user_ids).size] += np.bincount(user_ids).astype(np.int32)
        total_game_counts[: np.bincount(app_ids).size] += np.bincount(app_ids).astype(np.int32)

        if valid_mask.any():
            vu = user_ids[valid_mask]
            vg = app_ids[valid_mask]
            valid_user_counts = _grow(valid_user_counts, int(vu.max()) + 1)
            valid_game_counts = _grow(valid_game_counts, int(vg.max()) + 1)
            valid_user_counts[: np.bincount(vu).size] += np.bincount(vu).astype(np.int32)
            valid_game_counts[: np.bincount(vg).size] += np.bincount(vg).astype(np.int32)

        total_rows += len(chunk)
        valid_rows += int(valid_mask.sum())
        total_pos += int(targets.sum())
        valid_pos += int(targets[valid_mask].sum())
        total_bad_dates += int(dates.isna().sum())
        valid_bad_dates += int(dates[valid_mask].isna().sum())

        good_dates = dates.dropna()
        if not good_dates.empty:
            lo, hi = good_dates.min(), good_dates.max()
            total_date_min = lo if total_date_min is None else min(total_date_min, lo)
            total_date_max = hi if total_date_max is None else max(total_date_max, hi)
        valid_dates = dates[valid_mask].dropna()
        if not valid_dates.empty:
            lo, hi = valid_dates.min(), valid_dates.max()
            valid_date_min = lo if valid_date_min is None else min(valid_date_min, lo)
            valid_date_max = hi if valid_date_max is None else max(valid_date_max, hi)

        if chunk_no == 1 or chunk_no % 10 == 0:
            print(
                f"[scan] chunks={chunk_no:,} rows={total_rows:,} "
                f"valid={valid_rows:,} ({valid_rows / total_rows:.2%})",
                flush=True,
            )

    def build_summary(
        rows: int,
        positives: int,
        bad_dates: int,
        date_min: pd.Timestamp | None,
        date_max: pd.Timestamp | None,
        user_counts: np.ndarray,
        game_counts: np.ndarray,
    ) -> InteractionSummary:
        negatives = rows - positives
        return InteractionSummary(
            interactions=rows,
            unique_users=int(np.count_nonzero(user_counts)),
            unique_games=int(np.count_nonzero(game_counts)),
            positive_count=positives,
            negative_count=negatives,
            positive_ratio=positives / rows if rows else float("nan"),
            negative_ratio=negatives / rows if rows else float("nan"),
            date_min=None if date_min is None else str(date_min.date()),
            date_max=None if date_max is None else str(date_max.date()),
            date_missing_or_invalid=bad_dates,
            user_interactions_describe=_describe_nonzero(user_counts),
            game_interactions_describe=_describe_nonzero(game_counts),
        )

    return (
        build_summary(
            total_rows,
            total_pos,
            total_bad_dates,
            total_date_min,
            total_date_max,
            total_user_counts,
            total_game_counts,
        ),
        build_summary(
            valid_rows,
            valid_pos,
            valid_bad_dates,
            valid_date_min,
            valid_date_max,
            valid_user_counts,
            valid_game_counts,
        ),
        valid_user_counts,
    )


def choose_debug_users(
    valid_user_counts: np.ndarray,
    min_interactions: int,
    max_users: int,
    seed: int,
) -> np.ndarray:
    eligible = np.flatnonzero(valid_user_counts >= min_interactions).astype(np.int64)
    assert eligible.size > 0, "no eligible users after filtering"
    rng = np.random.default_rng(seed)
    if eligible.size > max_users:
        eligible = np.sort(rng.choice(eligible, size=max_users, replace=False))
    print(f"eligible users: {np.count_nonzero(valid_user_counts >= min_interactions):,}")
    print(f"selected DEBUG users: {eligible.size:,}")
    return eligible


def load_selected_interactions(
    recommendations_csv: Path,
    selected_users: np.ndarray,
    valid_game_ids: np.ndarray,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    valid_game_ids = np.sort(np.asarray(valid_game_ids, dtype=np.int64))
    selected_users = np.sort(np.asarray(selected_users, dtype=np.int64))
    for chunk in pd.read_csv(
        recommendations_csv,
        usecols=REQUIRED_COLUMNS,
        dtype={
            "user_id": "int64",
            "app_id": "int64",
            "is_recommended": "boolean",
            "hours": "float32",
            "review_id": "int64",
        },
        chunksize=chunksize,
    ):
        mask = np.isin(chunk.user_id.to_numpy(), selected_users) & np.isin(
            chunk.app_id.to_numpy(), valid_game_ids
        )
        if mask.any():
            pieces.append(chunk.loc[mask].copy())
    assert pieces, "selected users produced no interactions"
    data = pd.concat(pieces, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    before = len(data)
    data = data.dropna(subset=["date", "is_recommended"]).copy()
    print(f"selected interactions: {before:,}; dropped invalid date/target: {before-len(data):,}")
    return data


def chronological_split(data: pd.DataFrame) -> pd.DataFrame:
    """User-wise chronological split with at least one validation and test row."""
    data = data.sort_values(
        ["user_id", "date", "review_id"], kind="mergesort"
    ).reset_index(drop=True)
    sizes = data.groupby("user_id", sort=False)["user_id"].transform("size").to_numpy()
    positions = data.groupby("user_id", sort=False).cumcount().to_numpy()
    n_test = np.maximum(1, np.floor(sizes * 0.10).astype(np.int64))
    n_val = np.maximum(1, np.floor(sizes * 0.10).astype(np.int64))
    n_train = sizes - n_val - n_test
    assert np.all(n_train >= 3), "minimum interaction filter must guarantee >=3 train rows"

    split = np.full(len(data), "train", dtype=object)
    split[positions >= n_train] = "validation"
    split[positions >= (n_train + n_val)] = "test"
    data["split"] = split

    assert not data.duplicated("review_id").any(), "review_id overlap/duplicate found"
    pivot = data.groupby(["user_id", "split"])["date"].agg(["min", "max"])
    for user_id, group in pivot.groupby(level=0):
        rows = group.droplevel(0)
        assert {"train", "validation", "test"}.issubset(rows.index)
        assert rows.loc["train", "max"] <= rows.loc["validation", "min"]
        assert rows.loc["validation", "max"] <= rows.loc["test", "min"]
    return data


def split_summary(data: pd.DataFrame) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, frame in data.groupby("split", sort=False):
        result[name] = {
            "interactions": int(len(frame)),
            "users": int(frame.user_id.nunique()),
            "games": int(frame.app_id.nunique()),
            "positive_ratio": float(frame.is_recommended.mean()),
            "date_min": str(frame.date.min().date()),
            "date_max": str(frame.date.max().date()),
        }
    return result


def save_prepared_outputs(
    data: pd.DataFrame,
    total: InteractionSummary,
    valid: InteractionSummary,
    output_dir: Path,
    config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in data.groupby("split", sort=False):
        frame.to_parquet(output_dir / f"debug_{split}.parquet", index=False)
    summary = {
        "config": config,
        "all_interactions": asdict(total),
        "valid_game_interactions": asdict(valid),
        "debug_split": split_summary(data),
    }
    (output_dir / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

