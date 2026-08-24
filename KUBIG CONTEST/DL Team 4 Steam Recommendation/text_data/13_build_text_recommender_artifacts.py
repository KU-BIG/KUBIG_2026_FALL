"""
13_build_text_recommender_artifacts.py - 실제 추천용 text-only artifact 생성

검증 단계에서 가장 좋았던 log1p(hours) weighted mean 방식을 실제 추천 엔진이
바로 읽을 수 있는 파일로 저장한다.

생성 파일:
- item_embeddings_text_norm.npy: L2 normalize된 게임 text embedding
- item_catalog.csv: app_id, title, tags, train popularity, cold flag 등 게임 정보
- user_vectors_hours_weighted.npy: user별 취향 벡터
- user_vectors_index.csv: user_id와 user vector row 매핑
- user_train_seen.csv: 이미 본 게임 목록. 추천 시 제외용
- user_train_positive_history.csv: positive history와 hours. 추천 이유 생성용
- artifact_summary.json / README.md: 생성 설정과 요약
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BASELINE_SCRIPT = BASE_DIR / "11_text_recommendation_baseline.py"
OUT_DIR = BASE_DIR / "text_recommender_artifacts"

spec = importlib.util.spec_from_file_location("text_baseline", BASELINE_SCRIPT)
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return (vector / max(float(np.linalg.norm(vector)), 1e-12)).astype(np.float32)


def join_ids(app_ids: set[int]) -> str:
    return " ".join(str(app_id) for app_id in sorted(app_ids))


def first_scan_train_counts(
    chunksize: int,
) -> tuple[pd.Series, pd.Series, pd.Series, set[int], int]:
    train_user_count_chunks: list[pd.Series] = []
    item_train_count_chunks: list[pd.Series] = []
    item_train_positive_count_chunks: list[pd.Series] = []
    rec_app_ids: set[int] = set()
    total_rows = 0

    usecols = ["app_id", "date", "is_recommended", "user_id"]
    for step, chunk in enumerate(
        pd.read_csv(baseline.RECOMMENDATIONS_CSV, usecols=usecols, chunksize=chunksize),
        start=1,
    ):
        total_rows += len(chunk)
        rec_app_ids.update(chunk["app_id"].astype(int).unique().tolist())

        chunk["date"] = chunk["date"].astype(str)
        train_chunk = chunk.loc[chunk["date"].lt(baseline.TRAIN_END)].copy()
        if not train_chunk.empty:
            train_user_count_chunks.append(train_chunk["user_id"].value_counts())
            item_train_count_chunks.append(train_chunk["app_id"].value_counts())

            train_chunk["is_positive"] = baseline.str_to_bool(train_chunk["is_recommended"])
            positive_chunk = train_chunk.loc[train_chunk["is_positive"]]
            if not positive_chunk.empty:
                item_train_positive_count_chunks.append(positive_chunk["app_id"].value_counts())

        if step == 1 or step % 5 == 0:
            print(f"[scan1] chunks={step:,}, rows={total_rows:,}", flush=True)

    def combine_counts(chunks: list[pd.Series]) -> pd.Series:
        if not chunks:
            return pd.Series(dtype=np.int64)
        return pd.concat(chunks).groupby(level=0, sort=False).sum().astype(np.int64)

    return (
        combine_counts(train_user_count_chunks),
        combine_counts(item_train_count_chunks),
        combine_counts(item_train_positive_count_chunks),
        rec_app_ids,
        total_rows,
    )


def collect_sampled_train_history(
    sampled_users: np.ndarray,
    chunksize: int,
) -> tuple[
    dict[int, set[int]],
    dict[int, dict[int, float]],
    dict[int, dict[int, float]],
    dict[int, set[int]],
]:
    sampled_user_set = set(int(user_id) for user_id in sampled_users)
    train_seen: dict[int, set[int]] = defaultdict(set)
    positive_hours: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    positive_hours_weight: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    train_negative: dict[int, set[int]] = defaultdict(set)

    usecols = ["app_id", "date", "is_recommended", "hours", "user_id"]
    for step, chunk in enumerate(
        pd.read_csv(baseline.RECOMMENDATIONS_CSV, usecols=usecols, chunksize=chunksize),
        start=1,
    ):
        chunk = chunk.loc[chunk["user_id"].isin(sampled_user_set)].copy()
        if chunk.empty:
            continue

        chunk["date"] = chunk["date"].astype(str)
        chunk = chunk.loc[chunk["date"].lt(baseline.TRAIN_END)].copy()
        if chunk.empty:
            continue

        chunk["app_id"] = chunk["app_id"].astype(int)
        chunk["user_id"] = chunk["user_id"].astype(int)
        chunk["hours"] = pd.to_numeric(chunk["hours"], errors="coerce").fillna(0.0)
        chunk["hours_weight"] = np.log1p(chunk["hours"])
        chunk["is_positive"] = baseline.str_to_bool(chunk["is_recommended"])

        for user_id, app_ids in chunk.groupby("user_id")["app_id"]:
            train_seen[int(user_id)].update(int(app_id) for app_id in app_ids.to_numpy())

        positive_chunk = chunk.loc[chunk["is_positive"]]
        for row in positive_chunk.itertuples(index=False):
            user_id = int(row.user_id)
            app_id = int(row.app_id)
            positive_hours[user_id][app_id] += float(row.hours)
            positive_hours_weight[user_id][app_id] += float(row.hours_weight)

        negative_chunk = chunk.loc[~chunk["is_positive"]]
        for user_id, app_ids in negative_chunk.groupby("user_id")["app_id"]:
            train_negative[int(user_id)].update(int(app_id) for app_id in app_ids.to_numpy())

        if step == 1 or step % 20 == 0:
            print(
                f"[scan2] chunks={step:,}, users_with_positive={len(positive_hours_weight):,}",
                flush=True,
            )

    return (
        dict(train_seen),
        {user_id: dict(items) for user_id, items in positive_hours.items()},
        {user_id: dict(items) for user_id, items in positive_hours_weight.items()},
        dict(train_negative),
    )


def build_item_catalog(
    text_ready: pd.DataFrame,
    emb_index: pd.DataFrame,
    item_train_counts: pd.Series,
    item_train_positive_counts: pd.Series,
) -> pd.DataFrame:
    catalog = text_ready[
        [
            "app_id",
            "title_clean",
            "date_release",
            "rating",
            "positive_ratio",
            "user_reviews",
            "price_final",
            "tags_text",
        ]
    ].copy()
    catalog = catalog.merge(emb_index, on="app_id", how="left", validate="one_to_one")
    catalog["train_interactions"] = catalog["app_id"].map(item_train_counts).fillna(0).astype(np.int64)
    catalog["train_positive_interactions"] = (
        catalog["app_id"].map(item_train_positive_counts).fillna(0).astype(np.int64)
    )
    catalog["is_cold_train"] = catalog["train_interactions"].eq(0)
    catalog = catalog.rename(columns={"row": "embedding_row"})
    return catalog


def build_user_vectors(
    sampled_users: np.ndarray,
    positive_hours: dict[int, dict[int, float]],
    positive_hours_weight: dict[int, dict[int, float]],
    train_seen: dict[int, set[int]],
    train_negative: dict[int, set[int]],
    emb: np.ndarray,
    app_to_row: dict[int, int],
    min_positive_history: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectors = []
    index_rows = []
    seen_rows = []
    positive_history_rows = []

    for user_id in sampled_users:
        user_id = int(user_id)
        weights_by_app = positive_hours_weight.get(user_id, {})
        valid_app_ids = [app_id for app_id in weights_by_app if app_id in app_to_row]
        if len(valid_app_ids) < min_positive_history:
            continue

        rows = [app_to_row[app_id] for app_id in valid_app_ids]
        weights = np.asarray([max(float(weights_by_app[app_id]), 1e-6) for app_id in valid_app_ids], dtype=np.float32)
        vector = np.average(emb[rows], axis=0, weights=weights)
        vector = normalize_vector(vector)

        vector_row = len(vectors)
        vectors.append(vector)
        index_rows.append(
            {
                "user_id": user_id,
                "row": vector_row,
                "n_train_seen_items": len(train_seen.get(user_id, set())),
                "n_train_positive_items": len(valid_app_ids),
                "n_train_negative_items": len(train_negative.get(user_id, set())),
                "positive_hours_sum": sum(float(positive_hours.get(user_id, {}).get(app_id, 0.0)) for app_id in valid_app_ids),
                "positive_hours_weight_sum": float(weights.sum()),
            }
        )
        seen_rows.append({"user_id": user_id, "seen_app_ids": join_ids(train_seen.get(user_id, set()))})

        for app_id in valid_app_ids:
            positive_history_rows.append(
                {
                    "user_id": user_id,
                    "app_id": app_id,
                    "embedding_row": app_to_row[app_id],
                    "hours": float(positive_hours.get(user_id, {}).get(app_id, 0.0)),
                    "hours_weight": float(weights_by_app[app_id]),
                }
            )

    if not vectors:
        matrix = np.empty((0, emb.shape[1]), dtype=np.float32)
    else:
        matrix = np.vstack(vectors).astype(np.float32)

    return (
        matrix,
        pd.DataFrame(index_rows),
        pd.DataFrame(seen_rows),
        pd.DataFrame(positive_history_rows),
    )


def write_readme(path: Path, summary: dict[str, int | float | str | bool]) -> None:
    lines = [
        "# Text Recommender Artifacts",
        "",
        "이 폴더는 실제 text-only 추천을 실행하기 위한 중간 산출물입니다.",
        "",
        "## 생성 방식",
        "",
        f"- Train 기간: date < {baseline.TRAIN_END}",
        "- User vector: train positive 게임 embedding의 log1p(hours) weighted mean",
        "- Candidate set: text embedding이 있는 전체 game catalog",
        "- 추천 시 제외: train 기간에 user가 이미 본 게임",
        "",
        "## 주요 파일",
        "",
        "| 파일 | 설명 |",
        "|---|---|",
        "| `item_embeddings_text_norm.npy` | L2 normalize된 게임 text embedding |",
        "| `item_catalog.csv` | 게임 title/tags/popularity/cold flag/embedding row |",
        "| `user_vectors_hours_weighted.npy` | user별 취향 벡터 |",
        "| `user_vectors_index.csv` | user_id와 user vector row 매핑 |",
        "| `user_train_seen.csv` | 추천에서 제외할 user별 train seen 게임 목록 |",
        "| `user_train_positive_history.csv` | 추천 이유 생성에 쓸 positive history와 hours |",
        "| `artifact_summary.json` | 생성 설정과 row 수 요약 |",
        "",
        "## 요약",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--min-train-interactions", type=int, default=5)
    parser.add_argument("--min-positive-history", type=int, default=1)
    parser.add_argument("--max-users", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] static game/text files", flush=True)
    games, text_ready, emb_index, emb = baseline.load_static_frames()
    emb = baseline.l2_normalize(emb.astype(np.float32))
    app_to_row = {int(app_id): int(row) for app_id, row in zip(emb_index["app_id"], emb_index["row"])}

    print("[scan1] train counts", flush=True)
    train_user_counts, item_train_counts, item_train_positive_counts, rec_app_ids, total_rows = first_scan_train_counts(
        args.chunksize
    )
    alignment_stats = baseline.compute_alignment_stats(games, text_ready, emb_index, emb, rec_app_ids)

    sampled_users = baseline.sample_train_users(
        train_user_counts,
        args.min_train_interactions,
        args.max_users,
        args.seed,
    )
    print(f"[users] sampled={len(sampled_users):,}", flush=True)

    print("[scan2] sampled user train history", flush=True)
    train_seen, positive_hours, positive_hours_weight, train_negative = collect_sampled_train_history(
        sampled_users,
        args.chunksize,
    )

    print("[build] item catalog", flush=True)
    item_catalog = build_item_catalog(text_ready, emb_index, item_train_counts, item_train_positive_counts)

    print("[build] user vectors", flush=True)
    user_vectors, user_index, user_seen, user_positive_history = build_user_vectors(
        sampled_users,
        positive_hours,
        positive_hours_weight,
        train_seen,
        train_negative,
        emb,
        app_to_row,
        args.min_positive_history,
    )

    if not user_positive_history.empty:
        title_lookup = text_ready.set_index("app_id")["title_clean"].astype(str).to_dict()
        user_positive_history["title"] = user_positive_history["app_id"].map(title_lookup)
        user_positive_history = user_positive_history[
            ["user_id", "app_id", "title", "embedding_row", "hours", "hours_weight"]
        ]

    print("[save] artifacts", flush=True)
    np.save(OUT_DIR / "item_embeddings_text_norm.npy", emb.astype(np.float32))
    np.save(OUT_DIR / "user_vectors_hours_weighted.npy", user_vectors.astype(np.float32))
    emb_index.to_csv(OUT_DIR / "item_embedding_index.csv", index=False)
    item_catalog.to_csv(OUT_DIR / "item_catalog.csv", index=False)
    pd.DataFrame({"user_id": sampled_users}).to_csv(OUT_DIR / "sampled_user_ids.csv", index=False)
    user_index.to_csv(OUT_DIR / "user_vectors_index.csv", index=False)
    user_seen.to_csv(OUT_DIR / "user_train_seen.csv", index=False)
    user_positive_history.to_csv(OUT_DIR / "user_train_positive_history.csv", index=False)

    summary = {
        "recommendation_rows_scanned": int(total_rows),
        "catalog_games": int(len(item_catalog)),
        "embedding_games": int(len(emb_index)),
        "sampled_users": int(len(sampled_users)),
        "users_with_vector": int(len(user_index)),
        "user_vector_dim": int(user_vectors.shape[1]) if user_vectors.ndim == 2 and len(user_vectors) else int(emb.shape[1]),
        "positive_history_rows": int(len(user_positive_history)),
        "train_eligible_users": int((train_user_counts >= args.min_train_interactions).sum()),
        "min_train_interactions": int(args.min_train_interactions),
        "min_positive_history": int(args.min_positive_history),
        "max_users": int(args.max_users),
        "seed": int(args.seed),
        "train_end": baseline.TRAIN_END,
        "app_id_mapping_ok": bool(alignment_stats["embedding_app_id_order_matches_games_text_ready"]),
    }
    (OUT_DIR / "artifact_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(OUT_DIR / "README.md", summary)

    print("\n=== Artifact summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\n결과 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
