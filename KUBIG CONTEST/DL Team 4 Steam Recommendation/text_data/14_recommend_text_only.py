"""
14_recommend_text_only.py - 실제 text-only 추천 실행

13_build_text_recommender_artifacts.py가 만든 artifact를 읽어서 특정 user_id에 대해
Top-K 게임을 추천한다. 기본적으로 train 기간에 이미 본 게임은 제외한다.

예시:
python 14_recommend_text_only.py --user-id 58 --top-k 10
python 14_recommend_text_only.py --user-id 58 --top-k 20 --output-csv text_recommender_artifacts/recommend_user_58.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ARTIFACT_DIR = BASE_DIR / "text_recommender_artifacts"


def parse_seen_app_ids(value: object) -> set[int]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {int(token) for token in text.split()}


def load_artifacts(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_emb = np.load(artifact_dir / "item_embeddings_text_norm.npy").astype(np.float32)
    user_vectors = np.load(artifact_dir / "user_vectors_hours_weighted.npy").astype(np.float32)
    item_catalog = pd.read_csv(artifact_dir / "item_catalog.csv")
    user_index = pd.read_csv(artifact_dir / "user_vectors_index.csv")
    user_seen = pd.read_csv(artifact_dir / "user_train_seen.csv")

    if len(item_catalog) != len(item_emb):
        raise ValueError(f"item_catalog rows {len(item_catalog)} != item embeddings {len(item_emb)}")
    if len(user_index) != len(user_vectors):
        raise ValueError(f"user_index rows {len(user_index)} != user vectors {len(user_vectors)}")

    return item_emb, user_vectors, item_catalog, user_index, user_seen


def recommend_for_user(
    user_id: int,
    top_k: int,
    item_emb: np.ndarray,
    user_vectors: np.ndarray,
    item_catalog: pd.DataFrame,
    user_index: pd.DataFrame,
    user_seen: pd.DataFrame,
    exclude_seen: bool,
) -> pd.DataFrame:
    matched_user = user_index.loc[user_index["user_id"].eq(user_id)]
    if matched_user.empty:
        available = user_index["user_id"].head(10).tolist()
        raise ValueError(f"user_id={user_id}의 user vector가 없습니다. 예시 user_id: {available}")

    user_row = int(matched_user.iloc[0]["row"])
    user_vector = user_vectors[user_row]
    scores = item_emb @ user_vector

    seen_app_ids: set[int] = set()
    matched_seen = user_seen.loc[user_seen["user_id"].eq(user_id)]
    if not matched_seen.empty:
        seen_app_ids = parse_seen_app_ids(matched_seen.iloc[0]["seen_app_ids"])

    if exclude_seen and seen_app_ids:
        seen_rows = item_catalog.loc[item_catalog["app_id"].isin(seen_app_ids), "embedding_row"].dropna().astype(int)
        scores[seen_rows.to_numpy()] = -np.inf

    candidate_count = min(top_k, len(scores))
    top_rows = np.argpartition(-scores, kth=candidate_count - 1)[:candidate_count]
    top_rows = top_rows[np.argsort(-scores[top_rows])]

    result = item_catalog.iloc[top_rows].copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result.insert(3, "score", scores[top_rows])
    result["was_seen_in_train"] = result["app_id"].isin(seen_app_ids)

    columns = [
        "rank",
        "app_id",
        "title_clean",
        "embedding_row",
        "score",
        "rating",
        "positive_ratio",
        "user_reviews",
        "price_final",
        "train_interactions",
        "train_positive_interactions",
        "is_cold_train",
        "was_seen_in_train",
        "tags_text",
    ]
    return result[columns]


def add_recommendation_reasons(
    result: pd.DataFrame,
    user_id: int,
    item_emb: np.ndarray,
    artifact_dir: Path,
    max_history_items: int,
) -> pd.DataFrame:
    history_path = artifact_dir / "user_train_positive_history.csv"
    if not history_path.exists():
        result["reason"] = ""
        result["reason_app_id"] = np.nan
        result["reason_title"] = ""
        result["reason_similarity"] = np.nan
        return result

    usecols = ["user_id", "app_id", "title", "embedding_row", "hours", "hours_weight"]
    history = pd.read_csv(history_path, usecols=usecols)
    history = history.loc[history["user_id"].eq(user_id)].copy()
    if history.empty:
        result["reason"] = ""
        result["reason_app_id"] = np.nan
        result["reason_title"] = ""
        result["reason_similarity"] = np.nan
        return result

    history = history.sort_values("hours_weight", ascending=False).head(max_history_items)
    history_rows = history["embedding_row"].astype(int).to_numpy()
    history_emb = item_emb[history_rows]

    reason_app_ids = []
    reason_titles = []
    reason_scores = []
    reasons = []

    for row in result.itertuples(index=False):
        similarities = history_emb @ item_emb[int(row.embedding_row)]
        best_idx = int(np.argmax(similarities))
        best_history = history.iloc[best_idx]
        best_score = float(similarities[best_idx])
        best_title = str(best_history["title"])
        best_app_id = int(best_history["app_id"])

        reason_app_ids.append(best_app_id)
        reason_titles.append(best_title)
        reason_scores.append(best_score)
        reasons.append(f"유저가 좋아한 '{best_title}'와 text/tag가 비슷함")

    result = result.copy()
    result["reason"] = reasons
    result["reason_app_id"] = reason_app_ids
    result["reason_title"] = reason_titles
    result["reason_similarity"] = reason_scores
    return result


def print_history(user_id: int, artifact_dir: Path, max_items: int) -> None:
    history_path = artifact_dir / "user_train_positive_history.csv"
    if not history_path.exists():
        return

    usecols = ["user_id", "app_id", "title", "hours", "hours_weight"]
    history = pd.read_csv(history_path, usecols=usecols)
    history = history.loc[history["user_id"].eq(user_id)].copy()
    if history.empty:
        return

    history = history.sort_values("hours_weight", ascending=False).head(max_items)
    print("\n=== Train positive history sample ===")
    print(history[["app_id", "title", "hours", "hours_weight"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--include-seen", action="store_true")
    parser.add_argument("--show-history", action="store_true")
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--no-reasons", action="store_true")
    parser.add_argument("--reason-history-k", type=int, default=30)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    item_emb, user_vectors, item_catalog, user_index, user_seen = load_artifacts(args.artifact_dir)
    result = recommend_for_user(
        user_id=args.user_id,
        top_k=args.top_k,
        item_emb=item_emb,
        user_vectors=user_vectors,
        item_catalog=item_catalog,
        user_index=user_index,
        user_seen=user_seen,
        exclude_seen=not args.include_seen,
    )
    if not args.no_reasons:
        result = add_recommendation_reasons(
            result,
            args.user_id,
            item_emb,
            args.artifact_dir,
            args.reason_history_k,
        )

    if args.show_history:
        print_history(args.user_id, args.artifact_dir, args.history_k)

    print("\n=== Text-only recommendation ===")
    print(
        result[
            [
                "rank",
                "app_id",
                "title_clean",
                "score",
                "train_interactions",
                "is_cold_train",
                "reason",
                "reason_similarity",
                "tags_text",
            ]
        ].to_string(index=False)
    )

    if args.output_csv is not None:
        output_csv = args.output_csv
        if not output_csv.is_absolute():
            output_csv = BASE_DIR / output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"\n결과 저장 -> {output_csv}")


if __name__ == "__main__":
    main()
