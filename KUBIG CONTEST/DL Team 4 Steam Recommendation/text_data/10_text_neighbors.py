"""
10_text_neighbors.py - 텍스트 임베딩 이웃 점검

MiniLM 텍스트 임베딩이 비슷한 게임을 가까이 두는지 확인한다.
추천 모델에 넣기 전 임베딩 파일과 app_id 매핑을 점검하는 용도다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TEXT_READY = BASE_DIR / "games_text_ready.csv"
EMB_NPY = BASE_DIR / "emb_text_minilm.npy"
EMB_INDEX = BASE_DIR / "emb_text_minilm.csv"
OUT_CSV = BASE_DIR / "text_neighbors_sample.csv"

DEFAULT_APP_IDS = [
    13500,   # Prince of Persia: Warrior Within
    113020,  # Monaco: What's Yours Is Mine
    620,     # Portal 2
    413150,  # Stardew Valley
    292030,  # The Witcher 3: Wild Hunt
]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def find_query_app_ids(games: pd.DataFrame, query_titles: list[str]) -> list[int]:
    app_ids = []
    title_lower = games["title_clean"].fillna("").str.lower()

    for query in query_titles:
        mask = title_lower.str.contains(query.lower(), regex=False)
        matched = games.loc[mask, ["app_id", "title_clean"]].head(1)
        if matched.empty:
            print(f"[skip] '{query}' 포함 title 없음")
            continue
        app_id = int(matched.iloc[0]["app_id"])
        print(f"[query] {query} -> {app_id} / {matched.iloc[0]['title_clean']}")
        app_ids.append(app_id)

    return app_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--app-ids", nargs="*", type=int, default=DEFAULT_APP_IDS)
    parser.add_argument("--queries", nargs="*", default=None)
    args = parser.parse_args()

    games = pd.read_csv(TEXT_READY)
    index = pd.read_csv(EMB_INDEX)
    emb = np.load(EMB_NPY).astype(np.float32)
    emb = l2_normalize(emb)

    if len(index) != len(emb):
        raise ValueError(f"index rows {len(index)} != embedding rows {len(emb)}")

    app_to_row = {int(app_id): int(row) for app_id, row in zip(index["app_id"], index["row"])}
    meta = games.set_index("app_id")
    if args.queries:
        query_app_ids = find_query_app_ids(games, args.queries)
    else:
        query_app_ids = [app_id for app_id in args.app_ids if app_id in app_to_row]
        missing = [app_id for app_id in args.app_ids if app_id not in app_to_row]
        if missing:
            print(f"[skip] embedding 없는 app_id: {missing}")

    rows = []
    for query_app_id in query_app_ids:
        query_row = app_to_row[query_app_id]
        scores = emb @ emb[query_row]
        order = np.argsort(-scores)[: args.top_k + 1]

        for rank, neighbor_row in enumerate(order):
            neighbor_app_id = int(index.iloc[neighbor_row]["app_id"])
            if neighbor_app_id == query_app_id:
                continue

            rows.append(
                {
                    "query_app_id": query_app_id,
                    "query_title": meta.loc[query_app_id, "title_clean"],
                    "rank": rank,
                    "neighbor_app_id": neighbor_app_id,
                    "neighbor_title": meta.loc[neighbor_app_id, "title_clean"],
                    "cosine": float(scores[neighbor_row]),
                    "neighbor_tags": meta.loc[neighbor_app_id, "tags_text"],
                }
            )
            if len([row for row in rows if row["query_app_id"] == query_app_id]) >= args.top_k:
                break

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)

    print("\n=== Text neighbor sample ===")
    if result.empty:
        print("저장할 이웃 결과가 없습니다.")
    else:
        print(result[["query_title", "rank", "neighbor_title", "cosine"]].to_string(index=False))
        print(f"\n결과 -> {OUT_CSV}")


if __name__ == "__main__":
    main()
