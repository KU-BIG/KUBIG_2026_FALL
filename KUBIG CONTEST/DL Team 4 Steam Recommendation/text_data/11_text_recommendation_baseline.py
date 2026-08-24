"""
11_text_recommendation_baseline.py - MiniLM text embedding 추천 baseline 평가

게임 text embedding을 실제 사용자 interaction과 연결해 content-based 추천이
추천 signal로 쓸 수 있는지 확인한다.

절차:
1. games / text embedding / recommendations의 app_id alignment를 점검한다.
2. train 기간 interaction만 사용해 평가 대상 user를 고른다.
3. train positive 게임 embedding 평균으로 user text vector를 만든다.
4. Popularity baseline과 Text-only baseline을 같은 user set에서 평가한다.
5. metric CSV, figure, 정성 사례, 최종 요약 markdown을 저장한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_codex_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = BASE_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "steam_recommendations"

TEXT_READY = BASE_DIR / "games_text_ready.csv"
EMB_NPY = BASE_DIR / "emb_text_minilm.npy"
EMB_INDEX = BASE_DIR / "emb_text_minilm.csv"
GAMES_CSV = RAW_DIR / "games.csv"
RECOMMENDATIONS_CSV = RAW_DIR / "recommendations.csv"

OUT_DIR = BASE_DIR / "text_recommendation_results"

TRAIN_END = "2022-04-01"
VAL_END = "2022-07-01"
TEST_END = "2023-01-01"


def str_to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def dcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    score = 0.0
    for rank, app_id in enumerate(recommended[:k], start=1):
        if app_id in relevant:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return safe_div(dcg_at_k(recommended, relevant, k), ideal)


def make_metric_rows(
    model_name: str,
    recommendations: dict[int, list[int]],
    targets: dict[int, set[int]],
    cold_targets: dict[int, set[int]],
    warm_targets: dict[int, set[int]],
    catalog_size: int,
    item_train_counts: pd.Series,
    ks: list[int],
) -> list[dict[str, float | int | str]]:
    rows = []
    pop_lookup = item_train_counts.to_dict()

    for k in ks:
        recalls = []
        ndcgs = []
        cold_recalls = []
        cold_ndcgs = []
        warm_recalls = []
        warm_ndcgs = []
        covered_items: set[int] = set()
        rec_popularity = []

        for user_id, recs in recommendations.items():
            topk = recs[:k]
            relevant = targets[user_id]
            cold_relevant = cold_targets[user_id]
            warm_relevant = warm_targets[user_id]

            covered_items.update(topk)
            rec_popularity.extend(pop_lookup.get(app_id, 0) for app_id in topk)

            recalls.append(safe_div(len(set(topk) & relevant), len(relevant)))
            ndcgs.append(ndcg_at_k(topk, relevant, k))

            if cold_relevant:
                cold_recalls.append(safe_div(len(set(topk) & cold_relevant), len(cold_relevant)))
                cold_ndcgs.append(ndcg_at_k(topk, cold_relevant, k))

            if warm_relevant:
                warm_recalls.append(safe_div(len(set(topk) & warm_relevant), len(warm_relevant)))
                warm_ndcgs.append(ndcg_at_k(topk, warm_relevant, k))

        rows.append(
            {
                "model": model_name,
                "k": k,
                "n_eval_users": len(recommendations),
                "recall": float(np.mean(recalls)) if recalls else 0.0,
                "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
                "coverage": safe_div(len(covered_items), catalog_size),
                "avg_train_popularity": float(np.mean(rec_popularity)) if rec_popularity else 0.0,
                "cold_recall": float(np.mean(cold_recalls)) if cold_recalls else 0.0,
                "cold_ndcg": float(np.mean(cold_ndcgs)) if cold_ndcgs else 0.0,
                "n_cold_eval_users": len(cold_recalls),
                "warm_recall": float(np.mean(warm_recalls)) if warm_recalls else 0.0,
                "warm_ndcg": float(np.mean(warm_ndcgs)) if warm_ndcgs else 0.0,
                "n_warm_eval_users": len(warm_recalls),
            }
        )

    return rows


def plot_bar(
    data: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    colors: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    color_values = colors or ["#4C78A8", "#F58518"]
    ax.bar(data["label"], data[metric_col], color=color_values[: len(data)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(data[metric_col]):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_static_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    games = pd.read_csv(GAMES_CSV)
    text_ready = pd.read_csv(TEXT_READY)
    emb_index = pd.read_csv(EMB_INDEX)
    emb = np.load(EMB_NPY).astype(np.float32)
    emb = l2_normalize(emb)
    return games, text_ready, emb_index, emb


def compute_alignment_stats(
    games: pd.DataFrame,
    text_ready: pd.DataFrame,
    emb_index: pd.DataFrame,
    emb: np.ndarray,
    rec_app_ids: set[int],
) -> dict[str, int | bool]:
    games_app_ids = set(games["app_id"].astype(int))
    text_app_ids = set(text_ready["app_id"].astype(int))
    emb_app_ids = set(emb_index["app_id"].astype(int))
    rec_games_without_embedding = rec_app_ids - emb_app_ids
    zero_interaction_games = games_app_ids - rec_app_ids

    row_series = emb_index["row"].to_numpy()
    valid_row_numbers = (
        len(row_series) == len(emb)
        and np.array_equal(row_series, np.arange(len(row_series)))
        and row_series.min(initial=0) >= 0
        and row_series.max(initial=-1) < len(emb)
    )
    aligned_to_text_ready = (
        len(emb_index) == len(text_ready)
        and emb_index["app_id"].astype(int).reset_index(drop=True).equals(
            text_ready["app_id"].astype(int).reset_index(drop=True)
        )
    )

    return {
        "catalog_games": int(len(games)),
        "text_ready_games": int(len(text_ready)),
        "text_embedding_games": int(len(emb_index)),
        "embedding_rows": int(len(emb)),
        "recommendation_unique_games": int(len(rec_app_ids)),
        "recommendation_games_without_text_embedding": int(len(rec_games_without_embedding)),
        "zero_interaction_games": int(len(zero_interaction_games)),
        "zero_interaction_games_with_text_embedding": int(len(zero_interaction_games & emb_app_ids)),
        "duplicate_app_id_games_csv": int(games["app_id"].duplicated().sum()),
        "duplicate_app_id_games_text_ready": int(text_ready["app_id"].duplicated().sum()),
        "duplicate_app_id_emb_index": int(emb_index["app_id"].duplicated().sum()),
        "duplicate_row_emb_index": int(emb_index["row"].duplicated().sum()),
        "embedding_row_count_matches_index": bool(len(emb_index) == len(emb)),
        "embedding_row_numbers_valid": bool(valid_row_numbers),
        "embedding_app_id_order_matches_games_text_ready": bool(aligned_to_text_ready),
        "games_without_text_embedding": int(len(games_app_ids - emb_app_ids)),
        "text_embeddings_without_games_csv": int(len(emb_app_ids - games_app_ids)),
    }


def first_scan_recommendations(
    chunksize: int,
) -> tuple[pd.Series, pd.Series, set[int], int]:
    train_user_counts = pd.Series(dtype=np.int64)
    item_train_counts = pd.Series(dtype=np.int64)
    rec_app_ids: set[int] = set()
    total_rows = 0

    usecols = ["app_id", "date", "user_id"]
    for step, chunk in enumerate(
        pd.read_csv(RECOMMENDATIONS_CSV, usecols=usecols, chunksize=chunksize),
        start=1,
    ):
        total_rows += len(chunk)
        rec_app_ids.update(chunk["app_id"].astype(int).unique().tolist())

        train_mask = chunk["date"].astype(str).lt(TRAIN_END)
        train_chunk = chunk.loc[train_mask]
        if not train_chunk.empty:
            train_user_counts = train_user_counts.add(
                train_chunk["user_id"].value_counts(),
                fill_value=0,
            )
            item_train_counts = item_train_counts.add(
                train_chunk["app_id"].value_counts(),
                fill_value=0,
            )

        if step == 1 or step % 20 == 0:
            print(f"[scan1] chunks={step:,}, rows={total_rows:,}", flush=True)

    return (
        train_user_counts.astype(np.int64),
        item_train_counts.astype(np.int64),
        rec_app_ids,
        total_rows,
    )


def sample_train_users(
    train_user_counts: pd.Series,
    min_train_interactions: int,
    max_users: int,
    seed: int,
) -> np.ndarray:
    eligible = train_user_counts[train_user_counts >= min_train_interactions].index.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(eligible) > max_users:
        eligible = rng.choice(eligible, size=max_users, replace=False)
    eligible.sort()
    return eligible


def collect_user_sets(
    sampled_users: np.ndarray,
    eval_split: str,
    chunksize: int,
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
    sampled_user_set = set(int(user_id) for user_id in sampled_users)
    train_seen: dict[int, set[int]] = defaultdict(set)
    train_positive: dict[int, set[int]] = defaultdict(set)
    eval_positive: dict[int, set[int]] = defaultdict(set)

    if eval_split == "val":
        eval_start, eval_end = TRAIN_END, VAL_END
    elif eval_split == "test":
        eval_start, eval_end = VAL_END, TEST_END
    else:
        raise ValueError("eval_split은 'val' 또는 'test'만 가능합니다.")

    usecols = ["app_id", "date", "is_recommended", "user_id"]
    for step, chunk in enumerate(
        pd.read_csv(RECOMMENDATIONS_CSV, usecols=usecols, chunksize=chunksize),
        start=1,
    ):
        chunk = chunk.loc[chunk["user_id"].isin(sampled_user_set)].copy()
        if chunk.empty:
            continue

        chunk["app_id"] = chunk["app_id"].astype(int)
        chunk["user_id"] = chunk["user_id"].astype(int)
        chunk["date"] = chunk["date"].astype(str)
        chunk["is_positive"] = str_to_bool(chunk["is_recommended"])

        train_mask = chunk["date"].lt(TRAIN_END)
        eval_mask = chunk["date"].ge(eval_start) & chunk["date"].lt(eval_end) & chunk["is_positive"]

        train_chunk = chunk.loc[train_mask, ["user_id", "app_id", "is_positive"]]
        for user_id, app_ids in train_chunk.groupby("user_id")["app_id"]:
            train_seen[int(user_id)].update(int(app_id) for app_id in app_ids.to_numpy())

        train_pos_chunk = train_chunk.loc[train_chunk["is_positive"]]
        for user_id, app_ids in train_pos_chunk.groupby("user_id")["app_id"]:
            train_positive[int(user_id)].update(int(app_id) for app_id in app_ids.to_numpy())

        eval_chunk = chunk.loc[eval_mask, ["user_id", "app_id"]]
        for user_id, app_ids in eval_chunk.groupby("user_id")["app_id"]:
            eval_positive[int(user_id)].update(int(app_id) for app_id in app_ids.to_numpy())

        if step == 1 or step % 20 == 0:
            print(
                f"[scan2] chunks={step:,}, users_with_eval_pos={len(eval_positive):,}",
                flush=True,
            )

    return dict(train_seen), dict(train_positive), dict(eval_positive)


def build_user_vectors(
    eval_users: list[int],
    train_positive: dict[int, set[int]],
    emb: np.ndarray,
    app_to_row: dict[int, int],
) -> tuple[np.ndarray, list[int]]:
    vectors = []
    kept_users = []

    for user_id in eval_users:
        rows = [app_to_row[app_id] for app_id in train_positive[user_id] if app_id in app_to_row]
        if not rows:
            continue
        vector = emb[rows].mean(axis=0)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        vectors.append(vector.astype(np.float32))
        kept_users.append(user_id)

    return np.vstack(vectors).astype(np.float32), kept_users


def popularity_recommend(
    eval_users: Iterable[int],
    train_seen: dict[int, set[int]],
    sorted_app_ids: list[int],
    k_max: int,
) -> dict[int, list[int]]:
    recommendations = {}
    for user_id in eval_users:
        seen = train_seen.get(user_id, set())
        recs = []
        for app_id in sorted_app_ids:
            if app_id in seen:
                continue
            recs.append(app_id)
            if len(recs) >= k_max:
                break
        recommendations[user_id] = recs
    return recommendations


def text_recommend(
    user_vectors: np.ndarray,
    eval_users: list[int],
    train_seen: dict[int, set[int]],
    emb: np.ndarray,
    emb_index: pd.DataFrame,
    k_max: int,
    batch_size: int,
) -> dict[int, list[int]]:
    app_ids = emb_index["app_id"].astype(int).to_numpy()
    row_lookup = {int(app_id): idx for idx, app_id in enumerate(app_ids)}
    recommendations = {}

    for start in range(0, len(eval_users), batch_size):
        end = min(start + batch_size, len(eval_users))
        scores = user_vectors[start:end] @ emb.T

        for local_idx, user_id in enumerate(eval_users[start:end]):
            seen_rows = [row_lookup[app_id] for app_id in train_seen.get(user_id, set()) if app_id in row_lookup]
            if seen_rows:
                scores[local_idx, seen_rows] = -np.inf

            top_rows = np.argpartition(-scores[local_idx], kth=k_max - 1)[:k_max]
            top_rows = top_rows[np.argsort(-scores[local_idx, top_rows])]
            recommendations[user_id] = [int(app_ids[row]) for row in top_rows]

        if start == 0 or end % (batch_size * 10) == 0 or end == len(eval_users):
            print(f"[text-rec] users={end:,} / {len(eval_users):,}", flush=True)

    return recommendations


def make_qualitative_cases(
    eval_users: list[int],
    train_positive: dict[int, set[int]],
    targets: dict[int, set[int]],
    cold_targets: dict[int, set[int]],
    text_recs: dict[int, list[int]],
    popularity_recs: dict[int, list[int]],
    title_lookup: dict[int, str],
    item_train_counts: pd.Series,
    n_cases: int,
) -> pd.DataFrame:
    cold_users = [user_id for user_id in eval_users if cold_targets[user_id]]
    selected = cold_users[:n_cases]
    if len(selected) < n_cases:
        selected += [user_id for user_id in eval_users if user_id not in selected][: n_cases - len(selected)]

    rows = []
    train_count_lookup = item_train_counts.to_dict()
    for user_id in selected:
        history_titles = [title_lookup.get(app_id, str(app_id)) for app_id in list(train_positive[user_id])[:5]]
        target_titles = [title_lookup.get(app_id, str(app_id)) for app_id in list(targets[user_id])[:5]]
        cold_target_titles = [title_lookup.get(app_id, str(app_id)) for app_id in list(cold_targets[user_id])[:5]]
        text_titles = []
        for app_id in text_recs[user_id][:10]:
            marker = ""
            if app_id in targets[user_id]:
                marker += " [HIT]"
            if train_count_lookup.get(app_id, 0) == 0:
                marker += " [COLD]"
            text_titles.append(f"{title_lookup.get(app_id, str(app_id))}{marker}")
        popularity_titles = []
        for app_id in popularity_recs[user_id][:10]:
            marker = " [HIT]" if app_id in targets[user_id] else ""
            popularity_titles.append(f"{title_lookup.get(app_id, str(app_id))}{marker}")

        rows.append(
            {
                "user_id": user_id,
                "train_positive_history_sample": " | ".join(history_titles),
                "eval_positive_sample": " | ".join(target_titles),
                "cold_eval_positive_sample": " | ".join(cold_target_titles),
                "text_top10": " | ".join(text_titles),
                "popularity_top10": " | ".join(popularity_titles),
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    def format_cell(value: object) -> str:
        if isinstance(value, float):
            text = f"{value:.6f}"
        else:
            text = str(value)
        return text.replace("\n", " ").replace("|", "\\|")

    headers = [format_cell(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(dataframe_to_markdown(df), encoding="utf-8")


def decide_conclusion(metrics_k10: pd.DataFrame) -> str:
    pop = metrics_k10.loc[metrics_k10["model"] == "Popularity"].iloc[0]
    text = metrics_k10.loc[metrics_k10["model"] == "Text-only"].iloc[0]

    recall_delta = text["recall"] - pop["recall"]
    ndcg_delta = text["ndcg"] - pop["ndcg"]
    cold_delta = text["cold_recall"] - pop["cold_recall"]
    coverage_ratio = safe_div(text["coverage"], pop["coverage"])
    avg_pop_lower = text["avg_train_popularity"] < pop["avg_train_popularity"]

    if recall_delta >= 0 and ndcg_delta >= 0 and (cold_delta > 0 or coverage_ratio >= 1.5):
        return "유효함: text embedding이 실제 recommendation signal을 제공한다."
    if cold_delta > 0 or coverage_ratio >= 1.5 or avg_pop_lower:
        return "부분적으로 유효함: 일반 정확도는 popularity보다 약할 수 있지만 cold-start/coverage/long-tail 측면의 가치가 있다."
    return "현재 방식으로는 유효성이 낮음: popularity/simple baseline보다 의미 있는 이점이 확인되지 않았다."


def write_summary(
    output_path: Path,
    metrics_k10: pd.DataFrame,
    all_metrics: pd.DataFrame,
    stats: dict[str, int | bool],
    args: argparse.Namespace,
    qualitative_md_name: str,
) -> None:
    table = metrics_k10.copy()
    table["Model"] = table["model"]
    result_table = table[
        ["Model", "recall", "ndcg", "coverage", "cold_recall", "cold_ndcg", "avg_train_popularity"]
    ].rename(
        columns={
            "recall": "Recall@10",
            "ndcg": "NDCG@10",
            "coverage": "Coverage",
            "cold_recall": "Cold Recall@10",
            "cold_ndcg": "Cold NDCG@10",
            "avg_train_popularity": "Avg Train Popularity",
        }
    )

    diff_row = {
        "지표": "차이(Text - Popularity)",
        "Popularity": "",
        "Text-only": "",
        "차이": "",
    }
    pop = metrics_k10.loc[metrics_k10["model"] == "Popularity"].iloc[0]
    text = metrics_k10.loc[metrics_k10["model"] == "Text-only"].iloc[0]
    compact_rows = []
    for metric, label in [
        ("recall", "Recall@10"),
        ("ndcg", "NDCG@10"),
        ("coverage", "Coverage"),
        ("cold_recall", "Cold Recall@10"),
    ]:
        compact_rows.append(
            {
                "지표": label,
                "Popularity": f"{pop[metric]:.6f}",
                "Text-only": f"{text[metric]:.6f}",
                "차이": f"{text[metric] - pop[metric]:+.6f}",
            }
        )
    compact = pd.DataFrame(compact_rows)
    conclusion = decide_conclusion(metrics_k10)

    lines = [
        "# Text Recommendation Result Summary",
        "",
        "## 실험 목적",
        "",
        "MiniLM text embedding이 실제 사용자 추천에 사용할 수 있는지 검증.",
        "",
        "## 방법",
        "",
        (
            "사용자의 train positive 게임 embedding 평균을 user vector로 만들고, "
            "전체 game catalog와 cosine similarity를 계산해 Top-K를 추천했다. "
            "비교 기준은 train interaction count 기반 Popularity baseline이다."
        ),
        "",
        "## 데이터 구성",
        "",
        f"- Train: date < {TRAIN_END}",
        f"- Validation: {TRAIN_END} <= date < {VAL_END}",
        f"- Test: {VAL_END} <= date < {TEST_END}",
        f"- 평가 split: {args.eval_split}",
        f"- User filter: train 기간 interaction 수 >= {args.min_train_interactions}",
        f"- Train 기준 sampling seed: {args.seed}",
        f"- Train-eligible sampled users: {stats['sampled_train_users']:,}",
        f"- Final eval users: {stats['final_eval_users']:,}",
        "",
        "## App ID Alignment Check",
        "",
        f"- 전체 catalog game 수: {stats['catalog_games']:,}",
        f"- text embedding이 존재하는 game 수: {stats['text_embedding_games']:,}",
        f"- recommendations에 등장하는 game 중 text embedding이 없는 수: {stats['recommendation_games_without_text_embedding']:,}",
        f"- interaction 0건 게임 중 text embedding이 존재하는 수: {stats['zero_interaction_games_with_text_embedding']:,}",
        f"- duplicate app_id 여부: games={stats['duplicate_app_id_games_csv']}, text_ready={stats['duplicate_app_id_games_text_ready']}, embedding_index={stats['duplicate_app_id_emb_index']}",
        f"- embedding row와 app_id mapping mismatch 여부: {not stats['embedding_app_id_order_matches_games_text_ready']}",
        "",
        "## 핵심 결과",
        "",
        dataframe_to_markdown(compact),
        "",
        "## 전체 @K 결과",
        "",
        dataframe_to_markdown(
            all_metrics[
                [
                    "model",
                    "k",
                    "recall",
                    "ndcg",
                    "coverage",
                    "cold_recall",
                    "cold_ndcg",
                    "warm_recall",
                    "warm_ndcg",
                    "avg_train_popularity",
                ]
            ]
        ),
        "",
        "## 한 줄 결론",
        "",
        conclusion,
        "",
        "## 해석",
        "",
        (
            "Popularity는 인기 게임을 안정적으로 맞히는지 보는 기준이고, Text-only는 "
            "사용자 history와 게임 설명/tag/title의 의미적 유사도가 실제 추천으로 이어지는지 보는 기준이다. "
            "따라서 Recall/NDCG, Cold 지표, Coverage와 Avg Train Popularity를 함께 봐야 한다."
        ),
        "",
        f"정성 추천 사례는 `{qualitative_md_name}`에서 확인할 수 있다.",
        "",
        "## 생성 파일",
        "",
        "- `text_recommendation_metrics.csv`",
        "- `text_recommendation_alignment_stats.json`",
        "- `text_vs_popularity_accuracy.png`",
        "- `text_warm_cold_comparison.png`",
        "- `text_coverage_comparison.png`",
        "- `text_recommendation_qualitative_cases.csv`",
        f"- `{qualitative_md_name}`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--min-train-interactions", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=300_000)
    parser.add_argument("--max-eval-users", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--case-count", type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] static game/text files", flush=True)
    games, text_ready, emb_index, emb = load_static_frames()
    catalog_app_ids = emb_index["app_id"].astype(int).tolist()
    app_to_row = {int(app_id): int(row) for app_id, row in zip(emb_index["app_id"], emb_index["row"])}

    print("[scan1] recommendation stats and train counts", flush=True)
    train_user_counts, item_train_counts, rec_app_ids, total_rows = first_scan_recommendations(args.chunksize)

    stats = compute_alignment_stats(games, text_ready, emb_index, emb, rec_app_ids)
    stats["recommendation_rows"] = int(total_rows)

    sampled_users = sample_train_users(
        train_user_counts,
        args.min_train_interactions,
        args.max_users,
        args.seed,
    )
    stats["train_eligible_users"] = int((train_user_counts >= args.min_train_interactions).sum())
    stats["sampled_train_users"] = int(len(sampled_users))
    print(f"[users] sampled={len(sampled_users):,}", flush=True)

    print("[scan2] collect sampled user history and eval positives", flush=True)
    train_seen, train_positive, eval_positive = collect_user_sets(
        sampled_users,
        args.eval_split,
        args.chunksize,
    )

    eval_users = [
        int(user_id)
        for user_id in sampled_users
        if train_positive.get(int(user_id)) and eval_positive.get(int(user_id))
    ]
    if len(eval_users) > args.max_eval_users:
        rng = np.random.default_rng(args.seed + 1)
        eval_users = rng.choice(np.array(eval_users, dtype=np.int64), size=args.max_eval_users, replace=False).tolist()
        eval_users.sort()
    stats["final_eval_users"] = int(len(eval_users))

    if not eval_users:
        raise ValueError("평가 가능한 user가 없습니다. max-users 또는 split 설정을 확인하세요.")

    print(f"[vectors] build user text vectors for eval users={len(eval_users):,}", flush=True)
    user_vectors, eval_users = build_user_vectors(eval_users, train_positive, emb, app_to_row)
    stats["final_eval_users_with_text_vector"] = int(len(eval_users))

    item_train_counts_full = item_train_counts.reindex(catalog_app_ids, fill_value=0).astype(np.int64)
    cold_app_ids = set(item_train_counts_full[item_train_counts_full == 0].index.astype(int).tolist())
    raw_targets = {
        user_id: set(app_id for app_id in eval_positive[user_id] if app_id in app_to_row)
        for user_id in eval_users
    }
    keep_indices = [idx for idx, user_id in enumerate(eval_users) if raw_targets[user_id]]
    eval_users = [eval_users[idx] for idx in keep_indices]
    user_vectors = user_vectors[keep_indices]
    targets = {user_id: raw_targets[user_id] for user_id in eval_users}
    stats["final_eval_users"] = int(len(eval_users))

    if not eval_users:
        raise ValueError("embedding이 있는 positive target을 가진 평가 user가 없습니다.")

    cold_targets = {user_id: targets[user_id] & cold_app_ids for user_id in eval_users}
    warm_targets = {user_id: targets[user_id] - cold_app_ids for user_id in eval_users}
    stats["eval_positive_items"] = int(sum(len(value) for value in targets.values()))
    stats["eval_cold_positive_items"] = int(sum(len(value) for value in cold_targets.values()))
    stats["eval_users_with_cold_positive"] = int(sum(1 for value in cold_targets.values() if value))

    sorted_popularity = (
        item_train_counts_full.sort_values(ascending=False, kind="mergesort")
        .reset_index()
        .rename(columns={"index": "app_id", 0: "train_count"})
    )
    sorted_app_ids = sorted_popularity["app_id"].astype(int).tolist()

    k_values = [5, 10, 20]
    k_max = max(k_values)
    print("[recommend] popularity", flush=True)
    pop_recs = popularity_recommend(eval_users, train_seen, sorted_app_ids, k_max)

    print("[recommend] text-only", flush=True)
    text_recs = text_recommend(
        user_vectors,
        eval_users,
        train_seen,
        emb,
        emb_index,
        k_max,
        args.batch_size,
    )

    print("[metrics] compute", flush=True)
    metric_rows = []
    metric_rows.extend(
        make_metric_rows("Popularity", pop_recs, targets, cold_targets, warm_targets, len(catalog_app_ids), item_train_counts_full, k_values)
    )
    metric_rows.extend(
        make_metric_rows("Text-only", text_recs, targets, cold_targets, warm_targets, len(catalog_app_ids), item_train_counts_full, k_values)
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_DIR / "text_recommendation_metrics.csv", index=False)

    stats_path = OUT_DIR / "text_recommendation_alignment_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_k10 = metrics.loc[metrics["k"] == 10].copy()
    accuracy_df = pd.DataFrame(
        [
            {"label": f"{row.model}\nRecall@10", "value": row.recall}
            for row in metrics_k10.itertuples()
        ]
        + [
            {"label": f"{row.model}\nNDCG@10", "value": row.ndcg}
            for row in metrics_k10.itertuples()
        ]
    )
    plot_bar(
        accuracy_df,
        "value",
        "Model Performance Comparison",
        "Metric value",
        OUT_DIR / "text_vs_popularity_accuracy.png",
        colors=["#4C78A8", "#F58518", "#4C78A8", "#F58518"],
    )

    warm_cold_df = pd.DataFrame(
        [
            {"label": f"{row.model}\nWarm Recall@10", "value": row.warm_recall}
            for row in metrics_k10.itertuples()
        ]
        + [
            {"label": f"{row.model}\nCold Recall@10", "value": row.cold_recall}
            for row in metrics_k10.itertuples()
        ]
    )
    plot_bar(
        warm_cold_df,
        "value",
        "Warm vs Cold Recall@10",
        "Recall@10",
        OUT_DIR / "text_warm_cold_comparison.png",
        colors=["#54A24B", "#E45756", "#54A24B", "#E45756"],
    )

    coverage_df = pd.DataFrame(
        [{"label": row.model, "value": row.coverage} for row in metrics_k10.itertuples()]
    )
    plot_bar(
        coverage_df,
        "value",
        "Catalog Coverage@10",
        "Coverage",
        OUT_DIR / "text_coverage_comparison.png",
    )

    title_lookup = text_ready.set_index("app_id")["title_clean"].astype(str).to_dict()
    qualitative = make_qualitative_cases(
        eval_users,
        train_positive,
        targets,
        cold_targets,
        text_recs,
        pop_recs,
        title_lookup,
        item_train_counts_full,
        args.case_count,
    )
    qualitative.to_csv(OUT_DIR / "text_recommendation_qualitative_cases.csv", index=False)
    write_markdown_table(qualitative, OUT_DIR / "text_recommendation_qualitative_cases.md")

    write_summary(
        OUT_DIR / "text_recommendation_result_summary.md",
        metrics_k10,
        metrics,
        stats,
        args,
        "text_recommendation_qualitative_cases.md",
    )

    print("\n=== Alignment stats ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Metrics @10 ===")
    print(
        metrics_k10[
            [
                "model",
                "recall",
                "ndcg",
                "coverage",
                "cold_recall",
                "cold_ndcg",
                "warm_recall",
                "avg_train_popularity",
            ]
        ].to_string(index=False)
    )
    print(f"\n결과 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
