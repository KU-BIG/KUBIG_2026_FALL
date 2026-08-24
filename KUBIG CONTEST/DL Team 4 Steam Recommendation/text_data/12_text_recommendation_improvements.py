"""
12_text_recommendation_improvements.py - text-only 추천 user vector 개선 실험

기존 11_text_recommendation_baseline.py 결과는 보존하고, 같은 split / 같은
train 기준 user sample / 같은 candidate catalog / 같은 metric으로 user vector 생성법만
바꿔 비교한다.

실험:
- positive item embedding simple mean
- log1p(hours) weighted positive mean
- positive_mean - alpha * negative_mean, alpha in {0.2, 0.5}
- train positive history >= 1, >= 3, >= 5 조건 비교
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BASELINE_SCRIPT = BASE_DIR / "11_text_recommendation_baseline.py"
OUT_DIR = BASE_DIR / "text_recommendation_improvements"

spec = importlib.util.spec_from_file_location("text_baseline", BASELINE_SCRIPT)
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)


def collect_user_histories(
    sampled_users: np.ndarray,
    eval_split: str,
    chunksize: int,
) -> tuple[
    dict[int, set[int]],
    dict[int, set[int]],
    dict[int, dict[int, float]],
    dict[int, set[int]],
    dict[int, set[int]],
]:
    sampled_user_set = set(int(user_id) for user_id in sampled_users)
    train_seen: dict[int, set[int]] = defaultdict(set)
    train_positive: dict[int, set[int]] = defaultdict(set)
    train_positive_hours: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    train_negative: dict[int, set[int]] = defaultdict(set)
    eval_positive: dict[int, set[int]] = defaultdict(set)

    if eval_split == "val":
        eval_start, eval_end = baseline.TRAIN_END, baseline.VAL_END
    elif eval_split == "test":
        eval_start, eval_end = baseline.VAL_END, baseline.TEST_END
    else:
        raise ValueError("eval_split은 'val' 또는 'test'만 가능합니다.")

    usecols = ["app_id", "date", "is_recommended", "hours", "user_id"]
    for step, chunk in enumerate(
        pd.read_csv(baseline.RECOMMENDATIONS_CSV, usecols=usecols, chunksize=chunksize),
        start=1,
    ):
        chunk = chunk.loc[chunk["user_id"].isin(sampled_user_set)].copy()
        if chunk.empty:
            continue

        chunk["app_id"] = chunk["app_id"].astype(int)
        chunk["user_id"] = chunk["user_id"].astype(int)
        chunk["date"] = chunk["date"].astype(str)
        chunk["is_positive"] = baseline.str_to_bool(chunk["is_recommended"])
        chunk["hours_weight"] = np.log1p(pd.to_numeric(chunk["hours"], errors="coerce").fillna(0.0))

        train_mask = chunk["date"].lt(baseline.TRAIN_END)
        eval_mask = chunk["date"].ge(eval_start) & chunk["date"].lt(eval_end) & chunk["is_positive"]

        train_chunk = chunk.loc[train_mask, ["user_id", "app_id", "is_positive", "hours_weight"]]
        for user_id, app_ids in train_chunk.groupby("user_id")["app_id"]:
            train_seen[int(user_id)].update(int(app_id) for app_id in app_ids.to_numpy())

        train_pos_chunk = train_chunk.loc[train_chunk["is_positive"]]
        for row in train_pos_chunk.itertuples(index=False):
            user_id = int(row.user_id)
            app_id = int(row.app_id)
            train_positive[user_id].add(app_id)
            train_positive_hours[user_id][app_id] += float(row.hours_weight)

        train_neg_chunk = train_chunk.loc[~train_chunk["is_positive"]]
        for user_id, app_ids in train_neg_chunk.groupby("user_id")["app_id"]:
            train_negative[int(user_id)].update(int(app_id) for app_id in app_ids.to_numpy())

        eval_chunk = chunk.loc[eval_mask, ["user_id", "app_id"]]
        for user_id, app_ids in eval_chunk.groupby("user_id")["app_id"]:
            eval_positive[int(user_id)].update(int(app_id) for app_id in app_ids.to_numpy())

        if step == 1 or step % 20 == 0:
            print(
                f"[scan2] chunks={step:,}, users_with_eval_pos={len(eval_positive):,}",
                flush=True,
            )

    return (
        dict(train_seen),
        dict(train_positive),
        {user_id: dict(weights) for user_id, weights in train_positive_hours.items()},
        dict(train_negative),
        dict(eval_positive),
    )


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return (vector / max(float(np.linalg.norm(vector)), 1e-12)).astype(np.float32)


def mean_vector(app_ids: set[int], emb: np.ndarray, app_to_row: dict[int, int]) -> np.ndarray | None:
    rows = [app_to_row[app_id] for app_id in app_ids if app_id in app_to_row]
    if not rows:
        return None
    return normalize_vector(emb[rows].mean(axis=0))


def weighted_positive_vector(
    weights_by_app: dict[int, float],
    emb: np.ndarray,
    app_to_row: dict[int, int],
) -> np.ndarray | None:
    rows = []
    weights = []
    for app_id, weight in weights_by_app.items():
        if app_id not in app_to_row:
            continue
        rows.append(app_to_row[app_id])
        weights.append(max(float(weight), 1e-6))
    if not rows:
        return None
    weight_array = np.asarray(weights, dtype=np.float32)
    vector = np.average(emb[rows], axis=0, weights=weight_array)
    return normalize_vector(vector)


def pos_minus_neg_vector(
    positive_app_ids: set[int],
    negative_app_ids: set[int],
    emb: np.ndarray,
    app_to_row: dict[int, int],
    alpha: float,
) -> np.ndarray | None:
    pos_vector = mean_vector(positive_app_ids, emb, app_to_row)
    if pos_vector is None:
        return None
    neg_vector = mean_vector(negative_app_ids, emb, app_to_row)
    if neg_vector is None:
        return pos_vector
    return normalize_vector(pos_vector - alpha * neg_vector)


def build_vectors_for_experiment(
    eval_users: list[int],
    method: str,
    min_positive_history: int,
    emb: np.ndarray,
    app_to_row: dict[int, int],
    train_positive: dict[int, set[int]],
    train_positive_hours: dict[int, dict[int, float]],
    train_negative: dict[int, set[int]],
    alpha: float | None = None,
) -> tuple[np.ndarray, list[int]]:
    vectors = []
    kept_users = []

    for user_id in eval_users:
        positive_app_ids = train_positive.get(user_id, set())
        if len(positive_app_ids) < min_positive_history:
            continue

        if method == "simple_mean":
            vector = mean_vector(positive_app_ids, emb, app_to_row)
        elif method == "hours_weighted_mean":
            vector = weighted_positive_vector(train_positive_hours.get(user_id, {}), emb, app_to_row)
        elif method == "pos_minus_neg":
            if alpha is None:
                raise ValueError("pos_minus_neg에는 alpha가 필요합니다.")
            vector = pos_minus_neg_vector(
                positive_app_ids,
                train_negative.get(user_id, set()),
                emb,
                app_to_row,
                alpha,
            )
        else:
            raise ValueError(f"알 수 없는 method: {method}")

        if vector is None:
            continue
        vectors.append(vector)
        kept_users.append(user_id)

    if not vectors:
        return np.empty((0, emb.shape[1]), dtype=np.float32), []
    return np.vstack(vectors).astype(np.float32), kept_users


def base_eval_users(
    sampled_users: np.ndarray,
    train_positive: dict[int, set[int]],
    eval_positive: dict[int, set[int]],
    app_to_row: dict[int, int],
    max_eval_users: int,
    seed: int,
) -> list[int]:
    users = []
    for user_id in sampled_users:
        user_id = int(user_id)
        has_train_positive = bool(train_positive.get(user_id))
        has_eval_target = any(app_id in app_to_row for app_id in eval_positive.get(user_id, set()))
        if has_train_positive and has_eval_target:
            users.append(user_id)

    if len(users) > max_eval_users:
        rng = np.random.default_rng(seed + 1)
        users = rng.choice(np.array(users, dtype=np.int64), size=max_eval_users, replace=False).tolist()
        users.sort()
    return users


def targets_for_users(
    eval_users: list[int],
    eval_positive: dict[int, set[int]],
    app_to_row: dict[int, int],
) -> dict[int, set[int]]:
    return {
        user_id: set(app_id for app_id in eval_positive[user_id] if app_id in app_to_row)
        for user_id in eval_users
    }


def score_for_selection(row: pd.Series) -> float:
    return (
        float(row["recall"]) * 1.0
        + float(row["ndcg"]) * 1.0
        + float(row["cold_recall"]) * 1.0
        + float(row["coverage"]) * 0.01
    )


def write_summary(
    path: Path,
    comparison_k10: pd.DataFrame,
    best_row: pd.Series,
    baseline_row: pd.Series,
    stats: dict[str, int | float | str],
) -> None:
    display = comparison_k10[
        [
            "experiment",
            "method",
            "min_positive_history",
            "alpha",
            "n_eval_users",
            "recall",
            "ndcg",
            "coverage",
            "cold_recall",
            "avg_train_popularity",
            "selection_score",
        ]
    ].copy()
    display = display.rename(
        columns={
            "recall": "Recall@10",
            "ndcg": "NDCG@10",
            "coverage": "Coverage",
            "cold_recall": "Cold Recall@10",
            "avg_train_popularity": "Avg Train Popularity",
        }
    )

    recall_delta = best_row["recall"] - baseline_row["recall"]
    ndcg_delta = best_row["ndcg"] - baseline_row["ndcg"]
    cold_delta = best_row["cold_recall"] - baseline_row["cold_recall"]
    coverage_delta = best_row["coverage"] - baseline_row["coverage"]
    popularity_delta = best_row["avg_train_popularity"] - baseline_row["avg_train_popularity"]

    if best_row["experiment"] == baseline_row["experiment"]:
        verdict = "개선 실험이 baseline을 넘지는 못했다. 기존 simple mean이 이 sample에서는 가장 안정적이다."
    elif recall_delta > 0 or ndcg_delta > 0:
        verdict = "best 설정은 정확도 지표에서 baseline보다 개선됐다."
    elif cold_delta > 0 or coverage_delta > 0:
        verdict = "best 설정은 일반 정확도보다는 cold-start 또는 다양성 측면에서 baseline보다 낫다."
    else:
        verdict = "best 설정의 selection score는 높지만 핵심 지표별 개선은 제한적이다."

    lines = [
        "# Text Recommendation Improvement Experiments",
        "",
        "## 실험 설정",
        "",
        f"- Train: date < {baseline.TRAIN_END}",
        f"- Validation: {baseline.TRAIN_END} <= date < {baseline.VAL_END}",
        f"- Test: {baseline.VAL_END} <= date < {baseline.TEST_END}",
        f"- 평가 split: {stats['eval_split']}",
        f"- Train interaction >= {stats['min_train_interactions']} user를 seed {stats['seed']}로 sample",
        f"- Sampled train users: {stats['sampled_train_users']:,}",
        f"- Base eval users before positive-history filtering: {stats['base_eval_users']:,}",
        "",
        "## 비교 결과 (@10)",
        "",
        baseline.dataframe_to_markdown(display),
        "",
        "## 자동 선택 best setting",
        "",
        f"- experiment: `{best_row['experiment']}`",
        f"- method: `{best_row['method']}`",
        f"- min_positive_history: {int(best_row['min_positive_history'])}",
        f"- alpha: {best_row['alpha']}",
        f"- selection_score: {best_row['selection_score']:.6f}",
        "",
        "## Baseline 대비 변화",
        "",
        f"- Recall@10: {baseline_row['recall']:.6f} -> {best_row['recall']:.6f} ({recall_delta:+.6f})",
        f"- NDCG@10: {baseline_row['ndcg']:.6f} -> {best_row['ndcg']:.6f} ({ndcg_delta:+.6f})",
        f"- Coverage: {baseline_row['coverage']:.6f} -> {best_row['coverage']:.6f} ({coverage_delta:+.6f})",
        f"- Cold Recall@10: {baseline_row['cold_recall']:.6f} -> {best_row['cold_recall']:.6f} ({cold_delta:+.6f})",
        f"- Avg Train Popularity: {baseline_row['avg_train_popularity']:.2f} -> {best_row['avg_train_popularity']:.2f} ({popularity_delta:+.2f})",
        "",
        "## 해석",
        "",
        verdict,
        "",
        (
            "log1p(hours) weighting은 오래 플레이한 positive item을 user vector에 더 강하게 반영한다. "
            "성능이 좋아졌다면 단순 추천 여부보다 playtime이 취향 강도를 더 잘 표현했다는 뜻이고, "
            "나빠졌다면 hours가 취향보다 인기작/장시간 플레이 장르에 user vector를 과하게 끌고 갔을 가능성이 있다."
        ),
        "",
        (
            "positive_mean - alpha * negative_mean은 비추천한 게임 방향을 user vector에서 빼는 방식이다. "
            "개선되면 dislike signal이 의미 있었던 것이고, 악화되면 negative interaction이 적거나 noisy해서 "
            "좋아하는 장르와 가까운 방향까지 함께 깎았을 수 있다."
        ),
        "",
        (
            "positive history threshold를 높이면 user vector는 안정적이지만 평가 user가 줄어든다. "
            "따라서 threshold별 결과는 n_eval_users와 함께 해석해야 한다."
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--min-train-interactions", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=100_000)
    parser.add_argument("--max-eval-users", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] static game/text files", flush=True)
    games, text_ready, emb_index, emb = baseline.load_static_frames()
    catalog_app_ids = emb_index["app_id"].astype(int).tolist()
    app_to_row = {int(app_id): int(row) for app_id, row in zip(emb_index["app_id"], emb_index["row"])}

    print("[scan1] recommendation stats and train counts", flush=True)
    train_user_counts, item_train_counts, rec_app_ids, total_rows = baseline.first_scan_recommendations(args.chunksize)
    stats = baseline.compute_alignment_stats(games, text_ready, emb_index, emb, rec_app_ids)
    stats["recommendation_rows"] = int(total_rows)

    sampled_users = baseline.sample_train_users(
        train_user_counts,
        args.min_train_interactions,
        args.max_users,
        args.seed,
    )
    stats["train_eligible_users"] = int((train_user_counts >= args.min_train_interactions).sum())
    stats["sampled_train_users"] = int(len(sampled_users))
    print(f"[users] sampled={len(sampled_users):,}", flush=True)

    print("[scan2] collect histories with hours and negatives", flush=True)
    train_seen, train_positive, train_positive_hours, train_negative, eval_positive = collect_user_histories(
        sampled_users,
        args.eval_split,
        args.chunksize,
    )

    eval_users_base = base_eval_users(
        sampled_users,
        train_positive,
        eval_positive,
        app_to_row,
        args.max_eval_users,
        args.seed,
    )
    if not eval_users_base:
        raise ValueError("평가 가능한 user가 없습니다.")
    stats["base_eval_users"] = int(len(eval_users_base))

    item_train_counts_full = item_train_counts.reindex(catalog_app_ids, fill_value=0).astype(np.int64)
    cold_app_ids = set(item_train_counts_full[item_train_counts_full == 0].index.astype(int).tolist())
    sorted_app_ids = item_train_counts_full.sort_values(ascending=False, kind="mergesort").index.astype(int).tolist()

    experiments = []
    for min_positive_history in [1, 3, 5]:
        experiments.append(
            {
                "experiment": f"simple_mean_pos>={min_positive_history}",
                "method": "simple_mean",
                "min_positive_history": min_positive_history,
                "alpha": np.nan,
            }
        )
        experiments.append(
            {
                "experiment": f"hours_weighted_pos>={min_positive_history}",
                "method": "hours_weighted_mean",
                "min_positive_history": min_positive_history,
                "alpha": np.nan,
            }
        )
        for alpha in [0.2, 0.5]:
            experiments.append(
                {
                    "experiment": f"pos_minus_{alpha}_neg_pos>={min_positive_history}",
                    "method": "pos_minus_neg",
                    "min_positive_history": min_positive_history,
                    "alpha": alpha,
                }
            )

    k_values = [5, 10, 20]
    k_max = max(k_values)
    metric_rows = []
    user_count_rows = []
    popularity_thresholds_done: set[int] = set()

    for exp in experiments:
        print(f"[experiment] {exp['experiment']}", flush=True)
        min_positive_history = int(exp["min_positive_history"])
        if min_positive_history not in popularity_thresholds_done:
            pop_eval_users = [
                user_id
                for user_id in eval_users_base
                if len(train_positive.get(user_id, set())) >= min_positive_history
            ]
            pop_targets = targets_for_users(pop_eval_users, eval_positive, app_to_row)
            pop_eval_users = [user_id for user_id in pop_eval_users if pop_targets[user_id]]
            pop_targets = {user_id: pop_targets[user_id] for user_id in pop_eval_users}
            pop_cold_targets = {user_id: pop_targets[user_id] & cold_app_ids for user_id in pop_eval_users}
            pop_warm_targets = {user_id: pop_targets[user_id] - cold_app_ids for user_id in pop_eval_users}
            pop_recs = baseline.popularity_recommend(pop_eval_users, train_seen, sorted_app_ids, k_max)
            pop_rows = baseline.make_metric_rows(
                f"popularity_ref_pos>={min_positive_history}",
                pop_recs,
                pop_targets,
                pop_cold_targets,
                pop_warm_targets,
                len(catalog_app_ids),
                item_train_counts_full,
                k_values,
            )
            for row in pop_rows:
                row.update(
                    {
                        "method": "popularity_reference",
                        "min_positive_history": min_positive_history,
                        "alpha": np.nan,
                    }
                )
            metric_rows.extend(pop_rows)
            popularity_thresholds_done.add(min_positive_history)

        user_vectors, eval_users = build_vectors_for_experiment(
            eval_users_base,
            exp["method"],
            min_positive_history,
            emb,
            app_to_row,
            train_positive,
            train_positive_hours,
            train_negative,
            None if pd.isna(exp["alpha"]) else float(exp["alpha"]),
        )
        targets = targets_for_users(eval_users, eval_positive, app_to_row)
        keep_indices = [idx for idx, user_id in enumerate(eval_users) if targets[user_id]]
        eval_users = [eval_users[idx] for idx in keep_indices]
        user_vectors = user_vectors[keep_indices]
        targets = {user_id: targets[user_id] for user_id in eval_users}

        if not eval_users:
            print(f"[skip] {exp['experiment']} has no eval users", flush=True)
            continue

        cold_targets = {user_id: targets[user_id] & cold_app_ids for user_id in eval_users}
        warm_targets = {user_id: targets[user_id] - cold_app_ids for user_id in eval_users}

        text_recs = baseline.text_recommend(
            user_vectors,
            eval_users,
            train_seen,
            emb,
            emb_index,
            k_max,
            args.batch_size,
        )
        rows = baseline.make_metric_rows(
            exp["experiment"],
            text_recs,
            targets,
            cold_targets,
            warm_targets,
            len(catalog_app_ids),
            item_train_counts_full,
            k_values,
        )
        for row in rows:
            row.update(
                {
                    "method": exp["method"],
                    "min_positive_history": int(exp["min_positive_history"]),
                    "alpha": exp["alpha"],
                }
            )
        metric_rows.extend(rows)
        user_count_rows.append(
            {
                "experiment": exp["experiment"],
                "n_eval_users": len(eval_users),
                "n_cold_eval_users": sum(1 for value in cold_targets.values() if value),
                "n_users_with_negative_train": sum(1 for user_id in eval_users if train_negative.get(user_id)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics["experiment"] = metrics["model"]
    metrics["selection_score"] = metrics.apply(score_for_selection, axis=1)
    metrics.to_csv(OUT_DIR / "text_recommendation_improvement_metrics_all_k.csv", index=False)

    comparison_k10 = metrics.loc[metrics["k"] == 10].copy()
    comparison_k10 = comparison_k10.sort_values(
        ["selection_score", "recall", "ndcg", "cold_recall", "coverage"],
        ascending=False,
    )
    comparison_k10.to_csv(OUT_DIR / "text_recommendation_improvement_comparison_k10.csv", index=False)
    pd.DataFrame(user_count_rows).to_csv(OUT_DIR / "text_recommendation_improvement_user_counts.csv", index=False)

    baseline_row = comparison_k10.loc[comparison_k10["experiment"] == "simple_mean_pos>=1"].iloc[0]
    best_candidates = comparison_k10.loc[comparison_k10["method"] != "popularity_reference"]
    best_row = best_candidates.iloc[0]

    stats.update(
        {
            "eval_split": args.eval_split,
            "min_train_interactions": int(args.min_train_interactions),
            "max_users": int(args.max_users),
            "max_eval_users": int(args.max_eval_users),
            "seed": int(args.seed),
            "best_experiment": str(best_row["experiment"]),
            "best_selection_score": float(best_row["selection_score"]),
        }
    )
    (OUT_DIR / "text_recommendation_improvement_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_summary(
        OUT_DIR / "text_recommendation_improvement_summary.md",
        comparison_k10,
        best_row,
        baseline_row,
        stats,
    )

    print("\n=== Improvement comparison @10 ===")
    print(
        comparison_k10[
            [
                "experiment",
                "n_eval_users",
                "recall",
                "ndcg",
                "coverage",
                "cold_recall",
                "avg_train_popularity",
                "selection_score",
            ]
        ].to_string(index=False)
    )
    print("\n=== Best setting ===")
    print(best_row[["experiment", "recall", "ndcg", "coverage", "cold_recall", "avg_train_popularity", "selection_score"]])
    print(f"\n결과 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
