"""Downstream task 라벨 컬럼 진단 스크립트 — 설계 결정용 사전 확인, 정식 Step 아님.

실험계획서 §7.2가 zero-shot classification 라벨 후보로 school/type/timeframe을 언급하지만
실제 클래스 개수·분포를 본 적이 없어 어느 컬럼을 쓸지 결정하지 못한 상태다. 이 스크립트는 그
판단을 위한 순수 조회(read-only)이며, 정식 분석이 아니므로 JSON/그림 저장은 가볍게만 하고
pytest 테스트는 두지 않는다.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.gap import _relpath
from src.viz.style import BURGUNDY_CMAP, COLORS, new_figure, save_figure

# 막대 그라데이션 샘플링 상한 -- 0.0~1.0 전체를 쓰면 꼬리 막대(순위 최하위)가 흰 배경에 거의
# 안 보일 정도로 옅어지므로, 가장 옅은 색도 최소한의 존재감이 남게 0.85까지만 샘플링한다.
_GRADIENT_MAX = 0.85

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "semart_v1_modality_gap_dataset.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "downstream_label_distribution.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

CANDIDATE_COLUMNS = ["school", "type", "timeframe"]

RARE_CLASS_THRESHOLD = 10
PLOT_TOP_N = 20

# zero-shot classification 라벨 추천 기준 (PROPOSAL — 이 스크립트가 정하는 값, 팀 재논의 가능):
# - 클래스 수 5~30개: 너무 적으면(<5) 태스크 변별력이 없고, 너무 많으면(>30) CLIP 프롬프트
#   후보 풀이 커져 평가가 무거워지고 애매/희소 클래스가 늘어난다.
# - top1_ratio < 0.5: 최빈 클래스 비율이 너무 높으면 그 클래스만 찍어도 정확도가 높게 나와
#   태스크 자체가 무의미해진다 (분류기 성능이 아니라 클래스 불균형을 측정하게 됨).
MIN_CLASSES = 5
MAX_CLASSES = 30
MAX_TOP1_RATIO = 0.5

# school 등에서 국가/화파가 세미콜론·슬래시로 같이 들어있는 경우 "클래스" 정의가 모호해지므로 플래그.
_MULTIVALUE_PATTERN = re.compile(r"[;/]")


def find_rare_classes(
    value_counts: pd.Series,
    n_total: int,
    min_count: float | None = None,
    min_ratio: float | None = None,
) -> pd.Series:
    """value_counts 중 절대값(min_count) 또는 전체 대비 비율(min_ratio) 미만인 클래스를 찾는다.

    check_label_distribution의 기존 "<10개 절대값" rare-class 판정을 일반화한 것 —
    min_count=10을 주면 기존과 완전히 동일한 결과, min_ratio=0.01을 주면 전체 표본의 1% 미만인
    클래스를 찾는다 (zeroshot_baseline.py의 최종 클래스셋 필터가 이 방식 사용).
    둘 중 정확히 하나만 지정한다.
    """
    if (min_count is None) == (min_ratio is None):
        raise ValueError("exactly one of min_count or min_ratio must be given")
    threshold = min_count if min_count is not None else min_ratio * n_total
    return value_counts[value_counts < threshold]


def check_label_distribution(df: pd.DataFrame, column: str) -> dict:
    series = df[column]
    n_total = len(series)
    n_missing = int(series.isna().sum())

    value_counts = series.dropna().value_counts()
    n_unique = int(value_counts.shape[0])

    if n_unique > 0:
        top1_class = str(value_counts.index[0])
        top1_ratio = float(value_counts.iloc[0]) / n_total
        top3_ratio = float(value_counts.iloc[:3].sum()) / n_total
    else:
        top1_class = None
        top1_ratio = 0.0
        top3_ratio = 0.0

    rare = find_rare_classes(value_counts, n_total, min_count=RARE_CLASS_THRESHOLD)

    return {
        "column": column,
        "n_total": n_total,
        "n_missing": n_missing,
        "n_missing_ratio": n_missing / n_total if n_total > 0 else 0.0,
        "n_unique": n_unique,
        "top1_class": top1_class,
        "top1_ratio": top1_ratio,
        "top3_ratio": top3_ratio,
        "rare_classes": rare.index.tolist(),
        "n_rare_classes": int(len(rare)),
        "value_counts": {str(k): int(v) for k, v in value_counts.items()},
    }


def check_multivalue(df: pd.DataFrame, column: str) -> dict:
    """세미콜론/슬래시로 여러 값이 같이 들어간 셀이 있는지 확인 (있으면 "클래스" 정의가 모호해짐)."""
    non_null = df[column].dropna().astype(str)
    flagged = non_null[non_null.str.contains(_MULTIVALUE_PATTERN)]
    return {
        "column": column,
        "n_flagged": int(len(flagged)),
        "flagged_ratio": len(flagged) / len(non_null) if len(non_null) > 0 else 0.0,
        "examples": sorted(flagged.unique().tolist())[:10],
    }


def _plot_distribution(value_counts: dict, column: str, output_path: Path, top_n: int = PLOT_TOP_N) -> None:
    items = sorted(value_counts.items(), key=lambda kv: kv[1], reverse=True)
    if len(items) > top_n:
        top_items = items[:top_n]
        other_count = sum(v for _, v in items[top_n:])
        top_items = top_items + [(f"(other, {len(items) - top_n} classes)", other_count)]
    else:
        top_items = items

    labels = [k for k, _ in top_items]
    counts = [v for _, v in top_items]

    has_other_bucket = len(items) > top_n
    n_ranked = len(labels) - 1 if has_other_bucket else len(labels)
    # items는 이미 내림차순 정렬(1위가 index 0)이므로, 그라데이션도 진한 색(1위)->옅은 색 순으로
    # 뽑아야 한다 -- linspace(0, MAX)로 뽑으면 순서가 뒤집혀 1위가 가장 옅게 나온다.
    bar_colors = [BURGUNDY_CMAP(x) for x in np.linspace(_GRADIENT_MAX, 0.0, n_ranked)]
    if has_other_bucket:
        bar_colors.append(COLORS["neutral"])

    fig, ax = new_figure("single_panel")
    ax.bar(range(len(labels)), counts, color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(f"{column} distribution (n_unique={len(items)})")
    fig.tight_layout()

    save_figure(fig, output_path)


def _recommend_column(columns_results: dict) -> dict:
    """MIN_CLASSES<=n_unique<=MAX_CLASSES 그리고 top1_ratio<MAX_TOP1_RATIO를 만족하는 컬럼을
    우선하고, 그중에서는 top1_ratio가 낮은(더 균형잡힌) 컬럼을 상위로 정렬한다."""
    scored = []
    for column, result in columns_results.items():
        n_unique = result["n_unique"]
        top1_ratio = result["top1_ratio"]
        in_range = MIN_CLASSES <= n_unique <= MAX_CLASSES
        low_imbalance = top1_ratio < MAX_TOP1_RATIO
        scored.append({
            "column": column,
            "n_unique": n_unique,
            "top1_ratio": top1_ratio,
            "in_class_count_range": in_range,
            "top1_ratio_acceptable": low_imbalance,
            "meets_criteria": in_range and low_imbalance,
        })

    ranked = sorted(scored, key=lambda s: (not s["meets_criteria"], s["top1_ratio"]))
    return {
        "criteria": (
            f"class count in [{MIN_CLASSES}, {MAX_CLASSES}] AND top1_ratio < {MAX_TOP1_RATIO}, "
            "then prefer lower top1_ratio (more balanced) among qualifying columns"
        ),
        "ranking": ranked,
        "top_choice": ranked[0]["column"] if ranked else None,
    }


def run_label_check(
    parquet_path: Path = PARQUET_PATH,
    results_path: Path = RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> dict:
    df = pd.read_parquet(parquet_path)

    columns_results = {}
    multivalue_flags = {}
    for column in CANDIDATE_COLUMNS:
        result = check_label_distribution(df, column)

        fig_path = figures_dir / f"downstream_label_distribution_{column}.png"
        _plot_distribution(result["value_counts"], column, fig_path)
        result["figure"] = _relpath(fig_path)

        columns_results[column] = result
        multivalue_flags[column] = check_multivalue(df, column)

    recommendation = _recommend_column(columns_results)

    results = {
        "source_parquet": _relpath(parquet_path),
        "columns": columns_results,
        "multivalue_flags": multivalue_flags,
        "recommendation": recommendation,
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def _print_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))

    print("\n[Downstream Label Check] column comparison:")
    header = f"{'column':<12}{'n_unique':>10}{'top1_ratio':>12}{'rare_classes':>14}"
    print(header)
    print("-" * len(header))
    for column, result in results["columns"].items():
        print(
            f"{column:<12}{result['n_unique']:>10}{result['top1_ratio']:>12.3f}"
            f"{result['n_rare_classes']:>14}"
        )

    print("\n[Downstream Label Check] multivalue flag (e.g. ';' or '/' separated values in one cell):")
    for column, flag in results["multivalue_flags"].items():
        print(f"{column:<12} n_flagged={flag['n_flagged']} ({flag['flagged_ratio']:.1%})")

    rec = results["recommendation"]
    print(f"\n[Downstream Label Check] recommendation criteria: {rec['criteria']}")
    for r in rec["ranking"]:
        print(
            f"  {r['column']:<12} meets_criteria={r['meets_criteria']!s:<6} "
            f"(n_unique={r['n_unique']}, top1_ratio={r['top1_ratio']:.3f})"
        )
    print(f"\n[Downstream Label Check] TOP CHOICE: {rec['top_choice']}")


if __name__ == "__main__":
    _print_summary(run_label_check())
