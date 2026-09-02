"""Step 7a/7b: embedding shift(λ) 스윕에 따른 지표 변화 — 실험계획서 6절 개입 진단.

*** Step 7b: retrieval + zero-shot classification 추가 (FACT/TODO) ***
- FACT: Mind the Gap 5.1절/Appendix B.1에 따라 각 downstream task는 자신의 이미지-텍스트 쌍으로
  별도 gap 벡터를 계산한다. retrieval은 기존 4개 조합(캡션 기반) gap 벡터를 그대로 재사용하고,
  classification은 "이미지-정답라벨" 쌍으로 새 gap 벡터 하나를 별도로 계산한다 — 즉 retrieval은
  4-key 결과, classification은 1-key(단일 곡선, Stage A/B 구분 없음) 결과가 정상이다.
- FACT: retrieval 4곡선과 classification 1곡선은 서로 다른 gap 벡터를 쓰므로, 같은 λ값이라도
  실제 gap 축소 폭(delta_gap_norm)이 다르다. 두 태스크의 λ를 직접 비교하지 말고, 각자의
  gap이 최소가 되는 λ 대비 상대적 위치로 비교할 것.
- TODO: shift 개입이 순수 기하학적 조작이라는 실험계획서 6.2절의 한계는 classification에서
  above_majority가 음전환하는지 여부로 직접 관찰 가능해진다 — 리포트의 한계 4번 항목과 연결할 것.

*** Step 7a: 4개 지표(delta_gap/linear_sep/paired_cosine/margin) ***
λ를 0(원본)에서부터 스윕하며 shift_embeddings()로 gap을 인위적으로 조절하고, 각 λ에서
4개 지표를 재계산한다:
- delta_gap_norm: shift_embeddings()가 재정규화 후 재계산해 반환하는 centroid distance.
- linear_sep_accuracy: gap.py의 linear_separability() 재사용 (image vs text 분류 정확도).
- paired_cosine_mean: 대응하는 이미지-텍스트 쌍의 평균 코사인 유사도 (gap.py의
  paired_cosine_comparison 내부에서 쓰는 것과 같은 einsum 패턴, 단일 조건이라 그 함수 자체는
  재사용하지 않음 — paired_cosine_comparison은 두 조건을 비교하는 함수라 여기 용도와 다름).
- margin: pair_margin.py의 compute_pair_margin() 재사용 (진짜 쌍 vs 오답 쌍 마진, 양방향).

shift_embeddings()는 새로 만들지 않고 src.interventions.shift에서 그대로 재사용한다.
Step 6(temperature)과 같은 이유로 n=2,000 부분표본(seed=42)을 재사용 — 대표성은 Step 6에서
이미 검증됨(상대오차 0.06~0.15%). GPU 불필요 — frozen 임베딩에 대한 순수 후처리.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.downstream.zeroshot_baseline import compute_confusion_matrix
from src.embeddings.extract import encode_texts, load_model
from src.interventions.shift import shift_embeddings
from src.metrics.gap import _relpath, compute_delta_gap, linear_separability
from src.metrics.pair_margin import compute_pair_margin
from src.viz.style import COLORS, combo_display_name, new_figure, place_legend_outside, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "clip_vitb32_embeddings.npz"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "semart_v1_modality_gap_dataset.parquet"
ZEROSHOT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "downstream_zeroshot_baseline.json"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step7a_shift_diagnostics_results.json"
STEP7B_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step7b_downstream_sweep_results.json"
STEP7C_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step7c_retrieval_extended_results.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Step 6과 동일한 부분표본 설정 (대표성 검증 재사용, 새로 검증하지 않음).
N_SUBSAMPLE = 2000
SUBSAMPLE_SEED = 42

# PROPOSAL: 0(원본)에서 gap이 닫히는 지점을 넘어 살짝 반전되는 구간까지 볼 수 있는 정도의 폭.
# temperature.py의 back-to-back(distance~2) 전 구간까지는 필요 없음 — 여긴 downstream 진단이라
# "gap을 줄였을 때/과하게 줄였을 때" 근방만 보면 충분.
LAMBDAS = np.linspace(0, 2, 21)

# PROPOSAL: linear_separability는 5-fold 로지스틱회귀라 λ 21개 × 조합 4개 스윕에 비용이 크다.
# 스윕 진단 목적이므로 3-fold로 낮춰 비용을 줄인다 (Step 4/5의 정밀 측정은 그대로 5-fold 유지).
SWEEP_LINEAR_SEP_N_SPLITS = 3

# Step 7b retrieval sweep 후보 풀 대표성 관련 note (JSON에도 동일 문구로 기록).
RETRIEVAL_CANDIDATE_POOL_NOTE = (
    "This sweep uses a candidate pool of N_SUBSAMPLE samples (same seed/subsample as step6/step7a), "
    "which differs from step6b_pair_margin's full 9,356-sample candidate pool. Margin analysis is a "
    "single point estimate where a larger pool is affordable; this sweep repeats the computation once "
    "per lambda across 4 combos, so a smaller pool keeps it tractable. Absolute retrieval numbers are "
    "therefore not directly comparable to a 9,356-pool retrieval task -- the point of this sweep is "
    "the *relative* trend across lambda, not the absolute R@k value."
)

# λ=0에서 R@1이 이보다 낮으면 파이프라인 버그 의심 (경고만, assert 없음 — 사용자 지정).
RETRIEVAL_R1_SANITY_FLOOR = 0.01

# classification λ 그리드 자동 확장 (step7a의 _auto_lambda_grid 아이디어 참고 — temperature.py에는
# 없고 이 모듈에서 독립적으로 둠. 확장 조건: (a) 관측된 max distance가 목표(back-to-back 근방)
# 미달, (b) peak_lambda가 그리드 끝단에 걸림 — 둘 중 하나라도 해당하면 확장. 무한루프 방지로
# 최대 2회만).
CLASSIFICATION_BASE_LAMBDA_MAX = 2.0
CLASSIFICATION_N_LAMBDA_POINTS = 21
CLASSIFICATION_TARGET_MAX_DISTANCE = 1.9
CLASSIFICATION_MAX_LAMBDA_EXPANSIONS = 2

# retrieval 확장판(step7c)도 classification과 동일한 규칙/상수값을 쓴다. 공용 함수로 뽑아내지 않고
# classification의 인라인 while-루프 패턴을 그대로 복제한다 — 파일/구조 정리는 다음 단계에서 별도로
#하기로 했으므로, 지금은 기존 run_downstream_sweep()의 classification 경로를 건드리지 않고
# retrieval 쪽에 같은 패턴을 새로 추가하는 쪽이 더 안전하다고 판단함.
RETRIEVAL_BASE_LAMBDA_MAX = 2.0
RETRIEVAL_N_LAMBDA_POINTS = 21
RETRIEVAL_TARGET_MAX_DISTANCE = 1.9
RETRIEVAL_MAX_LAMBDA_EXPANSIONS = 2


def _paired_cosine_mean(x: np.ndarray, y: np.ndarray) -> float:
    """대응하는 행끼리의 평균 코사인 유사도. gap.py의 paired_cosine_comparison과 같은 einsum
    패턴이지만, 그 함수는 두 조건(A/B)을 비교하는 용도라 단일 조건 λ-스윕에는 맞지 않아 별도로 둠."""
    return float(np.einsum("ij,ij->i", x, y).mean())


def compute_shift_diagnostics(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    lambdas: np.ndarray,
    seed: int = SUBSAMPLE_SEED,
    n_splits: int = SWEEP_LINEAR_SEP_N_SPLITS,
    margin_n_bootstrap: int = 2000,
) -> dict:
    """lambdas 전체에 대해 4개 지표를 계산. shift_embeddings()는 그대로 재사용."""
    delta_gap = compute_delta_gap(image_emb, text_emb)
    original_distance = float(np.linalg.norm(delta_gap))

    points = []
    for lam in lambdas:
        x_shift, y_shift, dist = shift_embeddings(image_emb, text_emb, delta_gap, float(lam))

        linsep = linear_separability(x_shift, y_shift, n_splits=n_splits, random_state=seed)
        paired_cosine_mean = _paired_cosine_mean(x_shift, y_shift)
        margin = compute_pair_margin(x_shift, y_shift, n_bootstrap=margin_n_bootstrap, seed=seed)
        margin.pop("_raw")

        points.append({
            "lambda": float(lam),
            "delta_gap_norm": dist,
            "linear_sep_accuracy": linsep["mean"],
            "paired_cosine_mean": paired_cosine_mean,
            "margin_i2t_mean": margin["image_to_text"]["mean"],
            "margin_t2i_mean": margin["text_to_image"]["mean"],
        })

    return {
        "original_distance": original_distance,
        "n": int(image_emb.shape[0]),
        "n_lambda_points": len(lambdas),
        "linear_sep_n_splits": n_splits,
        "margin_n_bootstrap": margin_n_bootstrap,
        "points": points,
    }


def _plot_shift_diagnostics(diag: dict, output_path: Path, title: str) -> None:
    lambdas = [p["lambda"] for p in diag["points"]]
    fig, axes = new_figure("grid_2x2", nrows=2, ncols=2)
    fig.suptitle(title)

    metrics = [
        ("delta_gap_norm", "||delta_gap||", axes[0, 0]),
        ("linear_sep_accuracy", "linear separability accuracy", axes[0, 1]),
        ("paired_cosine_mean", "mean paired cosine", axes[1, 0]),
        ("margin_i2t_mean", "margin (image->text) mean", axes[1, 1]),
    ]
    for key, ylabel, ax in metrics:
        values = [p[key] for p in diag["points"]]
        ax.plot(lambdas, values, marker="o", markersize=3, linewidth=1, color=COLORS["primary"])
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    save_figure(fig, output_path)


def run_shift_diagnostics(
    npz_path: Path = NPZ_PATH,
    results_path: Path = RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
    n_subsample: int = N_SUBSAMPLE,
    seed: int = SUBSAMPLE_SEED,
    lambdas: np.ndarray = LAMBDAS,
) -> dict:
    data = np.load(npz_path)
    n_total = data["image_emb"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_subsample, replace=False)

    image_sub = data["image_emb"][idx]
    visual_a_sub = data["text_emb_visual_a"][idx]
    visual_b_sub = data["text_emb_visual_b"][idx]
    contextual_sub = data["text_emb_contextual"][idx]

    combos = {
        "stage_a_visual": (image_sub, visual_a_sub),
        "stage_a_contextual": (image_sub, contextual_sub),
        "stage_b_visual": (image_sub, visual_b_sub),
        "stage_b_contextual": (image_sub, contextual_sub),
    }

    combo_results = {}
    for name, (x, y) in combos.items():
        title = combo_display_name(name, suffix="shift sweep")
        diag = compute_shift_diagnostics(x, y, lambdas, seed=seed)
        fig_path = figures_dir / f"step7a_shift_{name}.png"
        _plot_shift_diagnostics(diag, fig_path, title)
        diag["figure"] = _relpath(fig_path)
        combo_results[name] = diag

    # Stage A/B contextual sanity check — 같은 서브샘플 인덱스의 동일 배열이라 사실상 bit-identical.
    a_ctx = combo_results["stage_a_contextual"]["points"]
    b_ctx = combo_results["stage_b_contextual"]["points"]
    max_abs_diff = max(
        abs(pa["delta_gap_norm"] - pb["delta_gap_norm"]) for pa, pb in zip(a_ctx, b_ctx)
    )
    sanity_check = {"max_abs_diff_delta_gap_norm": max_abs_diff, "passed": max_abs_diff < 1e-6}

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_npz": _relpath(npz_path),
        "n_subsample": n_subsample,
        "subsample_seed": seed,
        "lambdas": [float(lam) for lam in lambdas],
        "combos": combo_results,
        "sanity_check_stage_a_b_contextual": sanity_check,
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def _print_step7a_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))

    print("\n[Step 7a] shift diagnostics summary (lambda=0 vs lambda=max):")
    header = f"{'combo':<22}{'orig_dist':>10}{'d@0':>10}{'d@max':>10}{'linsep@0':>10}{'linsep@max':>12}"
    print(header)
    print("-" * len(header))
    for name, diag in results["combos"].items():
        p0, pmax = diag["points"][0], diag["points"][-1]
        print(
            f"{name:<22}{diag['original_distance']:>10.4f}{p0['delta_gap_norm']:>10.4f}"
            f"{pmax['delta_gap_norm']:>10.4f}{p0['linear_sep_accuracy']:>10.3f}"
            f"{pmax['linear_sep_accuracy']:>12.3f}"
        )

    sanity = results["sanity_check_stage_a_b_contextual"]
    print(
        f"\n[sanity] stage_a_contextual vs stage_b_contextual max abs diff (delta_gap_norm) = "
        f"{sanity['max_abs_diff_delta_gap_norm']:.2e} -> passed={sanity['passed']}"
    )


def _ranks_from_similarity(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S: (n, n) 유사도 행렬. 반환: (ranks_i2t, ranks_t2i), 각 길이 n, 1-indexed 진짜쌍 순위
    (진짜쌍보다 유사도가 큰 후보 개수 + 1 — 동점은 순위를 나눠주지 않고 보수적으로 처리)."""
    diag = np.diag(S)
    ranks_i2t = (S > diag[:, None]).sum(axis=1) + 1
    ranks_t2i = (S > diag[None, :]).sum(axis=0) + 1
    return ranks_i2t, ranks_t2i


def _retrieval_metrics_from_ranks(ranks: np.ndarray) -> dict:
    return {
        "r_at_1": float((ranks <= 1).mean()),
        "r_at_5": float((ranks <= 5).mean()),
        "r_at_10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
    }


def compute_retrieval_diagnostics(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    lambdas: np.ndarray,
    n_subsample: int = N_SUBSAMPLE,
    seed: int = SUBSAMPLE_SEED,
) -> dict:
    """실험계획서 7절 retrieval — 기존 캡션 기반 gap 벡터(해당 조합 자신의 image/text)를 그대로
    사용해 λ별 R@1/R@5/R@10/median_rank/MRR을 양방향(I->T, T->I) 계산한다.

    n_subsample개를 후보 풀로 뽑는다 (Step 6/7a와 동일 시드 — 같은 image_emb/text_emb에 대해
    항상 같은 인덱스가 재현됨). shift_embeddings()로 λ만큼 개입한 뒤 대각선 순위로 지표를 낸다.
    """
    n_total = image_emb.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_subsample, replace=False)
    x_sub, y_sub = image_emb[idx], text_emb[idx]

    delta_gap = compute_delta_gap(x_sub, y_sub)
    original_distance = float(np.linalg.norm(delta_gap))

    points = []
    for lam in lambdas:
        x_shift, y_shift, dist = shift_embeddings(x_sub, y_sub, delta_gap, float(lam))
        S = x_shift @ y_shift.T
        ranks_i2t, ranks_t2i = _ranks_from_similarity(S)
        points.append({
            "lambda": float(lam),
            "delta_gap_norm": dist,
            "i2t": _retrieval_metrics_from_ranks(ranks_i2t),
            "t2i": _retrieval_metrics_from_ranks(ranks_t2i),
        })

    r1_i2t = [p["i2t"]["r_at_1"] for p in points]
    peak_idx = int(np.argmax(r1_i2t))
    diffs = np.diff(r1_i2t)
    if np.all(diffs <= 1e-12):
        direction = "monotonic_decreasing"
    elif np.all(diffs >= -1e-12):
        direction = "monotonic_increasing"
    else:
        direction = "non_monotonic"

    return {
        "n_subsample": n_subsample,
        "subsample_seed": seed,
        "original_distance": original_distance,
        "points": points,
        "retrieval_r1_peak_lambda": points[peak_idx]["lambda"],
        "retrieval_monotonic_direction": direction,
        "notes": {"candidate_pool_size": RETRIEVAL_CANDIDATE_POOL_NOTE},
    }


_RETRIEVAL_K_COLORS = {"r_at_1": COLORS["primary"], "r_at_5": COLORS["tertiary"], "r_at_10": COLORS["secondary"]}


def _plot_retrieval(diag: dict, output_path: Path, title: str) -> None:
    lambdas = [p["lambda"] for p in diag["points"]]
    fig, ax = new_figure("single_panel")
    for key, label in [("r_at_1", "R@1"), ("r_at_5", "R@5"), ("r_at_10", "R@10")]:
        values = [p["i2t"][key] for p in diag["points"]]
        ax.plot(
            lambdas, values, marker="o", markersize=3, linewidth=1,
            color=_RETRIEVAL_K_COLORS[key], label=f"{label} (I->T)",
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, label="original (lambda=0)")
    ax.set_xlabel("lambda")
    ax.set_ylabel("recall")
    ax.set_title(title)
    place_legend_outside(ax)

    save_figure(fig, output_path)


def _monotonic_direction(values: list[float]) -> str:
    diffs = np.diff(values)
    if np.all(diffs <= 1e-12):
        return "monotonic_decreasing"
    elif np.all(diffs >= -1e-12):
        return "monotonic_increasing"
    return "non_monotonic"


def _summarize_retrieval_directions(points: list[dict]) -> dict:
    """i2t/t2i x r_at_1/r_at_5/r_at_10 6개 조합 전부에 대해 peak_lambda/peak_value/
    value_at_lambda_0/monotonic_direction을 계산한다. compute_retrieval_diagnostics()는 이미
    양방향 x 전체 R@k를 계산해 points에 담아두므로, 여기서는 그걸 요약만 한다(재계산 없음)."""
    summary = {}
    for direction in ("i2t", "t2i"):
        for k in ("r_at_1", "r_at_5", "r_at_10"):
            values = [p[direction][k] for p in points]
            peak_idx = int(np.argmax(values))
            summary[f"{direction}_{k}"] = {
                "peak_lambda": points[peak_idx]["lambda"],
                "peak_value": values[peak_idx],
                "value_at_lambda_0": values[0],
                "monotonic_direction": _monotonic_direction(values),
            }
    return summary


def compute_retrieval_diagnostics_extended(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    n_subsample: int = N_SUBSAMPLE,
    seed: int = SUBSAMPLE_SEED,
    base_lambda_max: float = RETRIEVAL_BASE_LAMBDA_MAX,
    n_points: int = RETRIEVAL_N_LAMBDA_POINTS,
    target_max_distance: float = RETRIEVAL_TARGET_MAX_DISTANCE,
    max_expansions: int = RETRIEVAL_MAX_LAMBDA_EXPANSIONS,
) -> dict:
    """compute_retrieval_diagnostics()를 그대로 재사용하면서, classification과 동일한 규칙으로
    λ 그리드를 자동 확장한다: I->T R@1의 peak_lambda가 그리드 끝단에 걸리거나 관측된 max distance가
    목표(back-to-back 근방) 미달이면 λ 상한을 배증 (최대 max_expansions회, 무한루프 방지).

    후보 풀 프로토콜은 compute_retrieval_diagnostics()와 완전히 동일 — n_subsample개를 뽑아
    이미지/텍스트 둘 다 그 n_subsample개로 제한한 뒤(즉 후보 풀 자체가 n_subsample), 그 안에서
    순위를 매긴다. 이 확장판도 같은 프로토콜을 그대로 유지한다(새로 만들지 않음).
    """
    max_lam = base_lambda_max
    n_expansions = 0
    while True:
        lambdas = np.linspace(0, max_lam, n_points)
        diag = compute_retrieval_diagnostics(image_emb, text_emb, lambdas, n_subsample=n_subsample, seed=seed)
        max_dist = max(p["delta_gap_norm"] for p in diag["points"])
        peak_lambda = diag["retrieval_r1_peak_lambda"]
        at_boundary = abs(peak_lambda - max_lam) < 1e-9
        needs_expansion = at_boundary or max_dist < target_max_distance
        if not needs_expansion or n_expansions >= max_expansions:
            break
        max_lam *= 2
        n_expansions += 1

    diag["lambda_grid_expansion"] = {
        "base_lambda_max": base_lambda_max,
        "final_lambda_max": max_lam,
        "n_expansions": n_expansions,
        "target_max_distance": target_max_distance,
        "peak_still_at_final_boundary": abs(peak_lambda - max_lam) < 1e-9,
    }
    diag["direction_k_summary"] = _summarize_retrieval_directions(diag["points"])
    return diag


def _plot_retrieval_extended(diag: dict, output_path: Path, title: str) -> None:
    """조합당 1장: I->T/T->I를 좌우 서브플롯으로 나누고, 각 서브플롯에 R@1/5/10을 겹쳐 그린다
    (28장에서 과하게 늘어나지 않도록 4개 조합 x 1장으로 유지)."""
    lambdas = [p["lambda"] for p in diag["points"]]
    fig, axes = new_figure("dual_panel_wide", nrows=1, ncols=2)
    fig.suptitle(title)

    for ax, direction, dlabel in [(axes[0], "i2t", "I->T"), (axes[1], "t2i", "T->I")]:
        for key, label in [("r_at_1", "R@1"), ("r_at_5", "R@5"), ("r_at_10", "R@10")]:
            values = [p[direction][key] for p in diag["points"]]
            ax.plot(lambdas, values, marker="o", markersize=3, linewidth=1, color=_RETRIEVAL_K_COLORS[key], label=label)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, label="original (lambda=0)")
        ax.set_xlabel("lambda")
        ax.set_ylabel("recall")
        ax.set_title(dlabel)

    # 두 서브플롯 다 legend 내용이 동일(R@1/5/10 + original)하므로 place_legend_outside와 같은
    # bbox_to_anchor 스타일을 서브플롯마다 적용 -- 가운데 legend(ax0)는 두 서브플롯 사이 wspace 여백에,
    # 오른쪽 legend(ax1)는 전체 그림 우측 바깥 여백에 들어가도록 wspace/right를 넉넉히 넓힌다
    # (내부 우하단에 두면 R@1 lambda~4 근방 마지막 점들과 겹쳐 색이 바래 보이는 문제가 있었음).
    fig.subplots_adjust(wspace=0.65, right=0.86, top=0.85)
    for ax in axes:
        ax.legend(bbox_to_anchor=(1.03, 1), loc="upper left", borderaxespad=0.0, fontsize=9)

    save_figure(fig, output_path)


def run_retrieval_extended_sweep(
    npz_path: Path = NPZ_PATH,
    results_path: Path = STEP7C_RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
    n_subsample: int = N_SUBSAMPLE,
    seed: int = SUBSAMPLE_SEED,
) -> dict:
    """retrieval 확장 스윕 — T->I 추가, R@5/R@10 노출, λ 자동 확장. step7b_downstream_sweep_results.json은
    건드리지 않고 별도 파일에 저장한다(기존 필드 보존 우선 — 사용자 지정).
    """
    data = np.load(npz_path)
    image_emb = data["image_emb"]

    combos = {
        "stage_a_visual": data["text_emb_visual_a"],
        "stage_a_contextual": data["text_emb_contextual"],
        "stage_b_visual": data["text_emb_visual_b"],
        "stage_b_contextual": data["text_emb_contextual"],
    }

    # 회귀 체크 — step7b_downstream_sweep_results.json 생성 당시 기록된 기존 I->T R@1@lambda=0 값
    # (검증 리포트 기준). 이 확장판이 같은 프로토콜을 유지한다면 정확히 재현되어야 한다.
    expected_r1_at_lambda0 = {
        "stage_a_visual": 0.1425,
        "stage_a_contextual": 0.1030,
        "stage_b_visual": 0.1220,
        "stage_b_contextual": 0.1030,
    }

    results_by_combo = {}
    regression_check = {}
    for name, text_emb in combos.items():
        diag = compute_retrieval_diagnostics_extended(image_emb, text_emb, n_subsample=n_subsample, seed=seed)

        r1_at_0 = diag["points"][0]["i2t"]["r_at_1"]
        expected = expected_r1_at_lambda0[name]
        regression_check[name] = {
            "expected_r1_at_lambda0": expected,
            "actual_r1_at_lambda0": r1_at_0,
            "abs_diff": abs(r1_at_0 - expected),
            "passed": abs(r1_at_0 - expected) < 1e-3,
        }

        fig_path = figures_dir / f"step7c_retrieval_extended_{name}.png"
        title = combo_display_name(name, suffix="retrieval, I->T & T->I")
        _plot_retrieval_extended(diag, fig_path, title)
        diag["figure"] = _relpath(fig_path)
        results_by_combo[name] = diag

    a_ctx_lams = [p["lambda"] for p in results_by_combo["stage_a_contextual"]["points"]]
    b_ctx_lams = [p["lambda"] for p in results_by_combo["stage_b_contextual"]["points"]]
    if a_ctx_lams == b_ctx_lams:
        a_ctx_pts = results_by_combo["stage_a_contextual"]["points"]
        b_ctx_pts = results_by_combo["stage_b_contextual"]["points"]
        max_diff = max(
            abs(pa["i2t"]["r_at_1"] - pb["i2t"]["r_at_1"]) for pa, pb in zip(a_ctx_pts, b_ctx_pts)
        )
        sanity = {"comparable": True, "max_abs_diff_r1": max_diff, "passed": max_diff < 1e-9}
    else:
        sanity = {
            "comparable": False,
            "note": "lambda grids differ between stage_a_contextual and stage_b_contextual after independent auto-expansion",
        }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_npz": _relpath(npz_path),
        "protocol_note": (
            "Candidate pool IS the n_subsample (2,000) draw itself: both image_emb and text_emb are "
            "subsampled to the same n_subsample indices before ranking (see compute_retrieval_diagnostics "
            "in this same module) -- this is NOT 'full 9,356 candidates with only queries subsampled'. "
            "Confirmed directly from source before writing this extension; resolves a prior contradictory "
            "note. The extended sweep below preserves this exact protocol unchanged."
        ),
        "regression_check_vs_step7b": regression_check,
        "combos": results_by_combo,
        "sanity_check_stage_a_b_contextual": sanity,
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def _print_step7c_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))

    print("\n[Step 7c] regression check vs step7b (I->T R@1 @ lambda=0):")
    for name, chk in results["regression_check_vs_step7b"].items():
        print(f"  {name:<22} expected={chk['expected_r1_at_lambda0']:.4f}  actual={chk['actual_r1_at_lambda0']:.4f}  passed={chk['passed']}")

    print("\n[Step 7c] lambda grid expansion per combo:")
    for name, diag in results["combos"].items():
        exp = diag["lambda_grid_expansion"]
        print(f"  {name:<22} base_max={exp['base_lambda_max']} -> final_max={exp['final_lambda_max']} ({exp['n_expansions']} exp.)")

    print("\n[Step 7c] direction x R@k summary (peak_lambda / peak_value / value@lambda=0 / direction):")
    header = f"{'combo':<22}{'metric':<14}{'peak_lam':>10}{'peak_val':>10}{'val@0':>10}{'shape':>22}"
    print(header)
    print("-" * len(header))
    for name, diag in results["combos"].items():
        for metric, s in diag["direction_k_summary"].items():
            print(
                f"{name:<22}{metric:<14}{s['peak_lambda']:>10.2f}{s['peak_value']:>10.4f}"
                f"{s['value_at_lambda_0']:>10.4f}{s['monotonic_direction']:>22}"
            )

    sanity = results["sanity_check_stage_a_b_contextual"]
    print(f"\n[sanity] stage_a_contextual vs stage_b_contextual: {sanity}")


def compute_classification_diagnostics(
    image_emb: np.ndarray,
    label_prompt_emb: np.ndarray,
    ground_truth_labels: np.ndarray,
    class_names: list[str],
    lambdas: np.ndarray,
    majority_baseline_accuracy: float,
) -> dict:
    """실험계획서 7절 zero-shot classification — "이미지-정답라벨" 쌍으로 별도 gap 벡터를 계산한다
    (Mind the Gap 5.1절/Appendix B.1: task마다 자신의 쌍으로 gap을 계산). Stage A/B 구분이 없는
    단일 곡선이다.

    label_prompt_emb: (n_classes, d) 클래스별 프롬프트 임베딩(class_names 순서와 1:1 대응).
    ground_truth_labels: (n,) 각 이미지의 정답 클래스명. 내부적으로 정답 클래스에 따라
    label_prompt_emb를 복제한 expanded_label_emb(n, d)를 만들어 기존 shift_embeddings()를
    그대로 호출한다 (새 shift 수식을 만들지 않음).
    """
    n = image_emb.shape[0]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    label_indices = np.array([class_to_idx[lbl] for lbl in ground_truth_labels])
    expanded_label_emb = label_prompt_emb[label_indices]

    delta_gap = compute_delta_gap(image_emb, expanded_label_emb)
    original_distance = float(np.linalg.norm(delta_gap))
    chance_accuracy = 1.0 / len(class_names)

    # 클래스별 첫 등장 인덱스 -- shift 후 프로토타입을 여기서 뽑는다.
    first_occurrence_idx = np.array([int(np.argmax(label_indices == i)) for i in range(len(class_names))])

    points = []
    for lam in lambdas:
        x_shift, y_shift, dist = shift_embeddings(image_emb, expanded_label_emb, delta_gap, float(lam))

        prototypes = y_shift[first_occurrence_idx]
        for i in range(len(class_names)):
            same_class_vectors = y_shift[label_indices == i]
            assert np.allclose(same_class_vectors, prototypes[i], atol=1e-5), (
                f"lambda={lam}: class '{class_names[i]}' shifted vectors are not identical -- "
                "identical inputs must produce identical shift+renormalize output"
            )

        sims = x_shift @ prototypes.T
        pred_idx = sims.argmax(axis=1)
        correct = pred_idx == label_indices
        micro_accuracy = float(correct.mean())

        recalls = [
            float(correct[label_indices == i].mean())
            for i in range(len(class_names)) if (label_indices == i).sum() > 0
        ]
        macro_accuracy = float(np.mean(recalls)) if recalls else None

        points.append({
            "lambda": float(lam),
            "delta_gap_norm": dist,
            "micro_accuracy": micro_accuracy,
            "macro_accuracy": macro_accuracy,
            "micro_accuracy_above_majority": micro_accuracy - majority_baseline_accuracy,
        })

    accs = [p["micro_accuracy"] for p in points]
    peak_lambda = points[int(np.argmax(accs))]["lambda"]

    always_positive = bool(all(p["micro_accuracy_above_majority"] > 0 for p in points))
    crossing_lambda = next(
        (p["lambda"] for p in points if p["micro_accuracy_above_majority"] <= 0), None,
    )

    return {
        "n": n,
        "n_classes": len(class_names),
        "class_names": class_names,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "chance_accuracy": chance_accuracy,
        "original_distance": original_distance,
        "points": points,
        "classification_accuracy_peak_lambda": peak_lambda,
        "above_majority_always_positive": always_positive,
        "above_majority_crosses_zero_at_lambda": crossing_lambda,
    }


def _plot_classification(result: dict, output_path: Path, title: str) -> None:
    lambdas = [p["lambda"] for p in result["points"]]
    micro = [p["micro_accuracy"] for p in result["points"]]
    macro = [p["macro_accuracy"] for p in result["points"]]

    fig, ax = new_figure("single_panel")
    ax.plot(lambdas, micro, marker="o", markersize=3, linewidth=1, color=COLORS["primary"], label="micro accuracy")
    ax.plot(lambdas, macro, marker="s", markersize=3, linewidth=1, color=COLORS["secondary"], label="macro accuracy")
    ax.axhline(
        result["majority_baseline_accuracy"], color=COLORS["neutral"], linestyle=":", linewidth=1,
        label="majority baseline (fixed)",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, label="original (lambda=0)")
    ax.set_xlabel("lambda")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    # "majority baseline (fixed)"가 다른 single_panel legend label보다 길어서 기본
    # right_margin(0.72)로는 잘림 -- pair_margin.py의 diagoffdiag와 같은 이유로 좁힌다.
    place_legend_outside(ax, right_margin=0.62)

    save_figure(fig, output_path)


def _classify_at_lambda(
    image_emb: np.ndarray,
    expanded_label_emb: np.ndarray,
    delta_gap: np.ndarray,
    class_names: list[str],
    label_indices: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float]:
    """compute_classification_diagnostics 내부 로직과 동일한 shift+프로토타입+argmax를 단일 λ에서
    한 번 실행하고 예측 라벨(문자열 배열)과 delta_gap_norm을 반환한다."""
    x_shift, y_shift, dist = shift_embeddings(image_emb, expanded_label_emb, delta_gap, lam)
    first_occurrence_idx = np.array([int(np.argmax(label_indices == i)) for i in range(len(class_names))])
    prototypes = y_shift[first_occurrence_idx]
    sims = x_shift @ prototypes.T
    pred_idx = sims.argmax(axis=1)
    pred_labels = np.array(class_names)[pred_idx]
    return pred_labels, dist


def compute_per_class_breakdown_at_lambdas(
    image_emb: np.ndarray,
    label_prompt_emb: np.ndarray,
    ground_truth_labels: np.ndarray,
    class_names: list[str],
    key_lambdas: dict[str, float],
) -> dict:
    """key_lambdas(예: {"lambda_0": 0.0, "trough": 0.9, "peak": 1.7}) 각각에서 클래스별 accuracy를
    계산한다. 예측→confusion matrix는 zeroshot_baseline.py의 compute_confusion_matrix를 그대로
    재사용 (새 로직을 만들지 않음). 관찰 전용 — 원인 해석 없음.
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    label_indices = np.array([class_to_idx[lbl] for lbl in ground_truth_labels])
    expanded_label_emb = label_prompt_emb[label_indices]
    delta_gap = compute_delta_gap(image_emb, expanded_label_emb)

    breakdown = {}
    for key, lam in key_lambdas.items():
        pred_labels, dist = _classify_at_lambda(
            image_emb, expanded_label_emb, delta_gap, class_names, label_indices, float(lam),
        )
        cm = compute_confusion_matrix(ground_truth_labels, pred_labels, class_names)

        per_class = {}
        for name in class_names:
            row = cm.loc[name]
            total = int(row.sum())
            correct = int(row[name])
            per_class[name] = {"n": total, "accuracy": correct / total if total > 0 else None}

        breakdown[key] = {
            "lambda": float(lam),
            "delta_gap_norm": dist,
            "per_class_accuracy": per_class,
        }
    return breakdown


def run_downstream_sweep(
    npz_path: Path = NPZ_PATH,
    parquet_path: Path = PARQUET_PATH,
    zeroshot_results_path: Path = ZEROSHOT_RESULTS_PATH,
    results_path: Path = STEP7B_RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
    n_subsample: int = N_SUBSAMPLE,
    seed: int = SUBSAMPLE_SEED,
    lambdas: np.ndarray = LAMBDAS,
) -> dict:
    data = np.load(npz_path)
    image_emb_full = data["image_emb"]
    filenames = data["filenames"]

    # --- 작업 A: retrieval, 기존 4개 조합의 캡션 기반 gap 벡터 재사용 ---
    retrieval_combos = {
        "stage_a_visual": (image_emb_full, data["text_emb_visual_a"]),
        "stage_a_contextual": (image_emb_full, data["text_emb_contextual"]),
        "stage_b_visual": (image_emb_full, data["text_emb_visual_b"]),
        "stage_b_contextual": (image_emb_full, data["text_emb_contextual"]),
    }

    retrieval_results = {}
    for name, (x, y) in retrieval_combos.items():
        diag = compute_retrieval_diagnostics(x, y, lambdas, n_subsample=n_subsample, seed=seed)

        r1_at_0 = diag["points"][0]["i2t"]["r_at_1"]
        if r1_at_0 < RETRIEVAL_R1_SANITY_FLOOR:
            print(
                f"[WARNING] {name}: R@1 at lambda=0 is {r1_at_0:.4f} "
                f"(< {RETRIEVAL_R1_SANITY_FLOOR:.0%}) -- given step6b_pair_margin's Cohen's d "
                "(1.6-2.0, very large effect), this is much lower than expected. Possible pipeline bug."
            )

        fig_path = figures_dir / f"step7b_retrieval_{name}.png"
        title = combo_display_name(name, suffix="retrieval, legacy snapshot")
        _plot_retrieval(diag, fig_path, title)
        diag["figure"] = _relpath(fig_path)
        retrieval_results[name] = diag

    a_ctx_r1 = [p["i2t"]["r_at_1"] for p in retrieval_results["stage_a_contextual"]["points"]]
    b_ctx_r1 = [p["i2t"]["r_at_1"] for p in retrieval_results["stage_b_contextual"]["points"]]
    retrieval_sanity = {
        "max_abs_diff_r1": max(abs(a - b) for a, b in zip(a_ctx_r1, b_ctx_r1)),
    }
    retrieval_sanity["passed"] = retrieval_sanity["max_abs_diff_r1"] < 1e-9

    # --- 작업 B: zero-shot classification, downstream_zeroshot_baseline.json의 final_prevalence_filtered 재사용 ---
    with open(zeroshot_results_path, "r", encoding="utf-8") as f:
        zeroshot_results = json.load(f)
    final_cfg = zeroshot_results["columns"]["type"]["final_prevalence_filtered"]

    class_names = list(final_cfg["per_class_accuracy"].keys())
    prompt_template = final_cfg["prompt_template"]
    majority_baseline_accuracy = final_cfg["majority_baseline_accuracy"]

    df = pd.read_parquet(parquet_path).set_index("filename")
    df_aligned = df.loc[filenames]
    all_labels = df_aligned["type"].to_numpy()
    keep_mask = np.isin(all_labels, class_names)
    image_emb_filtered = image_emb_full[keep_mask]
    ground_truth_labels = all_labels[keep_mask]

    model, processor = load_model("cpu")
    prompts = [prompt_template.format(name) for name in class_names]
    label_prompt_emb = encode_texts(prompts, model, processor, batch_size=len(prompts))

    # 작업 A: λ 그리드 자동 확장 — classification 자체 gap 벡터가 완전히 닫히는 지점과
    # back-to-back 구간까지 커버하도록. peak_lambda가 그리드 끝단에 걸리면 한 번 더 확장(최대 2회).
    max_lam = CLASSIFICATION_BASE_LAMBDA_MAX
    n_expansions = 0
    while True:
        classification_lambdas = np.linspace(0, max_lam, CLASSIFICATION_N_LAMBDA_POINTS)
        classification_result = compute_classification_diagnostics(
            image_emb_filtered, label_prompt_emb, ground_truth_labels, class_names,
            classification_lambdas, majority_baseline_accuracy,
        )
        max_dist = max(p["delta_gap_norm"] for p in classification_result["points"])
        peak_lambda = classification_result["classification_accuracy_peak_lambda"]
        at_boundary = abs(peak_lambda - max_lam) < 1e-9
        needs_expansion = at_boundary or max_dist < CLASSIFICATION_TARGET_MAX_DISTANCE
        if not needs_expansion or n_expansions >= CLASSIFICATION_MAX_LAMBDA_EXPANSIONS:
            break
        max_lam *= 2
        n_expansions += 1

    classification_result["lambda_grid_expansion"] = {
        "base_lambda_max": CLASSIFICATION_BASE_LAMBDA_MAX,
        "final_lambda_max": max_lam,
        "n_expansions": n_expansions,
        "target_max_distance": CLASSIFICATION_TARGET_MAX_DISTANCE,
        "peak_still_at_final_boundary": abs(peak_lambda - max_lam) < 1e-9,
    }

    # λ=0 sanity check — final_prevalence_filtered의 기존 정확값과 엄격히 일치해야 함 (assert).
    p0 = classification_result["points"][0]
    assert abs(p0["micro_accuracy"] - final_cfg["micro_accuracy"]) < 1e-4, (
        f"lambda=0 micro_accuracy mismatch: {p0['micro_accuracy']} vs "
        f"final_prevalence_filtered's {final_cfg['micro_accuracy']}"
    )
    assert abs(p0["macro_accuracy"] - final_cfg["macro_accuracy"]) < 1e-4, (
        f"lambda=0 macro_accuracy mismatch: {p0['macro_accuracy']} vs "
        f"final_prevalence_filtered's {final_cfg['macro_accuracy']}"
    )

    cls_fig_path = figures_dir / "step7b_classification_type9class.png"
    _plot_classification(classification_result, cls_fig_path, "type (9-class): accuracy vs lambda")
    classification_result["figure"] = _relpath(cls_fig_path)

    # 작업 B: 핵심 λ(원본/trough/peak) 지점에서 클래스별 accuracy 분해 — 관찰 전용, 원인 해석 없음.
    # 주의: 곡선이 λ=0에서 시작해 국소 최댓값(λ≈0.3)까지 올랐다가 다시 내려오는 모양이라, 그냥
    # 전체 최솟값을 찾으면 λ=0 자신이 뽑혀 "원본"과 "trough"가 같은 점이 되어버린다(실제로 한 번
    # 그렇게 나왔음). λ=0을 제외한 지점 중 최솟값을 진짜 trough(국소 저점)로 삼는다.
    nonzero_points = [p for p in classification_result["points"] if p["lambda"] > 0]
    trough_point = min(nonzero_points, key=lambda p: p["micro_accuracy"])
    peak_point = max(classification_result["points"], key=lambda p: p["micro_accuracy"])
    key_lambdas = {"lambda_0": 0.0, "trough": trough_point["lambda"], "peak": peak_point["lambda"]}
    classification_result["key_lambda_breakdown"] = compute_per_class_breakdown_at_lambdas(
        image_emb_filtered, label_prompt_emb, ground_truth_labels, class_names, key_lambdas,
    )

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_npz": _relpath(npz_path),
        "_schema_note": (
            "retrieval has 4 combo keys (stage_a/b x visual/contextual, each reusing that combo's "
            "own caption-based gap vector per Mind the Gap 5.1/Appendix B.1); classification has a "
            "single 'type_9class' key (separate image-vs-label gap vector, no Stage A/B distinction) "
            "-- this shape difference is intentional, not a bug."
        ),
        "retrieval": {**retrieval_results, "sanity_check_stage_a_b_contextual": retrieval_sanity},
        "classification": {"type_9class": classification_result},
        "notes": {
            "lambda_not_comparable_across_tasks": (
                "retrieval's 4 curves and classification's 1 curve use different gap vectors, so the "
                "same lambda value corresponds to a different actual gap-reduction magnitude in each. "
                "Do not compare lambda directly across the two tasks -- compare each task's position "
                "relative to its own gap-closing lambda instead."
            ),
            "shift_is_geometric_not_representational": (
                "Embedding shift is a purely geometric intervention (section 6.2 limitation): it moves "
                "centroids without improving representation quality. Whether above_majority flips "
                "negative in classification is a direct, observable symptom of this limitation -- "
                "connect this to report limitation item 4."
            ),
        },
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def _print_step7b_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))

    print("\n[Step 7b] retrieval summary (I->T):")
    header = f"{'combo':<22}{'R@1@0':>8}{'R@1@peak':>10}{'peak_lam':>10}{'direction':>22}"
    print(header)
    print("-" * len(header))
    for name, diag in results["retrieval"].items():
        if name == "sanity_check_stage_a_b_contextual":
            continue
        r1_0 = diag["points"][0]["i2t"]["r_at_1"]
        peak_lam = diag["retrieval_r1_peak_lambda"]
        r1_peak = next(p["i2t"]["r_at_1"] for p in diag["points"] if p["lambda"] == peak_lam)
        print(f"{name:<22}{r1_0:>8.3f}{r1_peak:>10.3f}{peak_lam:>10.2f}{diag['retrieval_monotonic_direction']:>22}")

    rsan = results["retrieval"]["sanity_check_stage_a_b_contextual"]
    print(
        f"\n[sanity] retrieval stage_a_contextual vs stage_b_contextual max abs diff (R@1) = "
        f"{rsan['max_abs_diff_r1']:.2e} -> passed={rsan['passed']}"
    )

    cls = results["classification"]["type_9class"]
    print("\n[Step 7b] classification (type, 9-class) summary:")
    print(
        f"  peak_lambda={cls['classification_accuracy_peak_lambda']:.2f}  "
        f"above_majority_always_positive={cls['above_majority_always_positive']}  "
        f"crosses_zero_at_lambda={cls['above_majority_crosses_zero_at_lambda']}"
    )
    exp = cls["lambda_grid_expansion"]
    print(
        f"  lambda grid: base_max={exp['base_lambda_max']} -> final_max={exp['final_lambda_max']} "
        f"({exp['n_expansions']} expansion(s)), peak_still_at_boundary={exp['peak_still_at_final_boundary']}"
    )

    print("\n[Step 7b] classification per-class accuracy at key lambda points (observation only):")
    breakdown = cls["key_lambda_breakdown"]
    keys = list(breakdown.keys())
    col_labels = [f"{k}(lam={breakdown[k]['lambda']:.2f})" for k in keys]
    header = f"{'class':<14}{'n':>6}" + "".join(f"{label:>20}" for label in col_labels)
    print(header)
    class_names_list = list(breakdown[keys[0]]["per_class_accuracy"].keys())
    for name in class_names_list:
        n = breakdown[keys[0]]["per_class_accuracy"][name]["n"]
        row = f"{name:<14}{n:>6}"
        for k in keys:
            acc = breakdown[k]["per_class_accuracy"][name]["accuracy"]
            row += f"{acc:>20.3f}" if acc is not None else f"{'--':>20}"
        print(row)


if __name__ == "__main__":
    import sys

    step = sys.argv[1] if len(sys.argv) > 1 else "step7a"
    if step == "step7a":
        _print_step7a_summary(run_shift_diagnostics())
    elif step == "step7b":
        _print_step7b_summary(run_downstream_sweep())
    elif step == "step7c":
        _print_step7c_summary(run_retrieval_extended_sweep())
    else:
        raise SystemExit(f"unknown step: {step!r} (expected 'step7a', 'step7b', or 'step7c')")
