"""Step 6: Temperature loss landscape probing (실험계획서 5절).

*** 5.1절 전제 (반드시 지킬 것) ***
frozen CLIP에서 임베딩만 추출하는 경우, temperature tau는 임베딩 벡터 자체에 아무 영향을
주지 않는다. tau는 contrastive loss의 softmax 스케일링 파라미터로 "학습 과정"에만 관여한다.
따라서 "temperature를 바꿔가며 gap을 측정한다"는 것은 성립하지 않는다. 여기서 하는 일은
임베딩을 고정한 채, 다양한 tau에서 contrastive loss 값이 어떻게 달라지는지 계산만 하는
loss landscape probing이다. gradient step이 하나도 없으므로 GPU가 전혀 필요 없다 (CPU 전용).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema
from scipy.special import logsumexp

from src.interventions.shift import shift_embeddings
from src.metrics.gap import _relpath, compute_delta_gap
from src.viz.style import combo_display_name, new_figure, place_legend_outside, save_figure, sequential_colors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "clip_vitb32_embeddings.npz"
STAGE_A_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step4_stage_a_results.json"
STAGE_B_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step5_stage_b_results.json"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step6_temperature_results.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# 실험계획서 5.2절 (3) — 논문(Mind the Gap)과 동일한 tau 후보. tau=1/100이 CLIP이 실제로 학습한 온도.
TAUS = [("1/100", 1 / 100), ("1/50", 1 / 50), ("1/30", 1 / 30), ("1/20", 1 / 20), ("1/10", 1 / 10), ("1", 1.0)]

# Mind the Gap도 계산량 때문에 n=5,000 부분표본을 씀 — 여기선 n=2,000, 고정 시드로 재현 가능하게.
N_SUBSAMPLE = 2000
SUBSAMPLE_SEED = 42

# repulsive structure(back-to-back local minimum) 판정 시 "상위 구간"으로 볼 distance 임계값
# (PROPOSAL — 원래 gap distance가 대략 0.85~0.89이므로 이보다 확실히 먼 구간)
REPULSIVE_REGION_THRESHOLD = 1.2

# 부분표본 대표성 체크 — 전체 표본 대비 상대오차가 이보다 크면 콘솔에 경고 (PROPOSAL)
REPRESENTATIVENESS_WARNING_THRESHOLD = 0.10


def contrastive_loss(x: np.ndarray, y: np.ndarray, tau: float) -> float:
    """실험계획서 5.2절 (2). L_{I->T}와 L_{T->I} 각각 계산 후 평균.

    N×N 유사도 행렬 x @ y.T / tau 사용. softmax를 직접 구현하면 overflow 위험이 있어
    scipy.special.logsumexp로 안정적으로 계산한다.
    """
    sim = (x @ y.T) / tau
    diag = np.diagonal(sim)
    loss_i2t = float((logsumexp(sim, axis=1) - diag).mean())
    loss_t2i = float((logsumexp(sim, axis=0) - diag).mean())
    return (loss_i2t + loss_t2i) / 2


def _compute_grid(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    delta_gap: np.ndarray,
    taus: list[float],
    lambdas: np.ndarray,
) -> dict[float, list[tuple[float, float, float]]]:
    """taus x lambdas 그리드를 lambda 증가 순서 그대로 계산 (정렬하지 않음).

    resulting_distance(lambda)는 실측 결과 단조가 아니다 — lambda=0에서 시작해 gap이 닫히는
    지점까지 감소했다가, 그 지점을 지나면 다시 증가해 back-to-back 방향으로 커진다(체크마크 모양).
    즉 같은 distance 값에 서로 다른 lambda(따라서 서로 다른 loss)가 대응할 수 있으므로, lambda
    순서를 보존해야 나중에 두 단조 구간(닫히는 구간/재개방 구간)으로 안전하게 나눌 수 있다.
    """
    result = {}
    for tau in taus:
        points = []
        for lam in lambdas:
            x_shift, y_shift, dist = shift_embeddings(image_emb, text_emb, delta_gap, lam)
            loss = contrastive_loss(x_shift, y_shift, tau)
            points.append((lam, dist, loss))
        result[tau] = points
    return result


def loss_landscape(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    delta_gap: np.ndarray,
    taus: list[float],
    lambdas: np.ndarray,
) -> dict[float, list[tuple[float, float]]]:
    """taus x lambdas 그리드 전체에 대해 (resulting_distance, loss) 쌍을 계산.

    tau를 키로 하는 dict를 반환하며, 각 tau의 값은 distance 오름차순으로 정렬된
    [(resulting_distance, loss), ...] 리스트다 (요청된 시그니처 그대로 — 전역 최솟값을 찾는 용도로는
    순서가 문제되지 않는다. 단, distance가 lambda에 대해 비단조라 이 정렬된 형태를 "distance의 함수"로
    그리거나 보간하면 안 된다 — 아래 _detect_repulsive_structure/plot은 별도로 lambda 순서를 보존한
    _compute_grid 결과를 쓴다).
    """
    grid = _compute_grid(image_emb, text_emb, delta_gap, taus, lambdas)
    result = {}
    for tau, points_ordered in grid.items():
        points = sorted(((dist, loss) for _, dist, loss in points_ordered), key=lambda p: p[0])
        result[tau] = points
    return result


def _auto_lambda_grid(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    delta_gap: np.ndarray,
    base_max: float = 3.0,
    n_points: int = 61,
    target_max_distance: float = 1.9,
    max_doublings: int = 5,
    safety_cap: float = 100.0,
) -> np.ndarray:
    """lambda 상한을 base_max에서 시작해, resulting distance가 target_max_distance(최대 가능값 2에
    근접)에 못 미치면 상한을 2배씩 늘려 재시도한다 — gap이 닫히는 지점 너머의 반전(back-to-back)
    구간까지 그리드에 포함시키기 위함(실험계획서 5.3절)."""
    max_lam = base_max
    for _ in range(max_doublings):
        lambdas = np.linspace(0, max_lam, n_points)
        distances = [shift_embeddings(image_emb, text_emb, delta_gap, lam)[2] for lam in lambdas]
        if max(distances) >= target_max_distance or max_lam >= safety_cap:
            return lambdas
        max_lam = min(max_lam * 2, safety_cap)
    return np.linspace(0, max_lam, n_points)


def _split_branches(
    points_ordered: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """points_ordered: [(lambda, distance, loss), ...] lambda 오름차순.

    distance(lambda)는 lambda=0에서 시작해 감소하다가 트로프(gap이 가장 많이 닫히는 지점)를 지나
    다시 증가하는 체크마크 모양이다(실측 확인). 트로프 인덱스를 기준으로 "닫히는 구간"(lambda 0~트로프,
    distance 단조감소)과 "재개방 구간"(트로프~lambda_max, distance 단조증가, back-to-back 방향)으로
    나눈다. 각 구간 안에서는 distance가 진짜 단조이므로 안전하게 보간할 수 있다.
    """
    distances = [d for _, d, _ in points_ordered]
    trough_idx = int(np.argmin(distances))
    branch_closing = points_ordered[: trough_idx + 1]
    branch_reopening = points_ordered[trough_idx:]
    return branch_closing, branch_reopening


def _detect_repulsive_structure(
    branch_reopening: list[tuple[float, float, float]],
    upper_region_threshold: float = REPULSIVE_REGION_THRESHOLD,
    n_interp: int = 200,
) -> tuple[bool, list[float] | None]:
    """branch_reopening: _split_branches()가 반환한 "재개방 구간"만 — 이 구간은 lambda 증가 순서가
    곧 distance 증가 순서이므로(단조) np.interp로 안전하게 균일 distance 그리드에 재샘플링한 뒤
    scipy.signal.argrelextrema로 극소점을 찾을 수 있다 (전체 구간을 distance로만 정렬해서 보간하면
    닫히는 구간과 재개방 구간이 뒤섞여 허위 극소점이 생긴다 — 실제로 그렇게 나와서 이 함수로 분리함).

    distance > upper_region_threshold 구간(원 논문의 "back-to-back" 위치에 해당, PROPOSAL 임계값)에
    극소점이 있으면 repulsive structure(back-to-back local minimum)로 판정한다.
    """
    if len(branch_reopening) < 3:
        return False, None

    distances = np.array([d for _, d, _ in branch_reopening])
    losses = np.array([l for _, _, l in branch_reopening])

    grid = np.linspace(distances.min(), distances.max(), n_interp)
    interp_losses = np.interp(grid, distances, losses)

    (local_min_idx,) = argrelextrema(interp_losses, np.less)
    if local_min_idx.size == 0:
        return False, None

    upper_mask = grid[local_min_idx] > upper_region_threshold
    if not upper_mask.any():
        return False, None

    region_distances = grid[local_min_idx][upper_mask]
    return True, [float(region_distances.min()), float(region_distances.max())]


def _summarize_condition(
    grid: dict[float, list[tuple[float, float, float]]],
    original_distance: float,
    tau_labels: dict[float, str],
) -> dict:
    per_tau = {}
    for tau, points_ordered in grid.items():
        distances = np.array([d for _, d, _ in points_ordered])
        losses = np.array([l for _, _, l in points_ordered])
        min_idx = int(np.argmin(losses))
        global_min_distance = float(distances[min_idx])

        _, branch_reopening = _split_branches(points_ordered)
        has_repulsive, region = _detect_repulsive_structure(branch_reopening)

        per_tau[tau_labels[tau]] = {
            "tau": tau,
            "global_min_distance": global_min_distance,
            "global_min_loss": float(losses[min_idx]),
            "original_minus_global_min_distance": global_min_distance - original_distance,
            "has_repulsive_structure": has_repulsive,
            "repulsive_region_distance": region,
        }

    # tau 값 오름차순으로 정렬해 has_repulsive_structure가 True->False로 바뀌는 지점을 bracket으로 추정
    # (6개 이산 tau만 테스트하므로 정확한 임계값이 아니라 근사 구간).
    ordered = sorted(per_tau.items(), key=lambda kv: kv[1]["tau"])
    critical_bracket = None
    for (label_lo, info_lo), (label_hi, info_hi) in zip(ordered, ordered[1:]):
        if info_lo["has_repulsive_structure"] and not info_hi["has_repulsive_structure"]:
            critical_bracket = f"{label_lo}~{label_hi}"
            break

    return {
        "original_distance": original_distance,
        "per_tau": per_tau,
        "repulsive_taus": [label for label, info in per_tau.items() if info["has_repulsive_structure"]],
        "critical_tau_bracket": critical_bracket,
    }


def _check_subsample_representativeness(
    checks: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> dict:
    """checks: {combo_name: (x_sub, y_sub, full_sample_reference_norm)}.

    부분표본(lambda=0, 즉 개입 없는 상태)의 ||Δgap||이 Step 4/5가 전체 9,356개로 낸 값과
    크게 다르면 부분표본이 대표성이 없을 수 있다는 경고를 출력한다 (n 증대 여부는 사용자 판단).
    """
    result = {}
    for name, (x_sub, y_sub, full_ref) in checks.items():
        dist_sub = float(np.linalg.norm(compute_delta_gap(x_sub, y_sub)))
        rel_diff = abs(dist_sub - full_ref) / full_ref
        result[name] = {
            "distance_sub": dist_sub,
            "distance_full_reference": full_ref,
            "relative_diff": rel_diff,
        }
        if rel_diff > REPRESENTATIVENESS_WARNING_THRESHOLD:
            print(
                f"[WARNING] subsample representativeness check failed for {name}: "
                f"relative_diff={rel_diff:.3f} > {REPRESENTATIVENESS_WARNING_THRESHOLD} "
                f"(distance_sub={dist_sub:.4f}, distance_full={full_ref:.4f}) "
                f"-- consider increasing N_SUBSAMPLE"
            )
    return result


def _plot_landscape(
    grid: dict[float, list[tuple[float, float, float]]],
    original_distance: float,
    tau_labels: dict[float, str],
    output_path: Path,
    title: str,
) -> None:
    """Mind the Gap Figure 3(b-d) 스타일: x=resulting gap distance, y=contrastive loss,
    tau별 곡선 + 원래(lambda=0) distance 위치에 세로 점선.

    grid는 _compute_grid()의 lambda 순서 그대로 그린다(distance로 재정렬하지 않음) — distance(lambda)가
    비단조라, distance로 정렬한 뒤 선으로 이으면 서로 다른 lambda 구간이 뒤섞여 지그재그가 생긴다.
    lambda 순서 그대로 그리면 "닫히는 구간"과 "재개방 구간"이 자연스럽게 이어진 체크마크 모양의
    연속 곡선이 된다.
    """
    fig, ax = new_figure("single_panel")
    colors = sequential_colors(len(grid))
    for color, (tau, points_ordered) in zip(colors, sorted(grid.items(), key=lambda kv: kv[0])):
        distances = [d for _, d, _ in points_ordered]
        losses = [l for _, _, l in points_ordered]
        ax.plot(distances, losses, marker="o", markersize=2, linewidth=1, color=color, label=f"tau={tau_labels[tau]}")

    ax.axvline(original_distance, color="black", linestyle="--", linewidth=1, label="original (lambda=0)")
    ax.set_xlabel("resulting gap distance")
    ax.set_ylabel("contrastive loss")
    ax.set_title(title)
    place_legend_outside(ax)

    save_figure(fig, output_path)


def run_temperature_analysis(
    npz_path: Path = NPZ_PATH,
    stage_a_results_path: Path = STAGE_A_RESULTS_PATH,
    stage_b_results_path: Path = STAGE_B_RESULTS_PATH,
    results_path: Path = RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
    n_subsample: int = N_SUBSAMPLE,
    seed: int = SUBSAMPLE_SEED,
) -> dict:
    data = np.load(npz_path)
    n_total = data["image_emb"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_subsample, replace=False)

    # 이미지/visual_a/visual_b/contextual 전부 같은 인덱스로 뽑는다 — Stage A/B의 contextual이
    # 뒤에서 비트 단위로 같아야 하는 sanity check가 이 구조로부터 자연스럽게 성립한다.
    image_sub = data["image_emb"][idx]
    visual_a_sub = data["text_emb_visual_a"][idx]
    visual_b_sub = data["text_emb_visual_b"][idx]
    contextual_sub = data["text_emb_contextual"][idx]

    with open(stage_a_results_path, "r", encoding="utf-8") as f:
        stage_a_results = json.load(f)
    with open(stage_b_results_path, "r", encoding="utf-8") as f:
        stage_b_results = json.load(f)

    subsample_check = _check_subsample_representativeness({
        "stage_a_visual": (image_sub, visual_a_sub, stage_a_results["delta_gap_visual_a_norm"]),
        "stage_a_contextual": (image_sub, contextual_sub, stage_a_results["delta_gap_contextual_norm"]),
        "stage_b_visual": (image_sub, visual_b_sub, stage_b_results["full"]["delta_gap_visual_b_norm"]),
        "stage_b_contextual": (image_sub, contextual_sub, stage_b_results["full"]["delta_gap_contextual_norm"]),
    })

    tau_labels = {tau: label for label, tau in TAUS}
    tau_values = [tau for _, tau in TAUS]

    combos = {
        "stage_a_visual": (image_sub, visual_a_sub, figures_dir / "step6_landscape_stageA_visual.png"),
        "stage_a_contextual": (image_sub, contextual_sub, figures_dir / "step6_landscape_stageA_contextual.png"),
        "stage_b_visual": (image_sub, visual_b_sub, figures_dir / "step6_landscape_stageB_visual.png"),
        "stage_b_contextual": (image_sub, contextual_sub, figures_dir / "step6_landscape_stageB_contextual.png"),
    }

    combo_results = {}
    for name, (x, y, fig_path) in combos.items():
        title = combo_display_name(name)
        delta_gap = compute_delta_gap(x, y)
        original_distance = float(np.linalg.norm(delta_gap))
        lambdas = _auto_lambda_grid(x, y, delta_gap)
        grid = _compute_grid(x, y, delta_gap, tau_values, lambdas)
        summary = _summarize_condition(grid, original_distance, tau_labels)
        _plot_landscape(grid, original_distance, tau_labels, fig_path, title)
        summary["n_lambda_points"] = len(lambdas)
        summary["lambda_max"] = float(lambdas.max())
        summary["figure"] = _relpath(fig_path)
        combo_results[name] = summary

    # Stage A/B contextual sanity check — 같은 서브샘플 인덱스의 동일 배열이라 사실상 bit-identical이어야 함.
    a_ctx = combo_results["stage_a_contextual"]
    b_ctx = combo_results["stage_b_contextual"]
    max_abs_diff = max(
        abs(a_ctx["per_tau"][label]["global_min_distance"] - b_ctx["per_tau"][label]["global_min_distance"])
        for label in a_ctx["per_tau"]
    )
    sanity_check = {
        "max_abs_diff_global_min_distance": max_abs_diff,
        "passed": max_abs_diff < 1e-4,
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_npz": _relpath(npz_path),
        "n_subsample": n_subsample,
        "seed": seed,
        "taus": {label: tau for label, tau in TAUS},
        "subsample_check": subsample_check,
        "combos": combo_results,
        "sanity_check_stage_a_b_contextual": sanity_check,
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def _print_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))

    print("\n[Temperature] tau=1/100 (CLIP-trained temperature): is the original gap distance near the global minimum?")
    print(f"{'combo':<22}{'original_dist':>14}{'global_min_dist':>18}{'diff':>10}")
    for name, combo in results["combos"].items():
        info = combo["per_tau"]["1/100"]
        print(
            f"{name:<22}{combo['original_distance']:>14.4f}"
            f"{info['global_min_distance']:>18.4f}{info['original_minus_global_min_distance']:>10.4f}"
        )

    sanity = results["sanity_check_stage_a_b_contextual"]
    print(
        f"\n[sanity] stage_a_contextual vs stage_b_contextual max abs diff (global_min_distance) = "
        f"{sanity['max_abs_diff_global_min_distance']:.2e} -> passed={sanity['passed']}"
    )


if __name__ == "__main__":
    _print_summary(run_temperature_analysis())
