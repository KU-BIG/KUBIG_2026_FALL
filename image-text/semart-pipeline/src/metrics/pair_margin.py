"""매칭 검증(Pair Margin) 분석 — 실험계획서 2.3절(쌍별 코사인 유사도) 확장.

기존에는 진짜 쌍 cos(I_i, T_i)만 봤다. 이 모듈은 진짜 쌍의 유사도를 같은 이미지의 나머지 모든
오답 텍스트(비대각)와 비교한 "마진"으로 본다 — "이 정렬이 CLIP이 실제로 학습한 의미 있는 대응인지,
narrow cone effect 때문에 아무 쌍이나 비슷하게 유사해서 그런 건 아닌지"를 검증하기 위함.
outputs/embeddings/*.npz의 기존 임베딩만 쓰는 순수 후처리 — GPU/재추출 불필요.

*** 이 분석이 검증하는 것과 안 하는 것 ***
- FACT: 이건 학습 과정 재현이 아니라 frozen 모델에 대한 post-hoc 검증이다.
- FACT: compute_delta_gap(centroid distance) = mean(image_emb) - mean(text_emb)는 페어링 순서에
  수학적으로 불변이다 (어느 이미지가 어느 텍스트와 짝지어지는지와 무관). 이 margin 분석 결과를
  "gap이 커졌다/작아졌다"는 Δgap 서술과 같은 의미로 혼동하지 말 것 — 정렬 품질(이 모듈) vs
  분포 중심 거리(gap.py)는 별개의 질문이다.
- TODO: retrieval R@k는 의도적으로 제외했다. 7절 downstream task(retrieval을 고를 경우)와
  계산 로직이 겹치고, 후보 풀 크기가 7.3절에 미결 사항으로 남아있어 지금 계산하면 팀 논의를
  선결정하게 된다.
- 한계: margin_i2t(그리고 margin_t2i) 값들은 서로 완전히 독립이 아니다 — 같은 텍스트 집합(고정된
  n개 텍스트 전체)을 공유하는 행들이라, Wilcoxon/부트스트랩 CI가 이 상관을 보정하지 않는다.
  n=9,356처럼 표본이 크면 p-value는 이 비독립성과 무관하게도 항상 유의하게 나오기 쉬우므로,
  해석은 p-value보다 Cohen's d/rank-biserial correlation(effect size) 위주로 할 것.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

from src.metrics.gap import _relpath, get_treated_mask_by_filename
from src.viz.style import COLORS, combo_display_name, new_figure, place_legend_outside, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "clip_vitb32_embeddings.npz"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "semart_v1_modality_gap_dataset.parquet"
STEP5_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step5_stage_b_results.json"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step6b_pair_margin_results.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# gap.py의 bootstrap_delta_gap_norm_ci와 동일한 값 재사용
N_BOOTSTRAP = 2000
SEED = 42

# 비대각(오답 쌍)은 n*(n-1) ~ 8,750만 개라 그림에는 무작위 서브샘플만 사용 (통계량은 전체로 계산됨)
N_OFFDIAG_SAMPLE = 50_000

_COMBO_FIGURE_SUFFIX = {
    "stage_a_visual": "stageA_visual",
    "stage_a_contextual": "stageA_contextual",
    "stage_b_visual": "stageB_visual",
    "stage_b_contextual": "stageB_contextual",
}

NOTES = {
    "post_hoc_not_training_reproduction": (
        "This is a post-hoc verification on the frozen model's output embeddings, "
        "not a reproduction of the CLIP training process."
    ),
    "delta_gap_pairing_invariant": (
        "compute_delta_gap (centroid distance) = mean(image_emb) - mean(text_emb) is mathematically "
        "invariant to pairing order -- it does not depend on which image is paired with which text. "
        "Do not conflate a change in pair margin with a change in delta_gap; they answer different "
        "questions (alignment quality vs. distribution-centroid distance)."
    ),
    "retrieval_excluded": (
        "Retrieval R@k is intentionally excluded here. It overlaps computationally with the section 7 "
        "downstream task (if retrieval is chosen), and candidate pool size is an open TODO in section "
        "7.3 -- computing it now would pre-decide that design before the team discussion."
    ),
    "margin_i2t_not_independent": (
        "margin_i2t (and margin_t2i) values are not fully independent across rows -- they all share the "
        "same fixed set of n texts (or images) as the off-diagonal comparison pool, so row-level "
        "off-diagonal means can be correlated through shared terms. Wilcoxon/bootstrap CI here do not "
        "correct for this. With n=9,356, p-values tend to be significant almost regardless of this "
        "non-independence, so interpretation should lean on Cohen's d / rank-biserial correlation "
        "(effect size), not p-value."
    ),
}


def _assert_l2_normalized(x: np.ndarray, name: str, atol: float = 1e-4) -> None:
    norms = np.linalg.norm(x, axis=1)
    max_dev = float(np.abs(norms - 1.0).max())
    assert max_dev <= atol, f"{name} is not L2-normalized (max |norm-1|={max_dev:.2e} > {atol})"


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = SEED,
) -> dict:
    """gap.py의 bootstrap_delta_gap_norm_ci와 동일한 percentile 방식 — 1차원 배열의 평균에 적용."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()

    point_estimate = float(values.mean())
    alpha = 1 - ci
    ci_low = float(np.percentile(means, 100 * alpha / 2))
    ci_high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def _rank_biserial_from_signed_ranks(diffs: np.ndarray) -> float:
    """Wilcoxon signed-rank test와 짝을 이루는 matched-pairs rank-biserial correlation.

    r = (W+ - W-) / (W+ + W-), W+/W-는 0이 아닌 차이의 절댓값 순위 합 중 양/음 방향 합.
    """
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    w_pos = ranks[nonzero > 0].sum()
    w_neg = ranks[nonzero < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total > 0 else 0.0


def _summarize_margin(margin: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    w_stat, w_p = wilcoxon(margin)
    cohens_d = float(margin.mean() / margin.std(ddof=1))
    rank_biserial = _rank_biserial_from_signed_ranks(margin)
    ci = _bootstrap_mean_ci(margin, n_bootstrap=n_bootstrap, seed=seed)

    return {
        "mean": float(margin.mean()),
        "std": float(margin.std(ddof=1)),
        "ci_95": [ci["ci_low"], ci["ci_high"]],
        "cohens_d": cohens_d,
        "rank_biserial_correlation": rank_biserial,
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": seed,
    }


def _sample_off_diag_values(S: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    n = S.shape[0]
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n, size=n_sample)
    cols = rng.integers(0, n, size=n_sample)
    same = rows == cols
    while same.any():
        cols[same] = rng.integers(0, n, size=int(same.sum()))
        same = rows == cols
    return S[rows, cols]


def compute_pair_margin(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    n_offdiag_sample: int = N_OFFDIAG_SAMPLE,
) -> dict:
    """실험계획서 2.3절 확장: 진짜 쌍 코사인 유사도 vs 같은 이미지의 오답 텍스트들과의 평균 유사도.

    S = image_emb @ text_emb.T (n x n, 이미 L2 정규화되어 있으므로 내적 = 코사인 유사도).
    margin_i2t[i] = S[i,i] - mean_{j!=i} S[i,j]  (이미지 i 기준, 오답 텍스트들과 비교)
    margin_t2i[i] = S[i,i] - mean_{j!=i} S[j,i]  (텍스트 i 기준, 오답 이미지들과 비교)
    두 방향을 반드시 따로 계산하고 따로 보고한다 (다를 수 있음).

    반환 dict는 JSON 직렬화 가능한 필드들 + "_raw"(numpy 배열, 그림 렌더링용 — 호출부가 pop해서
    JSON 저장 전에 제거해야 함)로 구성된다.
    """
    _assert_l2_normalized(image_emb, "image_emb")
    _assert_l2_normalized(text_emb, "text_emb")

    n = image_emb.shape[0]
    S = image_emb @ text_emb.T
    diag = np.diag(S).copy()

    row_off_mean = (S.sum(axis=1) - diag) / (n - 1)
    col_off_mean = (S.sum(axis=0) - diag) / (n - 1)
    margin_i2t = diag - row_off_mean
    margin_t2i = diag - col_off_mean

    off_diag_mean = float((S.sum() - diag.sum()) / (n * (n - 1)))
    off_diag_sample = _sample_off_diag_values(S, n_offdiag_sample, seed)

    return {
        "n": n,
        "diag_mean": float(diag.mean()),
        "off_diag_mean": off_diag_mean,
        "image_to_text": _summarize_margin(margin_i2t, n_bootstrap=n_bootstrap, seed=seed),
        "text_to_image": _summarize_margin(margin_t2i, n_bootstrap=n_bootstrap, seed=seed),
        "_raw": {
            "margin_i2t": margin_i2t,
            "margin_t2i": margin_t2i,
            "diag": diag,
            "off_diag_sample": off_diag_sample,
        },
    }


def _plot_margin_histogram(
    margin_i2t: np.ndarray,
    margin_t2i: np.ndarray,
    output_path: Path,
    title: str,
    bins: int = 60,
) -> None:
    fig, ax = new_figure("single_panel")
    ax.hist(margin_i2t, bins=bins, alpha=0.5, density=True, color=COLORS["primary"], label="margin (image->text)")
    ax.hist(margin_t2i, bins=bins, alpha=0.5, density=True, color=COLORS["secondary"], label="margin (text->image)")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1, label="margin=0")
    ax.set_xlabel("pair margin (diag cosine - mean off-diag cosine)")
    ax.set_ylabel("density")
    ax.set_title(title)
    place_legend_outside(ax)

    save_figure(fig, output_path)


def _plot_diag_offdiag(
    diag: np.ndarray,
    off_diag_sample: np.ndarray,
    output_path: Path,
    title: str,
    bins: int = 60,
) -> None:
    fig, ax = new_figure("single_panel")
    ax.hist(
        off_diag_sample, bins=bins, alpha=0.5, density=True, color=COLORS["neutral"],
        label="off-diagonal (wrong pairs)",
    )
    ax.hist(
        diag, bins=bins, alpha=0.6, density=True, color=COLORS["primary"],
        label="diagonal (true pairs)",
    )
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(title)
    # 이 family만 legend label이 다른 single_panel family(margin_hist 등)보다 길어서
    # (예: "off-diagonal (wrong pairs)") 기본 right_margin(0.72)로는 잘림 -- 축 영역을
    # 좀 더 좁혀 legend 폭을 넉넉히 확보한다.
    place_legend_outside(ax, right_margin=0.62)

    save_figure(fig, output_path)


def run_pair_margin_analysis(
    npz_path: Path = NPZ_PATH,
    parquet_path: Path = PARQUET_PATH,
    step5_results_path: Path = STEP5_RESULTS_PATH,
    results_path: Path = RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    n_offdiag_sample: int = N_OFFDIAG_SAMPLE,
) -> dict:
    data = np.load(npz_path)
    image_emb = data["image_emb"]

    combos = {
        "stage_a_visual": (image_emb, data["text_emb_visual_a"]),
        "stage_a_contextual": (image_emb, data["text_emb_contextual"]),
        "stage_b_visual": (image_emb, data["text_emb_visual_b"]),
        "stage_b_contextual": (image_emb, data["text_emb_contextual"]),
    }

    combo_results = {}
    for name, (img, txt) in combos.items():
        title = combo_display_name(name)
        margin = compute_pair_margin(
            img, txt, n_bootstrap=n_bootstrap, seed=seed, n_offdiag_sample=n_offdiag_sample,
        )
        raw = margin.pop("_raw")

        suffix = _COMBO_FIGURE_SUFFIX[name]
        hist_path = figures_dir / f"step6b_pair_margin_hist_{suffix}.png"
        diagoffdiag_path = figures_dir / f"step6b_pair_margin_diagoffdiag_{suffix}.png"
        _plot_margin_histogram(raw["margin_i2t"], raw["margin_t2i"], hist_path, title)
        _plot_diag_offdiag(raw["diag"], raw["off_diag_sample"], diagoffdiag_path, title)

        margin["margin_hist_figure"] = _relpath(hist_path)
        margin["diag_offdiag_figure"] = _relpath(diagoffdiag_path)
        combo_results[name] = margin

    # Stage A/B contextual sanity check — 같은 배열로 계산하므로 두 방향 평균이 사실상 일치해야 함.
    a_ctx = combo_results["stage_a_contextual"]
    b_ctx = combo_results["stage_b_contextual"]
    max_abs_diff = max(
        abs(a_ctx["image_to_text"]["mean"] - b_ctx["image_to_text"]["mean"]),
        abs(a_ctx["text_to_image"]["mean"] - b_ctx["text_to_image"]["mean"]),
    )
    sanity_check = {
        "max_abs_diff_mean_margin": max_abs_diff,
        "passed": max_abs_diff < 1e-6,
    }

    # --- Stage B treated-only 서브셋 (실험계획서 4.4절 처치 부분표본, step5와 동일한 treated_mask 재사용) ---
    filenames = data["filenames"]
    treated_mask = get_treated_mask_by_filename(filenames, parquet_path)

    with open(step5_results_path, "r", encoding="utf-8") as f:
        step5_results = json.load(f)
    expected_n_treated = step5_results["n_treated"]
    assert int(treated_mask.sum()) == expected_n_treated, (
        f"treated count mismatch: got {int(treated_mask.sum())}, "
        f"expected {expected_n_treated} (step5_stage_b_results.json)"
    )

    treated_combos = {
        "stage_b_visual": (image_emb[treated_mask], data["text_emb_visual_b"][treated_mask]),
        "stage_b_contextual": (image_emb[treated_mask], data["text_emb_contextual"][treated_mask]),
    }
    treated_suffix = {"stage_b_visual": "stageB_treated_visual", "stage_b_contextual": "stageB_treated_contextual"}

    treated_results = {}
    for name, (img, txt) in treated_combos.items():
        title = combo_display_name(name, treated=True)
        margin = compute_pair_margin(
            img, txt, n_bootstrap=n_bootstrap, seed=seed, n_offdiag_sample=n_offdiag_sample,
        )
        raw = margin.pop("_raw")

        suffix = treated_suffix[name]
        hist_path = figures_dir / f"step6b_pair_margin_hist_{suffix}.png"
        diagoffdiag_path = figures_dir / f"step6b_pair_margin_diagoffdiag_{suffix}.png"
        _plot_margin_histogram(raw["margin_i2t"], raw["margin_t2i"], hist_path, title)
        _plot_diag_offdiag(raw["diag"], raw["off_diag_sample"], diagoffdiag_path, title)

        margin["margin_hist_figure"] = _relpath(hist_path)
        margin["diag_offdiag_figure"] = _relpath(diagoffdiag_path)
        treated_results[name] = margin

    margin_diff_full = (
        combo_results["stage_b_contextual"]["image_to_text"]["mean"]
        - combo_results["stage_b_visual"]["image_to_text"]["mean"]
    )
    margin_diff_treated = (
        treated_results["stage_b_contextual"]["image_to_text"]["mean"]
        - treated_results["stage_b_visual"]["image_to_text"]["mean"]
    )

    stage_b_treated_only = {
        "n": int(treated_mask.sum()),
        "stage_b_visual": treated_results["stage_b_visual"],
        "stage_b_contextual": treated_results["stage_b_contextual"],
        "comparison": {
            "margin_diff_full": margin_diff_full,
            "margin_diff_treated": margin_diff_treated,
            "note": (
                "margin_diff = contextual.image_to_text.mean - visual.image_to_text.mean "
                "(same subtraction order as step5_stage_b_results.json's gap_diff, for direct "
                "comparability). Margin is a 'higher is better' quantity (unlike gap distance, which "
                "is 'lower is better'), so the H1-consistent direction here is NEGATIVE "
                "(visual margin > contextual margin), the mirror image of gap_diff's polarity -- "
                "this is the opposite convention from delta_gap and is called out explicitly to avoid "
                "confusion. Compare |margin_diff_treated| vs |margin_diff_full|: shrinking toward zero "
                "(or flipping sign) leans scenario 1 (length dominates); holding or growing leans "
                "scenario 2 (information type dominates) -- mirroring step5's interpretation "
                "framework (02_experiment_plan.md section 4.2). This script does not decide between "
                "them."
            ),
        },
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_npz": _relpath(npz_path),
        "bootstrap_n": n_bootstrap,
        "bootstrap_seed": seed,
        "combos": combo_results,
        "sanity_check_stage_a_b_contextual": sanity_check,
        "stage_b_treated_only": stage_b_treated_only,
        "notes": NOTES,
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def _print_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))

    print(
        "\n[Pair Margin] EFFECT SIZE FIRST (n=9,356 -> p-values are almost always significant; "
        "Cohen's d / rank-biserial correlation are the primary interpretive signal here):"
    )
    header = f"{'combo':<22}{'direction':<8}{'mean':>10}{'  COHENS_D':>12}{'  RANK_BISERIAL':>18}"
    print(header)
    print("=" * len(header))
    for name, combo in results["combos"].items():
        for key, label in [("image_to_text", "I->T"), ("text_to_image", "T->I")]:
            d = combo[key]
            print(
                f"{name:<22}{label:<8}{d['mean']:>10.4f}"
                f"{d['cohens_d']:>12.3f}{d['rank_biserial_correlation']:>18.3f}"
            )

    print("\n[Pair Margin] p-value (secondary -- reported for completeness, not the primary signal):")
    for name, combo in results["combos"].items():
        for key, label in [("image_to_text", "I->T"), ("text_to_image", "T->I")]:
            d = combo[key]
            print(f"{name:<22}{label:<6} wilcoxon_pvalue={d['wilcoxon_pvalue']:.3e}")

    sanity = results["sanity_check_stage_a_b_contextual"]
    print(
        f"\n[sanity] stage_a_contextual vs stage_b_contextual max abs diff (mean margin) = "
        f"{sanity['max_abs_diff_mean_margin']:.2e} -> passed={sanity['passed']}"
    )

    treated = results["stage_b_treated_only"]
    comp = treated["comparison"]
    diff_full = comp["margin_diff_full"]
    diff_treated = comp["margin_diff_treated"]
    # H1 방향(margin 관점) = visual margin > contextual margin = margin_diff(contextual - visual) < 0
    h1_full = "H1-consistent (visual > contextual)" if diff_full < 0 else "H1-reversed (contextual >= visual)"
    h1_treated = "H1-consistent (visual > contextual)" if diff_treated < 0 else "H1-reversed (contextual >= visual)"
    if abs(diff_full) > 1e-12:
        pct_change = (abs(diff_treated) - abs(diff_full)) / abs(diff_full) * 100
    else:
        pct_change = float("nan")
    widened_or_narrowed = "widened" if abs(diff_treated) > abs(diff_full) else "narrowed"
    if (diff_full < 0) != (diff_treated < 0):
        widened_or_narrowed = "REVERSED"

    print(f"\n[Pair Margin] Stage B full vs treated-only (n={treated['n']}) comparison:")
    print(f"{'sample':<16}{'margin_diff':>14}{'direction':>38}")
    print(f"{'full (9356)':<16}{diff_full:>14.4f}{h1_full:>38}")
    print(f"{'treated (' + str(treated['n']) + ')':<16}{diff_treated:>14.4f}{h1_treated:>38}")
    print(
        f"\n[Pair Margin] margin gap |diff| {widened_or_narrowed} from full to treated "
        f"({abs(diff_full):.4f} -> {abs(diff_treated):.4f}, {pct_change:+.1f}%)"
    )


if __name__ == "__main__":
    _print_summary(run_pair_margin_analysis())
