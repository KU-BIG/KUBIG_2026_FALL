"""Step 4/5: Gap 측정 지표 (실험계획서 2절) + Stage A(3절)/Stage B(4절) 분석.

GPU/torch 불필요 — frozen CLIP 임베딩(outputs/embeddings/*.npz)에 대한 순수 후처리.
Stage A/B는 analyze_condition_pair() 하나를 공용으로 호출한다 — 계산 방식이 두 곳에서
갈라지지 않게 하기 위함(사용자 요청).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.viz.style import COLORS, new_figure, place_legend_outside, save_figure

from src.data.dataset import build_stage_b, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _relpath(path) -> str:
    """PROJECT_ROOT 기준 상대경로 문자열(forward slash, OS 무관)로 변환.
    JSON 메타데이터에 로컬 절대경로(C:\\Users\\... 등)가 그대로 박히는 걸 방지한다.
    PROJECT_ROOT 바깥 경로(예: 테스트의 tmp_path)면 절대경로 문자열로 안전하게 폴백한다."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "semart_v1_modality_gap_dataset.parquet"
NPZ_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "clip_vitb32_embeddings.npz"
STEP1_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "results" / "step1_validation_summary.json"

RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step4_stage_a_results.json"
PCA_FIG_PATH = PROJECT_ROOT / "outputs" / "figures" / "stage_a_pca.png"
UMAP_FIG_PATH = PROJECT_ROOT / "outputs" / "figures" / "stage_a_umap.png"

STAGE_B_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step5_stage_b_results.json"
STAGE_B_PCA_FIG_PATH = PROJECT_ROOT / "outputs" / "figures" / "stage_b_full_pca.png"

H1_SUPPORTED_NOTE = (
    "h1_supported is a plain point-estimate direction flag "
    "(delta_gap_visual_a_norm < delta_gap_contextual_norm), not a statistical "
    "significance test. See 02_experiment_plan.md section 3.3: rejection of H1 "
    "is an equally meaningful result, not a failure."
)


def compute_delta_gap(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """실험계획서 2.1절: Δ_gap = mean(image_emb) - mean(text_emb)."""
    return image_emb.mean(axis=0) - text_emb.mean(axis=0)


def bootstrap_delta_gap_norm_ci(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """실험계획서 3.4절: ||Δ_gap|| 점추정 + 부트스트랩 95% percentile CI.

    복원추출은 이미지-텍스트 쌍 인덱스 단위로 수행(같은 리샘플이 image_emb/text_emb에 동일 적용).
    """
    rng = np.random.default_rng(seed)
    n = image_emb.shape[0]
    norms = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        image_mean = counts @ image_emb / n
        text_mean = counts @ text_emb / n
        norms[b] = np.linalg.norm(image_mean - text_mean)

    point_estimate = float(np.linalg.norm(compute_delta_gap(image_emb, text_emb)))
    alpha = 1 - ci
    ci_low = float(np.percentile(norms, 100 * alpha / 2))
    ci_high = float(np.percentile(norms, 100 * (1 - alpha / 2)))
    return {
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def linear_separability(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """실험계획서 2.2절: image_emb(label=0) vs text_emb(label=1) 로지스틱 회귀 분류.

    PROPOSAL: stratified 5-fold CV, fold별 test accuracy의 mean/std로 리포트
    (문서에 프로토콜 명시 안 됨).
    """
    X = np.concatenate([image_emb, text_emb], axis=0)
    y = np.concatenate([np.zeros(len(image_emb)), np.ones(len(text_emb))])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        fold_scores.append(clf.score(X[test_idx], y[test_idx]))

    fold_scores = np.array(fold_scores)
    return {
        "mean": float(fold_scores.mean()),
        "std": float(fold_scores.std()),
        "fold_scores": fold_scores.tolist(),
        "n_splits": n_splits,
        "random_state": random_state,
    }


def paired_cosine_comparison(
    image_emb: np.ndarray,
    text_emb_a: np.ndarray,
    text_emb_b: np.ndarray,
) -> dict:
    """실험계획서 2.3절: 같은 이미지 i에 대한 cos(image_i, a_i) vs cos(image_i, b_i) 대응표본 비교.

    임베딩이 이미 L2 정규화되어 있으므로 코사인 유사도 = 내적.
    paired t-test와 Wilcoxon signed-rank 둘 다 계산 (문서는 "또는"이나 계산 비용이 낮아 둘 다 리포트).
    """
    cos_a = np.einsum("ij,ij->i", image_emb, text_emb_a)
    cos_b = np.einsum("ij,ij->i", image_emb, text_emb_b)

    t_stat, t_p = ttest_rel(cos_a, cos_b)
    w_stat, w_p = wilcoxon(cos_a, cos_b)

    return {
        "cos_a_mean": float(cos_a.mean()),
        "cos_b_mean": float(cos_b.mean()),
        "ttest_statistic": float(t_stat),
        "ttest_pvalue": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
    }


def analyze_condition_pair(
    image_emb: np.ndarray,
    text_emb_a: np.ndarray,
    text_emb_b: np.ndarray,
    condition_a_name: str,
    condition_b_name: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """centroid distance(+CI) 두 조건, linear separability 두 조건, paired cosine(a vs b)을
    한 번에 계산하는 공용 진입점. Stage A/B가 이 함수 하나를 동일하게 호출해서, 두 Stage 간
    계산 방식이 실수로 갈라지지 않도록 한다.
    """
    gap_a = bootstrap_delta_gap_norm_ci(image_emb, text_emb_a, n_bootstrap=n_bootstrap, seed=seed)
    gap_b = bootstrap_delta_gap_norm_ci(image_emb, text_emb_b, n_bootstrap=n_bootstrap, seed=seed)
    linsep_a = linear_separability(image_emb, text_emb_a, random_state=seed)
    linsep_b = linear_separability(image_emb, text_emb_b, random_state=seed)
    paired = paired_cosine_comparison(image_emb, text_emb_a, text_emb_b)

    return {
        f"delta_gap_{condition_a_name}_norm": gap_a["point_estimate"],
        f"delta_gap_{condition_a_name}_ci_95": [gap_a["ci_low"], gap_a["ci_high"]],
        f"delta_gap_{condition_b_name}_norm": gap_b["point_estimate"],
        f"delta_gap_{condition_b_name}_ci_95": [gap_b["ci_low"], gap_b["ci_high"]],
        "bootstrap_n": n_bootstrap,
        "bootstrap_seed": seed,
        f"linear_sep_acc_{condition_a_name}_mean": linsep_a["mean"],
        f"linear_sep_acc_{condition_a_name}_std": linsep_a["std"],
        f"linear_sep_acc_{condition_b_name}_mean": linsep_b["mean"],
        f"linear_sep_acc_{condition_b_name}_std": linsep_b["std"],
        "linear_sep_n_splits": linsep_a["n_splits"],
        "linear_sep_random_state": seed,
        "paired_cosine": paired,
    }


def get_treated_mask_by_filename(npz_filenames: np.ndarray, parquet_path: Path = PARQUET_PATH) -> np.ndarray:
    """실험계획서 4.4절 treated_mask를 filename 기준 명시적 join으로 얻는다.

    npz의 filenames 순서와 build_stage_b(df)의 순서가 같다고 가정하지 않음 —
    filename을 키로 매핑해서 npz 순서에 맞춰 재배열한다.
    """
    df = load_dataset(str(parquet_path))
    stage_b_df = build_stage_b(df)
    treated_by_filename = dict(zip(stage_b_df["filename"], stage_b_df["treated_mask"]))

    missing = [f for f in npz_filenames if f not in treated_by_filename]
    assert not missing, f"{len(missing)}개 filename이 parquet에 없음 (예: {missing[:5]})"

    return np.array([treated_by_filename[f] for f in npz_filenames], dtype=bool)


def plot_projection_2d(
    image_emb: np.ndarray,
    text_emb_visual: np.ndarray,
    text_emb_contextual: np.ndarray,
    method: str,
    output_path: Path,
    seed: int = 42,
    title_prefix: str = "CLIP embeddings",
) -> np.ndarray:
    """실험계획서 3.4절: 이미지/visual텍스트/contextual텍스트 3색 2D 산점도. method: "pca" | "umap"."""
    X = np.concatenate([image_emb, text_emb_visual, text_emb_contextual], axis=0)
    n_image, n_visual, n_contextual = len(image_emb), len(text_emb_visual), len(text_emb_contextual)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
    elif method == "umap":
        import umap  # 선택적 의존성 — 사용 시점에만 import

        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError(f"unknown method: {method}")

    coords = reducer.fit_transform(X)

    fig, ax = new_figure("scatter_square")
    groups = [
        ("image", coords[:n_image], COLORS["secondary"]),
        ("visual_text", coords[n_image : n_image + n_visual], COLORS["primary"]),
        ("contextual_text", coords[n_image + n_visual :], COLORS["tertiary"]),
    ]
    for label, pts, color in groups:
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.4, label=label, color=color)
    place_legend_outside(ax)
    ax.set_title(f"{title_prefix} - {method.upper()} (2D)")

    save_figure(fig, output_path)
    return coords


def run_stage_a(
    npz_path: Path = NPZ_PATH,
    results_path: Path = RESULTS_PATH,
    pca_path: Path = PCA_FIG_PATH,
    umap_path: Path | None = UMAP_FIG_PATH,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    data = np.load(npz_path)
    image_emb = data["image_emb"]
    text_visual_a = data["text_emb_visual_a"]
    text_contextual = data["text_emb_contextual"]

    metrics = analyze_condition_pair(
        image_emb, text_visual_a, text_contextual, "visual_a", "contextual",
        n_bootstrap=n_bootstrap, seed=seed,
    )

    plot_projection_2d(
        image_emb, text_visual_a, text_contextual, method="pca", output_path=pca_path, seed=seed,
        title_prefix="Stage A embeddings",
    )
    if umap_path is not None:
        try:
            plot_projection_2d(
                image_emb, text_visual_a, text_contextual, method="umap", output_path=umap_path, seed=seed,
                title_prefix="Stage A embeddings",
            )
        except ImportError:
            umap_path = None

    h1_supported = metrics["delta_gap_visual_a_norm"] < metrics["delta_gap_contextual_norm"]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n": int(image_emb.shape[0]),
        "source_npz": _relpath(npz_path),
        **metrics,
        "h1_supported": h1_supported,
        "h1_supported_note": H1_SUPPORTED_NOTE,
        "pca_figure": _relpath(pca_path),
        "umap_figure": _relpath(umap_path) if umap_path is not None else None,
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def run_stage_b(
    npz_path: Path = NPZ_PATH,
    parquet_path: Path = PARQUET_PATH,
    step1_summary_path: Path = STEP1_SUMMARY_PATH,
    stage_a_results_path: Path = RESULTS_PATH,
    results_path: Path = STAGE_B_RESULTS_PATH,
    pca_path: Path = STAGE_B_PCA_FIG_PATH,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """실험계획서 4절: 전체 표본 + 처치 부분표본(treated_mask=True) 각각 분석 후 Stage A와 비교."""
    data = np.load(npz_path)
    image_emb = data["image_emb"]
    text_visual_b = data["text_emb_visual_b"]
    text_contextual = data["text_emb_contextual"]
    filenames = data["filenames"]

    treated_mask = get_treated_mask_by_filename(filenames, parquet_path)

    with open(step1_summary_path, "r", encoding="utf-8") as f:
        step1_summary = json.load(f)
    expected_treated_count = step1_summary["treated_count"]
    assert treated_mask.sum() == expected_treated_count, (
        f"treated count mismatch: got {int(treated_mask.sum())}, "
        f"expected {expected_treated_count} (Step 1 summary)"
    )

    full = analyze_condition_pair(
        image_emb, text_visual_b, text_contextual, "visual_b", "contextual",
        n_bootstrap=n_bootstrap, seed=seed,
    )
    full["n"] = int(len(image_emb))

    treated_only = analyze_condition_pair(
        image_emb[treated_mask], text_visual_b[treated_mask], text_contextual[treated_mask],
        "visual_b", "contextual", n_bootstrap=n_bootstrap, seed=seed,
    )
    treated_only["n"] = int(treated_mask.sum())

    plot_projection_2d(
        image_emb, text_visual_b, text_contextual, method="pca", output_path=pca_path, seed=seed,
        title_prefix="Stage B (full sample) embeddings",
    )

    with open(stage_a_results_path, "r", encoding="utf-8") as f:
        stage_a_results = json.load(f)
    stage_a_gap_diff = stage_a_results["delta_gap_contextual_norm"] - stage_a_results["delta_gap_visual_a_norm"]
    stage_b_full_gap_diff = full["delta_gap_contextual_norm"] - full["delta_gap_visual_b_norm"]
    stage_b_treated_gap_diff = treated_only["delta_gap_contextual_norm"] - treated_only["delta_gap_visual_b_norm"]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_npz": _relpath(npz_path),
        "n_full": int(len(image_emb)),
        "n_treated": int(treated_mask.sum()),
        "full": full,
        "treated_only": treated_only,
        "comparison": {
            "stage_a_gap_diff": stage_a_gap_diff,
            "stage_b_full_gap_diff": stage_b_full_gap_diff,
            "stage_b_treated_gap_diff": stage_b_treated_gap_diff,
            "note": (
                "gap_diff = delta_gap_contextual_norm - delta_gap_visual_norm (per row's own "
                "visual condition). Larger positive value = visual gap smaller than contextual "
                "gap (H1 direction). Compare direction/magnitude across the three rows to judge "
                "scenario 1 (length dominates -> gap_diff shrinks/reverses in Stage B) vs "
                "scenario 2 (information type dominates -> gap_diff holds/grows) - see "
                "02_experiment_plan.md section 4.2. This script does not decide between them."
            ),
        },
        "pca_figure": _relpath(pca_path),
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def _print_stage_a_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))
    direction = "<" if results["h1_supported"] else ">="
    print(
        f"\n[H1] ||delta_gap^visual_a|| ({results['delta_gap_visual_a_norm']:.6f}) "
        f"{direction} ||delta_gap^contextual|| ({results['delta_gap_contextual_norm']:.6f}) "
        f"=> h1_supported={results['h1_supported']}"
    )


def _print_stage_b_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))
    comp = results["comparison"]
    print("\n[Stage B] gap_diff comparison (delta_gap_contextual_norm - delta_gap_visual_norm):")
    rows = [
        ("Stage A", None, comp["stage_a_gap_diff"]),
        ("Stage B full", results["n_full"], comp["stage_b_full_gap_diff"]),
        ("Stage B treated", results["n_treated"], comp["stage_b_treated_gap_diff"]),
    ]
    print(f"{'condition':<20}{'n':>8}{'gap_diff':>14}")
    for name, n, diff in rows:
        n_str = "" if n is None else str(n)
        print(f"{name:<20}{n_str:>8}{diff:>14.6f}")

    linsep_visual = results["full"]["linear_sep_acc_visual_b_mean"]
    linsep_contextual = results["full"]["linear_sep_acc_contextual_mean"]
    print(
        f"\n[linear separability] full sample: visual_b={linsep_visual:.4f}, "
        f"contextual={linsep_contextual:.4f} "
        f"({'AT CEILING (1.0) - uninformative, same as Stage A' if linsep_visual >= 0.999 and linsep_contextual >= 0.999 else 'below ceiling - still informative'})"
    )


if __name__ == "__main__":
    import sys

    stage = sys.argv[1] if len(sys.argv) > 1 else "stage_a"
    if stage == "stage_a":
        _print_stage_a_summary(run_stage_a())
    elif stage == "stage_b":
        _print_stage_b_summary(run_stage_b())
    else:
        raise SystemExit(f"unknown stage: {stage!r} (expected 'stage_a' or 'stage_b')")
