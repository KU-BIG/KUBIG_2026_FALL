"""Step 4/5: gap 측정 로직 자체의 정합성 검증. 가설(H1) 채택 여부는 assert 대상이 아님."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.embeddings.extract import encode_texts, load_model
from src.interventions.step7a_shift_diagnostics import (
    RETRIEVAL_MAX_LAMBDA_EXPANSIONS,
    compute_classification_diagnostics,
    compute_retrieval_diagnostics,
    compute_retrieval_diagnostics_extended,
)
from src.metrics.gap import (
    analyze_condition_pair,
    bootstrap_delta_gap_norm_ci,
    compute_delta_gap,
    get_treated_mask_by_filename,
    linear_separability,
    paired_cosine_comparison,
    plot_projection_2d,
)
from src.metrics.pair_margin import compute_pair_margin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "clip_vitb32_embeddings.npz"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "semart_v1_modality_gap_dataset.parquet"
STEP1_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "results" / "step1_validation_summary.json"
STEP4_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step4_stage_a_results.json"
ZEROSHOT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "downstream_zeroshot_baseline.json"

# Step 3에서 실측된 ||Δ_gap^contextual|| — 회귀 앵커 값
STEP3_CONTEXTUAL_GAP_NORM = 0.860851

# 테스트 속도를 위한 부분표본 크기 (linear_separability/bootstrap은 O(n) 이상이라
# 전체 9,356개로 매 테스트 돌리면 느림 — 로직 정합성 검증이 목적이므로 부분표본으로 충분)
SUBSAMPLE_N = 1000


@pytest.fixture(scope="module")
def npz():
    return np.load(EMB_PATH)


@pytest.fixture(scope="module")
def image_emb(npz):
    return npz["image_emb"]


@pytest.fixture(scope="module")
def text_emb_visual_a(npz):
    return npz["text_emb_visual_a"]


@pytest.fixture(scope="module")
def text_emb_visual_b(npz):
    return npz["text_emb_visual_b"]


@pytest.fixture(scope="module")
def text_emb_contextual(npz):
    return npz["text_emb_contextual"]


@pytest.fixture(scope="module")
def filenames(npz):
    return npz["filenames"]


@pytest.fixture(scope="module")
def treated_mask(filenames):
    return get_treated_mask_by_filename(filenames, PARQUET_PATH)


def test_compute_delta_gap_matches_step3(image_emb, text_emb_contextual):
    gap = compute_delta_gap(image_emb, text_emb_contextual)
    assert np.linalg.norm(gap) == pytest.approx(STEP3_CONTEXTUAL_GAP_NORM, abs=1e-5)


def test_bootstrap_ci_contains_point_estimate(image_emb, text_emb_contextual):
    result = bootstrap_delta_gap_norm_ci(image_emb, text_emb_contextual, n_bootstrap=200, seed=42)
    assert result["ci_low"] < result["point_estimate"] < result["ci_high"]


def test_bootstrap_ci_reproducible_with_same_seed(image_emb, text_emb_visual_a):
    result_1 = bootstrap_delta_gap_norm_ci(image_emb, text_emb_visual_a, n_bootstrap=100, seed=7)
    result_2 = bootstrap_delta_gap_norm_ci(image_emb, text_emb_visual_a, n_bootstrap=100, seed=7)
    assert result_1["ci_low"] == result_2["ci_low"]
    assert result_1["ci_high"] == result_2["ci_high"]


def test_linear_separability_accuracy_range(image_emb, text_emb_contextual):
    result = linear_separability(image_emb[:SUBSAMPLE_N], text_emb_contextual[:SUBSAMPLE_N], n_splits=5, random_state=42)
    # gap이 존재하는 한 완전 랜덤(0.5) 근처거나 그 이상이어야 함.
    # 0.4 밑으로 나오면 라벨/피처가 뒤바뀐 버그를 의심할 수준.
    assert result["mean"] > 0.4
    assert result["mean"] <= 1.0
    assert len(result["fold_scores"]) == 5


def test_paired_cosine_comparison_self_identical(image_emb, text_emb_visual_a):
    # 같은 배열을 a/b에 둘 다 넣으면 쌍별 코사인 값 자체가 완전히 동일해야 한다 — 회귀 가드.
    # (t-test/wilcoxon statistic은 이 경우 차이의 분산이 0이라 0/0 = nan이 되는 게 통계적으로
    # 정상 동작이라 여기선 assert 대상이 아님 — scipy가 RuntimeWarning과 함께 nan을 반환함)
    result = paired_cosine_comparison(image_emb, text_emb_visual_a, text_emb_visual_a)
    assert result["cos_a_mean"] == pytest.approx(result["cos_b_mean"], abs=1e-9)


def test_paired_cosine_comparison_visual_vs_contextual_shapes(image_emb, text_emb_visual_a, text_emb_contextual):
    result = paired_cosine_comparison(image_emb, text_emb_visual_a, text_emb_contextual)
    for key in ("ttest_statistic", "ttest_pvalue", "wilcoxon_statistic", "wilcoxon_pvalue"):
        assert np.isfinite(result[key])


def test_pca_projection_shape(image_emb, text_emb_visual_a, text_emb_contextual, tmp_path):
    n_expected = len(image_emb) + len(text_emb_visual_a) + len(text_emb_contextual)
    coords = plot_projection_2d(
        image_emb, text_emb_visual_a, text_emb_contextual,
        method="pca", output_path=tmp_path / "test_pca.png",
    )
    assert coords.shape == (n_expected, 2)
    assert n_expected == 3 * len(image_emb)  # 9,356 * 3


# ---- Step 5: Stage B (실험계획서 4절) ----


def test_treated_mask_join_count_matches_step1(treated_mask):
    # get_treated_mask_by_filename은 filename 기준 명시적 join으로 만들어지므로,
    # Step 1이 build_stage_b(df) 원본 순서로 이미 검증해둔 treated_count와 정확히 같아야 한다
    # (순서만 다를 뿐 집합은 동일해야 함).
    with open(STEP1_SUMMARY_PATH, "r", encoding="utf-8") as f:
        step1_summary = json.load(f)
    assert int(treated_mask.sum()) == step1_summary["treated_count"]


def test_treated_mask_count_near_paper_estimate(treated_mask):
    # 전처리문서 6.3절의 근사치 3,825(분모 9,351)와 우리 파이프라인 실측 3,854(분모 9,356)는
    # 분모가 달라 정확히 같을 필요는 없음 — Step 1에서 이미 ±0.02 fraction tolerance로 검증됨.
    # 여기서는 같은 자릿수인지만 재확인.
    assert abs(int(treated_mask.sum()) - 3825) < 50


def test_treated_only_subsample_size_matches_treated_mask(image_emb, treated_mask):
    assert int(treated_mask.sum()) == treated_mask[treated_mask].shape[0]
    assert len(image_emb[treated_mask]) == int(treated_mask.sum())


def test_stage_b_full_contextual_matches_step4(image_emb, text_emb_visual_b, text_emb_contextual):
    # Stage A/B는 같은 text_emb_contextual 배열을 쓰고, 같은 analyze_condition_pair()로 계산하므로
    # Step 4(Stage A) 결과와 정확히 일치해야 한다 — 재확인용 sanity check(사용자 요청).
    with open(STEP4_RESULTS_PATH, "r", encoding="utf-8") as f:
        stage_a_results = json.load(f)

    metrics = analyze_condition_pair(
        image_emb, text_emb_visual_b, text_emb_contextual, "visual_b", "contextual",
        n_bootstrap=200, seed=42,
    )
    # point_estimate/linear_sep는 부트스트랩 표본 수와 무관하게 결정적이므로 그대로 비교 가능.
    # (CI 경계값은 n_bootstrap이 다르면 달라지므로 여기선 비교하지 않음)
    assert metrics["delta_gap_contextual_norm"] == pytest.approx(
        stage_a_results["delta_gap_contextual_norm"], abs=1e-6
    )
    assert metrics["linear_sep_acc_contextual_mean"] == pytest.approx(
        stage_a_results["linear_sep_acc_contextual_mean"], abs=1e-9
    )


def test_treated_only_analyze_condition_pair_sanity(image_emb, text_emb_visual_b, text_emb_contextual, treated_mask):
    metrics = analyze_condition_pair(
        image_emb[treated_mask], text_emb_visual_b[treated_mask], text_emb_contextual[treated_mask],
        "visual_b", "contextual", n_bootstrap=100, seed=42,
    )
    assert metrics["delta_gap_visual_b_ci_95"][0] < metrics["delta_gap_visual_b_norm"] < metrics["delta_gap_visual_b_ci_95"][1]
    assert metrics["delta_gap_contextual_ci_95"][0] < metrics["delta_gap_contextual_norm"] < metrics["delta_gap_contextual_ci_95"][1]
    assert 0.4 < metrics["linear_sep_acc_visual_b_mean"] <= 1.0
    assert 0.4 < metrics["linear_sep_acc_contextual_mean"] <= 1.0


# ---- Pair Margin 분석 (실험계획서 2.3절 확장) ----


def test_pair_margin_near_zero_for_random_embeddings():
    # narrow cone 없는 합성 데이터: 이미지/텍스트가 서로 무관한 독립 무작위 단위벡터라
    # 진짜 쌍이라고 특별히 더 유사할 이유가 없다 -> margin은 0 근처, Wilcoxon도 유의하지 않아야 함.
    rng = np.random.default_rng(0)
    n, d = 200, 64
    image = rng.normal(size=(n, d))
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    text = rng.normal(size=(n, d))
    text /= np.linalg.norm(text, axis=1, keepdims=True)

    result = compute_pair_margin(image, text, n_bootstrap=200, seed=0)

    assert abs(result["image_to_text"]["mean"]) < 0.05
    assert abs(result["text_to_image"]["mean"]) < 0.05
    assert result["image_to_text"]["wilcoxon_pvalue"] > 0.05
    assert result["text_to_image"]["wilcoxon_pvalue"] > 0.05


def test_pair_margin_detects_strong_diagonal():
    # 대각(진짜 쌍)만 인위적으로 강하게 정렬시킨 합성 데이터 -> margin이 뚜렷하게 양수,
    # Wilcoxon p-value가 매우 작고, Cohen's d가 크게 나와야 함 (탐지력 sanity check).
    rng = np.random.default_rng(1)
    n, d = 200, 64
    image = rng.normal(size=(n, d))
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    noise = 0.05 * rng.normal(size=(n, d))
    text = image + noise
    text /= np.linalg.norm(text, axis=1, keepdims=True)

    result = compute_pair_margin(image, text, n_bootstrap=200, seed=1)

    assert result["image_to_text"]["mean"] > 0.3
    assert result["text_to_image"]["mean"] > 0.3
    assert result["image_to_text"]["wilcoxon_pvalue"] < 0.01
    assert result["image_to_text"]["cohens_d"] > 1.0


def test_pair_margin_stage_a_b_contextual_matches(image_emb, text_emb_contextual):
    # Stage A/B는 같은 contextual 배열을 쓰므로 두 방향 margin 평균이 사실상 일치해야 한다
    # (같은 배열로 같은 함수를 두 번 부르는 셈 — 회귀 가드).
    result_a = compute_pair_margin(image_emb, text_emb_contextual, n_bootstrap=100, seed=42)
    result_b = compute_pair_margin(image_emb, text_emb_contextual, n_bootstrap=100, seed=42)
    assert result_a["image_to_text"]["mean"] == pytest.approx(result_b["image_to_text"]["mean"], abs=1e-9)
    assert result_a["text_to_image"]["mean"] == pytest.approx(result_b["text_to_image"]["mean"], abs=1e-9)


def test_pair_margin_treated_only_subset(image_emb, text_emb_visual_b, treated_mask):
    # step5의 n_treated=3,854와 일치하는지 + 대각선이 대칭이라 image_to_text/text_to_image
    # 평균이 (stage_a_visual 등 기존 조합에서 이미 관찰된 것처럼) 거의 같은 값인지 확인.
    assert int(treated_mask.sum()) == 3854

    result = compute_pair_margin(
        image_emb[treated_mask], text_emb_visual_b[treated_mask], n_bootstrap=100, seed=42,
    )
    assert result["n"] == 3854
    assert result["image_to_text"]["mean"] == pytest.approx(result["text_to_image"]["mean"], abs=1e-6)


# ---- Step 7b: downstream sweep (retrieval + zero-shot classification) ----


def test_classification_diagnostics_lambda_zero_matches_final_prevalence_filtered(image_emb, filenames):
    # final_prevalence_filtered(downstream_zeroshot_baseline.json)를 만든 것과 동일한 조건에서
    # compute_classification_diagnostics(lambda=0)이 그 기존 정확값과 엄격히 일치해야 한다
    # (같은 계산을 다른 경로로 재현하는 것이므로 atol=1e-4로 강하게 검증).
    with open(ZEROSHOT_RESULTS_PATH, "r", encoding="utf-8") as f:
        zeroshot_results = json.load(f)
    final_cfg = zeroshot_results["columns"]["type"]["final_prevalence_filtered"]
    class_names = list(final_cfg["per_class_accuracy"].keys())
    prompt_template = final_cfg["prompt_template"]
    majority_baseline_accuracy = final_cfg["majority_baseline_accuracy"]

    df = pd.read_parquet(PARQUET_PATH).set_index("filename")
    df_aligned = df.loc[filenames]
    all_labels = df_aligned["type"].to_numpy()
    keep_mask = np.isin(all_labels, class_names)
    image_emb_filtered = image_emb[keep_mask]
    ground_truth_labels = all_labels[keep_mask]

    model, processor = load_model("cpu")
    prompts = [prompt_template.format(name) for name in class_names]
    label_prompt_emb = encode_texts(prompts, model, processor, batch_size=len(prompts))

    # lambda=0(자명한 경우)과 0.5(실제 shift가 걸린 경우) 둘 다 — 클래스별 shift 동일성 assert가
    # compute_classification_diagnostics 내부에서 두 지점 모두에 대해 실행됨(아래 두 번째 확인 항목).
    result = compute_classification_diagnostics(
        image_emb_filtered, label_prompt_emb, ground_truth_labels, class_names,
        np.array([0.0, 0.5]), majority_baseline_accuracy,
    )
    p0 = result["points"][0]
    assert p0["micro_accuracy"] == pytest.approx(final_cfg["micro_accuracy"], abs=1e-4)
    assert p0["macro_accuracy"] == pytest.approx(final_cfg["macro_accuracy"], abs=1e-4)


def test_retrieval_diagnostics_lambda_zero_above_floor(image_emb, text_emb_contextual):
    # 약한 sanity check — step6b_pair_margin의 큰 Cohen's d를 감안하면 R@1은 1%보다 훨씬 높아야 함.
    result = compute_retrieval_diagnostics(
        image_emb, text_emb_contextual, np.array([0.0]), n_subsample=2000, seed=42,
    )
    r1 = result["points"][0]["i2t"]["r_at_1"]
    assert r1 > 0.01


# ---- Step 7c: retrieval 확장 (T->I, R@5/R@10, λ 자동 확장) ----


def test_retrieval_diagnostics_includes_t2i_and_all_k(image_emb, text_emb_contextual):
    result = compute_retrieval_diagnostics(
        image_emb, text_emb_contextual, np.array([0.0, 0.5]), n_subsample=500, seed=42,
    )
    for p in result["points"]:
        for direction in ("i2t", "t2i"):
            metrics = p[direction]
            for key in ("r_at_1", "r_at_5", "r_at_10", "median_rank", "mrr"):
                assert key in metrics
            # R@k는 k에 대해 비감소여야 한다 (정의상).
            assert metrics["r_at_1"] <= metrics["r_at_5"] <= metrics["r_at_10"]


def test_retrieval_extended_matches_base_at_final_grid(image_emb, text_emb_visual_a):
    # compute_retrieval_diagnostics_extended가 도달한 최종 λ 그리드로 compute_retrieval_diagnostics를
    # 직접 불러도 같은 값이 나와야 한다 (확장 로직이 값 자체를 왜곡하지 않는지 확인 — 재사용 검증).
    extended = compute_retrieval_diagnostics_extended(
        image_emb, text_emb_visual_a, n_subsample=500, seed=42,
        base_lambda_max=2.0, n_points=5, target_max_distance=1.9, max_expansions=2,
    )
    final_max = extended["lambda_grid_expansion"]["final_lambda_max"]
    direct = compute_retrieval_diagnostics(
        image_emb, text_emb_visual_a, np.linspace(0, final_max, 5), n_subsample=500, seed=42,
    )
    for pe, pd_ in zip(extended["points"], direct["points"]):
        assert pe["i2t"]["r_at_1"] == pytest.approx(pd_["i2t"]["r_at_1"], abs=1e-12)
        assert pe["t2i"]["r_at_1"] == pytest.approx(pd_["t2i"]["r_at_1"], abs=1e-12)


def test_retrieval_lambda_expansion_respects_max(image_emb, text_emb_visual_a):
    result = compute_retrieval_diagnostics_extended(
        image_emb, text_emb_visual_a, n_subsample=500, seed=42,
        base_lambda_max=2.0, n_points=5, target_max_distance=1.9, max_expansions=RETRIEVAL_MAX_LAMBDA_EXPANSIONS,
    )
    assert result["lambda_grid_expansion"]["n_expansions"] <= RETRIEVAL_MAX_LAMBDA_EXPANSIONS
    assert "direction_k_summary" in result
    assert len(result["direction_k_summary"]) == 6  # 2 directions x 3 R@k
