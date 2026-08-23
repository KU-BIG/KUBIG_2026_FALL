"""Step 6: temperature loss landscape 로직 정합성 검증."""
from pathlib import Path

import numpy as np
import pytest

from src.interventions.shift import shift_embeddings
from src.metrics.temperature import contrastive_loss, loss_landscape, run_temperature_analysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "clip_vitb32_embeddings.npz"
STAGE_A_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step4_stage_a_results.json"
STAGE_B_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "step5_stage_b_results.json"

SUBSAMPLE_N = 200
# 테스트용 소규모 실행(run_temperature_analysis)의 대표성 체크는 부분표본이 작아서 일부러 벗어날 수
# 있음 — 로직/구조 검증이 목적이므로 정확도 자체는 assert 대상이 아님.
TEST_RUN_N_SUBSAMPLE = 300


@pytest.fixture(scope="module")
def npz():
    return np.load(EMB_PATH)


@pytest.fixture(scope="module")
def image_sub(npz):
    return npz["image_emb"][:SUBSAMPLE_N]


@pytest.fixture(scope="module")
def text_sub(npz):
    return npz["text_emb_contextual"][:SUBSAMPLE_N]


@pytest.fixture(scope="module")
def delta_gap(image_sub, text_sub):
    return image_sub.mean(axis=0) - text_sub.mean(axis=0)


def test_shift_embeddings_lambda_zero_is_unchanged(image_sub, text_sub, delta_gap):
    # lam=0이면 shift가 없으니 재정규화만 적용됨 — 입력이 이미 단위벡터이므로 원본과 사실상 동일해야 함
    x_shift, y_shift, dist = shift_embeddings(image_sub, text_sub, delta_gap, lam=0.0)
    assert np.allclose(x_shift, image_sub, atol=1e-5)
    assert np.allclose(y_shift, text_sub, atol=1e-5)
    assert dist == pytest.approx(float(np.linalg.norm(delta_gap)), abs=1e-5)


def test_shift_embeddings_renormalizes(image_sub, text_sub, delta_gap):
    x_shift, y_shift, _ = shift_embeddings(image_sub, text_sub, delta_gap, lam=1.5)
    assert np.allclose(np.linalg.norm(x_shift, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(y_shift, axis=1), 1.0, atol=1e-5)


def test_contrastive_loss_aligned_pairs_lower_than_shuffled(image_sub, text_sub):
    # 정렬된(같은 이미지-텍스트) 쌍의 대각 유사도가 최댓값이라 loss가 낮아야 하고,
    # 무작위로 섞은 쌍은 대각 지배가 깨지므로 loss가 더 높아야 함.
    tau = 1 / 100
    loss_aligned = contrastive_loss(image_sub, text_sub, tau)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(image_sub))
    loss_shuffled = contrastive_loss(image_sub, text_sub[perm], tau)

    assert loss_aligned < loss_shuffled


def test_contrastive_loss_finite_and_nonnegative(image_sub, text_sub):
    for tau in (1 / 100, 1.0):
        loss = contrastive_loss(image_sub, text_sub, tau)
        assert np.isfinite(loss)
        assert loss >= 0


def test_loss_landscape_grouped_by_tau_and_distance_sorted(image_sub, text_sub, delta_gap):
    taus = [1 / 100, 1.0]
    lambdas = np.linspace(0, 2, 11)
    landscape = loss_landscape(image_sub, text_sub, delta_gap, taus, lambdas)

    assert set(landscape.keys()) == set(taus)
    for tau in taus:
        points = landscape[tau]
        assert len(points) == len(lambdas)
        distances = [p[0] for p in points]
        assert distances == sorted(distances)


@pytest.fixture(scope="module")
def small_run_results(tmp_path_factory):
    # 실제 run_temperature_analysis()를 작은 부분표본으로 돌려서 오케스트레이션/요약 로직 자체를
    # 검증 (전체 규모 2,000 실행은 별도로 `python -m src.metrics.temperature`에서 수행).
    tmp_dir = tmp_path_factory.mktemp("step6_test")
    return run_temperature_analysis(
        results_path=tmp_dir / "step6_temperature_results.json",
        figures_dir=tmp_dir,
        n_subsample=TEST_RUN_N_SUBSAMPLE,
        seed=42,
    )


def test_run_produces_all_four_combos(small_run_results):
    assert set(small_run_results["combos"].keys()) == {
        "stage_a_visual", "stage_a_contextual", "stage_b_visual", "stage_b_contextual",
    }


def test_stage_a_b_contextual_landscape_matches(small_run_results):
    # Stage A/B는 같은 contextual 배열, 같은 서브샘플 인덱스를 쓰므로 landscape 요약값이
    # 넉넉한 atol로 일치해야 한다 — 실수로 다른 배열을 참조하면 이 테스트가 깨지는 회귀 가드.
    a_ctx = small_run_results["combos"]["stage_a_contextual"]
    b_ctx = small_run_results["combos"]["stage_b_contextual"]
    for label in a_ctx["per_tau"]:
        assert a_ctx["per_tau"][label]["global_min_distance"] == pytest.approx(
            b_ctx["per_tau"][label]["global_min_distance"], abs=1e-4
        )
    sanity = small_run_results["sanity_check_stage_a_b_contextual"]
    assert sanity["passed"] is True


def test_subsample_check_present_and_numeric(small_run_results):
    checks = small_run_results["subsample_check"]
    assert set(checks.keys()) == {
        "stage_a_visual", "stage_a_contextual", "stage_b_visual", "stage_b_contextual",
    }
    for check in checks.values():
        assert np.isfinite(check["distance_sub"])
        assert np.isfinite(check["distance_full_reference"])
        assert np.isfinite(check["relative_diff"])


def test_per_tau_covers_all_six_taus(small_run_results):
    expected_labels = {"1/100", "1/50", "1/30", "1/20", "1/10", "1"}
    for combo in small_run_results["combos"].values():
        assert set(combo["per_tau"].keys()) == expected_labels
