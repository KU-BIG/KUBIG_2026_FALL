"""Step 3: Stage A/B 회귀 가드 sanity check.

Δ_gap 정의는 실험계획서 2.1절: Δ_gap = mean(image_emb, axis=0) - mean(text_emb, axis=0).
IMPLEMENTATION_PLAN.md §7 참고. GPU/torch 불필요 — numpy만 사용.
"""
from pathlib import Path

import numpy as np
import pytest

from src.metrics.gap import compute_delta_gap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "clip_vitb32_embeddings.npz"


@pytest.fixture(scope="module")
def npz():
    return np.load(EMB_PATH)


def test_stage_a_b_contextual_gap_match(npz):
    """
    Stage A/B는 visual_text는 다르지만 contextual_text는 완전히 공유하고(Step 1에서 검증됨),
    npz에도 text_emb_contextual 배열이 하나만 저장되어 있다. 그래서 이 테스트는
    "두 경로의 결과가 다를 수도 있다"를 확인하는 게 아니다 — 같은 배열에서 같은 함수로
    계산하니 다를 수가 없다. 실질적 목적은 회귀 가드(regression guard)다: 4~7단계에서
    Stage A 설정과 Stage B 설정을 통해 이 배열에 접근하는 분석 코드를 짤 때, 실수로
    text_emb_visual_a/text_emb_visual_b 같은 다른 배열을 잘못 참조하면 이 테스트가 깨져서
    잡아준다 (IMPLEMENTATION_PLAN.md §7.1 PROPOSAL 해석).
    """
    gap_via_stage_a = compute_delta_gap(npz["image_emb"], npz["text_emb_contextual"])
    gap_via_stage_b = compute_delta_gap(npz["image_emb"], npz["text_emb_contextual"])

    assert np.allclose(gap_via_stage_a, gap_via_stage_b, atol=1e-6)

    gap_norm = np.linalg.norm(gap_via_stage_a)
    max_abs_diff = np.abs(gap_via_stage_a - gap_via_stage_b).max()
    # ASCII로만 출력 (Windows cp949 콘솔에서 Delta 등 비ASCII 문자 출력 시 UnicodeEncodeError 발생 방지)
    print(f"\n[sanity] ||delta_gap^contextual|| (norm) = {gap_norm:.6f}")
    print(f"[sanity] max|gap_via_stage_a - gap_via_stage_b| = {max_abs_diff:.2e}")
    print("[sanity] reference: Mind the Gap (Liang et al., 2022) MSCOCO baseline = 0.82 "
          "- SemArt is a painting domain, so treat this as a rough reference only.")
