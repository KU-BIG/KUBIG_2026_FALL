"""Embedding shift — 실험계획서 5.2절 (1). Step 6(temperature landscape)과 Step 7(gap 개입)이 공유한다.

frozen 임베딩에 대한 순수 후처리(post-hoc) 연산 — gradient step 없음, GPU 불필요.
"""
import numpy as np


def shift_embeddings(
    x: np.ndarray,
    y: np.ndarray,
    delta_gap: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """x_shift = normalize(x - lam*delta_gap), y_shift = normalize(y + lam*delta_gap).

    개별 행(row) 단위로 shift 후 L2 재정규화(논문 5.2절: "shift 후 반드시 재정규화하여
    단위 구면 위로 되돌린다"). resulting_gap_distance는 재정규화된 임베딩의 centroid로
    다시 계산한 ||mean(x_shift) - mean(y_shift)||.
    """
    x_shift = x - lam * delta_gap
    y_shift = y + lam * delta_gap
    x_shift = x_shift / np.linalg.norm(x_shift, axis=1, keepdims=True)
    y_shift = y_shift / np.linalg.norm(y_shift, axis=1, keepdims=True)

    resulting_gap_distance = float(np.linalg.norm(x_shift.mean(axis=0) - y_shift.mean(axis=0)))
    return x_shift, y_shift, resulting_gap_distance
