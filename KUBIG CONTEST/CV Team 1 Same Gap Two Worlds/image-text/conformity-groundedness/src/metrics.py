"""gap 지표. PROJECT_v2.md §1.2(L2M), §1.3(RMG)."""

import numpy as np


def l2m(img_emb: np.ndarray, txt_emb: np.ndarray) -> float:
    """img_emb, txt_emb: (N, D), 이미 L2 정규화된 상태. 개별 정규화 후 평균 순서를 지킨다."""
    delta = img_emb.mean(axis=0) - txt_emb.mean(axis=0)
    return float(np.linalg.norm(delta))


def _cos_dist_scaled(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """코사인 비유사도를 [0,1]로 스케일: (1 - cos_sim) / 2."""
    sim = a @ b.T
    return (1.0 - sim) / 2.0


def rmg(img_emb: np.ndarray, txt_emb: np.ndarray) -> float:
    """PROJECT_v2.md §1.3 RMG. img_emb[i] <-> txt_emb[i] 페어링 가정."""
    n = img_emb.shape[0]
    cross = _cos_dist_scaled(img_emb, txt_emb)
    mean_pair_dist = float(np.diagonal(cross).mean())

    img_intra = _cos_dist_scaled(img_emb, img_emb)
    txt_intra = _cos_dist_scaled(txt_emb, txt_emb)
    iu = np.triu_indices(n, k=1)
    mean_intra = float(np.concatenate([img_intra[iu], txt_intra[iu]]).mean())

    return mean_pair_dist / (mean_intra + mean_pair_dist)
