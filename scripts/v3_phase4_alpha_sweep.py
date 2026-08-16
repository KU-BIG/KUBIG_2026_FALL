"""
PROJECT_v3.md Phase 4/5 — RQ4(핵심): "conformity 분포 정합"만으로 좋은 표현을 설명할 수 있는가.

오프셋 공식(§1.3, §12): v' = v - alpha * m  (원점 방향, 모달리티별 평균 벡터 m)
alpha를 -1~1로 쓸면서:
  1) 이미지/텍스트 conformity 분포의 KL divergence
  2) text->image retrieval (R@1, R@5)
둘을 동시에 관찰한다. 두 지점(KL 최소 alpha, retrieval 최대 alpha)이 다르면
"conformity 정합만으로는 불충분하다"는 직접 증거가 된다(§8 Phase 5, §9).

캐싱된 raw(정규화 전) 임베딩을 재사용 — 재인코딩 없음, 순수 배열 연산.
"""

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "results" / "embeddings"
N_SAMPLES = 500
SEED = 0
ALPHAS = np.round(np.arange(-1.0, 1.01, 0.2), 2).tolist()


def conformity_true(embs: torch.Tensor) -> torch.Tensor:
    E = torch.nn.functional.normalize(embs, dim=-1)
    S = E @ E.T
    n = S.shape[0]
    S.fill_diagonal_(0)
    return S.sum(1) / (n - 1)


def kl_divergence_hist(a: np.ndarray, b: np.ndarray, bins=50, rng=(-1.0, 1.0), eps=1e-6):
    """두 표본의 히스토그램 기반 KL(a || b)."""
    ha, _ = np.histogram(a, bins=bins, range=rng, density=False)
    hb, _ = np.histogram(b, bins=bins, range=rng, density=False)
    pa = (ha + eps) / (ha.sum() + eps * bins)
    pb = (hb + eps) / (hb.sum() + eps * bins)
    return float(np.sum(pa * np.log(pa / pb)))


def retrieval_at_k(sim: np.ndarray, k_list=(1, 5, 10)):
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] for i in range(n)])
    return {f"R@{k}": float((ranks < k).mean()) for k in k_list}


def main():
    img_raw = torch.from_numpy(np.load(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_img_raw.npy"))
    txt_raw = torch.from_numpy(np.load(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_txt_raw.npy"))

    m_img = img_raw.mean(0, keepdim=True)
    m_txt = txt_raw.mean(0, keepdim=True)

    print(f"{'alpha':>6} {'KL(img||txt)':>13} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    results = []
    for alpha in ALPHAS:
        img_shift = img_raw - alpha * m_img
        txt_shift = txt_raw - alpha * m_txt

        conf_img = conformity_true(img_shift).numpy()
        conf_txt = conformity_true(txt_shift).numpy()
        kl = kl_divergence_hist(conf_img, conf_txt)

        img_n = torch.nn.functional.normalize(img_shift, dim=-1).numpy()
        txt_n = torch.nn.functional.normalize(txt_shift, dim=-1).numpy()
        sim = txt_n @ img_n.T  # text -> image
        r = retrieval_at_k(sim)

        results.append({"alpha": alpha, "kl": kl, **r})
        print(f"{alpha:>6.2f} {kl:>13.5f} {r['R@1']:>7.3f} {r['R@5']:>7.3f} {r['R@10']:>7.3f}")

    kls = [r["kl"] for r in results]
    r1s = [r["R@1"] for r in results]
    alpha_kl_min = results[int(np.argmin(kls))]["alpha"]
    alpha_r1_max = results[int(np.argmax(r1s))]["alpha"]

    print(f"\nKL 최소 alpha = {alpha_kl_min}  (KL={min(kls):.5f})")
    print(f"R@1 최대 alpha = {alpha_r1_max}  (R@1={max(r1s):.3f})")
    if alpha_kl_min == alpha_r1_max:
        print("-> 두 지점이 일치. Double-Ellipsoid 설명이 SemArt에서도 성립 (확증)")
    else:
        print("-> 두 지점이 다름. conformity 정합만으로는 불충분 (§9 핵심 기여)")

    out_path = ROOT / "results" / "v3_phase4_alpha_sweep.json"
    out_path.write_text(json.dumps({
        "results": results,
        "alpha_kl_min": alpha_kl_min,
        "alpha_r1_max": alpha_r1_max,
    }, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
