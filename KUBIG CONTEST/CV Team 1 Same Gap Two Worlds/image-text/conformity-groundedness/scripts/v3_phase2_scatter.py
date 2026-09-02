"""
PROJECT_v3.md §8 Phase 2 — "이 한 장이 프로젝트의 얼굴이다"
x축 conformity(텍스트, 독특함), y축 groundedness(text->image, 접지성) 산점도.

Phase 1 / groundedness 스크립트가 캐싱해둔 raw 임베딩과 순위 기반 groundedness를 재사용한다.
재인코딩 없음 — 순수 배열 연산이라 즉시 끝난다.
"""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "results" / "embeddings"
DATA_DIR = ROOT / "data" / "SemArt"

N_SAMPLES = 500
SEED = 0


def load_pairs(csv_path, n, seed):
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def conformity_true(embs: torch.Tensor) -> torch.Tensor:
    E = torch.nn.functional.normalize(embs, dim=-1)
    S = E @ E.T
    n = S.shape[0]
    S.fill_diagonal_(0)
    return S.sum(1) / (n - 1)


def main():
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)

    txt_raw = torch.from_numpy(np.load(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_txt_raw.npy"))
    t2i_ground = np.load(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_t2i_groundedness.npy")

    conf = conformity_true(txt_raw).numpy()

    pearson = float(np.corrcoef(conf, t2i_ground)[0, 1])
    from scipy.stats import spearmanr
    spearman = float(spearmanr(conf, t2i_ground).correlation)
    print(f"conformity vs groundedness: Pearson={pearson:.4f}  Spearman={spearman:.4f}")
    print(f"conformity: mean={conf.mean():.4f} std={conf.std():.4f} min={conf.min():.4f} max={conf.max():.4f}")
    print(f"groundedness: mean={t2i_ground.mean():.4f} std={t2i_ground.std():.4f}")

    conf_med = float(np.median(conf))
    ground_med = float(np.median(t2i_ground))
    quadrants = {
        "typical_grounded": int(((conf >= conf_med) & (t2i_ground >= ground_med)).sum()),
        "typical_ungrounded": int(((conf >= conf_med) & (t2i_ground < ground_med)).sum()),
        "unique_grounded": int(((conf < conf_med) & (t2i_ground >= ground_med)).sum()),
        "unique_ungrounded": int(((conf < conf_med) & (t2i_ground < ground_med)).sum()),
    }
    print("quadrant counts (median split):", quadrants)

    # "독특 + 비접지" = 미탐색 영역 예시 문장 몇 개
    mask = (conf < conf_med) & (t2i_ground < np.percentile(t2i_ground, 25))
    idx = np.where(mask)[0]
    print(f"\n'독특+비접지' 후보 {len(idx)}개 중 샘플:")
    rng = random.Random(1)
    for i in rng.sample(list(idx), min(5, len(idx))):
        desc = rows[i]["DESCRIPTION"]
        print(f"  [{rows[i]['IMAGE_FILE']}] conf={conf[i]:.3f} ground={t2i_ground[i]:.3f}")
        print(f"    {desc[:150]}")

    # 대비: "전형 + 접지" 예시도
    mask2 = (conf >= np.percentile(conf, 75)) & (t2i_ground >= np.percentile(t2i_ground, 75))
    idx2 = np.where(mask2)[0]
    print(f"\n'전형+접지' 후보 {len(idx2)}개 중 샘플:")
    for i in rng.sample(list(idx2), min(3, len(idx2))):
        desc = rows[i]["DESCRIPTION"]
        print(f"  [{rows[i]['IMAGE_FILE']}] conf={conf[i]:.3f} ground={t2i_ground[i]:.3f}")
        print(f"    {desc[:150]}")

    out = {
        "n_samples": len(rows),
        "pearson": pearson,
        "spearman": spearman,
        "conformity": conf.tolist(),
        "groundedness": t2i_ground.tolist(),
        "quadrants": quadrants,
    }
    out_path = ROOT / "results" / "v3_phase2_scatter.json"
    out_path.write_text(json.dumps(out))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
