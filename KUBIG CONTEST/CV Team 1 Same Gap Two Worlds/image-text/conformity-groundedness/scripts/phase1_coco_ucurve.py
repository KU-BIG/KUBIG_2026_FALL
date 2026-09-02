"""
Phase 1 — PROJECT_v2.md §8: COCO에서 U자 곡선 파일럿 (⭐ 최우선).

- COCO val2017 1,000쌍 (Phase 0과 같은 시드로 뽑은 순서의 앞 1,000개 — 부분집합 일관성 유지)
- 텍스트 고정(원본 캡션), 이미지 열화만 스윕 (블러 σ 0~32, 7단계)
- sigma=0은 Phase 0에서 이미 인코딩한 전체 5,000장의 앞 1,000개를 그대로 재사용(중복 계산 방지)

결과가 나오면 멈추고 보고할 것 — U자인지, 단조 증가인지, 평탄한지.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from degrade import gaussian_blur  # noqa: E402
from encode import CACHE_DIR, encode_images, encode_texts, load_model  # noqa: E402
from metrics import l2m, rmg  # noqa: E402
from phase0_coco_reproduce import load_coco_pairs  # noqa: E402

COCO_DIR = ROOT / "data" / "coco"
IMAGES_DIR = COCO_DIR / "val2017"
N_PILOT = 1000
SIGMAS = [0, 1, 2, 4, 8, 16, 32]
SEED = 0
N_FULL_CACHE = 5000  # Phase 0에서 캐싱한 전체 규모, 여기서 앞 N_PILOT개만 재사용


def main():
    print(f"[1/3] loading COCO pairs (same seed/order as Phase 0) ...")
    pairs = load_coco_pairs()[:N_PILOT]
    print(f"  pilot pairs: {len(pairs)}")

    print("[2/3] loading OpenCLIP ViT-B-16 (openai) ...")
    model, preprocess, tokenizer = load_model("ViT-B-16", "openai", device="cpu")

    texts = [c for _, c in pairs]
    txt_cache = CACHE_DIR / f"coco_val_n{N_FULL_CACHE}_seed{SEED}_orig_txt.npy"
    if txt_cache.exists():
        txt_emb = np.load(txt_cache)[:N_PILOT]
    else:
        txt_emb = encode_texts(texts, model, tokenizer)

    print("[3/3] sweeping image blur sigma ...")
    results = []
    for sigma in SIGMAS:
        if sigma == 0:
            img_cache = CACHE_DIR / f"coco_val_n{N_FULL_CACHE}_seed{SEED}_orig_img.npy"
            if img_cache.exists():
                img_emb = np.load(img_cache)[:N_PILOT]
            else:
                images = [Image.open(IMAGES_DIR / f).convert("RGB") for f, _ in pairs]
                img_emb = encode_images(images, model, preprocess)
        else:
            images = [
                gaussian_blur(Image.open(IMAGES_DIR / f).convert("RGB"), sigma)
                for f, _ in pairs
            ]
            tag = f"coco_val_n{N_PILOT}_seed{SEED}_blur{sigma}"
            img_emb = encode_images(images, model, preprocess, cache_tag=tag)

        m_l2m = l2m(img_emb, txt_emb)
        m_rmg = rmg(img_emb, txt_emb)
        results.append({"sigma": sigma, "l2m": m_l2m, "rmg": m_rmg})
        print(f"  sigma={sigma:>3}  L2M={m_l2m:.4f}  RMG={m_rmg:.4f}")

    out_path = ROOT / "results" / "phase1_coco_ucurve.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")

    l2ms = [r["l2m"] for r in results]
    min_idx = int(np.argmin(l2ms))
    if min_idx == 0:
        verdict = "단조 증가 (원본이 최소, 열화할수록 계속 커짐)"
    elif min_idx == len(l2ms) - 1:
        verdict = "단조 감소 (예상 밖 — 재확인 필요)"
    else:
        verdict = f"U자 형태 가능성 (최소점: sigma={results[min_idx]['sigma']})"
    print(f"\n판정: {verdict}")


if __name__ == "__main__":
    main()
