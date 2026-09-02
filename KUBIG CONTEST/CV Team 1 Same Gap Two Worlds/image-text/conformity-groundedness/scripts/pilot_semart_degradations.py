"""
비공식 예비 파일럿 — SemArt, 열화 방식 3종(블러/다운샘플/center crop) 비교.

pilot_semart_blur.py에서 본 "단조 증가" 패턴이 블러라는 특정 방식의 왜곡 때문인지,
아니면 방식과 무관하게 나타나는 신호인지 확인하기 위한 것.
같은 200쌍 텍스트 임베딩(캐시)과 원본 이미지 세트를 그대로 재사용한다.
"""

import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from degrade import center_crop, downsample_upsample, gaussian_blur  # noqa: E402
from encode import encode_images, encode_texts, load_model  # noqa: E402
from metrics import l2m, rmg  # noqa: E402

DATA_DIR = ROOT / "data" / "SemArt"
IMAGES_DIR = DATA_DIR / "Images"
N_SAMPLES = 200
SEED = 0

SWEEPS = {
    "blur": {
        "fn": gaussian_blur,
        "levels": [0, 1, 2, 4, 8, 16],  # sigma, 이미 pilot_semart_blur.py에서 측정함
    },
    "downsample": {
        "fn": downsample_upsample,
        "levels": [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],  # scale, 작을수록 강한 열화
    },
    "crop": {
        "fn": center_crop,
        "levels": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],  # ratio, 작을수록 강한 열화 (분포 이동 통제군)
    },
}


def load_pairs(csv_path, n, seed):
    import random

    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def main():
    print(f"[1/3] loading {N_SAMPLES} SemArt val pairs ...")
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)
    images = [Image.open(IMAGES_DIR / r["IMAGE_FILE"]).convert("RGB") for r in rows]
    texts = [r["DESCRIPTION"] for r in rows]

    print("[2/3] loading OpenCLIP ViT-B-16 (openai) + encoding text (fixed, cached) ...")
    model, preprocess, tokenizer = load_model("ViT-B-16", "openai", device="cpu")
    txt_emb = encode_texts(
        texts, model, tokenizer, cache_tag=f"semart_val_n{N_SAMPLES}_seed{SEED}_text"
    )

    print("[3/3] sweeping each degradation method ...")
    all_results = {}
    for method, spec in SWEEPS.items():
        print(f"\n-- {method} --")
        method_results = []
        for level in spec["levels"]:
            degraded = [spec["fn"](im, level) for im in images]
            tag = f"semart_val_n{N_SAMPLES}_seed{SEED}_{method}{level}"
            img_emb = encode_images(degraded, model, preprocess, cache_tag=tag)
            m_l2m = l2m(img_emb, txt_emb)
            m_rmg = rmg(img_emb, txt_emb)
            method_results.append({"level": level, "l2m": m_l2m, "rmg": m_rmg})
            print(f"  level={level!s:>8}  L2M={m_l2m:.4f}  RMG={m_rmg:.4f}")
        all_results[method] = method_results

    out_path = ROOT / "results" / "pilot_semart_degradations.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
