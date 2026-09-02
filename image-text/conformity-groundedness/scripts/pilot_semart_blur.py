"""
비공식 예비 파일럿 — SemArt, 이미지 블러 스윕.

주의: PROJECT_v2.md의 정식 Phase 0/1은 MS-COCO 기준이다(§8). COCO가 아직 준비되지 않아
이 스크립트는 그 자리를 대신하지 않는다. 목적은 파이프라인(degrade/encode/metrics)이
실제로 동작하는지, SemArt에서 대략적인 신호가 보이는지를 빠르게 확인하는 것뿐이다.
"""

import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from degrade import gaussian_blur  # noqa: E402
from encode import encode_images, encode_texts, load_model  # noqa: E402
from metrics import l2m, rmg  # noqa: E402

DATA_DIR = ROOT / "data" / "SemArt"
IMAGES_DIR = DATA_DIR / "Images"
N_SAMPLES = 200
SIGMAS = [0, 1, 2, 4, 8, 16]
SEED = 0


def load_pairs(csv_path, n, seed):
    import random

    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def main():
    print(f"[1/4] loading {N_SAMPLES} SemArt val pairs ...")
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)
    images = [Image.open(IMAGES_DIR / r["IMAGE_FILE"]).convert("RGB") for r in rows]
    texts = [r["DESCRIPTION"] for r in rows]

    print("[2/4] loading OpenCLIP ViT-B-16 (openai) ...")
    model, preprocess, tokenizer = load_model("ViT-B-16", "openai", device="cpu")

    print("[3/4] encoding text (fixed) ...")
    txt_emb = encode_texts(
        texts, model, tokenizer, cache_tag=f"semart_val_n{N_SAMPLES}_seed{SEED}_text"
    )

    print("[4/4] sweeping image blur sigma ...")
    results = []
    for sigma in SIGMAS:
        degraded = [gaussian_blur(im, sigma) for im in images]
        img_emb = encode_images(
            degraded,
            model,
            preprocess,
            cache_tag=f"semart_val_n{N_SAMPLES}_seed{SEED}_blur{sigma}",
        )
        m_l2m = l2m(img_emb, txt_emb)
        m_rmg = rmg(img_emb, txt_emb)
        results.append({"sigma": sigma, "l2m": m_l2m, "rmg": m_rmg})
        print(f"  sigma={sigma:>3}  L2M={m_l2m:.4f}  RMG={m_rmg:.4f}")

    out_path = ROOT / "results" / "pilot_semart_blur.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
