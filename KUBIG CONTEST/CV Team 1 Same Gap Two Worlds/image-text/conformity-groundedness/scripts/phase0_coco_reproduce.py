"""
Phase 0 — PROJECT_v2.md §8: COCO validation에서 Liang의 L2M=0.82 재현.

- MS-COCO val2017, 5,000쌍, 이미지당 캡션 1개(각 image_id의 첫 캡션)
- OpenCLIP ViT-B/16, pretrained='openai' (QuickGELU 보정 적용, src/encode.py)
- N별 L2M 수렴 곡선(N=100,500,1000,5000)도 함께 본다(§7.3)
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from encode import encode_images, encode_texts, load_model  # noqa: E402
from metrics import l2m, rmg  # noqa: E402

COCO_DIR = ROOT / "data" / "coco"
IMAGES_DIR = COCO_DIR / "val2017"
CAPTIONS_JSON = COCO_DIR / "annotations" / "captions_val2017.json"
N_FULL = 5000
CONVERGENCE_NS = [100, 500, 1000, 5000]
SEED = 0


def load_coco_pairs():
    data = json.loads(CAPTIONS_JSON.read_text())
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    first_caption = {}
    for ann in data["annotations"]:
        first_caption.setdefault(ann["image_id"], ann["caption"])

    pairs = [
        (id_to_file[img_id], cap)
        for img_id, cap in first_caption.items()
        if img_id in id_to_file
    ]
    random.seed(SEED)
    random.shuffle(pairs)
    return pairs


def main():
    print(f"[1/4] loading COCO val2017 pairs ...")
    pairs = load_coco_pairs()
    print(f"  total pairs: {len(pairs)}")
    pairs = pairs[:N_FULL]

    print("[2/4] loading OpenCLIP ViT-B-16 (openai) ...")
    model, preprocess, tokenizer = load_model("ViT-B-16", "openai", device="cpu")

    print(f"[3/4] encoding {len(pairs)} images + texts (cached) ...")
    images = [Image.open(IMAGES_DIR / f).convert("RGB") for f, _ in pairs]
    texts = [c for _, c in pairs]

    img_emb = encode_images(
        images, model, preprocess, cache_tag=f"coco_val_n{len(pairs)}_seed{SEED}_orig_img"
    )
    txt_emb = encode_texts(
        texts, model, tokenizer, cache_tag=f"coco_val_n{len(pairs)}_seed{SEED}_orig_txt"
    )

    print("[4/4] computing L2M / RMG at increasing N (convergence) ...")
    results = []
    for n in CONVERGENCE_NS:
        if n > len(pairs):
            continue
        m_l2m = l2m(img_emb[:n], txt_emb[:n])
        m_rmg = rmg(img_emb[:n], txt_emb[:n])
        results.append({"n": n, "l2m": m_l2m, "rmg": m_rmg})
        print(f"  n={n:>5}  L2M={m_l2m:.4f}  RMG={m_rmg:.4f}")

    final = results[-1]
    print(f"\nFinal (n={final['n']}): L2M={final['l2m']:.4f} (Liang 기준값 0.82)")
    diff = abs(final["l2m"] - 0.82)
    print(f"|L2M - 0.82| = {diff:.4f}")

    out_path = ROOT / "results" / "phase0_coco_reproduce.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
