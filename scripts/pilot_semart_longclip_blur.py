"""
Long-CLIP으로 SemArt 전체 텍스트(자르지 않음)를 써서, 이미지 블러 스윕이 여전히
단조 증가인지 빠르게 확인하는 소규모 파일럿. 시간 제약으로 표본을 작게 잡는다.

원본 CLIP(ViT-B/16, 77토큰 실질 75토큰)에서는 SemArt 텍스트의 55%가 잘렸다(확인 완료).
Long-CLIP은 248토큰까지 커버하므로 대부분의 SemArt 설명을 안 자르고 쓸 수 있다.
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from degrade import gaussian_blur  # noqa: E402
from metrics import l2m, rmg  # noqa: E402

DATA_DIR = ROOT / "data" / "SemArt"
IMAGES_DIR = DATA_DIR / "Images"
N_SAMPLES = 50
SIGMAS = [0, 4, 8, 16, 32]
SEED = 0
MAX_LEN = 248


def load_pairs(csv_path, n, seed):
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def encode_images(images, model, processor):
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.numpy()


def encode_texts(texts, model, processor):
    with torch.no_grad():
        inputs = processor(
            text=texts, return_tensors="pt", max_length=MAX_LEN,
            padding="max_length", truncation=True,
        )
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.numpy()


def main():
    print(f"[1/4] loading {N_SAMPLES} SemArt val pairs (full DESCRIPTION, no manual truncation) ...")
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)
    images = [Image.open(IMAGES_DIR / r["IMAGE_FILE"]).convert("RGB") for r in rows]
    texts = [r["DESCRIPTION"] for r in rows]
    lens = [len(t.split()) for t in texts]
    print(f"  desc word-count: mean={np.mean(lens):.0f} max={max(lens)}")

    print("[2/4] loading Long-CLIP-B ...")
    t0 = time.time()
    model = AutoModel.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("[3/4] encoding text (fixed, full length up to 248 tokens) ...")
    txt_emb = encode_texts(texts, model, processor)

    print("[4/4] sweeping image blur sigma ...")
    results = []
    for sigma in SIGMAS:
        degraded = [gaussian_blur(im, sigma) for im in images]
        img_emb = encode_images(degraded, model, processor)
        m_l2m = l2m(img_emb, txt_emb)
        m_rmg = rmg(img_emb, txt_emb)
        results.append({"sigma": sigma, "l2m": m_l2m, "rmg": m_rmg})
        print(f"  sigma={sigma:>3}  L2M={m_l2m:.4f}  RMG={m_rmg:.4f}")

    out_path = ROOT / "results" / "pilot_semart_longclip_blur.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")

    l2ms = [r["l2m"] for r in results]
    min_idx = int(np.argmin(l2ms))
    if min_idx == 0:
        verdict = "단조 증가"
    elif min_idx == len(l2ms) - 1:
        verdict = "단조 감소 (예상 밖)"
    else:
        verdict = f"U자 형태 가능성 (최소점: sigma={results[min_idx]['sigma']})"
    print(f"판정: {verdict}")


if __name__ == "__main__":
    main()
