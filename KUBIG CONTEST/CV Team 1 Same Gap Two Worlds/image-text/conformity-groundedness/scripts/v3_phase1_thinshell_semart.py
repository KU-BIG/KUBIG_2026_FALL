"""
PROJECT_v3.md Phase 1 — SemArt에서 thin shell / conformity 근사식이 성립하는가.

Double-Ellipsoid(Levi & Gilboa, ICML 2025)의 핵심 추정 정리:
    conformity_est(v) = a * cos(mean, v) + b  가  conformity_true(v) (실제 leave-one-out 평균 코사인)를
    잘 근사한다 (COCO에서 피어슨 0.9998).

SemArt는 텍스트가 이질적(전기+시각서술+해석 혼합)이라 이 근사가 깨질 수 있다는 게 v3의 문제의식.
Long-CLIP을 써서 텍스트를 자르지 않고 검증한다. §6.1에 따라 정규화 "전" 임베딩을 사용한다.
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

DATA_DIR = ROOT / "data" / "SemArt"
IMAGES_DIR = DATA_DIR / "Images"
N_SAMPLES = 500
SEED = 0
MAX_LEN = 248


def load_pairs(csv_path, n, seed):
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def conformity_true(embs: torch.Tensor) -> torch.Tensor:
    """정의 그대로: leave-one-out 평균 코사인 유사도. O(N^2), 검증용."""
    E = torch.nn.functional.normalize(embs, dim=-1)
    S = E @ E.T
    n = S.shape[0]
    S.fill_diagonal_(0)
    return S.sum(1) / (n - 1)


def conformity_est_raw(embs: torch.Tensor):
    """추정식의 원재료: cos(mean, v). a,b는 이후 선형회귀로 구한다."""
    m = embs.mean(0, keepdim=True)
    return torch.nn.functional.cosine_similarity(embs, m, dim=-1)


def fit_ab(x: np.ndarray, y: np.ndarray):
    """y ~= a*x + b 선형회귀."""
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def encode_raw(items, encode_fn, batch_size=16):
    """정규화 전(raw) 임베딩. encode_fn(batch) -> tensor(B, D) unnormalized."""
    out = []
    for i in range(0, len(items), batch_size):
        out.append(encode_fn(items[i : i + batch_size]))
    return torch.cat(out, dim=0)


def main():
    print(f"[1/6] loading {N_SAMPLES} SemArt val pairs (full DESCRIPTION) ...")
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)
    images = [Image.open(IMAGES_DIR / r["IMAGE_FILE"]).convert("RGB") for r in rows]
    texts = [r["DESCRIPTION"] for r in rows]

    print("[2/6] loading Long-CLIP-B ...")
    t0 = time.time()
    model = AutoModel.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    def enc_img(batch):
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt")
            return model.get_image_features(**inputs)

    def enc_txt(batch):
        with torch.no_grad():
            inputs = processor(
                text=batch, return_tensors="pt", max_length=MAX_LEN,
                padding="max_length", truncation=True,
            )
            return model.get_text_features(**inputs)

    print("[3/6] encoding RAW (pre-normalization) image + text embeddings ...")
    img_raw = encode_raw(images, enc_img, batch_size=16)
    txt_raw = encode_raw(texts, enc_txt, batch_size=16)
    print(f"  img_raw {tuple(img_raw.shape)}  txt_raw {tuple(txt_raw.shape)}")

    print("[4/6] thin shell check — norm distribution (원점 근처에 질량이 없는가) ...")
    norms = {}
    for name, raw in [("image", img_raw), ("text", txt_raw)]:
        n = raw.norm(dim=-1).numpy()
        norms[name] = {
            "mean": float(n.mean()), "std": float(n.std()),
            "min": float(n.min()), "max": float(n.max()),
            "cv": float(n.std() / n.mean()),  # 변동계수 — 작을수록 thin shell(반경이 일정)
        }
        print(f"  {name:>5}  mean={n.mean():.3f} std={n.std():.3f} min={n.min():.3f} max={n.max():.3f} "
              f"cv={n.std()/n.mean():.4f}")

    print("[5/6] conformity: true vs estimated, Pearson correlation ...")
    conformity_results = {}
    for name, raw in [("image", img_raw), ("text", txt_raw)]:
        c_true = conformity_true(raw).numpy()
        cos_to_mean = conformity_est_raw(raw).numpy()
        a, b = fit_ab(cos_to_mean, c_true)
        c_est = a * cos_to_mean + b
        pearson = float(np.corrcoef(c_true, c_est)[0, 1])
        conformity_results[name] = {"a": a, "b": b, "pearson": pearson}
        print(f"  {name:>5}  a={a:.4f} b={b:.4f}  Pearson(true, est)={pearson:.4f}  "
              f"(COCO 기준 0.9998)")

    print("[6/6] saving ...")
    out = {
        "n_samples": len(rows),
        "norms": norms,
        "conformity": conformity_results,
    }
    out_path = ROOT / "results" / "v3_phase1_thinshell_semart.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")

    print("\n=== 판정 (§8 Phase 1 기준표) ===")
    for name in ["image", "text"]:
        p = conformity_results[name]["pearson"]
        if p >= 0.99:
            verdict = "thin shell 성립 — 추정식 사용 가능"
        elif p >= 0.9:
            verdict = "약화됨 — 실제 conformity 쓰되 한계 명시"
        else:
            verdict = "가정 붕괴 — 그 자체가 결과, 설계 재논의 필요"
        print(f"  {name}: Pearson={p:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
