"""
PROJECT_v3.md §6.3 — Groundedness(접지성): 절대 코사인 유사도 대신 "순위"로 측정한다.

groundedness(i) = 텍스트 i가 자기 짝 이미지를 갤러리 전체에서 몇 등으로 찾아내는가
                  (1 = 완벽히 특정, 0 = 전혀 못 찾음)

양방향(text->image, image->text) 모두 계산. Phase 1과 같은 표본(N_SAMPLES, SEED)을 써서
나중에 Phase 2(conformity vs groundedness 2차원 산점도)에서 바로 이어붙일 수 있게 한다.
정규화 전 임베딩도 .npy로 캐싱해서 재인코딩 없이 재사용 가능하게 한다.
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
CACHE_DIR = ROOT / "results" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 500  # Phase 1과 동일 표본
SEED = 0
MAX_LEN = 248

IMG_RAW_TAG = f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_img_raw"
TXT_RAW_TAG = f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_txt_raw"


def load_pairs(csv_path, n, seed):
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def encode_raw(items, encode_fn, batch_size=16):
    out = []
    for i in range(0, len(items), batch_size):
        out.append(encode_fn(items[i : i + batch_size]))
    return torch.cat(out, dim=0)


def normalized_rank_groundedness(sim_matrix: np.ndarray) -> np.ndarray:
    """sim_matrix[i, j] = query i와 candidate j의 유사도. 정답은 대각선(i==j).
    반환: 각 query i에 대해 1=완벽 매치, 0=최하위인 [0,1] 정규화 순위."""
    n = sim_matrix.shape[0]
    # 각 행에서 값 내림차순으로 candidate를 정렬했을 때, 정답(j=i)이 몇 번째(0-indexed)인지
    order = np.argsort(-sim_matrix, axis=1)  # (N, N), 각 행: 유사도 높은 순 candidate index
    rank0 = np.array([np.where(order[i] == i)[0][0] for i in range(n)])  # 0 = 1등
    return 1.0 - rank0 / (n - 1)


def main():
    print(f"[1/6] loading {N_SAMPLES} SemArt val pairs (same seed as Phase 1) ...")
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)
    images = [Image.open(IMAGES_DIR / r["IMAGE_FILE"]).convert("RGB") for r in rows]
    texts = [r["DESCRIPTION"] for r in rows]

    img_cache = CACHE_DIR / f"{IMG_RAW_TAG}.npy"
    txt_cache = CACHE_DIR / f"{TXT_RAW_TAG}.npy"

    if img_cache.exists() and txt_cache.exists():
        print("[2-3/6] cache hit — reusing raw embeddings from Phase 1 run")
        img_raw = torch.from_numpy(np.load(img_cache))
        txt_raw = torch.from_numpy(np.load(txt_cache))
    else:
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

        print("[3/6] encoding RAW image + text embeddings ...")
        img_raw = encode_raw(images, enc_img, batch_size=16)
        txt_raw = encode_raw(texts, enc_txt, batch_size=16)
        np.save(img_cache, img_raw.numpy())
        np.save(txt_cache, txt_raw.numpy())
        print(f"  cached -> {img_cache.name}, {txt_cache.name}")

    print("[4/6] normalizing (정규화 후 공간 — retrieval/groundedness용, §6.1) ...")
    img_emb = torch.nn.functional.normalize(img_raw, dim=-1).numpy()
    txt_emb = torch.nn.functional.normalize(txt_raw, dim=-1).numpy()

    print("[5/6] computing bidirectional rank-based groundedness (gallery size = "
          f"{N_SAMPLES}) ...")
    sim = txt_emb @ img_emb.T  # sim[i, j] = text_i vs image_j
    t2i_ground = normalized_rank_groundedness(sim)          # 텍스트 i가 이미지 i를 찾는 능력
    i2t_ground = normalized_rank_groundedness(sim.T)        # 이미지 i가 텍스트 i를 찾는 능력

    print(f"  text->image groundedness: mean={t2i_ground.mean():.4f} median={np.median(t2i_ground):.4f}")
    print(f"  image->text groundedness: mean={i2t_ground.mean():.4f} median={np.median(i2t_ground):.4f}")
    corr = float(np.corrcoef(t2i_ground, i2t_ground)[0, 1])
    print(f"  correlation(t2i, i2t) per item = {corr:.4f}")

    # R@1 등 표준 retrieval도 같이 보고 (§7.4 스타일 교차검증)
    ranks_t2i = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(len(rows))])
    ranks_i2t = np.array([np.where(np.argsort(-sim.T[i]) == i)[0][0] for i in range(len(rows))])
    for name, ranks in [("text->image", ranks_t2i), ("image->text", ranks_i2t)]:
        r1 = float((ranks < 1).mean())
        r5 = float((ranks < 5).mean())
        r10 = float((ranks < 10).mean())
        print(f"  {name}  R@1={r1:.3f}  R@5={r5:.3f}  R@10={r10:.3f}  (gallery={len(rows)})")

    print("[6/6] saving ...")
    np.save(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_t2i_groundedness.npy", t2i_ground)
    np.save(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_i2t_groundedness.npy", i2t_ground)

    out = {
        "n_samples": len(rows),
        "gallery_size": len(rows),
        "t2i_groundedness": {"mean": float(t2i_ground.mean()), "median": float(np.median(t2i_ground))},
        "i2t_groundedness": {"mean": float(i2t_ground.mean()), "median": float(np.median(i2t_ground))},
        "t2i_i2t_correlation": corr,
        "retrieval": {
            "text->image": {"R@1": float((ranks_t2i < 1).mean()), "R@5": float((ranks_t2i < 5).mean()), "R@10": float((ranks_t2i < 10).mean())},
            "image->text": {"R@1": float((ranks_i2t < 1).mean()), "R@5": float((ranks_i2t < 5).mean()), "R@10": float((ranks_i2t < 10).mean())},
        },
    }
    out_path = ROOT / "results" / "v3_groundedness_semart.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
