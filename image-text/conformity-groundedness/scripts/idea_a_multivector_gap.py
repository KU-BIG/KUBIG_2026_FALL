"""
아이디어 A — 긴 텍스트의 gap 팽창이 "진짜 정보 불균형"이 아니라
"여러 화제를 벡터 하나로 뭉갠 풀링의 부작용"일 수도 있다는 가설 검증.

같은 이미지에 대해:
  - doc  : SemArt 설명문 전체를 하나의 벡터로 (지금까지 하던 방식)
  - best : 설명문을 문장 단위로 쪼갠 뒤, 그 이미지와 가장 유사한 문장 하나만 사용
           (ColBERT류 multi-vector / late-interaction 검색의 "max-sim" 방식 차용)

doc 기준 gap이 best 기준 gap보다 훨씬 크면, "텍스트가 길어서 정보 불균형이 커졌다"가 아니라
"단일 벡터 풀링이 그라운딩된 부분을 희석시켰다"는 뜻이 된다.

소규모 파일럿(N_SAMPLES 작게), Long-CLIP-B 사용 (문서 전체가 77토큰을 넘으므로).
"""

import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import l2m, rmg  # noqa: E402

DATA_DIR = ROOT / "data" / "SemArt"
IMAGES_DIR = DATA_DIR / "Images"
N_SAMPLES = 50  # 이전 Long-CLIP 파일럿과 동일 표본(seed 동일) — 직접 비교 가능
SEED = 0
MAX_LEN = 248


def load_pairs(csv_path, n, seed):
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def split_sentences(text):
    text = text.strip()
    # 공백 있는 경우와, SemArt 데이터에 흔한 "...Bruges.The Agony..." 처럼 공백 없이 붙은 경우 둘 다 처리
    parts = re.split(r"(?<=[.!?])\s*(?=[A-Z])", text)
    parts = [p.strip() for p in parts if len(p.strip().split()) >= 3]
    return parts if parts else [text]


def encode_images(images, model, processor):
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.numpy()


def encode_texts(texts, model, processor, batch_size=32):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = processor(
                text=batch, return_tensors="pt", max_length=MAX_LEN,
                padding="max_length", truncation=True,
            )
            feats = model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embs.append(feats.numpy())
    return np.concatenate(embs, axis=0)


def main():
    print(f"[1/5] loading {N_SAMPLES} SemArt val pairs (same seed as previous Long-CLIP pilot) ...")
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)
    images = [Image.open(IMAGES_DIR / r["IMAGE_FILE"]).convert("RGB") for r in rows]
    docs = [r["DESCRIPTION"] for r in rows]
    sent_lists = [split_sentences(d) for d in docs]
    n_sents = [len(s) for s in sent_lists]
    print(f"  sentences per doc: mean={np.mean(n_sents):.1f} min={min(n_sents)} max={max(n_sents)}")

    print("[2/5] loading Long-CLIP-B ...")
    t0 = time.time()
    model = AutoModel.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("[3/5] encoding images (baseline, no degradation) ...")
    img_emb = encode_images(images, model, processor)

    print("[4/5] encoding full-document text ...")
    doc_emb = encode_texts(docs, model, processor)

    print("[5/5] encoding per-sentence text + picking max-sim sentence per image ...")
    best_emb = np.zeros_like(img_emb)
    best_sim = np.zeros(len(rows))
    best_sent_text = []
    for i, sents in enumerate(sent_lists):
        s_emb = encode_texts(sents, model, processor)
        sims = s_emb @ img_emb[i]
        j = int(np.argmax(sims))
        best_emb[i] = s_emb[j]
        best_sim[i] = sims[j]
        best_sent_text.append(sents[j])

    doc_l2m, doc_rmg = l2m(img_emb, doc_emb), rmg(img_emb, doc_emb)
    best_l2m, best_rmg = l2m(img_emb, best_emb), rmg(img_emb, best_emb)

    print("\n=== 결과 ===")
    print(f"  전체 문서(doc)     L2M={doc_l2m:.4f}  RMG={doc_rmg:.4f}")
    print(f"  최선 문장(best)    L2M={best_l2m:.4f}  RMG={best_rmg:.4f}")
    print(f"  ΔL2M (doc - best) = {doc_l2m - best_l2m:+.4f}")
    print(f"  평균 최선-문장 유사도: {best_sim.mean():.4f}")

    print("\n샘플 (최선 문장 3개):")
    for i in random.sample(range(len(rows)), 3):
        print(f"  [{rows[i]['IMAGE_FILE']}] sim={best_sim[i]:.3f}: {best_sent_text[i][:100]}")

    out = {
        "n_samples": len(rows),
        "doc": {"l2m": doc_l2m, "rmg": doc_rmg},
        "best_sentence": {"l2m": best_l2m, "rmg": best_rmg},
        "mean_sentences_per_doc": float(np.mean(n_sents)),
        "mean_best_sim": float(best_sim.mean()),
    }
    out_path = ROOT / "results" / "idea_a_multivector_gap.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {out_path}")

    if doc_l2m - best_l2m > 0.02:
        print("\n→ 문서 전체 풀링이 gap을 상당히 부풀림 (가설 지지: 풀링 아티팩트 가능성)")
    else:
        print("\n→ 큰 차이 없음 (가설 기각: 긴 텍스트 자체의 정보 불균형이 원인일 가능성 유지)")


if __name__ == "__main__":
    main()
