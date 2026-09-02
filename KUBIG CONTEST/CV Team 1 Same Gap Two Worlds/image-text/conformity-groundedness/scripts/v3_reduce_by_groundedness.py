"""
결론 활용 — 문장을 groundedness(자기 짝 이미지와의 유사도) 기준으로 골라 텍스트를 줄이고,
줄이기 전/후로 conformity, groundedness, L2M/RMG가 어떻게 달라지는지 비교한다.

절차:
  1. 각 SemArt 설명문을 문장 단위로 쪼갠다.
  2. 문장별로 "자기 이미지와의 코사인 유사도"를 groundedness 대리 점수로 쓴다.
  3. 문서마다 점수 상위 절반(최소 1문장)만 남겨 "줄인 텍스트"를 만든다 — conformity가 아니라
     groundedness로 고르는 이유는 Phase 2에서 conformity가 groundedness와 무관하다는 게 이미
     확인됐기 때문(§v3 Phase2). conformity는 결과 비교 항목으로만 같이 본다.
  4. 줄인 텍스트를 Long-CLIP으로 다시 인코딩해서 conformity/groundedness/L2M/RMG를 원본과 비교.
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
CACHE_DIR = ROOT / "results" / "embeddings"
N_SAMPLES = 500
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
    parts = re.split(r"(?<=[.!?])\s*(?=[A-Z])", text)
    parts = [p.strip() for p in parts if len(p.strip().split()) >= 3]
    return parts if parts else [text]


def conformity_true(embs: torch.Tensor) -> torch.Tensor:
    E = torch.nn.functional.normalize(embs, dim=-1)
    S = E @ E.T
    n = S.shape[0]
    S.fill_diagonal_(0)
    return S.sum(1) / (n - 1)


def rank_groundedness(sim: np.ndarray) -> np.ndarray:
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    rank0 = np.array([np.where(order[i] == i)[0][0] for i in range(n)])
    return 1.0 - rank0 / (n - 1)


def encode_texts(texts, model, processor, batch_size=16):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            inputs = processor(text=batch, return_tensors="pt", max_length=MAX_LEN,
                                padding="max_length", truncation=True)
            out.append(model.get_text_features(**inputs))
    return torch.cat(out, dim=0)


def main():
    print(f"[1/7] loading {N_SAMPLES} SemArt pairs + cached raw image embeddings ...")
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)
    img_raw = torch.from_numpy(np.load(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_img_raw.npy"))
    img_n = torch.nn.functional.normalize(img_raw, dim=-1)

    print("[2/7] loading Long-CLIP-B ...")
    model = AutoModel.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    model.eval()

    print("[3/7] splitting into sentences + scoring each by similarity to own image ...")
    all_sent_lists = [split_sentences(r["DESCRIPTION"]) for r in rows]
    flat_sents, owner = [], []
    for i, sents in enumerate(all_sent_lists):
        flat_sents.extend(sents)
        owner.extend([i] * len(sents))
    owner = np.array(owner)
    print(f"  total sentences: {len(flat_sents)} (avg {len(flat_sents)/len(rows):.1f}/doc)")

    sent_raw = encode_texts(flat_sents, model, processor)
    sent_n = torch.nn.functional.normalize(sent_raw, dim=-1)
    sent_score = (sent_n * img_n[torch.from_numpy(owner)]).sum(-1).numpy()  # 자기 이미지와의 코사인

    print("[4/7] keeping top half (min 1) of sentences per doc by score, building reduced text ...")
    reduced_texts = []
    orig_word_counts, reduced_word_counts = [], []
    for i, sents in enumerate(all_sent_lists):
        idx = np.where(owner == i)[0]
        scores = sent_score[idx]
        k = max(1, len(idx) // 2)
        keep = idx[np.argsort(-scores)[:k]]
        keep_sorted = sorted(keep.tolist())  # 원래 순서 유지
        reduced = " ".join(flat_sents[j] for j in keep_sorted)
        reduced_texts.append(reduced)
        orig_word_counts.append(len(rows[i]["DESCRIPTION"].split()))
        reduced_word_counts.append(len(reduced.split()))

    print(f"  avg words: {np.mean(orig_word_counts):.1f} -> {np.mean(reduced_word_counts):.1f} "
          f"({100*(1-np.mean(reduced_word_counts)/np.mean(orig_word_counts)):.0f}% 감소)")

    print("[5/7] encoding reduced texts (full pass, fresh) ...")
    red_raw = encode_texts(reduced_texts, model, processor)
    red_n = torch.nn.functional.normalize(red_raw, dim=-1)

    print("[6/7] computing conformity / groundedness / L2M / RMG, before vs after ...")

    def summarize(txt_raw_t, txt_n_np, label):
        conf = conformity_true(txt_raw_t).numpy()
        sim = txt_n_np @ img_n.numpy().T
        ground = rank_groundedness(sim)
        m_l2m = l2m(img_n.numpy(), txt_n_np)
        m_rmg = rmg(img_n.numpy(), txt_n_np)
        ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(len(rows))])
        r1 = float((ranks < 1).mean())
        print(f"  [{label}] conformity mean={conf.mean():.4f}  groundedness mean={ground.mean():.4f} "
              f"median={np.median(ground):.4f}  R@1={r1:.3f}  L2M={m_l2m:.4f}  RMG={m_rmg:.4f}")
        return {"conformity_mean": float(conf.mean()), "groundedness_mean": float(ground.mean()),
                "groundedness_median": float(np.median(ground)), "R@1": r1,
                "l2m": m_l2m, "rmg": m_rmg}

    # 원본 텍스트도 같은 500쌍, 같은 방식으로 다시 계산(캐시된 raw 재사용)
    txt_raw_orig = torch.from_numpy(np.load(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_txt_raw.npy"))
    txt_n_orig = torch.nn.functional.normalize(txt_raw_orig, dim=-1).numpy()

    before = summarize(txt_raw_orig, txt_n_orig, "원본 전체 텍스트")
    after = summarize(red_raw, red_n.numpy(), "groundedness로 줄인 텍스트")

    print("\n[7/7] saving ...")
    out = {
        "n_samples": len(rows),
        "avg_words_before": float(np.mean(orig_word_counts)),
        "avg_words_after": float(np.mean(reduced_word_counts)),
        "before": before,
        "after": after,
    }
    out_path = ROOT / "results" / "v3_reduce_by_groundedness.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")

    print("\n샘플 (원본 -> 축소):")
    for i in random.Random(2).sample(range(len(rows)), 3):
        print(f"  [{rows[i]['IMAGE_FILE']}]")
        print(f"    원본({orig_word_counts[i]}단어): {rows[i]['DESCRIPTION'][:120]}...")
        print(f"    축소({reduced_word_counts[i]}단어): {reduced_texts[i][:120]}")


if __name__ == "__main__":
    main()
