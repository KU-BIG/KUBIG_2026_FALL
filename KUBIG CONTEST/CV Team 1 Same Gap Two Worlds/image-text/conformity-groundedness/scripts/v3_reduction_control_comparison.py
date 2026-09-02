"""
groundedness 기반 문장 선택이 "정말 관련성 때문에" 좋아진 건지, 아니면 "그냥 짧아지기만 해도"
좋아지는 건지 가려낸다. 같은 개수의 문장을 세 가지 다른 기준으로 골라 비교한다:

  - groundedness : 자기 이미지와 코사인 유사도가 높은 문장 절반 (지금까지 한 것)
  - first_half   : 그냥 앞쪽 절반 문장 (위치 기준 — 단순 절단 대조군)
  - random_half  : 무작위 절반 문장 (완전 무작위 대조군)

셋 다 "몇 문장을 남기는지"는 완전히 동일하게 맞춘다 — 차이는 오직 "어떤 문장을 남기는지"뿐.
"""

import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import l2m, rmg  # noqa: E402

DATA_DIR = ROOT / "data" / "SemArt"
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


def summarize(txt_raw_t, txt_n_np, img_n_np, rows, label):
    conf = conformity_true(txt_raw_t).numpy()
    sim = txt_n_np @ img_n_np.T
    ground = rank_groundedness(sim)
    m_l2m = l2m(img_n_np, txt_n_np)
    m_rmg = rmg(img_n_np, txt_n_np)
    ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(len(rows))])
    r1 = float((ranks < 1).mean())
    r5 = float((ranks < 5).mean())
    print(f"  [{label:<12}] conformity={conf.mean():.4f}  groundedness={ground.mean():.4f}  "
          f"R@1={r1:.3f}  R@5={r5:.3f}  L2M={m_l2m:.4f}  RMG={m_rmg:.4f}")
    return {"conformity_mean": float(conf.mean()), "groundedness_mean": float(ground.mean()),
            "R@1": r1, "R@5": r5, "l2m": m_l2m, "rmg": m_rmg}


def main():
    print(f"[1/6] loading {N_SAMPLES} SemArt pairs + cached raw image embeddings ...")
    rows = load_pairs(DATA_DIR / "semart_val.csv", N_SAMPLES, SEED)
    img_raw = torch.from_numpy(np.load(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_img_raw.npy"))
    img_n = torch.nn.functional.normalize(img_raw, dim=-1).numpy()

    print("[2/6] loading Long-CLIP-B ...")
    model = AutoModel.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("creative-graphic-design/LongCLIP-B", trust_remote_code=True)
    model.eval()

    print("[3/6] splitting into sentences + scoring each by similarity to own image ...")
    all_sent_lists = [split_sentences(r["DESCRIPTION"]) for r in rows]
    flat_sents, owner = [], []
    for i, sents in enumerate(all_sent_lists):
        flat_sents.extend(sents)
        owner.extend([i] * len(sents))
    owner = np.array(owner)

    sent_cache = CACHE_DIR / f"v3_sent_raw_n{N_SAMPLES}_seed{SEED}.npy"
    if sent_cache.exists():
        sent_raw = torch.from_numpy(np.load(sent_cache))
    else:
        sent_raw = encode_texts(flat_sents, model, processor)
        np.save(sent_cache, sent_raw.numpy())
    sent_n = torch.nn.functional.normalize(sent_raw, dim=-1)
    sent_score = (sent_n * torch.from_numpy(img_n)[torch.from_numpy(owner)]).sum(-1).numpy()

    print("[4/6] building 3 variants (same #sentences kept, different selection rule) ...")
    variants = {"groundedness": [], "first_half": [], "random_half": []}
    word_counts = {k: [] for k in variants}
    rng = random.Random(SEED)

    for i, sents in enumerate(all_sent_lists):
        idx = np.where(owner == i)[0]
        k = max(1, len(idx) // 2)

        by_score = idx[np.argsort(-sent_score[idx])[:k]]
        gr_text = " ".join(flat_sents[j] for j in sorted(by_score.tolist()))

        fh_idx = idx[:k]
        fh_text = " ".join(flat_sents[j] for j in fh_idx)

        rnd_pick = rng.sample(list(idx), k)
        rnd_text = " ".join(flat_sents[j] for j in sorted(rnd_pick))

        variants["groundedness"].append(gr_text)
        variants["first_half"].append(fh_text)
        variants["random_half"].append(rnd_text)
        word_counts["groundedness"].append(len(gr_text.split()))
        word_counts["first_half"].append(len(fh_text.split()))
        word_counts["random_half"].append(len(rnd_text.split()))

    for k in variants:
        print(f"  {k:<12} avg words = {np.mean(word_counts[k]):.1f}")

    print("[5/6] encoding all 3 variants + original, computing metrics ...")
    txt_raw_orig = torch.from_numpy(np.load(CACHE_DIR / f"v3_semart_n{N_SAMPLES}_seed{SEED}_longclip_txt_raw.npy"))
    txt_n_orig = torch.nn.functional.normalize(txt_raw_orig, dim=-1).numpy()

    results = {"original": summarize(txt_raw_orig, txt_n_orig, img_n, rows, "original")}
    for name, texts in variants.items():
        raw = encode_texts(texts, model, processor)
        n = torch.nn.functional.normalize(raw, dim=-1).numpy()
        results[name] = summarize(raw, n, img_n, rows, name)
        results[name]["avg_words"] = float(np.mean(word_counts[name]))

    print("[6/6] saving ...")
    out_path = ROOT / "results" / "v3_reduction_control_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out_path}")

    print("\n=== R@1 비교 ===")
    for name in ["original", "first_half", "random_half", "groundedness"]:
        print(f"  {name:<12} R@1={results[name]['R@1']:.3f}  L2M={results[name]['l2m']:.4f}")


if __name__ == "__main__":
    main()
