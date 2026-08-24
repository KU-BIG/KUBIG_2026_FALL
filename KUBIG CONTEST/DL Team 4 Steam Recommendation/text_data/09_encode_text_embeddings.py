# %% [markdown]
# # Encode Steam text embeddings
#
# `games_text_ready.csv`의 `text_for_embedding`을 MiniLM 계열 encoder에 통과시켜
# 게임 1개당 텍스트 임베딩 1개를 만듭니다.
# 긴 텍스트는 tokenizer 단계에서 256 token으로 자릅니다.

# %% 1. Imports
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
INPUT_CSV = BASE_DIR / "games_text_ready.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 256


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-prefix", default="emb_text_minilm")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_CSV)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    texts = df["text_for_embedding"].fillna("").astype(str).tolist()
    app_ids = df["app_id"].tolist()

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = BASE_DIR / output_prefix
    output_npy = output_prefix.with_suffix(".npy")
    output_index_csv = output_prefix.with_suffix(".csv")

    device = pick_device(args.device)
    print("device:", device, flush=True)
    print("model:", MODEL_NAME, flush=True)
    print("rows:", len(texts), flush=True)

    print("loading tokenizer/model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("model loaded", flush=True)

    loader = DataLoader(texts, batch_size=args.batch_size, shuffle=False)
    chunks = []
    for step, batch_texts in enumerate(loader, start=1):
        emb = encode_batch(list(batch_texts), tokenizer, model, device)
        chunks.append(emb)
        if step == 1 or step % 20 == 0:
            done = min(step * args.batch_size, len(texts))
            pct = done / len(texts) * 100
            print(f"encoded {done:,} / {len(texts):,} ({pct:.1f}%)", flush=True)

    matrix = np.vstack(chunks).astype(np.float32)
    np.save(output_npy, matrix)
    pd.DataFrame({"app_id": app_ids, "row": range(len(app_ids))}).to_csv(output_index_csv, index=False)

    print("embedding shape:", matrix.shape, flush=True)
    print("saved:", output_npy, flush=True)
    print("index:", output_index_csv, flush=True)


if __name__ == "__main__":
    main()
