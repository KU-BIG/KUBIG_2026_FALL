"""임베딩 추출. PROJECT_v2.md §7.1 — 최종 projection 출력의 L2 정규화 직후(512차원)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_model_cache = {}


def load_model(model_name="ViT-B-16", pretrained="openai", device="cpu"):
    key = (model_name, pretrained, device)
    if key not in _model_cache:
        # OpenAI 체크포인트는 QuickGELU로 학습되었다. open_clip의 기본 ViT-B-16 config는
        # quick_gelu=False라서 명시하지 않으면 활성화 함수가 어긋나 재현이 깨진다.
        force_quick_gelu = pretrained == "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, force_quick_gelu=force_quick_gelu
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        _model_cache[key] = (model, preprocess, tokenizer)
    return _model_cache[key]


def _cache_key(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def encode_images(images, model, preprocess, device="cpu", batch_size=32, cache_tag=None):
    """images: list[PIL.Image]. cache_tag가 있으면 캐시에서 재사용."""
    if cache_tag is not None:
        cache_path = CACHE_DIR / f"{cache_tag}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    embs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([preprocess(im) for im in batch]).to(device)
            feats = F.normalize(model.encode_image(tensors), dim=-1)
            embs.append(feats.cpu().numpy())
    result = np.concatenate(embs, axis=0)

    if cache_tag is not None:
        np.save(CACHE_DIR / f"{cache_tag}.npy", result)
    return result


def encode_texts(texts, model, tokenizer, device="cpu", batch_size=64, cache_tag=None):
    if cache_tag is not None:
        cache_path = CACHE_DIR / f"{cache_tag}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = tokenizer(batch).to(device)
            feats = F.normalize(model.encode_text(tokens), dim=-1)
            embs.append(feats.cpu().numpy())
    result = np.concatenate(embs, axis=0)

    if cache_tag is not None:
        np.save(CACHE_DIR / f"{cache_tag}.npy", result)
    return result
