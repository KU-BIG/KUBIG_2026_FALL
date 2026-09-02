"""Step 2: frozen CLIP(openai/clip-vit-base-patch32)으로 이미지·텍스트 임베딩 1회 추출.

GPU가 필요한 유일한 모듈 (CLAUDE.md 규칙). npz 저장 스키마는 IMPLEMENTATION_PLAN.md §3 참고.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.data.dataset import build_stage_a, build_stage_b, load_config, load_dataset

MODEL_NAME = "openai/clip-vit-base-patch32"


def resolve_image_path(filename: str, images_dir: str = "data/raw/SemArt/Images") -> str:
    """images_dir/filename 반환. 파일 존재 여부는 호출부(encode_images)에서 확인."""
    return str(Path(images_dir) / filename)


class _ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[str]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        return Image.open(self.image_paths[idx]).convert("RGB")


def _identity_collate(batch):
    return batch


def load_model(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    # use_safetensors=True: transformers>=5.x는 pytorch_model.bin(torch.load 경로)을
    # torch<2.6에서 CVE-2025-32434 때문에 거부한다. safetensors 로딩은 이 취약점과 무관하므로
    # 명시적으로 강제한다 (가중치는 동일, 직렬화 포맷만 다름).
    model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def encode_images(
    image_paths: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    batch_size: int,
    num_workers: int = 4,
) -> np.ndarray:
    missing = [p for p in image_paths if not Path(p).exists()]
    assert not missing, f"{len(missing)}개 이미지 파일이 없음 (예: {missing[:5]})"

    device = next(model.parameters()).device
    dataset = _ImagePathDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_identity_collate,
    )

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode_images"):
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                # transformers>=5.x: get_image_features가 plain tensor 대신
                # BaseModelOutputWithPooling을 반환함. 프로젝션된 임베딩은 .pooler_output에 있음
                # (값은 구버전 API의 반환 텐서와 동일).
                feats = feats.pooler_output
            feats = F.normalize(feats, p=2, dim=-1)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def encode_texts(
    texts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer

    raw_lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
    assert max(raw_lengths) <= 77, f"truncation would occur: max_len={max(raw_lengths)} > 77"

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="encode_texts"):
            batch = texts[i : i + batch_size]
            # padding=True인 배치는 input_ids와 함께 attention_mask도 반환한다.
            # get_text_features에 dict를 통째로 넘겨야(**inputs) pooled output(EOT 위치 탐색)이
            # padding 토큰에 오염되지 않는다 — input_ids만 따로 넘기면 안 됨.
            inputs = tokenizer(batch, truncation=True, max_length=77, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_text_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                # transformers>=5.x: get_text_features가 plain tensor 대신
                # BaseModelOutputWithPooling을 반환함. 프로젝션된 임베딩은 .pooler_output에 있음
                # (값은 구버전 API의 반환 텐서와 동일).
                feats = feats.pooler_output
            feats = F.normalize(feats, p=2, dim=-1)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    assert cfg["model_name"] == MODEL_NAME, f"config model_name {cfg['model_name']} != {MODEL_NAME}"

    df = load_dataset(cfg["parquet_path"])
    stage_a_df = build_stage_a(df)
    stage_b_df = build_stage_b(df)

    filenames = stage_a_df["filename"].tolist()
    non_jpg = [f for f in filenames if not f.lower().endswith(".jpg")]
    print(f"[extract] {len(filenames)}개 파일 중 .jpg 아닌 확장자: {len(non_jpg)}건")
    if non_jpg:
        print(f"[extract] 예시: {non_jpg[:5]}")

    image_paths = [resolve_image_path(f, cfg["images_dir"]) for f in filenames]

    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] device={device}")

    model, processor = load_model(device)
    projection_dim = int(model.config.projection_dim)
    print(f"[extract] projection_dim(D) = {projection_dim}")

    t0 = time.perf_counter()
    image_emb = encode_images(image_paths, model, processor, cfg["batch_size"], cfg.get("num_workers", 4))
    t1 = time.perf_counter()
    print(f"[extract] image encoding: {t1 - t0:.1f}s for {len(image_paths)} images")

    text_emb_visual_a = encode_texts(stage_a_df["visual_text"].tolist(), model, processor, cfg["batch_size"])
    t2 = time.perf_counter()
    print(f"[extract] visual_a text encoding: {t2 - t1:.1f}s")

    text_emb_visual_b = encode_texts(stage_b_df["visual_text"].tolist(), model, processor, cfg["batch_size"])
    t3 = time.perf_counter()
    print(f"[extract] visual_b text encoding: {t3 - t2:.1f}s")

    text_emb_contextual = encode_texts(stage_a_df["contextual_text"].tolist(), model, processor, cfg["batch_size"])
    t4 = time.perf_counter()
    print(f"[extract] contextual text encoding: {t4 - t3:.1f}s")

    meta = json.dumps({
        "model": MODEL_NAME,
        "projection_dim": projection_dim,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(filenames),
    })

    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        filenames=np.array(filenames),
        image_emb=image_emb,
        text_emb_visual_a=text_emb_visual_a,
        text_emb_visual_b=text_emb_visual_b,
        text_emb_contextual=text_emb_contextual,
        meta=meta,
    )
    print(f"[extract] saved to {out_path}")
    print(f"[extract] total elapsed: {t4 - t0:.1f}s")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/extract.yaml"
    main(config_path)
