import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import build_stage_a, build_stage_b, load_dataset
from src.embeddings.extract import encode_images, encode_texts, load_model, resolve_image_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "semart_v1_modality_gap_dataset.parquet"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "SemArt" / "Images"
TIMING_PATH = PROJECT_ROOT / "outputs" / "results" / "step2_smoke_timing.json"

SMOKE_N = 25
BATCH_SIZE = 8

pytestmark = pytest.mark.skipif(
    not IMAGES_DIR.exists(),
    reason="SemArt 이미지가 로컬에 없음 — Lightning Studio에서 실행할 것",
)


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_and_processor(device):
    return load_model(device)


@pytest.fixture(scope="module")
def df():
    return load_dataset(str(PARQUET_PATH)).head(SMOKE_N)


@pytest.fixture(scope="module")
def stage_a_df(df):
    return build_stage_a(df)


@pytest.fixture(scope="module")
def stage_b_df(df):
    return build_stage_b(df)


@pytest.fixture(scope="module")
def image_paths(stage_a_df):
    return [resolve_image_path(f, str(IMAGES_DIR)) for f in stage_a_df["filename"].tolist()]


def _assert_valid_embeddings(emb: np.ndarray, n_expected: int) -> None:
    assert emb.shape[0] == n_expected
    assert emb.ndim == 2
    assert not np.isnan(emb).any()
    assert not np.isinf(emb).any()
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


@pytest.fixture(scope="module")
def encoded(model_and_processor, image_paths, stage_a_df, stage_b_df, device):
    model, processor = model_and_processor

    t0 = time.perf_counter()
    image_emb = encode_images(image_paths, model, processor, BATCH_SIZE)
    t1 = time.perf_counter()

    text_emb_visual_a = encode_texts(stage_a_df["visual_text"].tolist(), model, processor, BATCH_SIZE)
    text_emb_visual_b = encode_texts(stage_b_df["visual_text"].tolist(), model, processor, BATCH_SIZE)
    text_emb_contextual = encode_texts(stage_a_df["contextual_text"].tolist(), model, processor, BATCH_SIZE)
    t2 = time.perf_counter()

    summary = {
        "smoke_n": SMOKE_N,
        "batch_size": BATCH_SIZE,
        "device": device,
        "elapsed_seconds_image": t1 - t0,
        "elapsed_seconds_text": t2 - t1,
    }
    TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMING_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "image_emb": image_emb,
        "text_emb_visual_a": text_emb_visual_a,
        "text_emb_visual_b": text_emb_visual_b,
        "text_emb_contextual": text_emb_contextual,
    }


def test_encode_images_smoke(encoded):
    _assert_valid_embeddings(encoded["image_emb"], SMOKE_N)


def test_encode_texts_visual_a_smoke(encoded):
    _assert_valid_embeddings(encoded["text_emb_visual_a"], SMOKE_N)


def test_encode_texts_visual_b_smoke(encoded):
    _assert_valid_embeddings(encoded["text_emb_visual_b"], SMOKE_N)


def test_encode_texts_contextual_smoke(encoded):
    _assert_valid_embeddings(encoded["text_emb_contextual"], SMOKE_N)


def test_projection_dim_matches_embedding_width(model_and_processor, encoded):
    model, _ = model_and_processor
    assert encoded["image_emb"].shape[1] == model.config.projection_dim
