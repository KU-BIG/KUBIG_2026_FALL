"""
CLIP ViT의 최종 풀링 벡터 대신 '패치 토큰'을 직접 써서 이미지-텍스트 유사도를
공간적으로 시각화한다 (MaskCLIP 스타일 dense CLIP trick).

open_clip의 VisionTransformer.forward(output_tokens=True)는 최종 projection 전
패치별 토큰(14x14=196개, ViT-B/16 기준)을 그대로 돌려준다. 여기에 CLS와 같은
projection 행렬을 곱하면 텍스트와 같은 512차원 공유 공간에 패치별로 놓을 수 있다.
텍스트 임베딩과 각 패치의 코사인 유사도를 이미지 위에 히트맵으로 얹는다.

목적: groundedness가 높은/낮은 예시가 실제로 공간적으로 다르게 보이는지 정성적으로 확인.
"""

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "SemArt" / "Images"

EXAMPLES = [
    {
        "file": "17288-selfpo15.jpg",
        "text": "Catalogue numbers: F 269v, JH 1301",
        "ground": 0.004,
        "note": "비접지 최하위 — 카탈로그 번호일 뿐",
    },
    {
        "file": "21882-stillife.jpg",
        "text": "The painting depicts a still-life with grapes, a platter of peaches and a Chinese jar on a stone ledge.",
        "ground": 0.994,
        "note": "접지 높음 — 구체적 물체(포도/복숭아/항아리) 서술",
    },
    {
        "file": "03204-gatto.jpg",
        "text": "This painting represents the Virgin and Child with Saint Joseph and the Infant Baptist. The title refers to the cat (gatto) seen at lower left",
        "ground": 0.998,
        "note": "접지 높음 — 전문(全文), '왼쪽 아래 고양이' 언급",
    },
    {
        "file": "03204-gatto.jpg",
        "text": "a cat",
        "ground": None,
        "note": "같은 그림, 단어 하나만('a cat') — 실제로 왼쪽 아래를 짚는지 확인",
    },
]


def get_patch_similarity(model, preprocess, tokenizer, img: Image.Image, text: str):
    model.visual.output_tokens = True
    x = preprocess(img).unsqueeze(0)
    tokens_in = tokenizer([text])
    with torch.no_grad():
        _, patch_tokens = model.visual(x)  # (1, 196, 768)
        txt_emb = F.normalize(model.encode_text(tokens_in), dim=-1)  # (1, 512)
        patch_proj = patch_tokens @ model.visual.proj  # (1, 196, 512)
        patch_proj = F.normalize(patch_proj, dim=-1)
        sim = (patch_proj[0] @ txt_emb[0])  # (196,)
    model.visual.output_tokens = False
    grid = int(sim.shape[0] ** 0.5)  # 14
    sim_map = sim.reshape(grid, grid).numpy()
    return sim_map


def render_overlay(img: Image.Image, sim_map: np.ndarray) -> str:
    """224x224로 리사이즈한 원본 위에 히트맵을 겹쳐 base64 PNG로 반환."""
    img224 = img.resize((224, 224))
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img224)
    axes[0].axis("off")
    axes[0].set_title("original", fontsize=9)

    axes[1].imshow(img224)
    im = axes[1].imshow(sim_map, cmap="inferno", alpha=0.55, extent=(0, 224, 224, 0),
                         interpolation="bicubic")
    axes[1].axis("off")
    axes[1].set_title("patch similarity", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def main():
    print("loading OpenCLIP ViT-B-16 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", force_quick_gelu=True
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    results = []
    for ex in EXAMPLES:
        print(f"processing {ex['file']} :: {ex['text'][:50]}...")
        img = Image.open(IMAGES_DIR / ex["file"]).convert("RGB")
        sim_map = get_patch_similarity(model, preprocess, tokenizer, img, ex["text"])
        b64 = render_overlay(img, sim_map)
        results.append({**ex, "sim_min": float(sim_map.min()), "sim_max": float(sim_map.max()),
                         "sim_std": float(sim_map.std()), "b64": b64})
        print(f"  sim range [{sim_map.min():.3f}, {sim_map.max():.3f}] std={sim_map.std():.4f}")

    out_dir = Path("/tmp/claude-1000/-teamspace-studios-this-studio-cv-session-1/8bdaf5ca-ae9b-42ec-9531-62abf6aaadc5/scratchpad")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        p = out_dir / f"heatmap_{i}.png"
        p.write_bytes(base64.b64decode(r["b64"]))
        print("saved", p)

    import json
    json_out = [{k: v for k, v in r.items() if k != "b64"} for r in results]
    (out_dir / "heatmap_meta.json").write_text(json.dumps(json_out, indent=2))
    print("done")


if __name__ == "__main__":
    main()
