"""
CLIPSeg(CIDAS/clipseg-rd64-refined) — 텍스트 조건부 세그멘테이션 전용 모델로
앞서 만든 ViT 패치 히트맵(14x14, 거친 해상도)을 훨씬 정밀하게(352x352) 재현.

특히 "a cat"이 그림 왼쪽 아래 작은 동물을 정말로 찾아내는지 재검증한다.
"""

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "SemArt" / "Images"
OUT_DIR = Path("/tmp/claude-1000/-teamspace-studios-this-studio-cv-session-1/8bdaf5ca-ae9b-42ec-9531-62abf6aaadc5/scratchpad")

EXAMPLES = [
    {"file": "17288-selfpo15.jpg", "prompt": "catalogue numbers", "note": "비접지 텍스트를 짧은 구로 — 찾을 대상이 없어야 함"},
    {"file": "21882-stillife.jpg", "prompt": "grapes", "note": "포도만 콕 집어서"},
    {"file": "21882-stillife.jpg", "prompt": "a Chinese jar", "note": "항아리만 콕 집어서"},
    {"file": "03204-gatto.jpg", "prompt": "a cat", "note": "ViT 패치 히트맵에서 실패했던 그 테스트, 재도전"},
    {"file": "03204-gatto.jpg", "prompt": "the Virgin and Child", "note": "인물 주제"},
]


def render_overlay(img: Image.Image, prob_map: np.ndarray, prompt: str) -> str:
    img_disp = img.resize((352, 352))
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img_disp)
    axes[0].axis("off")
    axes[0].set_title("original", fontsize=9)

    axes[1].imshow(img_disp)
    axes[1].imshow(prob_map, cmap="inferno", alpha=0.6, extent=(0, 352, 352, 0))
    axes[1].axis("off")
    axes[1].set_title(f'"{prompt}"', fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def main():
    print("loading CLIPSeg ...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.eval()

    results = []
    for ex in EXAMPLES:
        print(f"  {ex['file']} :: '{ex['prompt']}'")
        img = Image.open(IMAGES_DIR / ex["file"]).convert("RGB")
        inputs = processor(text=[ex["prompt"]], images=[img], return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, interpolate_pos_encoding=True)
        prob = torch.sigmoid(out.logits[0]).numpy()
        b64 = render_overlay(img, prob, ex["prompt"])
        results.append({**ex, "prob_max": float(prob.max()), "prob_mean": float(prob.mean()), "b64": b64})

    for i, r in enumerate(results):
        (OUT_DIR / f"clipseg_{i}.png").write_bytes(base64.b64decode(r["b64"]))

    import json
    json_out = [{k: v for k, v in r.items() if k != "b64"} for r in results]
    (OUT_DIR / "clipseg_meta.json").write_text(json.dumps(json_out, indent=2))
    print("done")


if __name__ == "__main__":
    main()
