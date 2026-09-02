"""
Phase 1 보강 실험 — "눈먼" 열화(블러/다운샘플/crop) 대신, 캡션이 실제로 설명하는 대상을
정확히 가리는 타겟 마스킹으로 정보 불균형을 만든다.

COCO instances_val2017.json의 instance segmentation을 이용해 두 조건을 비교한다.
같은 이미지에서 같은 "면적"을 지우되:

  - object   : 캡션이 설명하는 물체(들)를 정확히 가림
  - bg_match : 물체와 무관한 배경에서 무작위로 같은 면적을 가림 (통제군)

두 조건의 gap 증가폭이 다르면, "정보가 줄어든 것 자체"가 아니라
"캡션과 관련된 정보가 줄어든 것"이 gap을 키운다는 훨씬 깔끔한 증거가 된다.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from degrade import apply_pixel_mask, random_area_matched_mask  # noqa: E402
from encode import encode_images, encode_texts, load_model  # noqa: E402
from metrics import l2m, rmg  # noqa: E402

COCO_DIR = ROOT / "data" / "coco"
IMAGES_DIR = COCO_DIR / "val2017"
CAPTIONS_JSON = COCO_DIR / "annotations" / "captions_val2017.json"
INSTANCES_JSON = COCO_DIR / "annotations" / "instances_val2017.json"

N_SAMPLES = 1000
SEED = 0
AREA_MIN, AREA_MAX = 0.05, 0.6  # 물체 면적 비율 필터 — 너무 작거나 너무 크면 제외


def load_shuffled_pairs_with_id():
    """phase0_coco_reproduce.load_coco_pairs()와 동일한 순서(같은 seed)를 재현하되 image_id를 유지."""
    data = json.loads(CAPTIONS_JSON.read_text())
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    first_caption = {}
    for ann in data["annotations"]:
        first_caption.setdefault(ann["image_id"], ann["caption"])

    pairs = [
        (img_id, id_to_file[img_id], cap)
        for img_id, cap in first_caption.items()
        if img_id in id_to_file
    ]
    random.seed(SEED)
    random.shuffle(pairs)
    return pairs


def main():
    print("[1/5] loading COCO captions + instance annotations ...")
    all_pairs = load_shuffled_pairs_with_id()
    coco = COCO(str(INSTANCES_JSON))

    print(f"[2/5] selecting {N_SAMPLES} images with object area in [{AREA_MIN}, {AREA_MAX}] ...")
    selected = []  # (image_id, file, caption, object_mask)
    area_fracs = []
    for img_id, file, cap in all_pairs:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if not ann_ids:
            continue
        anns = coco.loadAnns(ann_ids)
        img_meta = coco.imgs.get(img_id)
        if img_meta is None:
            continue
        h, w = img_meta["height"], img_meta["width"]
        mask = np.zeros((h, w), dtype=bool)
        for ann in anns:
            mask |= coco.annToMask(ann).astype(bool)
        frac = mask.mean()
        if AREA_MIN <= frac <= AREA_MAX:
            selected.append((img_id, file, cap, mask))
            area_fracs.append(frac)
        if len(selected) >= N_SAMPLES:
            break

    print(f"  selected: {len(selected)}  (mean object area frac: {np.mean(area_fracs):.3f})")

    print("[3/5] building masked image variants ...")
    baseline_imgs, object_imgs, bgmatch_imgs = [], [], []
    for i, (img_id, file, cap, obj_mask) in enumerate(selected):
        img = Image.open(IMAGES_DIR / file).convert("RGB")
        baseline_imgs.append(img)
        object_imgs.append(apply_pixel_mask(img, obj_mask))
        bg_mask = random_area_matched_mask(obj_mask, int(obj_mask.sum()), seed=SEED + i)
        bgmatch_imgs.append(apply_pixel_mask(img, bg_mask))

    texts = [cap for _, _, cap, _ in selected]

    print("[4/5] loading OpenCLIP ViT-B-16 (openai) + encoding ...")
    model, preprocess, tokenizer = load_model("ViT-B-16", "openai", device="cpu")

    txt_emb = encode_texts(
        texts, model, tokenizer, cache_tag=f"coco_objmask_n{N_SAMPLES}_seed{SEED}_text"
    )
    base_emb = encode_images(
        baseline_imgs, model, preprocess,
        cache_tag=f"coco_objmask_n{N_SAMPLES}_seed{SEED}_baseline",
    )
    obj_emb = encode_images(
        object_imgs, model, preprocess,
        cache_tag=f"coco_objmask_n{N_SAMPLES}_seed{SEED}_objectmasked",
    )
    bg_emb = encode_images(
        bgmatch_imgs, model, preprocess,
        cache_tag=f"coco_objmask_n{N_SAMPLES}_seed{SEED}_bgmatched",
    )

    print("[5/5] computing metrics ...")
    results = {}
    for name, img_emb in [("baseline", base_emb), ("object_masked", obj_emb), ("bg_matched", bg_emb)]:
        m_l2m = l2m(img_emb, txt_emb)
        m_rmg = rmg(img_emb, txt_emb)
        results[name] = {"l2m": m_l2m, "rmg": m_rmg}
        print(f"  {name:>14}  L2M={m_l2m:.4f}  RMG={m_rmg:.4f}")

    results["mean_area_frac_masked"] = float(np.mean(area_fracs))
    results["n_samples"] = len(selected)

    out_path = ROOT / "results" / "phase1b_coco_object_mask.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")

    d_obj = results["object_masked"]["l2m"] - results["baseline"]["l2m"]
    d_bg = results["bg_matched"]["l2m"] - results["baseline"]["l2m"]
    print(f"\nΔL2M (object_masked - baseline) = {d_obj:+.4f}")
    print(f"ΔL2M (bg_matched    - baseline) = {d_bg:+.4f}")
    if d_obj > d_bg:
        print("→ 같은 면적이어도 물체를 가렸을 때 gap이 더 크게 늘었음 (캡션-관련 정보 가설 지지)")
    else:
        print("→ 물체를 가려도 배경을 가린 것보다 gap이 더 크지 않음 (정보 종류 무관, 양만 중요할 가능성)")


if __name__ == "__main__":
    main()
