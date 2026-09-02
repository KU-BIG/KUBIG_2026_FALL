"""
Phase 1 보강 — 물체 마스킹을 "가림/안 가림" 이진이 아니라 강도로 스윕(0%→100%)해서
U자 곡선이 나오는지 본다. 배경을 같은 면적만큼 마스킹하는 통제 곡선도 같이 그린다.

phase1b_coco_object_mask.py와 동일한 1,000쌍(같은 seed)을 재사용하고,
레벨 0(무가공)과 레벨 1.0(물체/배경 전체 마스킹)은 그 스크립트의 캐시를 그대로 재사용한다.
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

from degrade import apply_pixel_mask, partial_mask, random_area_matched_mask  # noqa: E402
from encode import CACHE_DIR, encode_images, encode_texts, load_model  # noqa: E402
from metrics import l2m, rmg  # noqa: E402

COCO_DIR = ROOT / "data" / "coco"
IMAGES_DIR = COCO_DIR / "val2017"
CAPTIONS_JSON = COCO_DIR / "annotations" / "captions_val2017.json"
INSTANCES_JSON = COCO_DIR / "annotations" / "instances_val2017.json"

N_SAMPLES = 1000
SEED = 0
AREA_MIN, AREA_MAX = 0.05, 0.6
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

# phase1b_coco_object_mask.py가 만든 캐시 재사용
BASELINE_TAG = f"coco_objmask_n{N_SAMPLES}_seed{SEED}_baseline"
OBJ_FULL_TAG = f"coco_objmask_n{N_SAMPLES}_seed{SEED}_objectmasked"
BG_FULL_TAG = f"coco_objmask_n{N_SAMPLES}_seed{SEED}_bgmatched"
TEXT_TAG = f"coco_objmask_n{N_SAMPLES}_seed{SEED}_text"


def load_shuffled_pairs_with_id():
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


def select_images(coco, all_pairs):
    selected = []
    for img_id, file, cap in all_pairs:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if not ann_ids:
            continue
        anns = coco.loadAnns(ann_ids)
        meta = coco.imgs.get(img_id)
        if meta is None:
            continue
        h, w = meta["height"], meta["width"]
        mask = np.zeros((h, w), dtype=bool)
        for ann in anns:
            mask |= coco.annToMask(ann).astype(bool)
        frac = mask.mean()
        if AREA_MIN <= frac <= AREA_MAX:
            selected.append((img_id, file, cap, mask))
        if len(selected) >= N_SAMPLES:
            break
    return selected


def get_or_encode_images(images, model, preprocess, tag):
    cache_path = CACHE_DIR / f"{tag}.npy"
    if cache_path.exists():
        print(f"  [cache hit] {tag}")
        return np.load(cache_path)
    return encode_images(images, model, preprocess, cache_tag=tag)


def main():
    print("[1/4] loading COCO captions + instance annotations, selecting images ...")
    all_pairs = load_shuffled_pairs_with_id()
    coco = COCO(str(INSTANCES_JSON))
    selected = select_images(coco, all_pairs)
    print(f"  selected: {len(selected)}")

    texts = [cap for _, _, cap, _ in selected]

    print("[2/4] loading OpenCLIP ViT-B-16 (openai) + encoding text (cached) ...")
    model, preprocess, tokenizer = load_model("ViT-B-16", "openai", device="cpu")
    txt_emb = encode_texts(texts, model, tokenizer, cache_tag=TEXT_TAG)

    print("[3/4] sweeping mask strength 0% -> 100% for object-curve and bg-curve ...")
    obj_results, bg_results = [], []
    for level in LEVELS:
        if level == 0.0:
            img_emb_obj = get_or_encode_images(None, model, preprocess, BASELINE_TAG)
            img_emb_bg = img_emb_obj
        elif level == 1.0:
            img_emb_obj = get_or_encode_images(None, model, preprocess, OBJ_FULL_TAG)
            img_emb_bg = get_or_encode_images(None, model, preprocess, BG_FULL_TAG)
        else:
            obj_tag = f"coco_objmask_n{N_SAMPLES}_seed{SEED}_objfrac{level}"
            bg_tag = f"coco_objmask_n{N_SAMPLES}_seed{SEED}_bgfrac{level}"

            obj_cache = CACHE_DIR / f"{obj_tag}.npy"
            bg_cache = CACHE_DIR / f"{bg_tag}.npy"
            if obj_cache.exists():
                img_emb_obj = np.load(obj_cache)
            else:
                obj_imgs = []
                for i, (img_id, file, cap, obj_mask) in enumerate(selected):
                    img = Image.open(IMAGES_DIR / file).convert("RGB")
                    lvl_mask = partial_mask(obj_mask, level, seed=SEED + i)
                    obj_imgs.append(apply_pixel_mask(img, lvl_mask))
                img_emb_obj = encode_images(obj_imgs, model, preprocess, cache_tag=obj_tag)

            if bg_cache.exists():
                img_emb_bg = np.load(bg_cache)
            else:
                bg_imgs = []
                for i, (img_id, file, cap, obj_mask) in enumerate(selected):
                    img = Image.open(IMAGES_DIR / file).convert("RGB")
                    n_px = int(round(obj_mask.sum() * level))
                    lvl_bg_mask = random_area_matched_mask(obj_mask, n_px, seed=SEED + i + 10000)
                    bg_imgs.append(apply_pixel_mask(img, lvl_bg_mask))
                img_emb_bg = encode_images(bg_imgs, model, preprocess, cache_tag=bg_tag)

        m_l2m_obj, m_rmg_obj = l2m(img_emb_obj, txt_emb), rmg(img_emb_obj, txt_emb)
        m_l2m_bg, m_rmg_bg = l2m(img_emb_bg, txt_emb), rmg(img_emb_bg, txt_emb)
        obj_results.append({"level": level, "l2m": m_l2m_obj, "rmg": m_rmg_obj})
        bg_results.append({"level": level, "l2m": m_l2m_bg, "rmg": m_rmg_bg})
        print(f"  level={level:.2f}  object L2M={m_l2m_obj:.4f}  |  bg-matched L2M={m_l2m_bg:.4f}")

    print("[4/4] saving ...")
    out = {"object_curve": obj_results, "bg_curve": bg_results, "n_samples": len(selected)}
    out_path = ROOT / "results" / "phase1c_coco_object_mask_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")

    obj_l2ms = [r["l2m"] for r in obj_results]
    min_idx = int(np.argmin(obj_l2ms))
    if min_idx == 0:
        verdict = "단조 증가 (물체 마스킹도 U자 아님)"
    elif min_idx == len(obj_l2ms) - 1:
        verdict = "단조 감소 (예상 밖)"
    else:
        verdict = f"U자 형태 (최소점: level={obj_results[min_idx]['level']})"
    print(f"\n물체 마스킹 곡선 판정: {verdict}")


if __name__ == "__main__":
    main()
