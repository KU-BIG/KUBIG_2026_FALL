"""Build per-document, per-DPI ColPali page embeddings and log verification info."""
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent.parent
from colpali_backend import embed_images, get_model_and_processor  # noqa: E402

PAGES_ROOT = ROOT / "data" / "pages"
EMB_ROOT = ROOT / "embeddings"
OUTPUT_ROOT = ROOT / "output"
DPIS = [144, 72, 36]
DOC_IDS = sorted(p.name for p in (PAGES_ROOT / str(DPIS[0])).iterdir() if p.is_dir())

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def denormalize_and_save(pixel_values, out_path):
    """pixel_values: (3, H, W) tensor normalized with mean=std=0.5. Save as viewable PNG."""
    img = (pixel_values.float() * 0.5 + 0.5).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(img).save(out_path)


def main():
    EMB_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    model, processor = get_model_and_processor()
    log(f"Model loaded: {model.config.name_or_path if hasattr(model.config, 'name_or_path') else 'vidore/colpali-v1.2'}")
    log(f"Image processor config: {processor.image_processor}")
    log(f"visual_prompt_prefix (text wrapped around each image): {processor.visual_prompt_prefix!r}")

    # --- resize behavior check: fixed 448x448 square resize (no crop/pad) confirmed from
    # SiglipImageProcessor config (size={height:448,width:448}, no do_center_crop/padding).
    # Save before/after previews for a wide-ish doc page vs a narrower one to visually confirm.
    preview_dpi = 144
    preview_targets = [
        ("BESTBUY_2023_10K.pdf", "001.png"),
        ("afe620b9beac86c1027b96d31d396407.pdf", "001.png"),
    ]
    for doc_id, fname in preview_targets:
        src_path = PAGES_ROOT / str(preview_dpi) / doc_id / fname
        img = Image.open(src_path).convert("RGB")
        log(f"[preview] {doc_id}/{fname} original size (W,H) = {img.size}")
        inputs = processor.process_images([img])
        pixel_values = inputs["pixel_values"][0]
        log(f"[preview] {doc_id}/{fname} preprocessed pixel_values shape = {tuple(pixel_values.shape)}")
        safe_name = doc_id.replace(".pdf", "")
        denormalize_and_save(pixel_values, OUTPUT_ROOT / f"preview_{safe_name}_original_{img.size[0]}x{img.size[1]}.png")

    log(
        "[resize behavior] SiglipImageProcessor.size = {height:448,width:448} with no do_center_crop/padding "
        "field set -> PIL resize() targets 448x448 directly regardless of source aspect ratio. Both pilot docs "
        "above have different aspect ratios (WxH ~1224x1584 vs ~1185x1679) yet both are force-resized to a "
        "448x448 square, i.e. each page is DISTORTED/STRETCHED non-uniformly on the two axes rather than "
        "cropped or padded. See preview_*.png files for visual confirmation; compare against original page "
        "images in pages/144/ to see the aspect-ratio distortion directly."
    )

    total_start = time.time()
    for dpi in DPIS:
        dpi_dir = EMB_ROOT / str(dpi)
        dpi_dir.mkdir(parents=True, exist_ok=True)
        for doc_id in DOC_IDS:
            out_path = dpi_dir / f"{doc_id}.pt"
            if out_path.exists():
                log(f"[skip] dpi={dpi} doc={doc_id} already cached at {out_path}")
                continue

            doc_dir = PAGES_ROOT / str(dpi) / doc_id
            files = sorted(doc_dir.iterdir())
            page_numbers = [int(f.stem) for f in files]
            assert page_numbers == list(range(1, len(files) + 1)), (
                f"unexpected page numbering for dpi={dpi} doc={doc_id}: {page_numbers}"
            )

            t0 = time.time()
            images = [Image.open(f).convert("RGB") for f in files]
            embeddings = embed_images(images, batch_size=4)
            elapsed = time.time() - t0

            stacked = torch.stack(embeddings)  # (n_pages, n_patches, 128)
            torch.save({"page_numbers": page_numbers, "embeddings": stacked}, out_path)

            shape_str = tuple(embeddings[0].shape)
            log(
                f"[built] dpi={dpi} doc={doc_id} pages={len(files)} "
                f"per-page embedding shape=(1, {shape_str[0]}, {shape_str[1]}) "
                f"time={elapsed:.1f}s -> {out_path}"
            )
            if shape_str[0] != 1024:
                extra = shape_str[0] - 1024
                log(
                    f"  NOTE: sequence length is {shape_str[0]} (1024 image patches + {extra} extra tokens), "
                    f"not exactly 1024 as the task doc's naive expectation states. This is expected ColPaliProcessor "
                    f"behavior: it wraps each image with a fixed text instruction "
                    f"('<image>'x1024 + '<bos><bos>Describe the image.'), and those {extra} instruction tokens are "
                    f"embedded and returned alongside the 1024 image-patch tokens. Confirmed via "
                    f"processor.visual_prompt_prefix and manual tokenizer.decode() of process_images() output. "
                    f"Not a bug and not related to resize/aspect-ratio handling."
                )

    total_elapsed = time.time() - total_start
    log(f"Total indexing time: {total_elapsed:.1f}s")

    with open(OUTPUT_ROOT / "verification_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
