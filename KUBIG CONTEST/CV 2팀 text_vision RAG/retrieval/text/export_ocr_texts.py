import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
OCR_ROOT = BASE / "ocr" / "raw"
RESULT_DIR = BASE / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DPIS = [144, 72, 36]

DOCS = [
    x.strip()
    for x in (BASE / "data" / "full_docs.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

for dpi in DPIS:
    results = []

    for doc_id in DOCS:
        doc_dir = OCR_ROOT / str(dpi) / doc_id

        md_files = sorted(
            doc_dir.rglob("*.md"),
            key=lambda p: int(p.stem)
        )

        for md_path in md_files:
            page_number = int(md_path.stem)

            text = md_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            results.append({
                "doc_id": doc_id,
                "page_number": page_number,
                "text": text,
            })

    results = sorted(
        results,
        key=lambda x: (
            x["doc_id"],
            x["page_number"]
        )
    )

    output_path = RESULT_DIR / f"ocr_text_{dpi}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"{dpi} DPI: "
        f"{len(results)} pages -> {output_path}"
    )