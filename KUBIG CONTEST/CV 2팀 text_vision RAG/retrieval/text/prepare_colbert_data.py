from pathlib import Path
import json
import re
import csv

from bs4 import BeautifulSoup
from transformers import AutoTokenizer


BASE = Path(__file__).resolve().parent.parent.parent

OCR_ROOT = BASE / "ocr" / "raw"
OUT_ROOT = BASE / "colbert_data"

DPIS = [144, 72, 36]

DOCS = [
    x.strip()
    for x in (BASE / "data" / "full_docs.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

CHUNK_SIZE = 480
OVERLAP = 50
STRIDE = CHUNK_SIZE - OVERLAP

tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")


def linearize_table(html):
    """
    HTML table을 retrieval용 plain text로 변환.
    OCR 내용을 수정하지 않고 구조만 선형화한다.
    """
    soup = BeautifulSoup(html, "html.parser")

    rows = []

    for tr in soup.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
            for cell in tr.find_all(["th", "td"])
        ]

        if cells:
            rows.append(cells)

    if not rows:
        return " "

    # 첫 행은 표 header로 보존
    header = rows[0]

    output = [
        "[Table Header] " + " | ".join(header)
    ]

    # 이후 행은 header와 값을 함께 반복
    for row in rows[1:]:

        if len(row) == len(header):
            pairs = []

            for h, value in zip(header, row):
                if h and value:
                    pairs.append(f"{h}: {value}")
                elif value:
                    pairs.append(value)

            output.append(
                "[Table Row] " + " | ".join(pairs)
            )

        else:
            output.append(
                "[Table Row] " + " | ".join(row)
            )

    return "\n".join(output)


def clean_markdown(text):
    """
    OCR 내용 자체는 수정하지 않는다.
    retrieval에 불필요한 Markdown/HTML 형식만 정리한다.
    """

    # 이미지 링크 제거
    text = re.sub(
        r"!\[[^\]]*\]\([^)]+\)",
        " ",
        text
    )

    # HTML table 선형화
    table_pattern = re.compile(
        r"<table.*?</table>",
        flags=re.IGNORECASE | re.DOTALL
    )

    text = table_pattern.sub(
        lambda m: "\n" + linearize_table(m.group(0)) + "\n",
        text
    )

    # 혹시 남은 HTML tag가 있다면 내용만 유지
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text("\n")

    # Markdown heading marker 제거
    text = re.sub(
        r"(?m)^\s*#{1,6}\s*",
        "",
        text
    )

    # Markdown bold/italic marker 정도만 제거
    text = text.replace("**", "")
    text = text.replace("__", "")

    # 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = text.strip()

    return text


def make_chunks(text):
    """
    480-token chunk / 50-token overlap.

    character offset을 이용하므로
    tokenizer decode로 OCR text 자체를 다시 만들지 않는다.
    """

    if not text.strip():
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )

    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    chunks = []

    start = 0

    while start < len(ids):

        end = min(start + CHUNK_SIZE, len(ids))

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        chunk_text = text[char_start:char_end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "token_start": start,
                "token_end": end,
                "n_tokens": end - start,
            })

        if end >= len(ids):
            break

        start += STRIDE

    return chunks


summary = {}

for dpi in DPIS:

    summary[str(dpi)] = {}

    for doc in DOCS:

        input_dir = OCR_ROOT / str(dpi) / doc
        output_dir = OUT_ROOT / str(dpi) / doc

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # 001.md, 002.md, ... 순서 보장
        md_files = sorted(
            input_dir.rglob("*.md"),
            key=lambda p: int(p.stem)
        )

        collection = []
        metadata = []

        empty_pages = []
        page_chunk_counts = {}

        pid = 0

        for md_path in md_files:

            page_number = int(md_path.stem)

            raw_text = md_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            clean_text = clean_markdown(raw_text)

            chunks = make_chunks(clean_text)

            page_chunk_counts[str(page_number)] = len(chunks)

            if not chunks:
                empty_pages.append(page_number)
                continue

            for chunk_idx, chunk in enumerate(chunks):

                collection.append(
                    (pid, chunk["text"])
                )

                metadata.append({
                    "pid": pid,
                    "dpi": dpi,
                    "doc_id": doc,
                    "page_number": page_number,
                    "chunk_id": chunk_idx,
                    "n_tokens": chunk["n_tokens"],
                    "token_start": chunk["token_start"],
                    "token_end": chunk["token_end"],
                })

                pid += 1

        # collection.tsv
        collection_path = output_dir / "collection.tsv"

        with open(
            collection_path,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(
                f,
                delimiter="\t",
                lineterminator="\n"
            )

            for pid_, text in collection:
                # TSV 구조가 깨지지 않도록 줄바꿈/탭 제거
                text_tsv = re.sub(r"\s+", " ", text).strip()
                writer.writerow([pid_, text_tsv])

        # pid → page/chunk mapping
        metadata_path = output_dir / "metadata.json"

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=2
            )

        stats = {
            "pages": len(md_files),
            "empty_pages": empty_pages,
            "n_empty_pages": len(empty_pages),
            "chunks": len(collection),
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "page_chunk_counts": page_chunk_counts,
        }

        with open(
            output_dir / "stats.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                stats,
                f,
                ensure_ascii=False,
                indent=2
            )

        summary[str(dpi)][doc] = {
            "pages": len(md_files),
            "empty_pages": len(empty_pages),
            "chunks": len(collection),
        }

        print("=" * 70)
        print(f"DPI       : {dpi}")
        print(f"Document  : {doc}")
        print(f"Pages     : {len(md_files)}")
        print(f"Empty     : {len(empty_pages)}")
        print(f"Chunks    : {len(collection)}")
        print(f"Output    : {output_dir}")


summary_path = OUT_ROOT / "summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n" + "=" * 70)
print("ALL PREPROCESSING FINISHED")
print(f"Summary saved to: {summary_path}")