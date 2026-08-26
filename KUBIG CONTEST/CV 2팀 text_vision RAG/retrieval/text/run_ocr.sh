#!/bin/bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT="$BASE/data/pages"
OUTPUT="$BASE/ocr/raw"
API="http://127.0.0.1:8000"

DPIS=(144 72 36)

mapfile -t DOCS < <(
    sed 's/\r$//' "$BASE/data/full_docs.txt" |
    sed '/^[[:space:]]*$/d'
)

echo "========================================"
echo "Documents: ${#DOCS[@]}"
echo "DPIs: ${DPIS[*]}"
echo "========================================"

if [ "${#DOCS[@]}" -ne 60 ]; then
    echo "ERROR: Expected 60 documents, got ${#DOCS[@]}"
    exit 1
fi

for dpi in "${DPIS[@]}"; do
    for doc in "${DOCS[@]}"; do

        input_dir="$INPUT/$dpi/$doc"
        output_dir="$OUTPUT/$dpi/$doc"

        if [ ! -d "$input_dir" ]; then
            echo "ERROR: Input directory not found:"
            echo "$input_dir"
            exit 1
        fi

        expected_pages=$(find "$input_dir" -maxdepth 1 -type f -name "*.png" | wc -l)

        mkdir -p "$output_dir"

        existing_pages=$(find "$output_dir" -type f -name "*.md" 2>/dev/null | wc -l)

        echo
        echo "========================================"
        echo "DPI      : $dpi"
        echo "DOC      : $doc"
        echo "Expected : $expected_pages pages"
        echo "Existing : $existing_pages OCR pages"
        echo "========================================"

        if [ "$expected_pages" -gt 0 ] && [ "$existing_pages" -eq "$expected_pages" ]; then
            echo "SKIP: already completed"
            continue
        fi

        if [ "$existing_pages" -gt 0 ]; then
            echo "Partial output found. Re-running this document."
            rm -rf "$output_dir"
            mkdir -p "$output_dir"
        fi

        mineru \
          -p "$input_dir" \
          -o "$output_dir" \
          -b pipeline \
          -m ocr \
          --api-url "$API"

        completed_pages=$(find "$output_dir" -type f -name "*.md" | wc -l)

        echo "Completed: $completed_pages / $expected_pages"

        if [ "$completed_pages" -ne "$expected_pages" ]; then
            echo "ERROR: OCR page count mismatch"
            exit 1
        fi

    done
done

echo
echo "========================================"
echo "ALL OCR FINISHED"
echo "========================================"