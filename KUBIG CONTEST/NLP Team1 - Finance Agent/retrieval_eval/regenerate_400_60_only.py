"""One-off driver: rebuild documents (with PDF002 page filtering fixed) and
regenerate ONLY the 400/60 chunk variant. Does not touch 300/50 or 500/80.

This script lives in KUBIG_FINANCE_final_test only and is a throwaway driver
for this regeneration pass -- it imports and reuses the real pipeline
functions from prepare_retrieval_data.py rather than re-implementing them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_retrieval_data import (  # noqa: E402
    DOCUMENTS_PATH,
    CHUNKS_DIR,
    METADATA_DIR,
    read_jsonl,
    write_jsonl,
    write_json,
    rebuild_documents,
    load_tokenizer,
    build_chunks,
    build_corpus_statistics,
    build_chunk_statistics,
)

original_documents = read_jsonl(DOCUMENTS_PATH)
print(f"[1] read {len(original_documents)} template documents from {DOCUMENTS_PATH}")

documents, page_spans = rebuild_documents(original_documents)
print(f"[2] rebuilt {len(documents)} documents (PDF text re-extracted, web unchanged)")

tokenizer = load_tokenizer()
print("[3] loaded BGE-M3 tokenizer")

chunks_400_60 = build_chunks(
    documents, page_spans, tokenizer, target_tokens=400, overlap_tokens=60
)
print(f"[4] built {len(chunks_400_60)} chunks (400/60)")

write_jsonl(DOCUMENTS_PATH, documents)
write_jsonl(CHUNKS_DIR / "chunk_400_60" / "chunks.jsonl", chunks_400_60)
write_json(METADATA_DIR / "corpus_statistics.json", build_corpus_statistics(documents))
write_json(
    METADATA_DIR / "chunk_statistics_400_60.json",
    build_chunk_statistics("400_60", chunks_400_60, 400, 60),
)
print("[5] wrote documents.jsonl, chunk_400_60/chunks.jsonl, corpus_statistics.json, chunk_statistics_400_60.json")
print("NOTE: chunk_300_50 and chunk_500_80 were left untouched (not regenerated), per instructions.")
