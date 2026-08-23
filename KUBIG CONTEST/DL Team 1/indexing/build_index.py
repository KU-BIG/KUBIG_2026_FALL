"""Embed news chunks with BGE-M3 and upsert them into persistent Chroma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "news_chunks.jsonl"
DEFAULT_DB = PROJECT_ROOT / "data" / "chroma"
DEFAULT_COLLECTION = "kr_news_bge_m3"
DEFAULT_MODEL = "BAAI/bge-m3"
LIST_FIELDS = ("stock_names", "stock_codes", "source_ids")
METADATA_FIELDS = ("article_id", "chunk_index", "title", "date", "url", "doc_type")


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> Sequence[Sequence[float]]: ...


class SentenceTransformerEmbedder:
    """Lazy production adapter that always returns normalized BGE-M3 vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> Sequence[Sequence[float]]:
        vectors = self.model.encode(
            list(texts), batch_size=batch_size or 32, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors


def serialize_metadata(chunk: dict) -> dict:
    metadata = {field: chunk.get(field, "") for field in METADATA_FIELDS}
    for field in LIST_FIELDS:
        metadata[field] = json.dumps(chunk.get(field, []), ensure_ascii=False)
    return metadata


def deserialize_metadata(metadata: dict | None) -> dict:
    restored = dict(metadata or {})
    for field in LIST_FIELDS:
        value = restored.get(field, "[]")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON metadata in {field}") from exc
        restored[field] = value
    return restored


def iter_jsonl_batches(path: Path, batch_size: int) -> Iterator[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batch: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _collection(client: chromadb.PersistentClient, name: str, rebuild: bool):
    if rebuild:
        try:
            client.delete_collection(name)
        except Exception as exc:
            if exc.__class__.__name__ != "NotFoundError":
                raise
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def build_index(
    input_path: Path,
    db_path: Path,
    collection_name: str,
    embedder: Embedder,
    batch_size: int = 8,
    rebuild: bool = False,
):
    input_path = Path(input_path)
    db_path = Path(db_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"chunk file not found: {input_path}")
    db_path.mkdir(parents=True, exist_ok=True)
    collection = _collection(chromadb.PersistentClient(path=str(db_path)), collection_name, rebuild)
    processed = 0
    for batch in iter_jsonl_batches(input_path, batch_size):
        embeddings = embedder.encode([item["embedding_text"] for item in batch], batch_size=batch_size)
        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            embeddings=[list(vector) for vector in embeddings],
            documents=[item["content"] for item in batch],
            metadatas=[serialize_metadata(item) for item in batch],
        )
        processed += len(batch)
        print(f"indexed {processed} chunks", flush=True)
    print(f"collection {collection_name}: {collection.count()} records")
    return collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    embedder = SentenceTransformerEmbedder(args.model, args.device)
    build_index(args.input, args.db_path, args.collection, embedder, args.batch_size, args.rebuild)


if __name__ == "__main__":
    main()
