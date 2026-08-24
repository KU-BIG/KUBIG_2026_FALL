"""Query the persistent BGE-M3 news index with dense retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

try:
    from indexing.query_preprocess import normalize_query
except ModuleNotFoundError:  # Direct execution: python indexing/search.py
    from query_preprocess import normalize_query  # type: ignore[no-redef]

try:
    from indexing.build_index import (
        DEFAULT_COLLECTION,
        DEFAULT_DB,
        DEFAULT_MODEL,
        Embedder,
        SentenceTransformerEmbedder,
        deserialize_metadata,
    )
except ModuleNotFoundError:  # Direct execution: python indexing/search.py
    from build_index import (  # type: ignore[no-redef]
        DEFAULT_COLLECTION,
        DEFAULT_DB,
        DEFAULT_MODEL,
        Embedder,
        SentenceTransformerEmbedder,
        deserialize_metadata,
    )


def search_collection(collection, embedder: Embedder, query: str, top_k: int = 5) -> list[dict]:
    query = normalize_query(query)
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    count = collection.count()
    if count == 0:
        return []
    vector = embedder.encode([query], batch_size=1)[0]
    response = collection.query(
        query_embeddings=[list(vector)],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for chunk_id, content, metadata, distance in zip(
        response["ids"][0], response["documents"][0], response["metadatas"][0], response["distances"][0]
    ):
        item = deserialize_metadata(metadata)
        item.update(
            {
                "chunk_id": chunk_id,
                "content": content,
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            }
        )
        results.append(item)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Korean search question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=[DEFAULT_MODEL])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = chromadb.PersistentClient(path=str(args.db_path))
    try:
        collection = client.get_collection(args.collection)
    except Exception as exc:
        raise SystemExit(f"collection not found: {args.collection} ({args.db_path})") from exc
    embedder = SentenceTransformerEmbedder(args.model, args.device)
    for rank, result in enumerate(search_collection(collection, embedder, args.query, args.top_k), 1):
        print(f"\n[{rank}] distance={result['distance']:.6f} similarity={result['similarity']:.6f}")
        print(f"title: {result.get('title', '')}")
        print(f"date: {result.get('date', '')}")
        print(f"stocks: {result.get('stock_names', [])}")
        print(f"url: {result.get('url', '')}")
        print(f"content: {result.get('content', '')}")


if __name__ == "__main__":
    main()
