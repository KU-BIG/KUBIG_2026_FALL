"""Dense retrieval entry point: a question in, top-k news chunks out.

`search_collection()` needs a live Chroma collection and a loaded embedder handed
in. The LLM pipeline should not have to know either, and must not reload BGE-M3
(~4.5GB) on every question, so `DenseRetriever` owns both and loads them lazily
on first use.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from indexing.build_index import (
    DEFAULT_COLLECTION,
    DEFAULT_DB,
    DEFAULT_MODEL,
    Embedder,
    SentenceTransformerEmbedder,
)
from indexing.search import search_collection

DEFAULT_TOP_K = 5


class DenseRetriever:
    """Question -> top-k chunks, each carrying its source title/date/url."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        embedder: Embedder | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        self._embedder = embedder
        self._collection = None

    def _ensure_collection(self):
        if self._collection is None:
            if not self.db_path.is_dir():
                raise FileNotFoundError(
                    f"chroma db not found: {self.db_path} (run indexing/build_index.py --rebuild first)"
                )
            client = chromadb.PersistentClient(path=str(self.db_path))
            try:
                self._collection = client.get_collection(self.collection_name)
            except Exception as exc:
                raise RuntimeError(f"collection not found: {self.collection_name} ({self.db_path})") from exc
        return self._collection

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.model_name, self.device)
        return self._embedder

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        # Collection first: a missing index should fail before we pay for the model.
        collection = self._ensure_collection()
        return search_collection(collection, self._ensure_embedder(), question, top_k)


_RETRIEVERS: dict[tuple[str, str, str, str], DenseRetriever] = {}


def get_retriever(
    db_path: Path | str = DEFAULT_DB,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> DenseRetriever:
    """Reuse one retriever per configuration so BGE-M3 is loaded at most once."""
    key = (str(Path(db_path)), collection_name, model_name, device)
    if key not in _RETRIEVERS:
        _RETRIEVERS[key] = DenseRetriever(db_path, collection_name, model_name, device)
    return _RETRIEVERS[key]


def choose_device(has_cuda: bool, has_mps: bool) -> str:
    """Pick the fastest device the machine actually offers."""
    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def detect_device() -> str:
    """`choose_device` against the installed torch, or CPU if there is none."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return choose_device(torch.cuda.is_available(), torch.backends.mps.is_available())
