"""Read-only browsing of the cleaned article corpus."""

from __future__ import annotations

import json
import random
from pathlib import Path

DISPLAY_FIELDS = ("article_id", "title", "date", "content", "url", "doc_type")


class ArticleCatalog:
    def __init__(self, articles: list[dict]) -> None:
        self._articles = articles
        self._by_id = {article["id"]: article for article in articles}

    @classmethod
    def from_json(cls, path: Path | str) -> "ArticleCatalog":
        with Path(path).open(encoding="utf-8") as stream:
            articles = json.load(stream)
        if not isinstance(articles, list):
            raise ValueError("cleaned article data must be a JSON array")
        return cls(articles)

    @property
    def article_ids(self) -> set[int]:
        return set(self._by_id)

    def _display(self, article: dict) -> dict:
        return {
            "article_id": article["id"],
            "title": article.get("title", ""),
            "date": article.get("date", ""),
            "content": article.get("content", ""),
            "url": article.get("url", ""),
            "doc_type": article.get("doc_type", ""),
        }

    def get(self, article_id: int) -> dict:
        if article_id not in self._by_id:
            raise KeyError(f"article_id not found: {article_id}")
        return self._display(self._by_id[article_id])

    def search(self, keyword: str) -> list[dict]:
        keyword = keyword.strip().casefold()
        if not keyword:
            raise ValueError("keyword cannot be empty")
        return [
            self._display(article)
            for article in self._articles
            if keyword in article.get("title", "").casefold() or keyword in article.get("content", "").casefold()
        ]

    def sample(self, count: int, *, seed: int = 42) -> list[dict]:
        if count < 1 or count > len(self._articles):
            raise ValueError("count must be between 1 and corpus size")
        return [self._display(article) for article in random.Random(seed).sample(self._articles, count)]
