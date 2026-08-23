import json

from evaluation.articles import ArticleCatalog


def write_articles(path):
    articles = [
        {"id": 1, "title": "반도체 전망", "content": "HBM 수요 증가", "date": "2026.08.01", "url": "https://a", "doc_type": "article"},
        {"id": 2, "title": "배터리 동향", "content": "전기차 시장", "date": "2026.08.02", "url": "https://b", "doc_type": "article"},
        {"id": 3, "title": "금융 실적", "content": "은행 이익", "date": "2026.08.03", "url": "https://c", "doc_type": "article"},
    ]
    path.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")


def test_catalog_get_and_keyword_search_return_display_fields(tmp_path):
    path = tmp_path / "articles.json"
    write_articles(path)
    catalog = ArticleCatalog.from_json(path)

    assert catalog.get(1)["content"] == "HBM 수요 증가"
    assert [item["article_id"] for item in catalog.search("전기차")] == [2]
    assert set(catalog.get(1)) == {"article_id", "title", "date", "content", "url", "doc_type"}


def test_seeded_sampling_is_reproducible_and_contains_no_retrieval_score(tmp_path):
    path = tmp_path / "articles.json"
    write_articles(path)
    catalog = ArticleCatalog.from_json(path)

    first = catalog.sample(2, seed=42)
    second = catalog.sample(2, seed=42)

    assert first == second
    assert all("score" not in key for item in first for key in item)
