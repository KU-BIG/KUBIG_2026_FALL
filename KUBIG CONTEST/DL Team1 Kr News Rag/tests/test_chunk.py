from indexing.chunk import ChunkConfig, chunk_article, chunk_articles, split_units


def article(**overrides):
    base = {
        "id": 7,
        "title": "반도체 전망",
        "content": "첫 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다.",
        "date": "2026.08.08 10:00",
        "url": "https://example.com/7",
        "stock_names": ["삼성전자", "SK하이닉스"],
        "stock_codes": ["005930", "000660"],
        "source_ids": [11, 12],
        "doc_type": "article",
    }
    base.update(overrides)
    return base


def test_every_article_gets_a_chunk_and_lists_are_preserved():
    chunks = chunk_articles([article(id=1, content=""), article(id=2)])
    assert {chunk["article_id"] for chunk in chunks} == {1, 2}
    assert chunks[-1]["stock_names"] == ["삼성전자", "SK하이닉스"]
    assert chunks[-1]["stock_codes"] == ["005930", "000660"]
    assert chunks[-1]["source_ids"] == [11, 12]
    assert chunks[-1]["embedding_text"].startswith("반도체 전망\n")


def test_chunk_ids_and_order_are_unique_and_deterministic():
    config = ChunkConfig(target_min=30, target_max=45, max_length=55, overlap=10)
    doc = article(content="가" * 24 + "다. " + "나" * 24 + "다. " + "다" * 24 + "다.")
    first = chunk_article(doc, config)
    second = chunk_article(doc, config)
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    assert len({c["chunk_id"] for c in first}) == len(first)
    assert [c["chunk_index"] for c in first] == list(range(len(first)))


def test_maximum_length_and_long_sentence_safe_split():
    config = ChunkConfig(target_min=20, target_max=30, max_length=40, overlap=5)
    chunks = chunk_article(article(content=("긴문장, " * 20) + "끝입니다."), config)
    assert len(chunks) > 1
    assert max(len(c["content"]) for c in chunks) <= 40
    assert "".join(c["content"].replace(" ", "") for c in chunks).count("긴문장,") >= 20


def test_short_final_fragment_is_not_an_independent_chunk():
    config = ChunkConfig(target_min=20, target_max=30, max_length=50, overlap=5, min_tail=12)
    chunks = chunk_article(article(content="가" * 25 + "다. " + "짧다."), config)
    assert len(chunks) == 1
    assert chunks[0]["content"].endswith("짧다.")


def test_broadcast_speaker_markers_start_units():
    units = split_units("[앵커] 첫 소식입니다. [기자] 현장에 나와 있습니다.", "broadcast")
    assert units == ["[앵커] 첫 소식입니다.", "[기자] 현장에 나와 있습니다."]
