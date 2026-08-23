import pytest

from indexing.query_preprocess import normalize_query


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("삼전 전망", "삼성전자 전망"),
        ("하이닉스와 SK 하이닉스", "SK하이닉스와 SK하이닉스"),
        ("에스케이하이닉스 실적", "SK하이닉스 실적"),
        ("현대자동차 전략", "현대차 전략"),
        ("LG엔솔과 LG 엔솔", "LG에너지솔루션과 LG에너지솔루션"),
        ("엘지엔솔과 엘지에너지솔루션", "LG에너지솔루션과 LG에너지솔루션"),
        ("삼바 투자", "삼성바이오로직스 투자"),
        ("KB금융지주와 KB 금융", "KB금융와 KB금융"),
        ("케이비금융 실적", "KB금융 실적"),
        ("현대중공업과 HD현중", "HD현대중공업과 HD현대중공업"),
        ("한화에어로 수주", "한화에어로스페이스 수주"),
        ("두산중공업 전망", "두산에너빌리티 전망"),
        ("네이버와 nAvEr", "NAVER와 NAVER"),
    ],
)
def test_normalize_query_replaces_supported_stock_aliases(query, expected):
    assert normalize_query(query) == expected


def test_normalize_query_applies_nfkc_and_collapses_whitespace():
    assert normalize_query("  Ｎａｖｅｒ\tＡＩ   전망  ") == "NAVER AI 전망"


@pytest.mark.parametrize(
    "query",
    [
        "삼성전자 투자",
        "SK하이닉스 투자",
        "현대차 투자",
        "LG에너지솔루션 투자",
        "삼성바이오로직스 투자",
        "KB금융 투자",
        "HD현대중공업 투자",
        "한화에어로스페이스 투자",
        "두산에너빌리티 투자",
        "NAVER 투자",
    ],
)
def test_normalize_query_preserves_canonical_stock_names(query):
    assert normalize_query(query) == query


@pytest.mark.parametrize("query", ["삼성", "현대", "한화", "두산", "국민은행", "카카오"])
def test_normalize_query_does_not_replace_ambiguous_or_unsupported_names(query):
    assert normalize_query(f"{query} 실적 발표") == f"{query} 실적 발표"


def test_normalize_query_is_idempotent():
    normalized = normalize_query("네이버 AI와 현대자동차, LG엔솔 전망은?")
    assert normalize_query(normalized) == normalized


def test_normalize_query_preserves_non_stock_words_numbers_and_punctuation():
    assert normalize_query("삼전의 2026년 영업이익은 +12.5%?") == "삼성전자의 2026년 영업이익은 +12.5%?"


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_normalize_query_rejects_empty_input(query):
    with pytest.raises(ValueError, match="query cannot be empty"):
        normalize_query(query)
