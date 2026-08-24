from indexing.tokenize_ko import get_tokenizer, tokenize


def test_particles_are_split_off_the_noun():
    tokens = tokenize("삼성전자의 반도체 실적")

    # BM25 only matches whole tokens, so "삼성전자" must survive without its 조사.
    assert "삼성전자" in tokens
    assert "삼성전자의" not in tokens


def test_particles_and_endings_are_dropped():
    tokens = tokenize("삼성전자가 올랐다")

    assert "가" not in tokens
    assert "다" not in tokens
    assert "었" not in tokens


def test_latin_tokens_survive_and_are_lowercased():
    tokens = tokenize("NAVER와 HBM 수요")

    assert "naver" in tokens
    assert "hbm" in tokens


def test_query_and_document_casing_agree():
    assert tokenize("naver 실적") == tokenize("NAVER 실적")


def test_digits_survive_for_ticker_codes():
    assert "005930" in tokenize("삼성전자(005930)")


def test_verb_stem_survives_inflection():
    assert "오르" in tokenize("주가가 올랐다")


def test_punctuation_is_dropped():
    tokens = tokenize("코스피(005930), 반도체")

    assert "(" not in tokens
    assert ")" not in tokens
    assert "," not in tokens


def test_blank_text_yields_no_tokens():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenizing_is_deterministic():
    assert tokenize("삼성전자 반도체 전망") == tokenize("삼성전자 반도체 전망")


def test_tokenizer_is_loaded_once():
    # Kiwi costs ~0.5s to construct; rebuilding it per call would dominate indexing.
    assert get_tokenizer() is get_tokenizer()
