from ui_format import meta_line, provenance_line, source_heading


def test_the_title_links_to_the_article():
    heading = source_heading(1, {"title": "삼성전자 실적 개선", "url": "https://example.com/a"})

    assert heading == "**1. [삼성전자 실적 개선](https://example.com/a)**"


def test_brackets_in_the_title_do_not_break_the_link():
    # Korean news titles routinely start with a bracketed section marker
    # ("[마감시황]", "[칩톡]"). Unescaped, they terminate the link text early and
    # the markdown renders as literal brackets next to a bare URL.
    heading = source_heading(2, {"title": "[마감시황]코스피 6258 마감", "url": "https://ex.com/b"})

    assert heading == "**2. [\\[마감시황\\]코스피 6258 마감](https://ex.com/b)**"


def test_a_chunk_without_a_url_renders_as_plain_text():
    assert source_heading(3, {"title": "제목만 있음"}) == "**3. 제목만 있음**"


def test_a_missing_title_still_renders_something_clickable():
    heading = source_heading(1, {"url": "https://example.com/a"})

    assert "(제목 없음)" in heading
    assert "https://example.com/a" in heading


def test_meta_line_lists_date_and_stocks():
    line = meta_line({"date": "2026.08.08", "stock_names": ["삼성전자", "SK하이닉스"]})

    assert "2026.08.08" in line
    assert "삼성전자" in line and "SK하이닉스" in line


def test_meta_line_marks_broadcast_transcripts():
    assert "방송" in meta_line({"date": "2026.08.08", "doc_type": "broadcast"})


def test_provenance_line_shows_which_retriever_found_the_chunk():
    line = provenance_line(
        {"rrf_score": 0.0164, "dense_rank": 1, "bm25_rank": None, "similarity": 0.61}
    )

    assert "RRF 0.0164" in line
    assert "dense #1" in line
    assert "bm25 —" in line


def test_provenance_line_shows_which_expanded_queries_matched():
    line = provenance_line(
        {"expanded_queries": ["원본", "재작성", "가상 본문"], "matched_queries": [0, 2]}
    )

    assert "q0" in line and "q2" in line


def test_provenance_line_shows_the_citation_numbers_merged_into_a_source():
    line = provenance_line({"cited_as": [1, 3]})

    assert "[뉴스1]" in line and "[뉴스3]" in line


def test_provenance_line_is_empty_when_there_is_nothing_to_say():
    assert provenance_line({"title": "제목"}) == ""
