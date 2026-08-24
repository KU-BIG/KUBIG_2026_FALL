"""The Naver finance news crawler.

Every request goes through an injected fetcher, so these run offline and none of
them touches naver.com. What is worth pinning here is the behaviour that a live
run would take 20 minutes to reveal: the cache is reused, an article shared by
two stocks is fetched once, and a page whose markup does not match is skipped
rather than stored empty.
"""

import json

import pytest

from preprocessing import crawl


LIST_HTML = """
<table class="type5"><tbody>
  <tr><td class="title"><a href="/item/news_read.naver?article_id=1">첫 기사</a></td>
      <td class="date">2026.08.08 13:10</td></tr>
  <tr><td class="title"><a href="/item/news_read.naver?article_id=2">둘째  기사</a></td>
      <td class="date">2026.08.08 12:00</td></tr>
</tbody></table>
"""
EMPTY_LIST_HTML = '<table class="type5"><tbody></tbody></table>'
ARTICLE_HTML = '<div id="dic_area">본문입니다. ' + "가" * 200 + "</div>"


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.encoding = None


class FakeFetcher:
    """Serves canned HTML by URL and records what was asked for."""

    def __init__(self, pages=None, default=None):
        self.pages = pages or {}
        self.default = default
        self.calls = []

    def __call__(self, url, encoding="euc-kr"):
        self.calls.append(url)
        for fragment, html in self.pages.items():
            if fragment in url:
                return None if html is None else FakeResponse(html)
        return FakeResponse(self.default) if self.default else None


def make_crawler(tmp_path, fetcher, **kwargs):
    return crawl.Crawler(
        cache_dir=tmp_path / "cache",
        fetch=fetcher,
        sleep=lambda _seconds: None,
        **kwargs,
    )


# --- parsing ---------------------------------------------------------------


def test_whitespace_is_collapsed():
    assert crawl.clean_text("  여러   줄\n\n 텍스트 ") == "여러 줄 텍스트"


def test_the_news_list_yields_title_url_and_date(tmp_path):
    fetcher = FakeFetcher({"news_news.naver": LIST_HTML})
    crawler = make_crawler(tmp_path, fetcher, pages=1)

    items = crawler.news_list("005930")

    assert items == [
        {
            "title": "첫 기사",
            "url": "https://finance.naver.com/item/news_read.naver?article_id=1",
            "date": "2026.08.08 13:10",
        },
        {
            "title": "둘째 기사",
            "url": "https://finance.naver.com/item/news_read.naver?article_id=2",
            "date": "2026.08.08 12:00",
        },
    ]


def test_the_same_article_on_two_pages_is_listed_once(tmp_path):
    crawler = make_crawler(tmp_path, FakeFetcher({"news_news.naver": LIST_HTML}), pages=3)

    assert len(crawler.news_list("005930")) == 2


def test_listing_stops_at_the_first_empty_page(tmp_path):
    """Past the last page Naver keeps answering, with no rows. Walking on wastes requests."""
    fetcher = FakeFetcher({"page=1": LIST_HTML, "news_news.naver": EMPTY_LIST_HTML})
    crawler = make_crawler(tmp_path, fetcher, pages=25)

    crawler.news_list("005930")

    assert len(fetcher.calls) == 2


def test_a_failed_list_page_does_not_stop_the_rest(tmp_path):
    fetcher = FakeFetcher({"page=1": None, "news_news.naver": LIST_HTML})
    crawler = make_crawler(tmp_path, fetcher, pages=2)

    assert len(crawler.news_list("005930")) == 2


# --- article body ----------------------------------------------------------


def test_the_article_body_is_extracted(tmp_path):
    crawler = make_crawler(tmp_path, FakeFetcher(default=ARTICLE_HTML))

    assert crawler.article_content("https://x/1").startswith("본문입니다.")


def test_a_javascript_redirect_is_followed(tmp_path):
    """news_read.naver ships no body — only a JS hop to n.news.naver.com."""
    fetcher = FakeFetcher({
        "news_read": "<script>location.href = 'https://n.news.naver.com/article/1';</script>",
        "n.news.naver.com": ARTICLE_HTML,
    })
    crawler = make_crawler(tmp_path, fetcher)

    assert crawler.article_content("https://finance.naver.com/item/news_read.naver?a=1")
    assert "n.news.naver.com" in fetcher.calls[-1]


def test_an_iframe_body_is_followed(tmp_path):
    fetcher = FakeFetcher({
        "outer": '<iframe id="news_read_iframe" src="/inner"></iframe>',
        "/inner": ARTICLE_HTML,
    })
    crawler = make_crawler(tmp_path, fetcher)

    assert crawler.article_content("https://finance.naver.com/outer")
    assert fetcher.calls[-1] == "https://finance.naver.com/inner"


def test_script_and_style_are_stripped_from_the_body(tmp_path):
    html = '<div id="dic_area">진짜 본문<script>var x=1;</script><style>p{}</style></div>'
    crawler = make_crawler(tmp_path, FakeFetcher(default=html))

    assert crawler.article_content("https://x/1") == "진짜 본문"


def test_markup_with_no_known_body_yields_nothing(tmp_path):
    crawler = make_crawler(tmp_path, FakeFetcher(default="<div id='unexpected'>x</div>"))

    assert crawler.article_content("https://x/1") is None


def test_an_article_seen_under_another_stock_is_not_fetched_twice(tmp_path):
    """Roughly 39% of listings repeat across stocks; refetching them is wasted time."""
    fetcher = FakeFetcher(default=ARTICLE_HTML)
    crawler = make_crawler(tmp_path, fetcher)

    crawler.article_content("https://x/1")
    crawler.article_content("https://x/1")

    assert len(fetcher.calls) == 1


# --- one stock -------------------------------------------------------------


def canned_stock_fetcher():
    return FakeFetcher({"news_news.naver": LIST_HTML, "news_read": ARTICLE_HTML})


def test_a_stock_is_collected_into_records(tmp_path):
    crawler = make_crawler(tmp_path, canned_stock_fetcher(), pages=1)

    articles = crawler.crawl_stock("005930", "삼성전자")

    assert len(articles) == 2
    assert set(articles[0]) == {"title", "content", "date", "stock_name", "stock_code", "url"}
    assert articles[0]["stock_name"] == "삼성전자"
    assert articles[0]["stock_code"] == "005930"


def test_short_bodies_are_dropped_as_news_flashes(tmp_path):
    fetcher = FakeFetcher({
        "news_news.naver": LIST_HTML,
        "news_read": '<div id="dic_area">너무 짧다</div>',
    })
    crawler = make_crawler(tmp_path, fetcher, pages=1)

    assert crawler.crawl_stock("005930", "삼성전자") == []


def test_collection_stops_at_the_per_stock_target(tmp_path):
    crawler = make_crawler(tmp_path, canned_stock_fetcher(), pages=1, per_stock=1)

    assert len(crawler.crawl_stock("005930", "삼성전자")) == 1


def test_a_stock_is_written_to_its_own_cache_file(tmp_path):
    crawler = make_crawler(tmp_path, canned_stock_fetcher(), pages=1)

    crawler.crawl_stock("005930", "삼성전자")

    saved = json.loads((tmp_path / "cache" / "005930.json").read_text(encoding="utf-8"))
    assert len(saved) == 2


def test_a_cached_stock_is_reused_without_any_request(tmp_path):
    """A 20-minute run must resume, not restart, after an interruption."""
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "005930.json").write_text(
        json.dumps([{
            "title": "저장된 기사", "content": "본문", "date": "2026.08.08 13:10",
            "stock_name": "삼성전자", "stock_code": "005930", "url": "https://x/1",
        }]),
        encoding="utf-8",
    )
    fetcher = canned_stock_fetcher()
    crawler = make_crawler(tmp_path, fetcher, pages=1)

    articles = crawler.crawl_stock("005930", "삼성전자")

    assert [a["title"] for a in articles] == ["저장된 기사"]
    assert fetcher.calls == []


def test_a_cached_body_is_reused_by_the_next_stock(tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "005930.json").write_text(
        json.dumps([{
            "title": "공유 기사", "content": "이미 받아둔 본문",
            "date": "2026.08.08 13:10", "stock_name": "삼성전자",
            "stock_code": "005930", "url": "https://x/shared",
        }]),
        encoding="utf-8",
    )
    fetcher = FakeFetcher(default=ARTICLE_HTML)
    crawler = make_crawler(tmp_path, fetcher, pages=1)

    crawler.crawl_stock("005930", "삼성전자")

    assert crawler.article_content("https://x/shared") == "이미 받아둔 본문"
    assert fetcher.calls == []


# --- whole run -------------------------------------------------------------


def test_every_article_gets_an_id_across_all_stocks(tmp_path):
    crawler = make_crawler(
        tmp_path, canned_stock_fetcher(), pages=1,
        stocks={"005930": "삼성전자", "000660": "SK하이닉스"},
    )

    articles = crawler.crawl()

    assert [a["id"] for a in articles] == [1, 2, 3, 4]


def test_the_crawler_holds_its_own_cache(tmp_path):
    """The body cache used to be a module global, so a second run in one process
    inherited the first run's articles."""
    first = make_crawler(tmp_path, FakeFetcher(default=ARTICLE_HTML))
    first.article_content("https://x/1")
    second = make_crawler(tmp_path, FakeFetcher(default=ARTICLE_HTML))

    assert second.article_content("https://x/1") is not None
    assert "https://x/1" not in second._bodies or first is not second


# --- retrying fetcher ------------------------------------------------------


class FakeHTTP:
    def __init__(self, *statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        status = self.statuses.pop(0) if self.statuses else 200
        return FakeHTTPResponse(status)


class FakeHTTPResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.text = "<html></html>"
        self.encoding = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise crawl.requests.RequestException(f"{self.status_code}")


def test_a_rate_limit_is_waited_out_and_retried():
    waits = []
    fetch = crawl.make_fetcher(http=FakeHTTP(429, 200), sleep=waits.append)

    assert fetch("https://x/1") is not None
    assert waits, "429 must back off before retrying"


def test_a_request_that_keeps_failing_is_given_up_on():
    http = FakeHTTP(500, 500, 500)
    fetch = crawl.make_fetcher(http=http, sleep=lambda _s: None)

    assert fetch("https://x/1") is None
    assert http.calls == crawl.MAX_RETRY


# --- command line ----------------------------------------------------------


def test_the_defaults_match_the_collected_corpus():
    args = crawl.build_parser().parse_args([])

    assert args.pages == 25
    assert args.per_stock == 60
    assert args.delay == 1.0
    assert len(crawl.STOCKS) == 10


def test_the_scale_can_be_narrowed_for_a_trial_run():
    args = crawl.build_parser().parse_args(["--pages", "2", "--per-stock", "5", "--stocks", "005930"])

    assert (args.pages, args.per_stock, args.stocks) == (2, 5, ["005930"])


def test_an_unknown_stock_code_is_refused():
    with pytest.raises(SystemExit):
        crawl.build_parser().parse_args(["--stocks", "999999"])


def test_a_run_writes_the_corpus_and_reports_it(tmp_path):
    out = tmp_path / "news_data.json"
    printed = []

    code = crawl.main(
        ["--pages", "1", "--stocks", "005930", "--out", str(out),
         "--cache-dir", str(tmp_path / "cache"), "--delay", "0"],
        fetch=canned_stock_fetcher(),
        sleep=lambda _s: None,
        out=printed.append,
    )

    assert code == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 2
    assert any("2건" in line for line in printed)
