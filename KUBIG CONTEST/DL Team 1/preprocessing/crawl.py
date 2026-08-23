"""네이버 금융 종목 뉴스 크롤러.

종목별 뉴스 목록 → 개별 기사 본문 → `data/raw/news_data.json`.

    uv run python preprocessing/crawl.py                      # 전체 수집 (20~30분)
    uv run python preprocessing/crawl.py --stocks 005930 --pages 2 --per-stock 5

같은 코드를 `crawler.ipynb`에서도 씁니다. 노트북은 이 모듈을 불러 쓰기만 하므로
둘이 어긋날 일이 없습니다.

수집 규모와 그 이유
- 목록 25페이지 — 1차 수집이 8/4~8/7 나흘치에 그쳐 기간을 넓혔습니다.
- 종목당 60건 — 정제 단계에서 중복이 39% 걸러지므로 여유분이 필요합니다.
- 종목별 중간 저장 — 20~30분짜리 작업이라 끊겨도 이어받아야 합니다.
- 요청 실패 재시도 — 요청량이 5배로 늘어 타임아웃과 일시 차단 가능성이 커졌습니다.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://finance.naver.com/item/main.naver",
}
BASE_URL = "https://finance.naver.com"
REQUEST_DELAY = 1.0  # 서버 부담을 줄이기 위한 요청 간격(초)
LIST_PAGES = 25  # 종목당 목록 페이지 수 (뒤로 갈수록 과거 기사)
ARTICLES_PER_STOCK = 60  # 종목당 목표 수집 건수
MIN_CONTENT_LENGTH = 100  # 이보다 짧으면 단신으로 간주하고 제외
MAX_RETRY = 3  # 요청 실패 시 재시도 횟수
RETRY_STATUSES = (429, 503)  # 잠시 뒤 풀리는 경우가 많아 기다렸다 다시 시도

# 확정된 10개 종목 (코스피200 상위, 섹터 중복 제외)
STOCKS = {
    "005930": "삼성전자",           # 반도체(메모리)
    "000660": "SK하이닉스",         # 반도체(메모리)
    "005380": "현대차",             # 자동차
    "373220": "LG에너지솔루션",     # 2차전지
    "207940": "삼성바이오로직스",   # 바이오/CDMO
    "105560": "KB금융",             # 금융(은행)
    "329180": "HD현대중공업",       # 조선
    "012450": "한화에어로스페이스", # 방산
    "034020": "두산에너빌리티",     # 원전/발전
    "035420": "NAVER",              # 인터넷 플랫폼
}


def find_project_root(start: Path | str | None = None) -> Path:
    """`pyproject.toml`이 있는 상위 폴더를 프로젝트 루트로 봅니다.

    노트북에는 `__file__`이 없고 커널의 작업 디렉터리는 노트북을 어디서 열었는지에
    따라 달라집니다. 루트에서 열든 `preprocessing/`에서 열든 같은 `data/`를 쓰도록
    루트를 직접 찾습니다.
    """
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


PROJECT_ROOT = find_project_root(Path(__file__).parent)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "crawl_cache"  # 종목별 중간 저장 위치
OUT_PATH = RAW_DIR / "news_data.json"  # 수집 산출물


def clean_text(text: str) -> str:
    """공백과 개행을 하나로 정리합니다."""
    return re.sub(r"\s+", " ", text).strip()


def make_fetcher(http: Any = requests, sleep=time.sleep, delay: float = REQUEST_DELAY):
    """재시도가 붙은 GET을 만듭니다. 끝내 실패하면 `None`을 돌려줍니다.

    한 종목이 실패해도 전체 수집이 멈추면 안 되므로 예외 대신 `None`을 쓰고,
    호출 측이 그 항목만 건너뜁니다.
    """

    def fetch(url: str, encoding: str = "euc-kr"):
        for attempt in range(1, MAX_RETRY + 1):
            try:
                response = http.get(url, headers=HEADERS, timeout=15)
                if response.status_code in RETRY_STATUSES:
                    wait = delay * (2 ** attempt)
                    sleep(wait)
                    continue
                response.raise_for_status()
                response.encoding = encoding
                return response
            except requests.RequestException:
                if attempt == MAX_RETRY:
                    return None
                sleep(delay * (2 ** attempt))
        return None

    return fetch


class Crawler:
    """한 번의 수집 실행.

    본문 캐시를 인스턴스가 들고 있습니다. 같은 기사가 여러 종목에 걸쳐 나오므로
    (1차 수집 기준 39%) URL 하나당 한 번만 받습니다.
    """

    def __init__(
        self,
        stocks: dict[str, str] | None = None,
        *,
        pages: int = LIST_PAGES,
        per_stock: int = ARTICLES_PER_STOCK,
        cache_dir: Path | str = CACHE_DIR,
        delay: float = REQUEST_DELAY,
        fetch=None,
        sleep=time.sleep,
        out=print,
    ) -> None:
        self.stocks = stocks if stocks is not None else dict(STOCKS)
        self.pages = pages
        self.per_stock = per_stock
        self.cache_dir = Path(cache_dir)
        self.delay = delay
        self._fetch = fetch or make_fetcher(sleep=sleep, delay=delay)
        self._sleep = sleep
        self._out = out
        self._bodies: dict[str, str | None] = {}

    # --- 목록 ---------------------------------------------------------

    def news_list(self, code: str) -> list[dict]:
        """종목 뉴스 목록에서 (제목, 링크, 날짜)를 모읍니다."""
        links: list[dict] = []
        seen: set[str] = set()  # 페이지 사이에 같은 기사가 반복된다
        for page in range(1, self.pages + 1):
            response = self._fetch(f"{BASE_URL}/item/news_news.naver?code={code}&page={page}")
            if response is None:
                continue
            soup = BeautifulSoup(response.text, "html.parser")

            # 목록은 table.type5 안의 tr. 구조가 바뀌면 여기부터 확인한다.
            rows = soup.select("table.type5 tbody tr")
            if not rows:
                break  # 마지막 페이지를 지나면 빈 목록이 온다

            for row in rows:
                title_tag = row.select_one("td.title a")
                date_tag = row.select_one("td.date")
                if not title_tag:
                    continue
                url = self._absolute(title_tag.get("href", ""))
                if not url or url in seen:
                    continue
                seen.add(url)
                links.append({
                    "title": clean_text(title_tag.get_text()),
                    "url": url,
                    "date": clean_text(date_tag.get_text()) if date_tag else None,
                })

            self._sleep(self.delay)
        return links

    @staticmethod
    def _absolute(href: Any) -> str:
        """Site-relative hrefs to absolute. bs4 types attributes as possibly
        multi-valued, so the value is normalised to a string first."""
        value = "" if href is None else str(href)
        return f"{BASE_URL}{value}" if value.startswith("/") else value

    # --- 본문 ---------------------------------------------------------

    def article_content(self, url: str) -> str | None:
        """기사 페이지에서 본문 텍스트를 뽑습니다."""
        if url in self._bodies:  # 다른 종목에서 이미 받아둔 기사
            return self._bodies[url]

        response = self._fetch(url)
        if response is None:
            return None

        # news_read.naver는 본문 없이 JS로 n.news.naver.com에 넘길 뿐이다
        redirect = re.search(r"location\.href\s*=\s*'([^']+)'", response.text)
        if redirect:
            response = self._fetch(redirect.group(1), encoding="utf-8")
            if response is None:
                return None

        soup = BeautifulSoup(response.text, "html.parser")

        # iframe으로 본문을 싣는 구버전도 대비한다
        iframe = soup.select_one("iframe#news_read_iframe, iframe#mainFrame")
        if iframe and iframe.get("src"):
            response = self._fetch(self._absolute(iframe["src"]), encoding="utf-8")
            if response is None:
                return None
            soup = BeautifulSoup(response.text, "html.parser")

        # 네이버뉴스 통합형 / 금융뉴스 자체 페이지
        body = (
            soup.select_one("#dic_area")
            or soup.select_one("#articleBodyContents")
            or soup.select_one("#content")
        )
        if not body:
            self._bodies[url] = None
            return None

        for tag in body.select("script, style"):
            tag.decompose()

        content = clean_text(body.get_text())
        self._bodies[url] = content
        return content

    # --- 종목 / 전체 ---------------------------------------------------

    def crawl_stock(self, code: str, name: str) -> list[dict]:
        """종목 하나를 수집합니다. 캐시가 있으면 요청 없이 그대로 씁니다."""
        cache_path = self.cache_dir / f"{code}.json"
        if cache_path.is_file():
            done = json.loads(cache_path.read_text(encoding="utf-8"))
            for article in done:  # 캐시된 본문도 재사용 대상에 넣는다
                self._bodies.setdefault(article["url"], article["content"])
            self._out(f"[{name}] 이미 수집됨 → {len(done)}건 (건너뜀)")
            return done

        self._out(f"[{name}] 뉴스 목록 수집 중…")
        listed = self.news_list(code)
        self._out(f"  목록 {len(listed)}건 확보, 본문 수집 시작")

        articles = []
        for item in listed:
            if len(articles) >= self.per_stock:
                break
            if not item["url"]:
                continue

            already_had = item["url"] in self._bodies
            content = self.article_content(item["url"])
            if not content or len(content) < MIN_CONTENT_LENGTH:
                continue

            articles.append({
                "title": item["title"],
                "content": content,
                "date": item["date"],
                "stock_name": name,
                "stock_code": code,
                "url": item["url"],
            })
            if not already_had:  # 캐시에서 꺼냈으면 요청이 없었으니 쉴 필요도 없다
                self._sleep(self.delay)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._out(f"  → {len(articles)}건 수집")
        return articles

    def crawl(self) -> list[dict]:
        articles: list[dict] = []
        for code, name in self.stocks.items():
            articles.extend(self.crawl_stock(code, name))
        for index, article in enumerate(articles, 1):  # id는 마지막에 한 번에 매긴다
            article["id"] = index
        return articles


def summarize(articles: list[dict], out=print) -> None:
    """수집 결과를 한눈에 보여줍니다."""
    dates = sorted(a["date"][:10] for a in articles if a.get("date"))
    if dates:
        out(f"수집 기간: {dates[0]} ~ {dates[-1]}")
    out(f"고유 URL: {len({a['url'] for a in articles})}건")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="네이버 금융 종목 뉴스를 수집해 data/raw/news_data.json에 저장합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stocks", nargs="+", choices=sorted(STOCKS), metavar="CODE",
        help=f"수집할 종목 코드 (기본: 전체 {len(STOCKS)}개)",
    )
    parser.add_argument("--pages", type=int, default=LIST_PAGES, help="종목당 목록 페이지 수")
    parser.add_argument("--per-stock", type=int, default=ARTICLES_PER_STOCK, help="종목당 목표 건수")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="요청 간격(초)")
    parser.add_argument("--out", default=str(OUT_PATH), help="산출물 경로")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help="종목별 중간 저장 위치")
    return parser


def main(argv: list[str] | None = None, *, fetch=None, sleep=time.sleep, out=print) -> int:
    args = build_parser().parse_args(argv)
    stocks = {code: STOCKS[code] for code in args.stocks} if args.stocks else dict(STOCKS)

    crawler = Crawler(
        stocks, pages=args.pages, per_stock=args.per_stock, cache_dir=args.cache_dir,
        delay=args.delay, fetch=fetch, sleep=sleep, out=out,
    )
    articles = crawler.crawl()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")

    out(f"\n총 {len(articles)}건 저장 완료 → {out_path}")
    summarize(articles, out)
    out("\n다음: uv run python preprocessing/clean.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
