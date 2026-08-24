"""Rebuild the blind candidate pool from the articles it pointed at.

`evaluation/pools/blind_pool_50.jsonl` held 2.4M characters of Naver Finance
article text, so it is not tracked. What the repository keeps instead is
`blind_pool_50_sources.json`: for each of the 371 candidate articles, its URL,
title, date, and the SHA-256 of the body it had when the evaluation was frozen.

    uv run python -m evaluation.rebuild_pool                 # URL에서 다시 받기 (6~7분)
    uv run python -m evaluation.rebuild_pool --from-corpus   # 정제본이 있으면 즉시

Refetching does not always return what was collected. Articles get edited: the
day after the freeze, 365 of 371 came back byte-identical and 6 had drifted. So
this checks each article against its own hash and reports which ones it could
not reproduce, rather than passing or failing the file as a whole.

The rebuilt pool is for reading the article a judgment was made about. The
metrics do not need it — `final_evaluation` computes those from the frozen
questions, the mapping and the judgments, none of which carry article text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = ROOT / "evaluation" / "pools" / "blind_pool_50_sources.json"
DEFAULT_FREEZE = ROOT / "evaluation" / "retrieval_eval_50.jsonl"
DEFAULT_MAPPING = ROOT / "evaluation" / "pools" / "blind_pool_50_mapping.json"
DEFAULT_MANIFEST = ROOT / "evaluation" / "pools" / "blind_pool_50_manifest.json"
DEFAULT_CORPUS = ROOT / "data" / "processed" / "news_data_clean.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "pools" / "blind_pool_50.jsonl"

ARTICLE_FIELDS = ("title", "date", "content", "url", "doc_type")
REQUEST_DELAY = 1.0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def make_fetcher():
    """URL → 정제된 본문. 수집 때와 같은 추출·정제 규칙을 그대로 쓴다."""
    from preprocessing.clean import clean_content
    from preprocessing.crawl import Crawler

    crawler = Crawler(pages=0)

    def fetch(url: str) -> str | None:
        raw = crawler.article_content(url)
        return clean_content(raw) if raw else None

    return fetch


def refetch(sources: dict, *, fetch, sleep=time.sleep, delay: float = REQUEST_DELAY, out=None):
    """각 기사를 다시 받고 기록된 해시와 대조한다.

    드리프트한 기사도 본문을 그대로 쓴다 — 후보가 통째로 빠지면 읽을 수 없는
    pool이 되고, 어디가 달라졌는지는 report가 알려준다.
    """
    bodies: dict[int, str] = {}
    report: dict[str, list[int]] = {"exact": [], "drifted": [], "dead": []}
    articles = sources["articles"]

    for index, (key, meta) in enumerate(articles.items(), 1):
        article_id = int(key)
        content = fetch(meta["url"])
        if content is None:
            report["dead"].append(article_id)
        else:
            bodies[article_id] = content
            bucket = "exact" if _sha(content) == meta["content_sha256"] else "drifted"
            report[bucket].append(article_id)
        if out and index % 50 == 0:
            out(f"  {index}/{len(articles)} · 일치 {len(report['exact'])} "
                f"드리프트 {len(report['drifted'])} 실패 {len(report['dead'])}")
        sleep(delay)
    return bodies, report


def build_articles(sources: dict, bodies: dict[int, str]) -> list[dict]:
    """sources의 메타데이터 + 받아온 본문 → 코퍼스와 같은 모양의 기사 레코드."""
    return [
        {
            "id": int(key),
            "title": meta["title"],
            "date": meta["date"],
            "content": bodies[int(key)],
            "url": meta["url"],
            "doc_type": meta["doc_type"],
        }
        for key, meta in sources["articles"].items()
        if int(key) in bodies
    ]


def rebuild_packets(records: list[dict], mappings: list[dict], articles: list[dict]) -> list[dict]:
    """질의당 packet 하나, 후보 순서는 mapping이 기록한 그대로.

    원래 순서는 pool을 만들 때의 seed 셔플에서 나왔다. 여기서 다시 계산하지
    않는다 — mapping이 이미 그 순서를 담고 있고, 재계산하면 셔플 구현에 묶인다.
    """
    by_id = {article["id"]: article for article in articles}
    by_query = {record["query_id"]: record for record in records}

    grouped: dict[str, list[dict]] = {}
    for item in mappings:
        grouped.setdefault(item["query_id"], []).append(item)

    packets = []
    for query_id, items in grouped.items():
        if query_id not in by_query:
            raise KeyError(f"query_id is not in the frozen question file: {query_id}")
        record = by_query[query_id]
        candidates = []
        for item in items:
            article_id = item["article_id"]
            if article_id not in by_id:
                raise KeyError(f"article body is not available: {article_id}")
            article = by_id[article_id]
            candidates.append({
                "candidate_key": item["candidate_key"],
                **{field: article[field] for field in ARTICLE_FIELDS},
            })
        packets.append({
            "query_id": query_id,
            "question": record["question"],
            "category": record["category"],
            "candidates": candidates,
        })
    return packets


def serialize(packets: list[dict]) -> bytes:
    """`batch_pooling.write_pool_artifacts`가 쓴 바이트 모양 그대로."""
    return "".join(
        json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n"
        for packet in packets
    ).encode()


def write_pool(records, mappings, articles, output_path) -> str:
    """pool을 쓰고 그 SHA-256을 돌려준다.

    기록된 해시와 대조하지 않는다 — 기사 한 건만 수정돼도 파일 해시는 어긋나는데,
    그때 쓰기를 거부하면 들여다볼 것이 없어진다. 대조는 호출 측이 기사별로 한다.
    """
    payload = serialize(rebuild_packets(records, mappings, articles))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bodies_from_corpus(corpus_path: Path, sources: dict) -> dict[int, str]:
    corpus = {a["id"]: a for a in json.load(corpus_path.open(encoding="utf-8"))}
    wanted = {int(k) for k in sources["articles"]}
    return {i: corpus[i]["content"] for i in wanted if i in corpus}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="blind pool을 기사 출처에서 다시 만듭니다.")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--from-corpus", action="store_true",
                        help="정제본이 이미 있으면 URL 수집 없이 거기서 본문을 가져옵니다")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="요청 간격(초)")
    return parser


def main(argv: list[str] | None = None, *, fetch=None, sleep=time.sleep, out=print) -> int:
    args = build_parser().parse_args(argv)
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    total = len(sources["articles"])

    if args.from_corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.is_file():
            out(f"{corpus_path} 이(가) 없습니다. --from-corpus 없이 실행하면 URL에서 받습니다.")
            return 1
        bodies = _bodies_from_corpus(corpus_path, sources)
        report = {"exact": [], "drifted": [], "dead": []}
        for key, meta in sources["articles"].items():
            aid = int(key)
            if aid not in bodies:
                report["dead"].append(aid)
            else:
                report["exact" if _sha(bodies[aid]) == meta["content_sha256"] else "drifted"].append(aid)
    else:
        out(f"기사 {total}건을 URL에서 다시 받습니다 (요청 간격 {args.delay}초)")
        bodies, report = refetch(sources, fetch=fetch or make_fetcher(),
                                 sleep=sleep, delay=args.delay, out=out)

    out(f"\n기사 {total}건 · 일치 {len(report['exact'])} · "
        f"드리프트 {len(report['drifted'])} · 못 받음 {len(report['dead'])}")
    for aid in report["drifted"][:20]:
        meta = sources["articles"][str(aid)]
        out(f"  드리프트 id={aid} {meta['title'][:40]} — 동결 당시와 본문이 다릅니다")
    for aid in report["dead"][:20]:
        out(f"  못 받음 id={aid} {sources['articles'][str(aid)]['url']}")

    if report["dead"]:
        out("\n받지 못한 기사가 있어 pool을 조립할 수 없습니다.")
        return 1

    digest = write_pool(_jsonl(Path(args.freeze)),
                        json.loads(Path(args.mapping).read_text(encoding="utf-8")),
                        build_articles(sources, bodies), args.output)
    expected = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected = expected["generated_file_sha256"]["blind_pool_sha256"]

    out(f"\n{args.output} 기록 완료")
    if digest == expected:
        out(f"  sha256 {digest}  (동결 당시와 동일)")
        return 0
    out(f"  sha256 {digest}")
    out(f"  동결 당시 {expected}")
    out("  드리프트한 기사 때문에 파일 해시가 다릅니다. 지표 계산은 이 파일을 읽지 않으므로")
    out("  evaluation/results/ 의 값은 영향받지 않습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
