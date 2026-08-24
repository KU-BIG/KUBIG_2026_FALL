"""Rebuilding the blind candidate pool from article URLs.

The pool held 2.4M characters of news text, so the repository keeps only where
each candidate came from: url, title, date, and a SHA-256 of the body it had
when the evaluation was frozen. The bodies are fetched back.

Refetching is not guaranteed to return what was collected — articles get
edited. Measured on the real pool the day after freezing, 365 of 371 came back
byte-identical and 6 had drifted, so the per-article hash is the point: a
rebuild has to say which articles it could not reproduce instead of reporting
one pass/fail for the whole file. Every fetch here is injected; nothing in this
file touches the network.
"""

import hashlib
import json

import pytest

from evaluation.rebuild_pool import (
    build_articles,
    rebuild_packets,
    refetch,
    serialize,
    write_pool,
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


SOURCES = {
    "articles": {
        "11": {"url": "https://x/11", "title": "첫 기사", "date": "2026.08.08",
               "doc_type": "article", "content_sha256": sha("본문 하나"), "content_chars": 5},
        "22": {"url": "https://x/22", "title": "둘째 기사", "date": "2026.08.07",
               "doc_type": "broadcast", "content_sha256": sha("본문 둘"), "content_chars": 4},
    }
}
RECORDS = [
    {"query_id": "K001", "question": "첫 질문", "category": "factoid"},
    {"query_id": "R001", "question": "둘째 질문", "category": "abstract"},
]
MAPPING = [
    {"query_id": "K001", "candidate_key": "aaa", "article_id": 22},
    {"query_id": "K001", "candidate_key": "bbb", "article_id": 11},
    {"query_id": "R001", "candidate_key": "ccc", "article_id": 11},
]
BODIES = {"https://x/11": "본문 하나", "https://x/22": "본문 둘"}


def fake_fetch(bodies=None):
    bodies = BODIES if bodies is None else bodies
    calls = []

    def fetch(url):
        calls.append(url)
        return bodies.get(url)

    fetch.calls = calls
    return fetch


# --- refetching ------------------------------------------------------------


def test_every_article_is_fetched_once():
    fetch = fake_fetch()

    bodies, report = refetch(SOURCES, fetch=fetch, sleep=lambda _s: None)

    assert sorted(fetch.calls) == ["https://x/11", "https://x/22"]
    assert bodies == {11: "본문 하나", 22: "본문 둘"}
    assert report["exact"] == [11, 22]
    assert report["drifted"] == [] and report["dead"] == []


def test_an_edited_article_is_reported_but_still_used():
    """Its body changed since the freeze; the candidate still has to exist."""
    fetch = fake_fetch({**BODIES, "https://x/22": "고쳐진 본문"})

    bodies, report = refetch(SOURCES, fetch=fetch, sleep=lambda _s: None)

    assert bodies[22] == "고쳐진 본문"
    assert report["drifted"] == [22]
    assert report["exact"] == [11]


def test_an_unreachable_article_is_reported_and_left_out():
    fetch = fake_fetch({"https://x/11": "본문 하나"})

    bodies, report = refetch(SOURCES, fetch=fetch, sleep=lambda _s: None)

    assert 22 not in bodies
    assert report["dead"] == [22]


def test_the_fetch_is_paced():
    waits = []
    refetch(SOURCES, fetch=fake_fetch(), sleep=waits.append, delay=1.0)

    assert waits == [1.0, 1.0]


# --- assembling ------------------------------------------------------------


def test_metadata_comes_from_the_sources_file_and_the_body_from_the_fetch():
    articles = build_articles(SOURCES, {11: "본문 하나", 22: "본문 둘"})

    assert articles[0] == {
        "id": 11, "title": "첫 기사", "date": "2026.08.08", "content": "본문 하나",
        "url": "https://x/11", "doc_type": "article",
    }


def test_each_candidate_takes_its_body_by_article_id():
    articles = build_articles(SOURCES, {11: "본문 하나", 22: "본문 둘"})

    packets = rebuild_packets(RECORDS, MAPPING, articles)

    assert packets[0]["candidates"][0] == {
        "candidate_key": "aaa", "title": "둘째 기사", "date": "2026.08.07",
        "content": "본문 둘", "url": "https://x/22", "doc_type": "broadcast",
    }


def test_candidate_and_query_order_follow_the_mapping():
    """The pool came from a seeded shuffle; the mapping is what preserves it."""
    articles = build_articles(SOURCES, {11: "본문 하나", 22: "본문 둘"})

    packets = rebuild_packets(RECORDS, MAPPING[2:] + MAPPING[:2], articles)

    assert [p["query_id"] for p in packets] == ["R001", "K001"]
    assert [c["candidate_key"] for c in packets[1]["candidates"]] == ["aaa", "bbb"]


def test_a_candidate_whose_article_could_not_be_fetched_is_named():
    articles = build_articles(SOURCES, {11: "본문 하나"})

    with pytest.raises(KeyError, match="22"):
        rebuild_packets(RECORDS, MAPPING, articles)


def test_a_query_missing_from_the_frozen_file_is_named():
    articles = build_articles(SOURCES, {11: "본문 하나", 22: "본문 둘"})

    with pytest.raises(KeyError, match="R001"):
        rebuild_packets(RECORDS[:1], MAPPING, articles)


# --- writing ---------------------------------------------------------------


def test_serialization_matches_the_writer_that_produced_the_pool():
    """batch_pooling wrote compact separators and no ASCII escaping."""
    articles = build_articles(SOURCES, {11: "본문 하나", 22: "본문 둘"})

    payload = serialize(rebuild_packets(RECORDS, MAPPING, articles))

    assert payload.endswith(b"\n")
    assert "첫 질문".encode() in payload
    assert b'", "' not in payload


def test_the_pool_is_written_and_its_digest_returned(tmp_path):
    out = tmp_path / "pool.jsonl"
    articles = build_articles(SOURCES, {11: "본문 하나", 22: "본문 둘"})

    digest = write_pool(RECORDS, MAPPING, articles, out)

    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()


def test_a_drifted_body_is_written_rather_than_refused(tmp_path):
    """A drifted article changes the file hash. Refusing to write would leave
    nothing to inspect, and the metrics do not read this file anyway."""
    out = tmp_path / "pool.jsonl"
    intact = write_pool(RECORDS, MAPPING, build_articles(SOURCES, {11: "본문 하나", 22: "본문 둘"}), out)

    drifted = write_pool(RECORDS, MAPPING, build_articles(SOURCES, {11: "다른 본문", 22: "본문 둘"}), out)

    assert out.exists()
    assert drifted != intact


def test_the_written_pool_is_valid_jsonl(tmp_path):
    out = tmp_path / "pool.jsonl"
    articles = build_articles(SOURCES, {11: "본문 하나", 22: "본문 둘"})
    write_pool(RECORDS, MAPPING, articles, out)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]

    assert {r["query_id"] for r in rows} == {"K001", "R001"}
