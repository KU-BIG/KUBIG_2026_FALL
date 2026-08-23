# -*- coding: utf-8 -*-
"""
뉴스 데이터 정제 (1주차 3번)

data/raw/news_data.json (원본, 수정하지 않음)
  -> data/processed/news_data_clean.json  (정제 결과)
  -> data/processed/clean_report.json     (규칙별 적용 건수)

실행: uv run python preprocessing/clean.py

정제 범위
  1) 꼬리 boilerplate 제거 : 제보 안내, 기자 이메일 서명, 저작권, 방송 크레딧
  2) 짧은 기사 필터
  3) 중복 기사 병합        : 종목 정보를 리스트로 보존

앞머리(부제목/캡션) 제거는 하지 않는다.
  - 첫 청크 오염도가 0.6%에 그쳐 임베딩에 유의미한 영향이 없고,
  - 부제목은 기자가 쓴 요약문이라 오히려 검색에 유용하기 때문.

본문 중간 사진 캡션("<캡션문장>. [사진=매체]")도 건드리지 않는다.
  - 캡션 문장이 마침표로 끝나 마커만 떨어져 나갈 뿐, 정작 지우려던
    "삼성전자 서초사옥 전경."은 본문에 그대로 남아 효과가 없었다.
  - 전체 본문의 0.25%(1,140/459,448자)로 앞머리보다도 비중이 작다.
  - "/사진=매체" 형태는 매체명 끝을 특정할 수 없어 뒤 단어를 깎아먹었다
    ("...고승민코스피가" -> "...스피가").
"""

import json
import re
import hashlib
from collections import Counter, OrderedDict
from pathlib import Path

# 데이터는 모두 프로젝트 루트의 data/ 아래에 둔다.
# 실행 위치와 무관하게 찾도록 이 파일(preprocessing/clean.py) 기준 경로를 쓴다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SRC = RAW_DIR / "news_data.json"
DST = PROCESSED_DIR / "news_data_clean.json"
REPORT = PROCESSED_DIR / "clean_report.json"

MIN_LENGTH = 200   # 정제 후 이 길이 미만이면 탈락
TAIL_ZONE = 400    # 꼬리 마커를 이 범위 안에서만 탐색 (본문 중간 이메일 오탐 방지)

stats = Counter()


# ---------------------------------------------------------------- 꼬리 제거
# 아래 마커가 나타나면 "그 지점부터 끝까지" 버린다.
# 꼬리에는 마커가 줄줄이 이어지므로, 순차 치환이 아니라 '가장 이른 지점'을 찾는다.
TAIL_MARKERS = [
    ("jebo",         r"[^.]{0,25}제보(가|를|로)[^.]{0,20}(됩니다|만들어집니다|기다리|바랍니다|함께)"),
    ("jebo_label",   r"[▶※]\s*(카카오톡|이메일|전화|뉴스 홈페이지)\s*[:：]"),
    ("resale",       r"\*?\s*재판매 및 DB\s*금지"),
    ("byline_mail",  r"[가-힣]{2,4}(\s*[가-힣]{2,6})?\s*기자\s*[\w.\-]+@[\w.\-]+\.[a-zA-Z]{2,}"),
    ("byline_box",   r"\[[^\]]{0,30}기자[^\]]{0,10}[\w.\-]+@[\w.\-]+\.[a-zA-Z]{2,}[^\]]{0,5}\]"),
    ("magazine",     r"\[본\s*기사는[^\]]{0,60}기사입니다\]"),
    ("copyright",    r"\[?Copyright\s*\(c\)|저작권자\s*[ⓒ©]|무단\s*전재|재배포\s*금지"),
    ("video_credit", r"영상(기자|취재|편집|디자인)\s*[:：]"),
    ("bare_mail",    r"\s*[\w.\-]+@[\w.\-]+\.[a-zA-Z]{2,}\s*$"),
    ("promo_url",    r"(사이트\s*[:：]\s*)?https?://\S+\s*$"),
    ("photo_credit", r"[^.]{0,40}(제공|촬영)\.?\s*$"),
]
TAIL_COMPILED = [(name, re.compile(p)) for name, p in TAIL_MARKERS]


def strip_tail(text):
    zone_start = max(0, len(text) - TAIL_ZONE)
    cut = len(text)
    hits = []
    for name, rx in TAIL_COMPILED:
        m = rx.search(text, zone_start)
        if m is None or m.start() > cut:
            continue
        if m.start() < cut:   # 더 이른 지점을 찾았으면 집계를 다시 시작한다
            cut = m.start()
            hits = []
        hits.append(name)     # 같은 지점에서 겹친 마커는 함께 센다
    for name in hits:
        stats[f"tail:{name}"] += 1
    return text[:cut].strip()


# ---------------------------------------------------------------- 공통 정리
def normalize(text):
    text = re.sub(r"[​﻿\xad]", "", text)   # 제로폭/BOM
    return re.sub(r"\s+", " ", text).strip()


def clean_content(text):
    # strip_tail은 잘라내기만 하므로 정규화는 앞에서 한 번이면 된다.
    return strip_tail(normalize(text))


# ------------------------------------------------------------------ 중복 병합
def dup_key(text):
    """공백/기호를 무시한 본문 해시. 완전일치만 병합한다.

    근접중복(자카드 0.35↑)은 184건 중 1건뿐인 반면, 제목이 같아도 내용이
    다른 연재물([서울데이터랩] 등)이 있어 제목 기준 병합은 오히려 위험하다.
    """
    return hashlib.md5(re.sub(r"[^가-힣a-zA-Z0-9]", "", text).encode()).hexdigest()


def merge_group(group):
    base = max(group, key=lambda a: len(a["title"]))   # 덜 잘린 제목 우선
    names, codes = OrderedDict(), OrderedDict()
    for a in group:
        names[a["stock_name"]] = None
        codes[a["stock_code"]] = None
    return {
        "title":       base["title"],
        "content":     base["_clean"],
        "date":        base["date"],
        "url":         base["url"],
        "stock_names": list(names),
        "stock_codes": list(codes),
        "source_ids":  sorted(a["id"] for a in group),
        "doc_type":    base["_type"],
    }


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SRC, encoding="utf-8") as f:
        raw = json.load(f)
    stats["input"] = len(raw)

    for a in raw:
        a["_type"] = "broadcast" if re.search(r"\[앵커\]|\[기자\]", a["content"]) else "article"
        a["_clean"] = clean_content(a["content"])

    kept = []
    for a in raw:
        if len(a["_clean"]) < MIN_LENGTH:
            stats["dropped_short"] += 1
        else:
            kept.append(a)

    groups = OrderedDict()
    for a in kept:
        groups.setdefault(dup_key(a["_clean"]), []).append(a)
    stats["merged_away"] = len(kept) - len(groups)

    docs = [merge_group(g) for g in groups.values()]
    for i, d in enumerate(docs, 1):
        d["id"] = i
    stats["output"] = len(docs)

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, ensure_ascii=False, indent=2)

    print(f"{stats['input']}건 -> {stats['output']}건")
    print(f"  짧아서 탈락 {stats['dropped_short']}건 / 중복 병합 {stats['merged_away']}건")
    for k in sorted(stats):
        if k.startswith("tail:"):
            print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
