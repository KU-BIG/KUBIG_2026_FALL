"""Normalize dense-search queries without changing their meaning."""

from __future__ import annotations

import re
import unicodedata


CANONICAL_STOCK_NAMES = (
    "삼성전자",
    "SK하이닉스",
    "현대차",
    "LG에너지솔루션",
    "삼성바이오로직스",
    "KB금융",
    "HD현대중공업",
    "한화에어로스페이스",
    "두산에너빌리티",
    "NAVER",
)

_STOCK_ALIASES = {
    "삼전": "삼성전자",
    "하이닉스": "SK하이닉스",
    "SK 하이닉스": "SK하이닉스",
    "에스케이하이닉스": "SK하이닉스",
    "현대자동차": "현대차",
    "LG엔솔": "LG에너지솔루션",
    "LG 엔솔": "LG에너지솔루션",
    "엘지엔솔": "LG에너지솔루션",
    "엘지에너지솔루션": "LG에너지솔루션",
    "삼바": "삼성바이오로직스",
    "KB금융지주": "KB금융",
    "KB 금융": "KB금융",
    "케이비금융": "KB금융",
    "현대중공업": "HD현대중공업",
    "HD현중": "HD현대중공업",
    "한화에어로": "한화에어로스페이스",
    "두산중공업": "두산에너빌리티",
    "네이버": "NAVER",
    "Naver": "NAVER",
}

_NORMALIZED_STOCK_NAMES = {
    **{name.casefold(): name for name in CANONICAL_STOCK_NAMES},
    **{alias.casefold(): canonical for alias, canonical in _STOCK_ALIASES.items()},
}
_STOCK_NAME_PATTERN = re.compile(
    "|".join(
        re.escape(name)
        for name in sorted((*CANONICAL_STOCK_NAMES, *_STOCK_ALIASES), key=len, reverse=True)
    ),
    flags=re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """Return an NFKC-normalized query with supported stock aliases canonicalized."""
    normalized = " ".join(unicodedata.normalize("NFKC", query).split())
    if not normalized:
        raise ValueError("query cannot be empty")

    return _STOCK_NAME_PATTERN.sub(
        lambda match: _NORMALIZED_STOCK_NAMES[match.group(0).casefold()],
        normalized,
    )
