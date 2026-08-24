"""Korean tokenizer for BM25.

BM25 matches whole tokens, and Korean is agglutinative: splitting on whitespace
leaves "삼성전자의" in the document and "삼성전자" in the query, so the two never
match and BM25 contributes nothing. Kiwi separates the 조사 and returns the stem,
which is what makes lexical retrieval work at all here.

Kiwi ships as a pure wheel (no Java, no system libraries), so it stays inside the
`uv.lock` reproducibility guarantee the rest of the project relies on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiwipiepy import Kiwi

# Content-bearing parts of speech. `SL` matters as much as the noun tags here:
# NAVER, SK, HBM, AI are all tagged SL, and dropping them would blind BM25 to the
# ticker and technology terms these questions are mostly about. `SN` keeps stock
# codes (005930) and index levels searchable.
KEEP_TAG_PREFIXES = (
    "NNG",  # 일반명사
    "NNP",  # 고유명사
    "SL",  # 외국어
    "SN",  # 숫자
    "SH",  # 한자
    "VV",  # 동사 어간 (올랐다 -> 오르)
    "VA",  # 형용사 어간
    "XR",  # 어근
)

_TOKENIZER: Kiwi | None = None


def get_tokenizer() -> Kiwi:
    """Reuse one Kiwi instance — construction costs ~0.5s."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from kiwipiepy import Kiwi

        _TOKENIZER = Kiwi()
    return _TOKENIZER


def tokenize(text: str) -> list[str]:
    """Split Korean text into lowercased content tokens for BM25."""
    if not text or not text.strip():
        return []
    return [
        token.form.lower()
        for token in get_tokenizer().tokenize(text)
        if token.tag.startswith(KEEP_TAG_PREFIXES)
    ]
