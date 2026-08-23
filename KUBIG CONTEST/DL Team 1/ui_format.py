"""Pure formatting helpers for the Streamlit source cards.

Kept out of `app.py` so they can be tested directly — importing the app module
would execute the whole Streamlit script.
"""

from __future__ import annotations


def _escape_link_text(text: str) -> str:
    """Escape brackets so a title cannot terminate its own markdown link.

    Korean news titles routinely open with a bracketed section marker —
    "[마감시황]", "[칩톡]", "[서울데이터랩]". Dropped into `[title](url)` as-is,
    the first `]` closes the link text and the rest renders as literal junk next
    to a bare URL.
    """
    return text.replace("[", "\\[").replace("]", "\\]")


def source_heading(rank: int, chunk: dict) -> str:
    """Numbered title, linked to the article when there is a URL."""
    title = chunk.get("title") or "(제목 없음)"
    url = chunk.get("url")
    if not url:
        return f"**{rank}. {title}**"
    return f"**{rank}. [{_escape_link_text(title)}]({url})**"


def meta_line(chunk: dict) -> str:
    """Date, tagged stocks, and whether the chunk is a broadcast transcript."""
    bits = []
    if chunk.get("date"):
        bits.append(chunk["date"])
    if chunk.get("stock_names"):
        bits.append(" · ".join(chunk["stock_names"]))
    if chunk.get("doc_type") == "broadcast":
        bits.append("방송")
    return " | ".join(bits)


def provenance_line(chunk: dict) -> str:
    """Why this chunk surfaced: which query, which retriever, what score."""
    parts = []

    if chunk.get("matched_queries") is not None:
        matched = chunk["matched_queries"]
        queries = chunk.get("expanded_queries") or []
        marks = " ".join(
            f"q{i}" if i in matched else "·" for i in range(len(queries))
        )
        if marks:
            parts.append(f"질의 {marks}")

    if chunk.get("rrf_score") is not None:
        parts.append(f"RRF {chunk['rrf_score']:.4f}")
        dense_rank = chunk.get("dense_rank")
        bm25_rank = chunk.get("bm25_rank")
        parts.append(f"dense #{dense_rank}" if dense_rank else "dense —")
        parts.append(f"bm25 #{bm25_rank}" if bm25_rank else "bm25 —")

    if chunk.get("similarity") is not None:
        parts.append(f"유사도 {chunk['similarity']:.4f}")
    if chunk.get("bm25_score") is not None:
        parts.append(f"BM25 {chunk['bm25_score']:.2f}")
    if chunk.get("cited_as"):
        parts.append("인용 " + ", ".join(f"[뉴스{n}]" for n in chunk["cited_as"]))

    return " · ".join(parts)
