"""Ask the Korean financial-news index a question from the terminal.

The same pipeline the Streamlit app drives, with every retrieval technique
reachable from a flag instead of a widget:

    uv run python cli.py "삼성전자 반도체 실적 전망은?"
    uv run python cli.py "005930 주가" --mode bm25
    uv run python cli.py "HBM 수요" --mode hybrid --expand hyde
    uv run python cli.py "코스피 시황" --retrieval-only      # API 크레딧 안 씀
    uv run python cli.py --chat --gate                      # 후속 질문 가능

One command can run several techniques over the same question and print them
side by side. `--compare` names the combinations outright; `--mode`/`--expand`
take comma-separated lists and sweep every pairing:

    uv run python cli.py "HBM 수요" --compare dense,hybrid,dense+hyde,dense+multi_query
    uv run python cli.py "HBM 수요" --mode dense,bm25,hybrid --retrieval-only
    uv run python cli.py "HBM 수요" --mode hybrid --expand none,multi_query,hyde

`--retrieval-only` needs no API key at all, which makes it the cheap way to tune
retrieval; everything else streams the answer as Claude writes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product

from indexing.hybrid import DEFAULT_CANDIDATE_K, DEFAULT_RRF_K
from rag import DEFAULT_TOP_K, RAGPipeline
from ui_format import meta_line, provenance_line

MODES = ("dense", "bm25", "hybrid")
EXPANDERS = ("none", "multi_query", "hyde")
EXIT_WORDS = {"exit", "quit", "q", "종료", "나가기"}
SNIPPET_CHARS = 160
TITLE_CHARS = 46
RULE = "─" * 60
HEAVY_RULE = "═" * 60


@dataclass(frozen=True)
class Variant:
    """One retrieval configuration to run the question through."""

    mode: str
    expand: str

    @property
    def label(self) -> str:
        return self.mode if self.expand == "none" else f"{self.mode}+{self.expand}"


def _split(raw: str, allowed: tuple[str, ...], flag: str) -> list[str]:
    values = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(f"unknown {flag}: {value} (choose from {', '.join(allowed)})")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{flag} cannot be empty")
    return values


def parse_variants(modes: str, expands: str) -> list[Variant]:
    """Every pairing of the requested modes and expanders, in the order given."""
    return [
        Variant(mode, expand)
        for mode, expand in product(
            _split(modes, MODES, "--mode"), _split(expands, EXPANDERS, "--expand")
        )
    ]


def parse_variant_label(label: str) -> Variant:
    """`dense`, `hybrid+hyde` — the same spelling the comparison table prints."""
    parts = label.strip().split("+")
    if len(parts) > 2:
        raise ValueError(f"unknown variant: {label} (expected mode or mode+expand)")
    mode = _split(parts[0], MODES, "variant mode")[0]
    expand = _split(parts[1], EXPANDERS, "variant expander")[0] if len(parts) == 2 else "none"
    return Variant(mode, expand)


def parse_variant_list(raw: str) -> list[Variant]:
    variants = []
    for label in raw.split(","):
        if not label.strip():
            continue
        variant = parse_variant_label(label)
        if variant not in variants:
            variants.append(variant)
    if not variants:
        raise ValueError("--compare cannot be empty")
    return variants


class Loaders:
    """Builds the heavy pieces once, on first use.

    Injected rather than imported at the point of use so the CLI can be wired
    and tested without loading a 4.5GB embedding model or holding an API key.
    Sharing one instance across variants is what keeps a three-way comparison
    from loading BGE-M3 three times.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._dense = None
        self._bm25 = None
        self._llm = None

    def dense(self):
        if self._dense is None:
            from indexing.retriever import get_retriever

            self._dense = get_retriever(device=self.device)
        return self._dense

    def bm25(self):
        if self._bm25 is None:
            from indexing.bm25 import get_bm25_retriever

            self._bm25 = get_bm25_retriever()
        return self._bm25

    def llm(self):
        if self._llm is None:
            from generation.llm import get_llm

            self._llm = get_llm()
        return self._llm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="한국 금융 뉴스 RAG — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="?", help="질문 (--chat 이면 생략)")
    parser.add_argument("--chat", action="store_true", help="후속 질문이 가능한 대화 모드")
    parser.add_argument(
        "--mode",
        default="hybrid",
        metavar="LIST",
        help=f"검색 방식, 쉼표로 여러 개 ({'/'.join(MODES)}, 기본 hybrid)",
    )
    parser.add_argument(
        "--expand",
        default="none",
        metavar="LIST",
        help=f"질의 확장, 쉼표로 여러 개 ({'/'.join(EXPANDERS)}, 기본 none)",
    )
    parser.add_argument(
        "--compare",
        metavar="LIST",
        help="비교할 조합을 직접 지정 (예: dense,hybrid,dense+hyde,dense+multi_query). "
        "주면 --mode/--expand 대신 이 목록을 씁니다",
    )
    parser.add_argument("--gate", action="store_true", help="LLM이 검색 필요 여부를 판단")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="검색만 하고 Claude는 호출하지 않음"
    )
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")

    args = parser.parse_args(argv)
    try:
        args.variants = (
            parse_variant_list(args.compare)
            if args.compare
            else parse_variants(args.mode, args.expand)
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not args.chat and not args.question:
        parser.error("question is required unless --chat is given")
    if args.chat and args.retrieval_only:
        # 대화 모드의 핵심은 직전 답변을 기억하는 것인데, 답변을 만들지 않으면
        # 기억할 것도 없다. 검색만 볼 거라면 단발 질문으로 충분하다.
        parser.error("--retrieval-only is a single-question mode; drop --chat")
    if args.chat and len(args.variants) > 1:
        # 변형마다 답이 다르면 어느 답을 이력에 남길지 정할 수 없다.
        parser.error("--chat runs one variant at a time; name a single variant")
    return args


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    from indexing.retriever import detect_device

    return detect_device()


def build_retriever(variant: Variant, args: argparse.Namespace, loaders: Loaders):
    """Assemble the retrieval stack this variant describes."""
    if variant.mode == "dense":
        base = loaders.dense()
    elif variant.mode == "bm25":
        base = loaders.bm25()
    else:
        from indexing.hybrid import HybridRetriever

        base = HybridRetriever(
            dense=loaders.dense(),
            bm25=loaders.bm25(),
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )

    if variant.expand == "none":
        # 확장을 안 쓰면 LLM을 만들지 않는다 — --retrieval-only가 API 키 없이 도는 이유.
        return base

    from indexing.expanding import ExpandingRetriever
    from indexing.query_expand import get_expander

    return ExpandingRetriever(
        base=base,
        expander=get_expander(variant.expand, llm=loaders.llm()),
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )


def build_pipeline(
    variant: Variant, args: argparse.Namespace, loaders: Loaders
) -> RAGPipeline:
    gate = None
    if args.gate:
        from generation.router import SearchGate

        gate = SearchGate(llm=loaders.llm())
    return RAGPipeline(
        retriever=build_retriever(variant, args, loaders), llm=loaders.llm(), gate=gate
    )


def format_source(rank: int, source: dict) -> str:
    """One retrieved chunk as an indented block: title, meta, why, link, snippet."""
    lines = [f"  [{rank}] {source.get('title') or '(제목 없음)'}"]
    for line in (meta_line(source), provenance_line(source), source.get("url") or ""):
        if line:
            lines.append(f"      {line}")
    content = " ".join((source.get("content") or "").split())
    if content:
        lines.append(f"      {content[:SNIPPET_CHARS]}…")
    return "\n".join(lines)


def write_sources(sources: list[dict], out) -> None:
    if not sources:
        out.write("출처 없음 (검색을 건너뛰었거나 결과가 없습니다)\n")
        return
    out.write(f"출처 {len(sources)}건\n")
    for rank, source in enumerate(sources, 1):
        out.write(format_source(rank, source) + "\n")


def _article_key(source: dict, fallback: str) -> object:
    key = source.get("article_id")
    if key is not None:
        return key
    return source.get("title") or fallback


def comparison_table(runs: list[tuple[Variant, list[dict]]]) -> str:
    """Which articles each variant surfaced, and at what rank.

    The point of running several techniques over one question is to see where
    they disagree, which a stack of separate result lists does not show. Rows
    every variant found sit at the top; the interesting ones are the gaps.
    """
    ranks: dict[object, dict[str, int]] = {}
    titles: dict[object, str] = {}
    order: list[object] = []

    for variant, sources in runs:
        for rank, source in enumerate(sources, 1):
            key = _article_key(source, f"{variant.label}#{rank}")
            if key not in ranks:
                ranks[key] = {}
                titles[key] = source.get("title") or "(제목 없음)"
                order.append(key)
            ranks[key].setdefault(variant.label, rank)

    labels = [variant.label for variant, _ in runs]
    widths = [max(len(label), 3) for label in labels]
    lines = [
        "변형별 순위 (· = 그 변형은 못 찾음)",
        "  ".join(label.rjust(w) for label, w in zip(labels, widths)) + "  기사",
    ]
    for key in sorted(
        order, key=lambda k: (-len(ranks[k]), min(ranks[k].values()), order.index(k))
    ):
        cells = [
            (str(ranks[key][label]) if label in ranks[key] else "·").rjust(width)
            for label, width in zip(labels, widths)
        ]
        title = titles[key]
        if len(title) > TITLE_CHARS:
            title = title[: TITLE_CHARS - 1] + "…"
        lines.append("  ".join(cells) + "  " + title)
    return "\n".join(lines)


def render_answer(response, out) -> str:
    """Sources first — retrieval is already done — then the text as it arrives."""
    write_sources(response.sources, out)
    out.write(RULE + "\n")
    pieces = []
    for piece in response.stream:
        out.write(piece)
        out.flush()
        pieces.append(piece)
    out.write("\n")
    return "".join(pieces)


def run_once(
    pipeline,
    question: str,
    args: argparse.Namespace,
    out=None,
    history: list[dict] | None = None,
) -> str:
    """Ask one question. Returns the answer so a chat turn can remember it."""
    out = out or sys.stdout
    response = pipeline.ask_stream(question, top_k=args.top_k, history=history)
    if args.json:
        # 대화는 턴마다 한 덩어리씩 흘려보낸다. 세션 전체를 하나로 묶으면
        # 마지막 턴이 끝나기 전에는 아무것도 기록되지 않는다.
        answer = "".join(response.stream)
        json.dump(
            {
                "question": response.question,
                "answer": answer,
                "searched": response.searched,
                "sources": response.sources,
            },
            out,
            ensure_ascii=False,
            indent=2,
        )
        out.write("\n")
        return answer
    return render_answer(response, out)


def run_chat(pipeline, args: argparse.Namespace, out=None, input_fn=input) -> None:
    """Multi-turn loop. Each turn sees the previous ones, so follow-ups work."""
    out = out or sys.stdout
    history: list[dict] = []
    while True:
        try:
            question = input_fn("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            out.write("\n")
            return
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            return
        answer = run_once(pipeline, question, args, out=out, history=history)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def run_variants(
    args: argparse.Namespace, loaders: Loaders, out
) -> list[tuple[Variant, list[dict]]]:
    """Run the question through each variant, sharing one set of loaded models."""
    total = len(args.variants)
    runs: list[tuple[Variant, list[dict]]] = []
    payload = []

    for index, variant in enumerate(args.variants, 1):
        if total > 1 and not args.json:
            out.write(f"\n{HEAVY_RULE}\n[{index}/{total}] {variant.label}\n{HEAVY_RULE}\n")

        if args.retrieval_only:
            sources = build_retriever(variant, args, loaders).retrieve(
                args.question, top_k=args.top_k
            )
            answer = None
            if not args.json:
                write_sources(sources, out)
        else:
            response = build_pipeline(variant, args, loaders).ask_stream(
                args.question, top_k=args.top_k
            )
            sources = response.sources
            answer = (
                "".join(response.stream) if args.json else render_answer(response, out)
            )

        runs.append((variant, sources))
        payload.append(
            {
                "variant": variant.label,
                "mode": variant.mode,
                "expand": variant.expand,
                "answer": answer,
                "sources": sources,
            }
        )

    if args.json:
        json.dump(
            {"question": args.question, "runs": payload}, out, ensure_ascii=False, indent=2
        )
        out.write("\n")
    return runs


def main(argv: list[str] | None = None, loaders: Loaders | None = None, out=None) -> int:
    args = parse_args(argv)
    out = out or sys.stdout
    if loaders is None:
        loaders = Loaders(device=resolve_device(args.device))

    try:
        if args.chat:
            run_chat(build_pipeline(args.variants[0], args, loaders), args, out=out)
            return 0
        runs = run_variants(args, loaders, out)
        if len(runs) > 1 and not args.json:
            out.write("\n" + HEAVY_RULE + "\n" + comparison_table(runs) + "\n")
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        # 인덱스 없음 / API 키 없음 / 컬렉션 불일치 — 스택 트레이스 대신 한 줄로.
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
