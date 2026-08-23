"""Streamlit UI for the Korean financial-news RAG pipeline.

Two UI modes over the same pipeline:

- **단일 질문** — one question, one answer, nothing remembered. This is the mode
  for comparing retrieval settings, because each run is independent.
- **대화** — a chat that keeps its history, so follow-ups work. An optional LLM
  gate decides per turn whether the news index is needed at all.

Answers stream as they are written. Retrieval-only skips the LLM call; use it
while tuning retrieval, since every answer costs API credits.
"""

from __future__ import annotations

import logging

import streamlit as st

# Streamlit's source watcher walks every entry in sys.modules and touches
# __path__ to decide what to watch. transformers resolves its submodules lazily,
# so that touch *imports* them — including ~95 image processors that need
# torchvision, which this project has no use for and does not install. Each one
# logs a traceback, on every rerun. Nothing actually fails: the watcher catches
# the error and moves on. Only the log is affected, so only the log is fixed.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

from generation.llm import PROVIDERS, active_provider, api_key_env_var, has_api_key
from indexing.build_index import DEFAULT_DB
from indexing.retriever import detect_device
from rag import RAGPipeline
from ui_format import meta_line, provenance_line, source_heading

st.set_page_config(page_title="한국 금융 뉴스 RAG", page_icon="📰", layout="wide")




@st.cache_resource(show_spinner="BGE-M3 로딩 중… (최초 1회, 10초쯤 걸립니다)")
def load_dense(device: str):
    from indexing.retriever import get_retriever

    retriever = get_retriever(device=device)
    retriever.retrieve("워밍업")  # 모델·컬렉션을 첫 질문 전에 올려둔다
    return retriever


@st.cache_resource(show_spinner="BM25 인덱스 구축 중… (최초 1회, 5초쯤 걸립니다)")
def load_bm25():
    from indexing.bm25 import get_bm25_retriever

    retriever = get_bm25_retriever()
    retriever.retrieve("워밍업")
    return retriever


@st.cache_resource
def load_hybrid(device: str, candidate_k: int, rrf_k: int):
    from indexing.hybrid import HybridRetriever

    return HybridRetriever(
        dense=load_dense(device),
        bm25=load_bm25(),
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )


@st.cache_resource
def load_llm():
    from generation.llm import get_llm

    return get_llm()


@st.cache_resource
def load_gate():
    from generation.router import SearchGate

    return SearchGate(llm=load_llm())


EXPANDERS = {
    "없음": None,
    "Multi-Query": "multi_query",
    "HyDE": "hyde",
}


def wrap_with_expansion(base, expander_name: str, candidate_k: int, rrf_k: int):
    """Wrap the retriever so each question is searched under several phrasings."""
    kind = EXPANDERS[expander_name]
    if kind is None:
        return base
    from indexing.expanding import ExpandingRetriever
    from indexing.query_expand import get_expander

    expander = get_expander(kind, llm=load_llm())
    if expander is None:
        return base
    return ExpandingRetriever(
        base=base, expander=expander, candidate_k=candidate_k, rrf_k=rrf_k
    )


def show_expanded_queries(results: list[dict]) -> None:
    """Show what the expander actually searched with."""
    queries = next((r.get("expanded_queries") for r in results if r.get("expanded_queries")), None)
    if not queries:
        return
    with st.expander(f"실제 검색에 쓴 질의 {len(queries)}개", expanded=False):
        for i, query in enumerate(queries):
            label = "원본" if i == 0 else f"q{i}"
            st.markdown(f"**{label}** · {query}")
        st.caption(
            "q1 이후는 LLM이 생성한 것입니다. HyDE의 가상 본문은 검색에만 쓰이고 "
            "답변 근거로는 들어가지 않습니다."
        )


def render_chunk(rank: int, chunk: dict) -> None:
    """One source card: linked title, why it surfaced, and the passage."""
    st.markdown(source_heading(rank, chunk))

    meta = meta_line(chunk)
    if meta:
        st.caption(meta)
    provenance = provenance_line(chunk)
    if provenance:
        st.caption(provenance)

    content = chunk.get("content", "")
    if content:
        with st.expander("본문 보기", expanded=rank <= 2):
            st.write(content)


def render_sources(sources: list[dict], searched: bool) -> None:
    """Source cards under an answer, or why there are none."""
    if not searched:
        st.caption("이 답변은 검색 없이 대화 내용만으로 작성됐습니다.")
        return
    if not sources:
        st.info("관련 뉴스를 찾지 못했습니다.")
        return
    with st.expander(f"출처 {len(sources)}건", expanded=False):
        for source in sources:
            render_chunk(source["news_number"], source)


def build_retriever(is_hybrid, resolved_device, candidate_k, rrf_k, expander_name):
    base = (
        load_hybrid(resolved_device, candidate_k, rrf_k)
        if is_hybrid
        else load_dense(resolved_device)
    )
    return wrap_with_expansion(base, expander_name, candidate_k, rrf_k)


# --- page -----------------------------------------------------------------

# Rendered before the sidebar so the page still has a heading when a later check
# stops the script — a bare st.stop() with nothing drawn yet is a blank page.
st.title("📰 한국 금융 뉴스 RAG")

with st.sidebar:
    # Which key matters depends on LLM_PROVIDER — a Claude key is no help to an
    # OpenAI run, and naming the wrong variable sends the reader off to fix the
    # one that is already set.
    provider = active_provider()
    key_var = api_key_env_var(provider) if provider in PROVIDERS else "LLM_PROVIDER"
    has_key = has_api_key(provider)

    st.header("UI 모드")
    ui_mode = st.radio(
        "화면",
        ["단일 질문 (검색 실험)", "대화"],
        index=0,
        key="ui_mode",
        label_visibility="collapsed",
        help=(
            "단일 질문: 매 질문이 독립적. 검색 설정을 바꿔가며 비교할 때 씁니다.\n\n"
            "대화: 이력을 기억해 후속 질문이 됩니다."
        ),
    )
    is_chat = ui_mode == "대화"

    st.divider()
    st.header("검색 설정")
    mode = st.radio(
        "검색 방식", ["Hybrid (Dense + BM25)", "Dense only"], index=0, key="search_mode"
    )
    is_hybrid = mode.startswith("Hybrid")

    top_k = st.slider("top_k (최종 청크 수)", 1, 10, 5)
    candidate_k = st.slider("후보 풀 (각 검색기당)", 5, 50, 20, disabled=not is_hybrid)
    rrf_k = st.slider("RRF k", 10, 120, 60, disabled=not is_hybrid)

    st.divider()
    expander_name = st.radio(
        "질의 확장",
        list(EXPANDERS),
        index=0,
        key="expander",
        disabled=not has_key,
        help=(
            "Multi-Query: 질문을 여러 표현으로 바꿔 각각 검색.\n\n"
            "HyDE: 답이 될 법한 가상 기사 본문을 지어내 그것으로 검색.\n\n"
            "둘 다 질문당 LLM 호출이 1회 더 듭니다."
        ),
    )
    if not has_key:
        expander_name = "없음"

    st.divider()
    retrieval_only = st.checkbox(
        "검색만 (LLM 호출 안 함)",
        key="retrieval_only",
        value=not has_key,
        disabled=not has_key or is_chat,
        help="검색 결과만 봅니다. 답변 생성에는 LLM API 비용이 듭니다.",
    )
    if is_chat:
        retrieval_only = False
    use_gate = st.checkbox(
        "검색 게이트 (LLM)",
        key="use_gate",
        value=True,
        disabled=not has_key or not is_chat,
        help=(
            "매 턴 검색이 필요한지 LLM이 먼저 판단합니다. 인사나 "
            "'더 쉽게 설명해줘' 같은 요청은 검색 없이 대화 내용만으로 답합니다.\n\n"
            "턴당 LLM 호출이 1회 더 듭니다."
        ),
    )
    if not has_key:
        st.warning(f"{key_var}가 없어 검색 전용으로 동작합니다.")
    if is_chat and st.button("대화 초기화", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

    device = st.selectbox(
        "임베딩 device", ["auto", "cpu", "cuda", "mps"], index=0
    )
    resolved_device = detect_device() if device == "auto" else device
    st.caption(f"device: `{resolved_device}`")

st.caption(
    "네이버 금융 뉴스 432건 · 1,377청크 · BGE-M3 임베딩 + BM25 · "
    f"검색: **{mode}** · 질의 확장: **{expander_name}**"
    + (f" · 검색 게이트: **{'켜짐' if use_gate else '꺼짐'}**" if is_chat else "")
)

if not DEFAULT_DB.is_dir():
    st.error(
        f"Chroma 인덱스가 없습니다 (`{DEFAULT_DB}`). 먼저 인덱스를 만들어 주세요.\n\n"
        "```\n"
        "uv run python indexing/chunk.py\n"
        "uv run python indexing/build_index.py --rebuild --device mps\n"
        "```"
    )
    st.stop()

if not is_chat:
    # --- 단일 질문 -----------------------------------------------------------
    question = st.text_input(
        "질문", placeholder="예: 삼전 반도체 실적 전망은?", label_visibility="collapsed"
    )

    if question:
        retriever = build_retriever(
            is_hybrid, resolved_device, candidate_k, rrf_k, expander_name
        )
        if retrieval_only:
            with st.spinner("검색 중…"):
                chunks = retriever.retrieve(question, top_k=top_k)
            show_expanded_queries(chunks)
            st.subheader(f"검색 결과 {len(chunks)}개")
            if not chunks:
                st.info("검색 결과가 없습니다.")
            for rank, chunk in enumerate(chunks, 1):
                render_chunk(rank, chunk)
        else:
            with st.spinner("검색 중…"):
                response = RAGPipeline(retriever=retriever, llm=load_llm()).ask_stream(
                    question, top_k=top_k
                )
            show_expanded_queries(response.sources)
            st.subheader("답변")
            st.write_stream(response.stream)
            st.subheader(f"출처 {len(response.sources)}건")
            if not response.sources:
                st.info("관련 뉴스를 찾지 못했습니다.")
            for source in response.sources:
                render_chunk(source["news_number"], source)

else:
    # --- 대화 ---------------------------------------------------------------
    st.session_state.setdefault("chat", [])

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn["role"] == "assistant":
                render_sources(turn.get("sources", []), turn.get("searched", True))

    prompt = st.chat_input("질문을 입력하세요" if has_key else "API 키가 없어 대화를 쓸 수 없습니다")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat.append({"role": "user", "content": prompt})

        # Only the plain turns go to the model; sources are UI-side state.
        history = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state.chat[:-1]
        ]
        retriever = build_retriever(
            is_hybrid, resolved_device, candidate_k, rrf_k, expander_name
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            llm=load_llm(),
            gate=load_gate() if use_gate else None,
        )
        with st.chat_message("assistant"):
            with st.spinner("생각 중…"):
                response = pipeline.ask_stream(prompt, top_k=top_k, history=history)
            if response.searched:
                show_expanded_queries(response.sources)
            # write_stream renders as the text arrives and returns the whole thing,
            # which is what goes into the transcript.
            answer = st.write_stream(response.stream)
            render_sources(response.sources, response.searched)

        st.session_state.chat.append({
            "role": "assistant",
            "content": answer,
            "sources": response.sources,
            "searched": response.searched,
        })
