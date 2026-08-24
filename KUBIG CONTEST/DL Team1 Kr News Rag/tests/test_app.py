"""Smoke tests for the Streamlit UI.

No question is submitted, so no retriever is constructed and BGE-M3 never loads —
these stay fast. They catch the failure that matters most in a Streamlit script:
a layout or import error that only surfaces when the script body runs.
"""

import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# AppTest resolves relative paths against the calling file, not the project root.
APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(autouse=True)
def pinned_provider(monkeypatch):
    """Pin the provider and key these tests run against.

    The sidebar reads the live environment, which `generation.llm` fills from
    `.env`. Left alone, what a developer happens to have configured decides what
    these tests assert: an empty `.env` locks the widgets and a machine set to
    another provider unlocks them for the wrong reason. Tests that care about a
    different provider override this.
    """
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def run_app():
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_renders_without_raising():
    app = run_app()

    assert not app.exception
    assert app.title[0].value.endswith("한국 금융 뉴스 RAG")


def test_the_module_watcher_noise_is_silenced():
    """Streamlit's watcher logs ~95 torchvision tracebacks per rerun without this."""
    run_app()

    watcher = logging.getLogger("streamlit.watcher.local_sources_watcher")
    assert watcher.level >= logging.ERROR


def test_both_retrieval_modes_are_offered_with_hybrid_first():
    app = run_app()

    assert app.radio(key="search_mode").options == ["Hybrid (Dense + BM25)", "Dense only"]
    assert app.radio(key="search_mode").value == "Hybrid (Dense + BM25)"


def test_retrieval_only_is_forced_when_no_api_key_is_configured(monkeypatch):
    # Answering costs API credits, so the UI must not offer generation it cannot do.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = run_app()

    checkbox = app.checkbox(key="retrieval_only")
    assert checkbox.value is True
    assert checkbox.disabled is True
    assert any("ANTHROPIC_API_KEY" in w.value for w in app.sidebar.warning)


def test_the_key_check_follows_the_configured_provider(monkeypatch):
    # A Claude key is useless when LLM_PROVIDER points elsewhere; naming the wrong
    # variable would send the reader off to fix a key that is already set.
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = run_app()

    assert app.checkbox(key="retrieval_only").disabled is True
    assert any("OPENAI_API_KEY" in w.value for w in app.sidebar.warning)


def test_an_openai_key_unlocks_generation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = run_app()

    assert app.checkbox(key="retrieval_only").disabled is False
    assert not app.sidebar.warning


def test_no_question_runs_no_retrieval():
    app = run_app()

    assert not app.subheader
    assert not app.expander


def test_missing_index_reports_the_problem_instead_of_rendering_nothing(monkeypatch, tmp_path):
    # data/chroma is a gitignored build artifact, so a fresh clone has no index.
    # Stopping before the title renders leaves a blank page and turns `pytest`
    # red for anyone who has not built the index yet.
    import indexing.build_index

    monkeypatch.setattr(indexing.build_index, "DEFAULT_DB", tmp_path / "absent")
    app = run_app()

    assert not app.exception
    assert app.title[0].value.endswith("한국 금융 뉴스 RAG")
    assert any("build_index" in e.value for e in app.error)


def test_single_question_mode_is_the_default_and_offers_chat():
    app = run_app()

    assert app.radio(key="ui_mode").options == ["단일 질문 (검색 실험)", "대화"]
    # Retrieval comparisons need independent runs, so the stateless mode leads.
    assert app.radio(key="ui_mode").value == "단일 질문 (검색 실험)"


def test_single_question_mode_takes_a_text_box_and_no_chat_input():
    app = run_app()

    assert app.text_input
    assert not app.chat_input


def test_chat_mode_swaps_in_a_chat_input():
    app = run_app()
    app.radio(key="ui_mode").set_value("대화").run()

    assert not app.exception
    assert app.chat_input
    assert not app.text_input


def test_the_search_gate_is_only_offered_in_chat_mode():
    app = run_app()
    assert app.checkbox(key="use_gate").disabled is True

    app.radio(key="ui_mode").set_value("대화").run()
    assert app.checkbox(key="use_gate").disabled is False
    assert app.checkbox(key="use_gate").value is True


def test_retrieval_only_is_not_offered_in_chat_mode():
    # A chat turn without an answer would just be a dead end in the transcript.
    app = run_app()
    app.radio(key="ui_mode").set_value("대화").run()

    assert app.checkbox(key="retrieval_only").disabled is True
