"""OpenAI-style provider adapter and the provider factory.

Kept apart from test_generation.py, which is already 450 lines of Claude and
prompt coverage. The adapters share a base class, so the shared behaviour
(prompt guards, empty-response errors) is asserted here too — a regression in
the base shows up in both files.
"""

from types import SimpleNamespace

import pytest

from generation.llm import (
    ClaudeClient,
    LLMError,
    OpenAIChatClient,
    active_provider,
    api_key_env_var,
    get_llm,
    has_api_key,
)


class FakeCompletions:
    def __init__(self, content="테스트 답변", chunks=None):
        self.content = content
        self.chunks = chunks
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.chunks or [])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeOpenAI:
    def __init__(self, content="테스트 답변", chunks=None):
        self.chat = SimpleNamespace(completions=FakeCompletions(content, chunks))

    @property
    def calls(self):
        return self.chat.completions.calls


def delta(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


# --- request shape ---------------------------------------------------------


def test_the_system_prompt_becomes_the_first_message():
    """Anthropic takes `system=` beside the messages; OpenAI wants it inside."""
    fake = FakeOpenAI()

    OpenAIChatClient(client=fake, model="gpt-test").generate("system", "user")

    call = fake.calls[0]
    assert "system" not in call
    assert call["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_history_sits_between_the_system_prompt_and_the_question():
    fake = FakeOpenAI()
    history = [{"role": "user", "content": "이전 질문"}]

    OpenAIChatClient(client=fake, model="gpt-test").generate("system", "질문", history=history)

    assert fake.calls[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "이전 질문"},
        {"role": "user", "content": "질문"},
    ]


def test_the_model_and_token_limit_are_sent():
    fake = FakeOpenAI()

    OpenAIChatClient(client=fake, model="gpt-test", max_tokens=123).generate("system", "user")

    assert fake.calls[0]["model"] == "gpt-test"
    assert fake.calls[0]["max_tokens"] == 123


def test_the_token_limit_field_can_be_renamed():
    """OpenAI deprecated `max_tokens`; o-series models reject it outright."""
    fake = FakeOpenAI()

    OpenAIChatClient(
        client=fake, model="gpt-test", max_tokens=99, max_tokens_field="max_completion_tokens"
    ).generate("system", "user")

    assert fake.calls[0]["max_completion_tokens"] == 99
    assert "max_tokens" not in fake.calls[0]


# --- shared base behaviour -------------------------------------------------


def test_the_answer_comes_back_as_text():
    client = OpenAIChatClient(client=FakeOpenAI("삼성전자는 ... [뉴스1]"), model="gpt-test")

    assert client.generate("system", "user") == "삼성전자는 ... [뉴스1]"


@pytest.mark.parametrize("content", ["", "   ", None])
def test_an_empty_answer_is_an_error(content):
    client = OpenAIChatClient(client=FakeOpenAI(content), model="gpt-test")

    with pytest.raises(LLMError, match="empty"):
        client.generate("system", "user")


def test_empty_prompts_are_rejected_before_the_request():
    fake = FakeOpenAI()
    client = OpenAIChatClient(client=fake, model="gpt-test")

    with pytest.raises(ValueError):
        client.generate("", "질문")
    with pytest.raises(ValueError):
        client.generate("system", "   ")
    assert fake.calls == []


def test_sdk_failures_surface_as_llm_error():
    fake = FakeOpenAI()
    fake.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    client = OpenAIChatClient(client=fake, model="gpt-test")

    with pytest.raises(LLMError, match="boom"):
        client.generate("system", "user")


# --- streaming -------------------------------------------------------------


def test_streaming_yields_the_delta_pieces():
    fake = FakeOpenAI(chunks=[delta("삼성전자는 "), delta("반도체 "), delta("개선 전망입니다")])
    client = OpenAIChatClient(client=fake, model="gpt-test")

    assert list(client.stream("system", "질문")) == ["삼성전자는 ", "반도체 ", "개선 전망입니다"]
    assert fake.calls[0]["stream"] is True


def test_streaming_skips_the_chunks_that_carry_no_text():
    """The first and last chunks of an OpenAI stream have `delta.content = None`."""
    fake = FakeOpenAI(chunks=[delta(None), delta("답변"), delta(""), delta(None)])

    assert list(OpenAIChatClient(client=fake, model="gpt-test").stream("system", "질문")) == ["답변"]


def test_a_stream_that_produced_nothing_is_an_error():
    fake = FakeOpenAI(chunks=[delta(None)])

    with pytest.raises(LLMError, match="empty"):
        list(OpenAIChatClient(client=fake, model="gpt-test").stream("system", "질문"))


def test_streaming_sends_the_same_message_shape_as_a_plain_call():
    fake = FakeOpenAI(chunks=[delta("답변")])

    list(OpenAIChatClient(client=fake, model="gpt-test").stream("system", "질문"))

    assert fake.calls[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "질문"},
    ]


# --- factory ---------------------------------------------------------------


def test_the_default_provider_is_claude(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert isinstance(get_llm(), ClaudeClient)


def test_openai_is_selected_by_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    client = get_llm()

    assert isinstance(client, OpenAIChatClient)
    assert client.model == "gpt-test"
    # o-series models reject the deprecated name, so OpenAI proper gets the new one.
    assert client.max_tokens_field == "max_completion_tokens"


def test_openrouter_is_the_same_adapter_pointed_at_another_host(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/model-test")

    client = get_llm()

    assert isinstance(client, OpenAIChatClient)
    assert "openrouter.ai" in str(client.base_url)
    # OpenRouter normalises the legacy name across every model it proxies.
    assert client.max_tokens_field == "max_tokens"


def test_an_explicit_provider_beats_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert isinstance(get_llm(provider="anthropic"), ClaudeClient)


def test_an_explicit_model_beats_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "from-env")

    assert get_llm(provider="openai", model="explicit").model == "explicit"


def test_an_unknown_provider_names_the_ones_that_exist(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(ValueError, match="openrouter"):
        get_llm()


def test_openai_style_providers_need_a_model(monkeypatch):
    """Claude has a sane default; guessing an OpenAI model ID would fail at request time."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        get_llm(provider="openai")


def test_a_missing_key_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        get_llm(provider="openrouter", model="vendor/model-test")


# --- key detection used by the Streamlit sidebar ---------------------------


def test_the_active_provider_falls_back_to_claude(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert active_provider() == "anthropic"

    monkeypatch.setenv("LLM_PROVIDER", "  OpenAI  ")
    assert active_provider() == "openai"


def test_each_provider_names_its_own_key_variable():
    assert api_key_env_var("anthropic") == "ANTHROPIC_API_KEY"
    assert api_key_env_var("openai") == "OPENAI_API_KEY"
    assert api_key_env_var("openrouter") == "OPENROUTER_API_KEY"


def test_key_detection_follows_the_active_provider(monkeypatch):
    """app.py used to hardcode ANTHROPIC_API_KEY, locking the UI for OpenAI users."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert has_api_key() is False

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert has_api_key() is True
