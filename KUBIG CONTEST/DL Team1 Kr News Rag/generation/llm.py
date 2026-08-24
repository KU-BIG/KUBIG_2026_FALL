"""LLM adapters used by the RAG generation stage.

Two providers, one interface. Everything downstream — the pipeline, the search
gate, the query expanders — only ever calls `generate` and `stream`, so a
provider swap is a factory call rather than a change at each site.

Claude and the OpenAI-style APIs disagree on more than the endpoint: the system
prompt is a separate field for one and the first message for the other, the
token-limit field has two names, and the streaming payloads are shaped
differently. `LLMClient` holds what is genuinely shared and leaves those four
differences to small hooks.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_PROVIDER = "anthropic"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# One row per provider: which environment variables carry its key and model,
# and whether a model can be defaulted. setup_api.py reads the same table.
PROVIDERS: dict[str, dict[str, str | None]] = {
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "model_env": "CLAUDE_MODEL",
        "default_model": DEFAULT_MODEL,
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": None,
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": None,
    },
}


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce an answer."""


@dataclass(frozen=True)
class LLMResult:
    """An answer plus what the run cost, for callers that record provenance."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None


class LLMClient:
    """Shared behaviour for every provider adapter.

    Subclasses supply the four things that actually differ: how a request is
    made, how a stream is opened, how text is pulled out of a response, and
    whether the system prompt belongs in the message list.
    """

    label = "LLM"

    def __init__(self, model: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.model = model
        self.max_tokens = max_tokens

    # --- hooks --------------------------------------------------------

    def _create(
        self,
        system_prompt: str,
        messages: list[Any],
        *,
        temperature: float | None = None,
        output_schema: dict | None = None,
    ) -> Any:
        raise NotImplementedError

    def _open_stream(self, system_prompt: str, messages: list[Any]) -> Iterator[str]:
        raise NotImplementedError

    def _extract_text(self, response: Any) -> str:
        raise NotImplementedError

    def _usage(self, response: Any) -> tuple[int | None, int | None, int | None, int | None]:
        """(input, output, cached input, reasoning) tokens. Providers name these differently."""
        raise NotImplementedError

    def _build_messages(self, system_prompt: str, user_prompt: str, history) -> list[Any]:
        return [*(history or []), {"role": "user", "content": user_prompt}]

    # --- shared -------------------------------------------------------

    @staticmethod
    def _guard(system_prompt: str, user_prompt: str) -> None:
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt and user_prompt cannot be empty")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict] | None = None,
        *,
        temperature: float | None = None,
        output_schema: dict | None = None,
        return_metadata: bool = False,
    ) -> str | LLMResult:
        """Generate one text answer.

        `history` is prior `{"role", "content"}` turns, sent ahead of the current
        question so follow-ups ("그럼 SK하이닉스는?") have something to refer to.

        `output_schema` constrains the reply to a JSON schema and `return_metadata`
        swaps the plain string for an `LLMResult` carrying the model that answered
        and the token counts — both needed by callers that have to record which
        model produced a result, not just what it said.
        """
        self._guard(system_prompt, user_prompt)
        messages = self._build_messages(system_prompt, user_prompt, history)
        try:
            response = self._create(
                system_prompt, messages, temperature=temperature, output_schema=output_schema
            )
        except Exception as exc:
            raise LLMError(f"{self.label} API request failed: {exc}") from exc

        answer = self._extract_text(response).strip()
        if not answer:
            raise LLMError(f"{self.label} returned an empty text response")
        if not return_metadata:
            return answer
        input_tokens, output_tokens, cached, reasoning = self._usage(response)
        return LLMResult(
            answer,
            str(getattr(response, "model", self.model)),
            input_tokens,
            output_tokens,
            cached,
            reasoning,
        )

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Yield the answer in pieces as it is written.

        Same request as `generate`; the difference is that the caller can render
        text while it arrives instead of waiting out the whole completion.
        """
        self._guard(system_prompt, user_prompt)
        messages = self._build_messages(system_prompt, user_prompt, history)
        produced = False
        try:
            for piece in self._open_stream(system_prompt, messages):
                if piece:
                    produced = True
                    yield piece
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.label} API request failed: {exc}") from exc

        if not produced:
            raise LLMError(f"{self.label} returned an empty text response")


class ClaudeClient(LLMClient):
    """Adapter around Anthropic's Messages API.

    The Anthropic client is imported lazily so prompt/retrieval unit tests do not
    require an API key or an active network connection.
    """

    label = "Claude"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        super().__init__(model or os.getenv("CLAUDE_MODEL", DEFAULT_MODEL), max_tokens)

        if client is not None:
            self._client = client
            return

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Put your Claude API key in the environment."
            )

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Run: uv add anthropic"
            ) from exc

        self._client = Anthropic(api_key=key)

    def _create(
        self,
        system_prompt: str,
        messages: list[Any],
        *,
        temperature: float | None = None,
        output_schema: dict | None = None,
    ) -> Any:
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if output_schema is not None:
            options["output_config"] = {
                "format": {"type": "json_schema", "schema": output_schema}
            }
        return self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
            **options,
        )

    def _open_stream(self, system_prompt: str, messages: list[Any]) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        ) as stream:
            yield from stream.text_stream

    def _extract_text(self, response: Any) -> str:
        return "\n".join(
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", "")
        )

    def _usage(self, response: Any):
        usage = getattr(response, "usage", None)
        return (
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
            None,
        )


class OpenAIChatClient(LLMClient):
    """Adapter around any OpenAI-compatible Chat Completions endpoint.

    OpenAI proper and OpenRouter are the same wire format; only `base_url`, the
    key, and the token-limit field name differ, so one class covers both.

    `generation/openai_llm.py` also talks to OpenAI, over the Responses API, for
    blind adjudication: it refuses history and has no streaming because the
    evaluation run needs neither. This one is the conversational path. The two
    are kept apart because the frozen evaluation artifacts record which client
    produced them.
    """

    label = "OpenAI"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
        base_url: str | None = None,
        max_tokens_field: str = "max_tokens",
        key_env: str = "OPENAI_API_KEY",
    ) -> None:
        if not model:
            raise ValueError("model is required for OpenAI-style providers")
        super().__init__(model, max_tokens)
        self.base_url = base_url
        self.max_tokens_field = max_tokens_field

        if client is not None:
            self._client = client
            return

        key = api_key or os.getenv(key_env)
        if not key:
            raise ValueError(f"{key_env} is not set. Put your API key in the environment.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required. Run: uv add openai") from exc

        self._client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)

    def _build_messages(self, system_prompt: str, user_prompt: str, history) -> list[Any]:
        """Anthropic takes `system=` beside the messages; OpenAI wants it inside."""
        return [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": user_prompt},
        ]

    def _request(self, messages: list[Any], **extra: Any) -> Any:
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **{self.max_tokens_field: self.max_tokens},
            **extra,
        )

    def _create(
        self,
        system_prompt: str,
        messages: list[Any],
        *,
        temperature: float | None = None,
        output_schema: dict | None = None,
    ) -> Any:
        extra: dict[str, Any] = {}
        if temperature is not None:
            extra["temperature"] = temperature
        if output_schema is not None:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": output_schema,
                },
            }
        return self._request(messages, **extra)

    def _open_stream(self, system_prompt: str, messages: list[Any]) -> Iterator[str]:
        for chunk in self._request(messages, stream=True):
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            yield getattr(choices[0].delta, "content", None)

    def _extract_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        return getattr(choices[0].message, "content", None) or ""

    def _usage(self, response: Any):
        usage = getattr(response, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        return (
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(prompt_details, "cached_tokens", None),
            getattr(completion_details, "reasoning_tokens", None),
        )


# --- provider selection ----------------------------------------------------


def active_provider() -> str:
    """The provider named by `LLM_PROVIDER`, or Claude when it is unset."""
    return (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()


def _provider_config(provider: str) -> dict:
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r}. Choose one of: {', '.join(sorted(PROVIDERS))}"
        )
    return PROVIDERS[provider]


def api_key_env_var(provider: str) -> str:
    """Name of the environment variable holding this provider's API key."""
    return str(_provider_config(provider)["key_env"])


def has_api_key(provider: str | None = None) -> bool:
    """Whether the active provider has a key. Never raises — the UI calls it on every rerun."""
    provider = provider or active_provider()
    if provider not in PROVIDERS:
        return False
    return bool(os.getenv(api_key_env_var(provider)))


def resolve_model(provider: str, model: str | None = None) -> str:
    """The model to use, from the argument, the environment, or the provider default."""
    config = _provider_config(provider)
    resolved = model or os.getenv(str(config["model_env"])) or config["default_model"]
    if not resolved:
        raise ValueError(
            f"no model configured for {provider}. Set {config['model_env']} "
            f"or run: uv run python setup_api.py"
        )
    return str(resolved)


def get_llm(provider: str | None = None, model: str | None = None, **kwargs: Any) -> LLMClient:
    """Build the client for the configured provider.

    This is the single construction point; call sites should not name a concrete
    adapter, so switching providers stays an environment change.
    """
    provider = (provider or active_provider()).strip().lower()
    config = _provider_config(provider)

    # Key before model: a missing key is the more basic problem, and reporting
    # the model first sends the reader to fix the wrong thing.
    key_env = str(config["key_env"])
    if not kwargs.get("client") and not kwargs.get("api_key") and not os.getenv(key_env):
        raise ValueError(f"{key_env} is not set. Put your API key in the environment.")

    resolved = resolve_model(provider, model)
    if provider == "anthropic":
        return ClaudeClient(model=resolved, **kwargs)

    if provider == "openrouter":
        # OpenRouter normalises the legacy field name across every model it proxies.
        kwargs.setdefault("base_url", OPENROUTER_BASE_URL)
        kwargs.setdefault("max_tokens_field", "max_tokens")
    else:
        # OpenAI deprecated max_tokens; the o-series rejects it outright.
        kwargs.setdefault("base_url", os.getenv("OPENAI_BASE_URL"))
        kwargs.setdefault("max_tokens_field", "max_completion_tokens")

    return OpenAIChatClient(model=resolved, key_env=key_env, **kwargs)
