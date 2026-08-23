"""OpenAI Responses API adapter for structured evaluation tasks."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from generation.llm import LLMError, LLMResult

load_dotenv()

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_MAX_TOKENS = 1024


class OpenAIClient:
    """Responses API adapter compatible with the adjudication LLM interface."""

    provider = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS, client: Any | None = None) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
            return
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set. Put your OpenAI API key in .env.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required. Run: uv add openai") from exc
        self._client = OpenAI(api_key=key)

    def generate(self, system_prompt: str, user_prompt: str, history: list[dict] | None = None,
                 *, temperature: float | None = None, output_schema: dict | None = None,
                 return_metadata: bool = False, reasoning_effort: str | None = None) -> str | LLMResult:
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt and user_prompt cannot be empty")
        if history:
            raise ValueError("OpenAIClient adjudication adapter does not accept conversation history")
        options: dict[str, Any] = {
            "model": self.model, "instructions": system_prompt, "input": user_prompt,
            "max_output_tokens": self.max_tokens, "store": False,
        }
        if temperature is not None:
            options["temperature"] = temperature
        if reasoning_effort is not None:
            options["reasoning"] = {"effort": reasoning_effort}
        if output_schema is not None:
            options["text"] = {"format": {
                "type": "json_schema", "name": "structured_output",
                "strict": True, "schema": output_schema,
            }}
        try:
            response = self._client.responses.create(**options)
        except Exception as exc:
            raise LLMError(f"OpenAI API request failed: {exc}") from exc
        answer = str(getattr(response, "output_text", "")).strip()
        if not answer:
            raise LLMError("OpenAI returned an empty text response")
        if not return_metadata:
            return answer
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        return LLMResult(answer, str(getattr(response, "model", self.model)),
                         getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None),
                         getattr(input_details, "cached_tokens", None),
                         getattr(output_details, "reasoning_tokens", None))
