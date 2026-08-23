"""Optional live Claude API smoke test.

Marked `live` and deselected by default (see `addopts` in pyproject.toml) so a
plain `uv run pytest` never spends API credits. Run it deliberately:

    uv run pytest -m live -q -s

It is additionally skipped when ANTHROPIC_API_KEY is unset.
"""

import os

import pytest

from generation.llm import ClaudeClient
from generation.prompt import build_messages


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
def test_live_claude_generation():
    results = [
        {
            "title": "삼성전자 반도체 실적 개선 전망",
            "date": "2026.08.08",
            "url": "https://example.com/test",
            "stock_names": ["삼성전자"],
            "content": "삼성전자의 반도체 사업 실적이 개선될 수 있다는 전망이 나왔다.",
        }
    ]
    system, user = build_messages("삼성전자 반도체 실적에 대한 전망은?", results)
    answer = ClaudeClient().generate(system, user)

    assert answer.strip()
    print("\nClaude answer:\n", answer)
