"""Integration: OpenAI facade end-to-end with live gpt-4o-mini + Compresr APIs."""

from __future__ import annotations

import os

import pytest
from langchain_core.tools import tool

from compresr import WebSearchTool

pytestmark = pytest.mark.integration

LONG_TEXT = "This is a long document about machine learning. " * 80


@tool
def long_search(query: str) -> str:
    """Search the corpus. Always call this tool with the user's query."""
    return f"Results for {query}: {LONG_TEXT}"


class TestOpenAIChatCompletions:
    def test_simple_response(self, openai_client):
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=128,
            messages=[{"role": "user", "content": "Say the word 'pong' and nothing else."}],
        )
        assert r.object == "chat.completion"
        msg = r.choices[0].message
        assert msg.role == "assistant"
        assert "pong" in (msg.content or "").lower()
        assert r.usage.prompt_tokens > 0
        assert r.usage.completion_tokens > 0
        assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens

    def test_function_tool_output_is_compressed(self, openai_client, compress_spy):
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": "Use the long_search tool with query='ml'. Summarize what it returns.",
                }
            ],
            tools=[long_search],
        )
        assert r.object == "chat.completion"
        # TODO(wave 3): if real LLM occasionally refuses to call the tool, mark flaky.
        assert len(compress_spy) >= 1

    def test_web_search_via_tavily(self, openai_client, tavily_required, compress_spy):
        search = WebSearchTool(
            provider="tavily",
            max_results=3,
            api_key=os.environ["TAVILY_API_KEY"],
        )
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Use the tavily_search tool to find the latest news about "
                        "OpenAI. Summarize the top result."
                    ),
                }
            ],
            tools=[search],
        )
        assert r.object == "chat.completion"
        # TODO(wave 3): if real LLM occasionally refuses to call the tool, mark flaky.
        assert len(compress_spy) >= 1
