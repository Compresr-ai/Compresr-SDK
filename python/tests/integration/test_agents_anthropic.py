"""Integration: Anthropic facade end-to-end with live Claude + Compresr APIs."""

from __future__ import annotations

import os

import pytest
from langchain_core.tools import tool

from compresr import WebSearchTool

pytestmark = pytest.mark.integration

LONG_TEXT = "This is a long document about machine learning. " * 80  # ~3500 chars


@tool
def long_search(query: str) -> str:
    """Search the corpus. Always call this tool with the user's query."""
    return f"Results for {query}: {LONG_TEXT}"


class TestAnthropicMessagesCreate:
    def test_simple_response(self, anthropic_client):
        r = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=128,
            messages=[{"role": "user", "content": "Say the word 'pong' and nothing else."}],
        )
        # Anthropic Message shape
        assert r.role == "assistant"
        assert r.type == "message"
        assert r.content
        text = r.content[0].text.lower()
        assert "pong" in text
        assert r.usage.input_tokens > 0
        assert r.usage.output_tokens > 0

    def test_function_tool_output_is_compressed(self, anthropic_client, compress_spy):
        r = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": "Use the long_search tool with query='ml'. Summarize what it returns.",
                }
            ],
            tools=[long_search],
        )
        assert r.role == "assistant"
        # At least one compress call fired — the tool output was long enough.
        # TODO(wave 3): if real LLM occasionally refuses to call the tool, mark flaky.
        assert (
            len(compress_spy) >= 1
        ), "Expected CompresrToolMiddleware to call compress on the tool output"
        # The compressed call should reference our query somehow (latte_v1 query passthrough)
        # query may be None for some paths — don't hard-assert it.

    def test_web_search_via_tavily(self, anthropic_client, tavily_required, compress_spy):
        search = WebSearchTool(
            provider="tavily",
            max_results=3,
            api_key=os.environ["TAVILY_API_KEY"],
        )
        r = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Use the tavily_search tool to find the latest news about "
                        "Anthropic Claude. Summarize the top result."
                    ),
                }
            ],
            tools=[search],
        )
        assert r.role == "assistant"
        # Tavily output is reliably long (raw_content + snippets) — compress should fire.
        # TODO(wave 3): if real LLM occasionally refuses to call the tool, mark flaky.
        assert len(compress_spy) >= 1, "Tavily output should have triggered Compresr compression"
