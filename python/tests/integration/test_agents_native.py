"""Integration: native ``run()`` facade end-to-end with live Claude + Compresr APIs."""

from __future__ import annotations

import os

import pytest

from compresr import WebSearchTool
from compresr.agents.normalized import NormalizedResult

pytestmark = pytest.mark.integration


class TestNativeRun:
    def test_run_returns_normalized_result(self, anthropic_client):
        r = anthropic_client.run(
            prompt="Say 'pong' and nothing else.",
            max_tokens=64,
        )
        assert isinstance(r, NormalizedResult)
        assert "pong" in r.text.lower()
        assert r.stop_reason in ("end_turn", "stop")
        assert r.usage.get("input_tokens", 0) > 0 or r.usage.get("output_tokens", 0) > 0

    def test_run_with_tavily(self, anthropic_client, tavily_required, compress_spy):
        search = WebSearchTool(
            provider="tavily",
            max_results=3,
            api_key=os.environ["TAVILY_API_KEY"],
        )
        r = anthropic_client.run(
            prompt=(
                "Use the tavily_search tool to find news about Compresr.ai "
                "(a YC-backed AI startup) and summarize."
            ),
            tools=[search],
            max_tokens=512,
        )
        assert isinstance(r, NormalizedResult)
        assert r.text
        # TODO(wave 3): if real LLM occasionally refuses to call the tool, mark flaky.
        assert len(compress_spy) >= 1
