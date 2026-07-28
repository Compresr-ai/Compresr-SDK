"""Live tests for CompressionClient.compress_tool_output against the real API."""

import os

import pytest

from compresr import CompressionClient

TOOL_OUTPUT = (
    "Search results for 'context compression':\n"
    + "\n".join(
        f"{i}. https://example.com/article-{i} — Filler result about caching, "
        f"benchmarks, and unrelated infrastructure details number {i}."
        for i in range(40)
    )
    + "\nThe key finding: query-aware compression reduces LLM API costs by 30-70%."
)


class TestCompressToolOutputLive:
    def test_single_output(self, live_client):
        resp = live_client.compress_tool_output(
            tool_output=TOOL_OUTPUT,
            tool_name="web_search",
            query="How much does query-aware compression reduce costs?",
        )
        assert resp.success is True
        assert resp.data is not None
        assert isinstance(resp.data.compressed_output, str)
        assert resp.data.compressed_output.strip()
        assert resp.data.original_tokens > 0
        assert resp.data.compressed_tokens > 0

    def test_target_ratio_forwarded(self, live_client):
        resp = live_client.compress_tool_output(
            tool_output=TOOL_OUTPUT,
            tool_name="web_search",
            query="key finding about cost reduction",
            target_compression_ratio=3.0,
        )
        assert resp.success is True
        assert resp.data.compressed_tokens < resp.data.original_tokens

    @pytest.mark.asyncio
    async def test_async_single_output(self, live_client):
        # Own client per event loop: the session-scoped live_client's pooled
        # async connections are bound to whichever loop used them first.
        async with CompressionClient(
            api_key=os.environ["COMPRESR_API_KEY"],
            base_url=os.environ.get("COMPRESR_BASE_URL"),
        ) as client:
            resp = await client.compress_tool_output_async(
                tool_output=TOOL_OUTPUT,
                tool_name="web_search",
                query="How much does compression reduce costs?",
            )
        assert resp.success is True
        assert isinstance(resp.data.compressed_output, str)
