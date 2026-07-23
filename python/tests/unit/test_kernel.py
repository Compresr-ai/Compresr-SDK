"""Tests for compresr.integrations._shared.kernel.ToolOutputCompressor."""

from __future__ import annotations

import asyncio

import pytest

from compresr.integrations._shared import (
    CompressionPolicy,
    ToolOutputCompressor,
)

LONG = "x " * 2000  # matches the LONG_OUTPUT pattern from test_integrations_langchain.py
SHORT = "tiny"


class TestProcessSync:
    def test_long_string_is_compressed(self, fake_client):
        kernel = ToolOutputCompressor(client=fake_client, policy=CompressionPolicy())
        out = kernel.process(tool_name="search", output=LONG, query="find X")
        assert "<<C>>" in out
        assert fake_client.calls[0]["query"] == "find X"
        assert fake_client.calls[0]["compression_model_name"] == "latte_v1"

    def test_non_string_passthrough(self, fake_client):
        kernel = ToolOutputCompressor(client=fake_client, policy=CompressionPolicy())
        out = kernel.process(tool_name="search", output={"a": 1}, query="q")
        assert out == {"a": 1}
        assert fake_client.calls == []

    def test_short_string_below_min_tokens_passes(self, fake_client):
        policy = CompressionPolicy(min_tokens=200)
        kernel = ToolOutputCompressor(client=fake_client, policy=policy)
        out = kernel.process(tool_name="search", output=SHORT, query="q")
        assert out == SHORT
        assert fake_client.calls == []

    def test_allow_list_filters(self, fake_client):
        policy = CompressionPolicy(allow_tools={"search"})
        kernel = ToolOutputCompressor(client=fake_client, policy=policy)
        out_ok = kernel.process(tool_name="search", output=LONG, query="q")
        out_skip = kernel.process(tool_name="fetch", output=LONG, query="q")
        assert "<<C>>" in out_ok
        assert out_skip == LONG
        assert len(fake_client.calls) == 1

    def test_ignore_list_filters(self, fake_client):
        policy = CompressionPolicy(ignore_tools={"search"})
        kernel = ToolOutputCompressor(client=fake_client, policy=policy)
        assert kernel.process(tool_name="search", output=LONG, query="q") == LONG
        assert fake_client.calls == []

    def test_model_name_passes_through(self, fake_client):
        policy = CompressionPolicy(compression_model_name="future_v3")
        kernel = ToolOutputCompressor(client=fake_client, policy=policy)
        kernel.process(tool_name="t", output=LONG, query="q")
        assert fake_client.calls[0]["compression_model_name"] == "future_v3"

    def test_on_error_passthrough(self, failing_client):
        policy = CompressionPolicy(on_error="passthrough")
        kernel = ToolOutputCompressor(client=failing_client, policy=policy)
        out = kernel.process(tool_name="t", output=LONG, query="q")
        assert out == LONG  # original returned

    def test_on_error_raise(self, failing_client):
        policy = CompressionPolicy(on_error="raise")
        kernel = ToolOutputCompressor(client=failing_client, policy=policy)
        with pytest.raises(RuntimeError):
            kernel.process(tool_name="t", output=LONG, query="q")

    def test_empty_tool_name_compresses(self, fake_client):
        kernel = ToolOutputCompressor(client=fake_client, policy=CompressionPolicy())
        out = kernel.process(tool_name="", output=LONG, query="q")
        assert "<<C>>" in out


class TestAprocessAsync:
    def test_async_compress(self, fake_client):
        kernel = ToolOutputCompressor(client=fake_client, policy=CompressionPolicy())
        out = asyncio.run(kernel.aprocess(tool_name="search", output=LONG, query="x"))
        assert "<<C>>" in out
        assert fake_client.calls[0]["query"] == "x"

    def test_async_non_string(self, fake_client):
        kernel = ToolOutputCompressor(client=fake_client, policy=CompressionPolicy())
        out = asyncio.run(kernel.aprocess(tool_name="t", output=42, query="q"))
        assert out == 42
        assert fake_client.calls == []

    def test_async_passthrough_on_error(self, failing_client):
        policy = CompressionPolicy(on_error="passthrough")
        kernel = ToolOutputCompressor(client=failing_client, policy=policy)
        out = asyncio.run(kernel.aprocess(tool_name="t", output=LONG, query="q"))
        assert out == LONG
