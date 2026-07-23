"""Tests for compresr.integrations.llamaindex — postprocessor + tool wrapper."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode  # noqa: E402
from llama_index.core.tools import FunctionTool  # noqa: E402

from compresr.integrations.llamaindex import (  # noqa: E402
    CompresrNodePostprocessor,
    wrap_tool_with_compresr,
)

LONG = "x " * 2000


# ---------------------------------------------------------------------------
# CompresrNodePostprocessor
# ---------------------------------------------------------------------------


def _nws(text: str, score: float = 1.0) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text), score=score)


class TestCompresrNodePostprocessor:
    def test_compresses_long_nodes(self, fake_client):
        pp = CompresrNodePostprocessor(client=fake_client)
        out = pp.postprocess_nodes(
            [_nws(LONG), _nws(LONG)],
            query_bundle=QueryBundle(query_str="find X"),
        )
        assert len(out) == 2
        for n in out:
            assert "<<C>>" in n.node.get_content()
        assert len(fake_client.batch_calls) == 1
        assert fake_client.batch_calls[0]["queries"] == "find X"
        assert fake_client.batch_calls[0]["compression_model_name"] == "latte_v1"

    def test_target_token_overrides_ratio(self, fake_client):
        from compresr.integrations._shared.tokens import estimate_tokens

        pp = CompresrNodePostprocessor(
            client=fake_client,
            target_token=100,
            target_compression_ratio=0.5,  # ignored when target_token is set
        )
        pp.postprocess_nodes(
            [_nws(LONG), _nws(LONG)],
            query_bundle=QueryBundle(query_str="find X"),
        )
        # ratio = avg_chunk_tokens / target_token
        expected = estimate_tokens(LONG) / 100.0
        assert fake_client.batch_calls[0]["target_compression_ratio"] == expected
        assert fake_client.batch_calls[0]["target_compression_ratio"] > 1.0

    def test_short_nodes_passthrough(self, fake_client):
        pp = CompresrNodePostprocessor(client=fake_client)
        out = pp.postprocess_nodes(
            [_nws("tiny")],
            query_bundle=QueryBundle(query_str="q"),
        )
        assert out[0].node.get_content() == "tiny"
        assert fake_client.batch_calls == []

    def test_no_query_skips_with_warning(self, fake_client):
        pp = CompresrNodePostprocessor(client=fake_client)
        out = pp.postprocess_nodes([_nws(LONG)], query_bundle=None)
        assert out[0].node.get_content() == LONG
        assert fake_client.batch_calls == []

    def test_static_query_override(self, fake_client):
        pp = CompresrNodePostprocessor(client=fake_client, query="STATIC")
        pp.postprocess_nodes([_nws(LONG)], query_bundle=QueryBundle(query_str="ignored"))
        assert fake_client.batch_calls[0]["queries"] == "STATIC"

    def test_passthrough_on_batch_failure(self, failing_client):
        pp = CompresrNodePostprocessor(client=failing_client, on_error="passthrough")
        out = pp.postprocess_nodes([_nws(LONG)], query_bundle=QueryBundle(query_str="q"))
        assert out[0].node.get_content() == LONG


# ---------------------------------------------------------------------------
# wrap_tool_with_compresr
# ---------------------------------------------------------------------------


class TestWrapToolWithCompresr:
    def test_wraps_function_tool(self, fake_client):
        def search(query: str) -> str:
            """Search."""
            return LONG

        tool = FunctionTool.from_defaults(fn=search)
        wrapped = wrap_tool_with_compresr(
            tool, client=fake_client, compression_model="latte_v1", query_arg="query"
        )
        out = wrapped.call(query="find X").raw_output
        assert "<<C>>" in out
        assert fake_client.calls[0]["query"] == "find X"
        assert fake_client.calls[0]["compression_model_name"] == "latte_v1"

    def test_short_output_passes(self, fake_client):
        def search(query: str) -> str:
            """Search."""
            return "tiny"

        tool = FunctionTool.from_defaults(fn=search)
        wrapped = wrap_tool_with_compresr(tool, client=fake_client)
        out = wrapped.call(query="x").raw_output
        assert out == "tiny"
        assert fake_client.calls == []

    def test_preserves_metadata(self, fake_client):
        def search(query: str) -> str:
            """Find stuff."""
            return LONG

        tool = FunctionTool.from_defaults(fn=search, name="my_search")
        wrapped = wrap_tool_with_compresr(tool, client=fake_client)
        assert wrapped.metadata.name == "my_search"


# ---------------------------------------------------------------------------
# CompresrMemoryBlock
# ---------------------------------------------------------------------------

try:
    from llama_index.core.llms import ChatMessage  # noqa: E402

    from compresr.integrations.llamaindex import CompresrMemoryBlock  # noqa: E402

    _HAS_MEMORY_BLOCK = True
except ImportError:
    _HAS_MEMORY_BLOCK = False


@pytest.mark.skipif(not _HAS_MEMORY_BLOCK, reason="llama-index-core too old (no Memory API)")
class TestCompresrMemoryBlock:
    @pytest.mark.asyncio
    async def test_aput_accumulates_buffer(self, fake_client):
        block = CompresrMemoryBlock(client=fake_client, min_tokens=10)
        await block._aput([ChatMessage(role="user", content="hello")])
        await block._aput([ChatMessage(role="assistant", content="hi there")])
        buf = await block._aget()
        assert "user: hello" in buf
        assert "assistant: hi there" in buf

    @pytest.mark.asyncio
    async def test_atruncate_compresses_above_threshold(self, fake_client):
        block = CompresrMemoryBlock(client=fake_client, min_tokens=10, target_token=100)
        out = await block.atruncate(LONG, tokens_to_truncate=500)
        assert "<<C>>" in out
        assert len(fake_client.calls) == 1
        call = fake_client.calls[0]
        # target_token=100 with LONG ≈ 1000–2001 tokens → ratio ≥ 10
        assert call["target_compression_ratio"] >= 10.0

    @pytest.mark.asyncio
    async def test_atruncate_skips_short_content(self, fake_client):
        block = CompresrMemoryBlock(client=fake_client, min_tokens=10_000)
        out = await block.atruncate(LONG, tokens_to_truncate=500)
        assert out == LONG
        assert fake_client.calls == []

    @pytest.mark.asyncio
    async def test_atruncate_uses_tokens_to_truncate_when_no_target(self, fake_client):
        block = CompresrMemoryBlock(client=fake_client, min_tokens=10)
        await block.atruncate(LONG, tokens_to_truncate=200)
        assert len(fake_client.calls) == 1
        # Without target_token, ratio falls back to current/(current - tokens_to_truncate)
        assert fake_client.calls[0]["target_compression_ratio"] > 1.0

    @pytest.mark.asyncio
    async def test_passthrough_on_error(self, failing_client):
        block = CompresrMemoryBlock(
            client=failing_client,
            min_tokens=10,
            target_token=100,
            on_error="passthrough",
        )
        out = await block.atruncate(LONG, tokens_to_truncate=500)
        assert out == LONG

    @pytest.mark.asyncio
    async def test_atruncate_empty_content_passthrough(self, fake_client):
        block = CompresrMemoryBlock(client=fake_client, min_tokens=10)
        assert await block.atruncate("", tokens_to_truncate=100) == ""
        assert await block.atruncate("   ", tokens_to_truncate=100) == "   "
        assert fake_client.calls == []

    @pytest.mark.asyncio
    async def test_aput_skips_empty_content(self, fake_client):
        block = CompresrMemoryBlock(client=fake_client, min_tokens=10)
        await block._aput([ChatMessage(role="user", content="hello")])
        await block._aput([ChatMessage(role="assistant", content="")])
        buf = await block._aget()
        assert "user: hello" in buf
        # Empty content was skipped — no `assistant:` line.
        assert "assistant:" not in buf
