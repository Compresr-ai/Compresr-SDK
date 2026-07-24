"""Tests for compresr.integrations.langchain — middleware, wrappers, retriever."""

from __future__ import annotations

import asyncio

import pytest

langchain_core = pytest.importorskip("langchain_core")
pytest.importorskip("langchain")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import StructuredTool, tool  # noqa: E402

from compresr.integrations.langchain import (  # noqa: E402
    CompresrExtractor,
    CompresrSummarizationMiddleware,
    CompresrToolMiddleware,
    compress_tool_output,
    wrap_tool_with_compression,
)

LONG_OUTPUT = "x " * 2000  # ~4000 chars => well above 200-token threshold


# ---------------------------------------------------------------------------
# CompresrToolMiddleware
# ---------------------------------------------------------------------------


def _make_request(*, tool_call: dict, messages: list):
    class _Req:
        pass

    r = _Req()
    r.tool_call = tool_call
    r.messages = messages
    return r


class TestCompresrToolMiddleware:
    def test_compresses_long_tool_output(self, fake_client):
        mw = CompresrToolMiddleware(client=fake_client)

        def handler(_req):
            return ToolMessage(content=LONG_OUTPUT, tool_call_id="t1", name="search")

        req = _make_request(
            tool_call={"id": "t1", "name": "search", "args": {"query": "find me X"}},
            messages=[HumanMessage(content="please find X")],
        )
        out = mw.wrap_tool_call(req, handler)
        assert "<<C>>" in out.content
        assert len(fake_client.calls) == 1
        # latte_v1 + query was picked from args, not history
        assert fake_client.calls[0]["query"] == "find me X"
        assert fake_client.calls[0]["compression_model_name"] == "latte_v1"

    def test_short_output_passes_through(self, fake_client):
        mw = CompresrToolMiddleware(client=fake_client, min_tokens=200)

        def handler(_req):
            return ToolMessage(content="tiny", tool_call_id="t1", name="search")

        req = _make_request(tool_call={"id": "t1", "name": "search"}, messages=[])
        out = mw.wrap_tool_call(req, handler)
        assert out.content == "tiny"
        assert fake_client.calls == []

    def test_allow_list_excludes_other_tools(self, fake_client):
        mw = CompresrToolMiddleware(client=fake_client, allow_tools={"search"})

        def handler(_req):
            return ToolMessage(content=LONG_OUTPUT, tool_call_id="t1", name="fetch_url")

        req = _make_request(tool_call={"id": "t1", "name": "fetch_url"}, messages=[])
        out = mw.wrap_tool_call(req, handler)
        assert out.content == LONG_OUTPUT  # untouched
        assert fake_client.calls == []

    def test_static_query_overrides_args(self, fake_client):
        mw = CompresrToolMiddleware(client=fake_client, query="FIXED")

        def handler(_req):
            return ToolMessage(content=LONG_OUTPUT, tool_call_id="t1", name="search")

        req = _make_request(
            tool_call={"id": "t1", "name": "search", "args": {"query": "ignored"}},
            messages=[],
        )
        mw.wrap_tool_call(req, handler)
        assert fake_client.calls[0]["query"] == "FIXED"

    def test_query_arg_picks_named_key(self, fake_client):
        mw = CompresrToolMiddleware(client=fake_client, query_arg="url")

        def handler(_req):
            return ToolMessage(content=LONG_OUTPUT, tool_call_id="t1", name="fetch")

        req = _make_request(
            tool_call={"id": "t1", "name": "fetch", "args": {"url": "https://example.com"}},
            messages=[],
        )
        mw.wrap_tool_call(req, handler)
        assert fake_client.calls[0]["query"] == "https://example.com"

    def test_arbitrary_model_name_passes_through(self, fake_client):
        # SDK is permissive — backend validates model names.
        mw = CompresrToolMiddleware(client=fake_client, compression_model="future_v3")

        def handler(_req):
            return ToolMessage(content=LONG_OUTPUT, tool_call_id="t1", name="t")

        req = _make_request(
            tool_call={"id": "t1", "name": "t", "args": {"query": "anything"}},
            messages=[],
        )
        mw.wrap_tool_call(req, handler)
        assert fake_client.calls[0]["compression_model_name"] == "future_v3"

    def test_passthrough_on_error(self, failing_client):
        mw = CompresrToolMiddleware(client=failing_client, on_error="passthrough")

        def handler(_req):
            return ToolMessage(content=LONG_OUTPUT, tool_call_id="t1", name="t")

        req = _make_request(tool_call={"id": "t1", "name": "t"}, messages=[])
        out = mw.wrap_tool_call(req, handler)
        assert out.content == LONG_OUTPUT  # original returned

    def test_raise_policy(self, failing_client):
        mw = CompresrToolMiddleware(client=failing_client, on_error="raise")

        def handler(_req):
            return ToolMessage(content=LONG_OUTPUT, tool_call_id="t1", name="t")

        req = _make_request(tool_call={"id": "t1", "name": "t"}, messages=[])
        with pytest.raises(RuntimeError):
            mw.wrap_tool_call(req, handler)

    def test_awrap_tool_call_runs_async(self, fake_client):
        mw = CompresrToolMiddleware(client=fake_client)

        async def handler(_req):
            return ToolMessage(content=LONG_OUTPUT, tool_call_id="t1", name="search")

        req = _make_request(
            tool_call={"id": "t1", "name": "search", "args": {"query": "x"}},
            messages=[],
        )
        out = asyncio.run(mw.awrap_tool_call(req, handler))
        assert "<<C>>" in out.content


# ---------------------------------------------------------------------------
# CompresrSummarizationMiddleware
# ---------------------------------------------------------------------------


class TestCompresrSummarizationMiddleware:
    def _build_messages(self, n: int) -> list:
        """n turn-pairs of HumanMessage + ToolMessage with long content."""
        out = []
        for i in range(n):
            out.append(HumanMessage(content=f"question {i}"))
            out.append(
                ToolMessage(
                    content=LONG_OUTPUT,
                    tool_call_id=f"t{i}",
                    name="search",
                )
            )
        return out

    def test_under_threshold_is_noop(self, fake_client):
        mw = CompresrSummarizationMiddleware(
            client=fake_client,
            max_tokens_before_summary=1_000_000,
            messages_to_keep=2,
        )
        out = mw.before_model({"messages": self._build_messages(3)}, runtime=None)
        assert out is None
        assert fake_client.calls == []

    def test_few_messages_is_noop(self, fake_client):
        mw = CompresrSummarizationMiddleware(
            client=fake_client, max_tokens_before_summary=1, messages_to_keep=20
        )
        out = mw.before_model({"messages": self._build_messages(2)}, runtime=None)
        assert out is None
        assert fake_client.calls == []

    def test_above_threshold_summarizes_and_keeps_recent(self, fake_client):
        from langchain_core.messages import RemoveMessage

        msgs = self._build_messages(10)  # 20 messages, >>4000 tokens
        mw = CompresrSummarizationMiddleware(
            client=fake_client,
            max_tokens_before_summary=100,
            messages_to_keep=4,
        )
        out = mw.before_model({"messages": msgs}, runtime=None)
        assert out is not None
        new_messages = out["messages"]
        # First entry: RemoveMessage; second: summary HumanMessage; then recent tail.
        assert isinstance(new_messages[0], RemoveMessage)
        assert isinstance(new_messages[1], HumanMessage)
        assert new_messages[1].content.startswith("[Earlier conversation summary]")
        # Recent 4 messages preserved verbatim
        assert new_messages[-4:] == msgs[-4:]
        # Exactly ONE compress call (not per-message)
        assert len(fake_client.calls) == 1

    def test_idempotent_re_run_folds_into_single_summary(self, fake_client):
        msgs = self._build_messages(10)
        mw = CompresrSummarizationMiddleware(
            client=fake_client,
            max_tokens_before_summary=100,
            messages_to_keep=4,
        )
        first = mw.before_model({"messages": msgs}, runtime=None)
        # Simulate state after the previous summary is applied + 2 new turns
        new_state = first["messages"][1:] + self._build_messages(2)  # drop RemoveMessage marker
        # First message is the summary; subsequent are recent + 2 new pairs.
        second = mw.before_model({"messages": new_state}, runtime=None)
        # Second summary still starts with the prefix — no nested wrapping.
        new_messages = second["messages"]
        assert new_messages[1].content.startswith("[Earlier conversation summary]")
        # And it doesn't double-prefix.
        assert new_messages[1].content.count("[Earlier conversation summary]") == 1

    def test_trigger_and_keep_aliases_override_long_names(self, fake_client):
        # LangChain `SummarizationMiddleware`-style param names.
        mw = CompresrSummarizationMiddleware(
            client=fake_client,
            trigger=100,
            keep=4,
        )
        out = mw.before_model({"messages": self._build_messages(10)}, runtime=None)
        assert out is not None
        assert len(fake_client.calls) == 1

    def test_token_counter_used(self, fake_client):
        # A counter that always returns 1 keeps us under any threshold.
        mw = CompresrSummarizationMiddleware(
            client=fake_client,
            trigger=100,
            keep=4,
            token_counter=lambda _s: 1,
        )
        out = mw.before_model({"messages": self._build_messages(10)}, runtime=None)
        assert out is None
        assert fake_client.calls == []


# ---------------------------------------------------------------------------
# CompresrPromptMiddleware
# ---------------------------------------------------------------------------


from compresr.integrations.langchain import CompresrPromptMiddleware  # noqa: E402


class _FakeModelRequest:
    """Mimics LangChain's ModelRequest enough for wrap_model_call."""

    def __init__(self, messages):
        self.messages = messages


class TestCompresrPromptMiddleware:
    def _msgs(self, n: int, body: str) -> list:
        return [HumanMessage(content=body) for _ in range(n)]

    def test_under_budget_is_noop(self, fake_client):
        mw = CompresrPromptMiddleware(client=fake_client, max_tokens=1_000_000)
        req = _FakeModelRequest(self._msgs(3, LONG_OUTPUT))
        original = list(req.messages)

        def handler(r):
            return ("ok", r.messages)

        out = mw.wrap_model_call(req, handler)
        assert out[0] == "ok"
        assert out[1] == original
        assert fake_client.calls == []

    def test_over_budget_compresses_largest_first(self, fake_client):
        # 3 long messages; budget tight → at least one compressed.
        mw = CompresrPromptMiddleware(
            client=fake_client,
            max_tokens=500,
            min_tokens=50,
            token_counter=lambda s: len(s),  # 1 char = 1 "token" for predictability
        )
        # 3 messages of length 1000 chars each = 3000 chars total, budget 500.
        msgs = [HumanMessage(content="z" * 1000) for _ in range(3)]
        req = _FakeModelRequest(msgs)

        captured = {}

        def handler(r):
            captured["messages"] = list(r.messages)
            return "done"

        out = mw.wrap_model_call(req, handler)
        assert out == "done"
        # At least one compress call fired.
        assert len(fake_client.calls) >= 1
        # Total content tokens after shrink should be lower than before.
        new_total = sum(len(m.content) for m in captured["messages"])
        assert new_total < 3000

    def test_skips_short_messages(self, fake_client):
        mw = CompresrPromptMiddleware(
            client=fake_client,
            max_tokens=10,
            min_tokens=1000,
            token_counter=lambda s: len(s),
        )
        # All messages below min_tokens — nothing eligible to compress.
        msgs = [HumanMessage(content="hello") for _ in range(3)]
        req = _FakeModelRequest(msgs)
        mw.wrap_model_call(req, lambda r: None)
        assert fake_client.calls == []

    def test_passthrough_on_error(self, failing_client):
        mw = CompresrPromptMiddleware(
            client=failing_client,
            max_tokens=10,
            min_tokens=10,
            token_counter=lambda s: len(s),
            on_error="passthrough",
        )
        msgs = [HumanMessage(content="z" * 1000)]
        req = _FakeModelRequest(msgs)
        captured = {}

        def handler(r):
            captured["messages"] = list(r.messages)
            return "ok"

        mw.wrap_model_call(req, handler)
        # Original content survives — backend failure passed through.
        assert captured["messages"][0].content == "z" * 1000

    def test_async_path(self, fake_client):
        mw = CompresrPromptMiddleware(
            client=fake_client,
            max_tokens=500,
            min_tokens=50,
            token_counter=lambda s: len(s),
        )
        msgs = [HumanMessage(content="z" * 1000) for _ in range(3)]
        req = _FakeModelRequest(msgs)

        async def handler(r):
            return ("async-ok", len(r.messages))

        out = asyncio.run(mw.awrap_model_call(req, handler))
        assert out[0] == "async-ok"
        assert len(fake_client.calls) >= 1

    def test_preserves_message_types_and_metadata(self, fake_client):
        mw = CompresrPromptMiddleware(
            client=fake_client,
            max_tokens=100,
            min_tokens=50,
            token_counter=lambda s: len(s),
        )
        msgs = [
            HumanMessage(content="z" * 1000),
            ToolMessage(content="z" * 1000, tool_call_id="tc-keep", name="search"),
        ]
        req = _FakeModelRequest(msgs)

        captured = {}

        def handler(r):
            captured["messages"] = list(r.messages)
            return None

        mw.wrap_model_call(req, handler)
        out_msgs = captured["messages"]
        assert isinstance(out_msgs[0], HumanMessage)
        assert isinstance(out_msgs[1], ToolMessage)
        # tool_call_id and name preserved through rebuild.
        assert out_msgs[1].tool_call_id == "tc-keep"
        assert out_msgs[1].name == "search"


# ---------------------------------------------------------------------------
# wrap_tool_with_compression
# ---------------------------------------------------------------------------


class TestWrapToolWithCompression:
    def test_preserves_name_description(self, fake_client):
        @tool
        def my_search(query: str) -> str:
            """Search the web."""
            return LONG_OUTPUT

        wrapped = wrap_tool_with_compression(my_search, client=fake_client)
        assert wrapped.name == "my_search"
        assert "Search" in wrapped.description

    def test_compresses_output(self, fake_client):
        @tool
        def my_search(query: str) -> str:
            """Search the web."""
            return LONG_OUTPUT

        wrapped = wrap_tool_with_compression(
            my_search, client=fake_client, compression_model="latte_v1", query_arg="query"
        )
        out = wrapped.invoke({"query": "find me X"})
        assert "<<C>>" in out
        assert fake_client.calls[0]["query"] == "find me X"

    def test_short_output_passes(self, fake_client):
        @tool
        def my_search(query: str) -> str:
            """Search the web."""
            return "tiny"

        wrapped = wrap_tool_with_compression(my_search, client=fake_client)
        assert wrapped.invoke({"query": "anything"}) == "tiny"
        assert fake_client.calls == []

    def test_decorator_form(self, fake_client):
        @compress_tool_output(client=fake_client)
        @tool
        def my_search(query: str) -> str:
            """Search the web."""
            return LONG_OUTPUT

        assert isinstance(my_search, StructuredTool) or hasattr(my_search, "invoke")
        out = my_search.invoke({"query": "x"})
        assert "<<C>>" in out


# ---------------------------------------------------------------------------
# CompresrExtractor
# ---------------------------------------------------------------------------


class TestCompresrExtractor:
    def test_compresses_long_docs(self, fake_client):
        comp = CompresrExtractor(client=fake_client)
        docs = [
            Document(page_content=LONG_OUTPUT, metadata={"src": "a"}),
            Document(page_content=LONG_OUTPUT, metadata={"src": "b"}),
        ]
        out = comp.compress_documents(docs, query="my query")
        assert len(out) == 2
        for d in out:
            assert "<<C>>" in d.page_content
            assert d.metadata["compresr"] is True
        # One batch call, two contexts
        assert len(fake_client.batch_calls) == 1
        assert fake_client.batch_calls[0]["queries"] == "my query"
        assert fake_client.batch_calls[0]["compression_model_name"] == "latte_v1"

    def test_short_docs_pass_through(self, fake_client):
        comp = CompresrExtractor(client=fake_client)
        docs = [Document(page_content="short", metadata={})]
        out = comp.compress_documents(docs, query="q")
        assert out[0].page_content == "short"
        assert fake_client.batch_calls == []

    def test_passthrough_on_batch_failure(self, failing_client):
        comp = CompresrExtractor(client=failing_client, on_error="passthrough")
        docs = [Document(page_content=LONG_OUTPUT, metadata={})]
        out = comp.compress_documents(docs, query="q")
        assert out[0].page_content == LONG_OUTPUT

    def test_raise_policy_propagates(self, failing_client):
        comp = CompresrExtractor(client=failing_client, on_error="raise")
        docs = [Document(page_content=LONG_OUTPUT, metadata={})]
        with pytest.raises(RuntimeError):
            comp.compress_documents(docs, query="q")

    def test_acompress_documents_async(self, fake_client):
        comp = CompresrExtractor(client=fake_client)
        docs = [Document(page_content=LONG_OUTPUT, metadata={"src": "a"})]
        out = asyncio.run(comp.acompress_documents(docs, query="q"))
        assert "<<C>>" in out[0].page_content
        assert len(fake_client.batch_calls) == 1

    def test_batch_boundary_101_docs(self, fake_client):
        comp = CompresrExtractor(client=fake_client)
        docs = [Document(page_content=LONG_OUTPUT, metadata={"i": i}) for i in range(101)]
        out = comp.compress_documents(docs, query="q")
        assert len(out) == 101
        # Two batch calls: 100 + 1
        assert len(fake_client.batch_calls) == 2
        assert len(fake_client.batch_calls[0]["contexts"]) == 100
        assert len(fake_client.batch_calls[1]["contexts"]) == 1
        for d in out:
            assert "<<C>>" in d.page_content


# ---------------------------------------------------------------------------
# Wrapper async path + TypeError for non-StructuredTool
# ---------------------------------------------------------------------------


class TestWrapToolAsync:
    def test_wrap_compresses_async_coroutine(self, fake_client):
        async def search_impl(query: str) -> str:
            return LONG_OUTPUT

        async_tool = StructuredTool.from_function(
            coroutine=search_impl, name="search", description="search"
        )
        wrapped = wrap_tool_with_compression(
            async_tool,
            client=fake_client,
            compression_model="latte_v1",
            query_arg="query",
        )
        out = asyncio.run(wrapped.ainvoke({"query": "find X"}))
        assert "<<C>>" in out
        assert fake_client.calls[0]["query"] == "find X"

    def test_rejects_non_structured_tool(self, fake_client):
        from langchain_core.tools import BaseTool

        class _Custom(BaseTool):
            name: str = "x"
            description: str = "x"

            def _run(self, *a, **kw):
                return "hi"

        with pytest.raises(TypeError, match="StructuredTool"):
            wrap_tool_with_compression(_Custom(), client=fake_client)
