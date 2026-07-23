"""Tests for compresr.integrations.langgraph — node + re-exported middleware."""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402

from compresr.integrations.langgraph import (  # noqa: E402
    CompresrSummarizationMiddleware,
    CompresrToolMiddleware,
    compresr_node,
    make_compresr_node,
)


def test_compresr_node_alias_identical():
    assert compresr_node is make_compresr_node


LONG = "x " * 2000


class TestMakeCompresrNode:
    def test_node_compresses_state_field(self, fake_client):
        node = make_compresr_node(
            client=fake_client,
            context_key="ctx",
            compression_model="latte_v1",
            query_key="q",
        )
        out = node({"ctx": LONG, "q": "find X"})
        assert "ctx" in out and "<<C>>" in out["ctx"]
        assert fake_client.calls[0]["query"] == "find X"

    def test_node_passes_short_through(self, fake_client):
        node = make_compresr_node(client=fake_client, context_key="ctx")
        assert node({"ctx": "tiny"}) == {}
        assert fake_client.calls == []

    def test_static_query_takes_priority(self, fake_client):
        node = make_compresr_node(
            client=fake_client,
            context_key="ctx",
            compression_model="latte_v1",
            query="FIXED",
            query_key="q",
        )
        node({"ctx": LONG, "q": "ignored"})
        assert fake_client.calls[0]["query"] == "FIXED"

    def test_query_extractor_callable(self, fake_client):
        node = make_compresr_node(
            client=fake_client,
            context_key="ctx",
            compression_model="latte_v1",
            query_extractor=lambda s: f"derived:{s.get('topic')}",
        )
        node({"ctx": LONG, "topic": "ml"})
        assert fake_client.calls[0]["query"] == "derived:ml"

    def test_passthrough_on_error(self, failing_client):
        node = make_compresr_node(client=failing_client, context_key="ctx", on_error="passthrough")
        out = node({"ctx": LONG})
        assert out == {} or out["ctx"] == LONG


class TestMiddlewareReExport:
    def test_tool_middleware_works_through_langgraph_namespace(self, fake_client):
        # Smoke test: middleware imported via langgraph namespace works.
        mw = CompresrToolMiddleware(client=fake_client)

        def handler(_req):
            return ToolMessage(content=LONG, tool_call_id="t1", name="search")

        class _Req:
            pass

        req = _Req()
        req.tool_call = {"id": "t1", "name": "search", "args": {"query": "X"}}
        req.messages = [HumanMessage(content="X")]
        out = mw.wrap_tool_call(req, handler)
        assert "<<C>>" in out.content

    def test_summarization_middleware_works_through_langgraph_namespace(self, fake_client):
        mw = CompresrSummarizationMiddleware(
            client=fake_client,
            max_tokens_before_summary=100,
            messages_to_keep=2,
        )
        msgs = []
        for i in range(8):
            msgs.append(HumanMessage(content=f"q{i}"))
            msgs.append(ToolMessage(content=LONG, tool_call_id=f"t{i}", name="search"))
        out = mw.before_model({"messages": msgs}, runtime=None)
        assert out is not None
        new_msgs = out["messages"]
        # summary at index 1 (index 0 is RemoveMessage marker)
        assert "[Earlier conversation summary]" in new_msgs[1].content
        # last 2 messages preserved
        assert new_msgs[-2:] == msgs[-2:]


# ---------------------------------------------------------------------------
# CompresrCheckpointSerializer
# ---------------------------------------------------------------------------


from langgraph.types import Command  # noqa: E402

from compresr.integrations.langgraph import compresr_handoff_tool  # noqa: E402
from compresr.integrations.langgraph import (  # noqa: E402
    CompresrCheckpointSerializer,
)


def _tool_call(name: str, args: dict, tool_call_id: str) -> dict:
    """Build the ToolCall envelope LangChain requires when InjectedToolCallId is used."""
    return {
        "name": name,
        "args": args,
        "type": "tool_call",
        "id": tool_call_id,
    }


class TestCompresrHandoffTool:
    def test_tool_metadata(self, fake_client):
        t = compresr_handoff_tool("researcher", client=fake_client)
        assert t.name == "transfer_to_researcher"
        assert "researcher" in t.description

    def test_handoff_compresses_long_task_and_context(self, fake_client):
        t = compresr_handoff_tool("researcher", client=fake_client, min_tokens=10)
        cmd = t.invoke(
            _tool_call(
                "transfer_to_researcher",
                {"task_description": LONG, "context": LONG},
                "tc1",
            )
        )
        assert isinstance(cmd, Command)
        assert cmd.goto == "researcher"
        assert cmd.graph == Command.PARENT
        update = cmd.update
        assert "<<C>>" in update["task_description"]
        assert "<<C>>" in update["context"]
        # Two compress calls — one per long field.
        assert len(fake_client.calls) == 2
        # The ack ToolMessage carries the tool_call_id.
        assert update["messages"][0].tool_call_id == "tc1"

    def test_short_payload_passes_through(self, fake_client):
        t = compresr_handoff_tool("writer", client=fake_client, min_tokens=10_000)
        cmd = t.invoke(_tool_call("transfer_to_writer", {"task_description": "tiny"}, "tc2"))
        assert isinstance(cmd, Command)
        # No compress call — body below threshold.
        assert fake_client.calls == []
        # Empty context is not compressed.
        assert cmd.update["context"] == ""

    def test_passthrough_on_error(self, failing_client):
        t = compresr_handoff_tool(
            "writer", client=failing_client, min_tokens=10, on_error="passthrough"
        )
        cmd = t.invoke(_tool_call("transfer_to_writer", {"task_description": LONG}, "tc3"))
        # Original survives.
        assert cmd.update["task_description"] == LONG

    def test_only_task_long_no_context(self, fake_client):
        t = compresr_handoff_tool("researcher", client=fake_client, min_tokens=10)
        cmd = t.invoke(_tool_call("transfer_to_researcher", {"task_description": LONG}, "tc4"))
        # Only the task was sent through Compresr.
        assert len(fake_client.calls) == 1
        assert cmd.update["context"] == ""

    def test_custom_description_override(self, fake_client):
        t = compresr_handoff_tool(
            "researcher",
            client=fake_client,
            description="Custom handoff description",
        )
        assert t.description == "Custom handoff description"


# ---------------------------------------------------------------------------
# CompresrStore
# ---------------------------------------------------------------------------


from langgraph.store.memory import InMemoryStore  # noqa: E402

from compresr.integrations.langgraph import CompresrStore  # noqa: E402


class TestCompresrStore:
    def test_put_compresses_long_string_fields(self, fake_client):
        inner = InMemoryStore()
        store = CompresrStore(inner, client=fake_client, min_tokens=10)
        store.put(("u1",), "doc1", {"long": LONG, "short": "hi"})
        item = inner.get(("u1",), "doc1")
        assert item is not None
        assert "<<C>>" in item.value["long"]
        assert item.value["short"] == "hi"
        assert len(fake_client.calls) == 1

    def test_field_allowlist(self, fake_client):
        inner = InMemoryStore()
        store = CompresrStore(inner, client=fake_client, min_tokens=10, fields={"history"})
        store.put(("u1",), "doc1", {"history": LONG, "other_big": LONG})
        assert len(fake_client.calls) == 1
        item = inner.get(("u1",), "doc1")
        assert item is not None
        assert "<<C>>" in item.value["history"]
        # other_big not in allowlist → untouched.
        assert item.value["other_big"] == LONG

    def test_short_values_passthrough(self, fake_client):
        inner = InMemoryStore()
        store = CompresrStore(inner, client=fake_client, min_tokens=10_000)
        store.put(("u1",), "doc1", {"value": LONG})
        assert fake_client.calls == []
        item = inner.get(("u1",), "doc1")
        assert item is not None
        assert item.value["value"] == LONG

    def test_get_is_passthrough(self, fake_client):
        inner = InMemoryStore()
        # Seed directly via the inner store.
        inner.put(("u1",), "doc1", {"long": LONG})
        store = CompresrStore(inner, client=fake_client, min_tokens=10)
        item = store.get(("u1",), "doc1")
        assert item is not None
        # No compression on get — original survives untouched.
        assert item.value["long"] == LONG
        assert fake_client.calls == []

    def test_passthrough_on_error(self, failing_client):
        inner = InMemoryStore()
        store = CompresrStore(inner, client=failing_client, min_tokens=10, on_error="passthrough")
        store.put(("u1",), "doc1", {"long": LONG})
        item = inner.get(("u1",), "doc1")
        assert item is not None
        assert item.value["long"] == LONG

    def test_nested_dict_and_list_values(self, fake_client):
        inner = InMemoryStore()
        store = CompresrStore(inner, client=fake_client, min_tokens=10)
        store.put(("u1",), "doc1", {"outer": {"inner": LONG}, "lst": [LONG, "tiny"]})
        item = inner.get(("u1",), "doc1")
        assert item is not None
        assert "<<C>>" in item.value["outer"]["inner"]
        assert "<<C>>" in item.value["lst"][0]
        assert item.value["lst"][1] == "tiny"
        assert len(fake_client.calls) == 2

    def test_batch_rewrites_putop(self, fake_client):
        from langgraph.store.base import PutOp

        inner = InMemoryStore()
        store = CompresrStore(inner, client=fake_client, min_tokens=10)
        op = PutOp(namespace=("u1",), key="doc1", value={"long": LONG}, index=None, ttl=None)
        store.batch([op])
        item = inner.get(("u1",), "doc1")
        assert item is not None
        assert "<<C>>" in item.value["long"]

    @pytest.mark.asyncio
    async def test_aput_async_path(self, fake_client):
        inner = InMemoryStore()
        store = CompresrStore(inner, client=fake_client, min_tokens=10)
        await store.aput(("u1",), "doc1", {"long": LONG})
        item = await inner.aget(("u1",), "doc1")
        assert item is not None
        assert "<<C>>" in item.value["long"]


class TestCompresrCheckpointSerializer:
    def test_compresses_long_field_only(self, fake_client):
        ser = CompresrCheckpointSerializer(client=fake_client, min_tokens=10)
        obj = {"long": LONG, "short": "hi"}
        type_tag, encoded = ser.dumps_typed(obj)
        # Round-trip through the same serializer so the format tag matches.
        decoded = ser.loads_typed((type_tag, encoded))
        assert decoded["short"] == "hi"
        # Round-trip yields the compressed *string*, not the sentinel wrapper:
        # loads_typed unwraps it so resumed state stays text.
        assert isinstance(decoded["long"], str)
        assert "<<C>>" in decoded["long"]
        # Exactly one compress call (long only).
        assert len(fake_client.calls) == 1

    def test_field_allowlist_respected(self, fake_client):
        ser = CompresrCheckpointSerializer(
            client=fake_client,
            min_tokens=10,
            fields={"history"},
        )
        obj = {"history": LONG, "other_big": LONG}
        ser.dumps_typed(obj)
        # Only `history` was compressed.
        assert len(fake_client.calls) == 1
        assert fake_client.calls[0]["context"] == LONG

    def test_short_field_unchanged(self, fake_client):
        ser = CompresrCheckpointSerializer(client=fake_client, min_tokens=10_000)
        obj = {"foo": LONG}
        ser.dumps_typed(obj)
        assert fake_client.calls == []

    def test_nested_dict_and_list(self, fake_client):
        ser = CompresrCheckpointSerializer(client=fake_client, min_tokens=10)
        obj = {"outer": {"inner": LONG, "tiny": "x"}, "list": [LONG, "tiny"]}
        ser.dumps_typed(obj)
        # `inner` + list[0] both compressed → 2 calls.
        assert len(fake_client.calls) == 2

    def test_passthrough_on_error(self, failing_client):
        ser = CompresrCheckpointSerializer(
            client=failing_client, min_tokens=10, on_error="passthrough"
        )
        type_tag, encoded = ser.dumps_typed({"long": LONG})
        decoded = ser.loads_typed((type_tag, encoded))
        # Original string survives untouched.
        assert decoded["long"] == LONG

    def test_non_string_values_unchanged(self, fake_client):
        ser = CompresrCheckpointSerializer(client=fake_client, min_tokens=10)
        obj = {"count": 42, "ratio": 0.5, "flag": True, "missing": None}
        type_tag, encoded = ser.dumps_typed(obj)
        decoded = ser.loads_typed((type_tag, encoded))
        assert decoded == obj
        assert fake_client.calls == []
