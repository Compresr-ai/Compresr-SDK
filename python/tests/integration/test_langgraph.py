"""End-to-end LangGraph integration tests against the real Compresr backend.

Requires ``COMPRESR_API_KEY``. Skips cleanly otherwise.

Workflow:
    ``make_compresr_node`` inside a 3-node ``StateGraph``
    (retrieve → compress → consume).
"""

from __future__ import annotations

from typing import TypedDict

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langgraph.graph import END, START, StateGraph  # noqa: E402

from compresr.integrations.langgraph import make_compresr_node  # noqa: E402


class _State(TypedDict, total=False):
    user_question: str
    retrieved_text: str


def test_compresr_node_in_full_graph(live_client, live_long_text, live_query):
    def retrieve(_state: _State) -> dict:
        return {"retrieved_text": live_long_text}

    def consume(state: _State) -> dict:
        return {"retrieved_text": state["retrieved_text"]}

    compress = make_compresr_node(
        client=live_client,
        context_key="retrieved_text",
        query_key="user_question",
        compression_model="latte_v1",
        target_compression_ratio=0.5,
        min_tokens=100,
    )

    graph = StateGraph(_State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("compress", compress)
    graph.add_node("consume", consume)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "compress")
    graph.add_edge("compress", "consume")
    graph.add_edge("consume", END)
    app = graph.compile()

    out = app.invoke({"user_question": live_query, "retrieved_text": ""})
    assert "retrieved_text" in out
    assert 0 < len(out["retrieved_text"]) < len(live_long_text)
