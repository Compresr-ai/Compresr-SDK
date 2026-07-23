"""End-to-end LlamaIndex integration tests against the real Compresr backend.

Requires ``COMPRESR_API_KEY``. Skips cleanly otherwise.

Workflows exercised:
    1. ``CompresrNodePostprocessor`` shrinks retrieved nodes using a query
       from ``QueryBundle``.
    2. ``wrap_tool_with_compresr`` on a ``FunctionTool``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode  # noqa: E402
from llama_index.core.tools import FunctionTool  # noqa: E402

from compresr.integrations.llamaindex import (  # noqa: E402
    CompresrNodePostprocessor,
    wrap_tool_with_compresr,
)


def _nws(text: str, score: float = 1.0) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text), score=score)


def test_postprocessor_latte_with_query_bundle(live_client, live_long_text, live_query):
    """Real backend: query-aware compression via QueryBundle."""
    pp = CompresrNodePostprocessor(
        client=live_client,
        compression_model="latte_v1",
        target_compression_ratio=0.5,
        min_tokens=100,
    )
    nodes = [_nws(live_long_text) for _ in range(3)]
    out = pp.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=live_query))
    assert len(out) == 3
    for n in out:
        compressed = n.node.get_content()
        assert 0 < len(compressed) < len(live_long_text)


def test_wrap_tool_with_compresr_end_to_end(live_client, live_long_text, live_query):
    """Real backend: wrapped FunctionTool returns a shrunken string."""

    def search(query: str) -> str:
        """Noisy mock search."""
        return live_long_text

    tool = FunctionTool.from_defaults(fn=search)
    wrapped = wrap_tool_with_compresr(
        tool,
        client=live_client,
        compression_model="latte_v1",
        query_arg="query",
        target_compression_ratio=0.6,
        min_tokens=100,
    )

    out = wrapped.call(query=live_query).raw_output
    assert isinstance(out, str)
    assert 0 < len(out) < len(live_long_text)
