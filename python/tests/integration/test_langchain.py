"""End-to-end LangChain integration tests against the real Compresr backend.

Requires ``COMPRESR_API_KEY`` in the environment. Skips cleanly when the
key isn't set, so CI without secrets remains green.

Run::

    pytest tests/integration/test_langchain.py --prod -v

Workflows exercised:
    1. ``CompresrToolMiddleware`` on a noisy ``StructuredTool``.
    2. ``wrap_tool_with_compression`` HOF on the same tool — token math.
    3. ``CompresrExtractor`` inside ``ContextualCompressionRetriever``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from compresr.integrations.langchain import (  # noqa: E402
    CompresrExtractor,
    CompresrToolMiddleware,
    wrap_tool_with_compression,
)

# ---------------------------------------------------------------------------
# Workflow 1 — middleware on a real tool call
# ---------------------------------------------------------------------------


def test_tool_middleware_compresses_real_output(live_client, live_long_text, live_query):
    """Real backend: middleware receives a long ToolMessage and shrinks it."""

    @tool
    def web_search(query: str) -> str:
        """Noisy mock search."""
        return live_long_text

    mw = CompresrToolMiddleware(
        client=live_client,
        compression_model="latte_v1",
        query_arg="query",
        target_compression_ratio=0.5,
        min_tokens=100,
    )

    class _Req:
        pass

    req = _Req()
    req.tool_call = {"id": "t1", "name": "web_search", "args": {"query": live_query}}
    req.messages = [HumanMessage(content=live_query)]

    def handler(_req):
        return ToolMessage(content=live_long_text, tool_call_id="t1", name="web_search")

    result = mw.wrap_tool_call(req, handler)
    assert isinstance(result, ToolMessage)
    assert isinstance(result.content, str)
    assert (
        0 < len(result.content) < len(live_long_text)
    ), "expected compressed content to be strictly shorter than original"


# ---------------------------------------------------------------------------
# Workflow 2 — wrapper HOF on a real tool call
# ---------------------------------------------------------------------------


def test_wrap_tool_with_compression_end_to_end(live_client, live_long_text, live_query):
    """Real backend: wrap a tool and invoke it; output should shrink."""

    @tool
    def web_search(query: str) -> str:
        """Noisy mock search."""
        return live_long_text

    wrapped = wrap_tool_with_compression(
        web_search,
        client=live_client,
        compression_model="latte_v1",
        query_arg="query",
        target_compression_ratio=0.6,
        min_tokens=100,
    )

    out = wrapped.invoke({"query": live_query})
    assert isinstance(out, str)
    assert 0 < len(out) < len(live_long_text)


# ---------------------------------------------------------------------------
# Workflow 3 — Document compressor inside a retriever
# ---------------------------------------------------------------------------


class _StaticRetriever(BaseRetriever):
    """Returns the same long doc 3x for any query — useful for assertions."""

    def _get_relevant_documents(self, query, *, run_manager=None):
        from tests.integration.conftest import LIVE_LONG_TEXT

        return [
            Document(page_content=LIVE_LONG_TEXT, metadata={"src": "demo", "i": i})
            for i in range(3)
        ]


def test_document_compressor_in_contextual_retriever(live_client, live_query):
    """Real backend: documents passed through a ContextualCompressionRetriever
    shrink before reaching the LLM."""
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        try:
            from langchain_classic.retrievers import (
                ContextualCompressionRetriever,
            )
        except ImportError:
            pytest.skip("ContextualCompressionRetriever not available in this LangChain install")

    compressor = CompresrExtractor(
        client=live_client,
        compression_model="latte_v1",
        target_compression_ratio=0.6,
        min_tokens=100,
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=_StaticRetriever(),
    )

    docs = retriever.invoke(live_query)
    assert len(docs) == 3
    # Every returned doc must be shorter than the source AND carry our marker.
    from tests.integration.conftest import LIVE_LONG_TEXT

    for d in docs:
        assert 0 < len(d.page_content) < len(LIVE_LONG_TEXT)
        assert d.metadata.get("compresr") is True
