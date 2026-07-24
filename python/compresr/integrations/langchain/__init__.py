"""LangChain integration for Compresr.

    pip install compresr[langchain]

Exports:
    - ``CompresrToolMiddleware``: compress each tool output before it
      enters agent state (LangChain 1.0+ ``create_agent`` middleware).
    - ``CompresrSummarizationMiddleware``: when state grows past a token
      threshold, compress the older block into one summary message and
      replace it in state. KV-cache-friendly alternative to LangChain's
      built-in ``SummarizationMiddleware`` (Compresr instead of an LLM call).
    - ``wrap_tool_with_compression``: HOF that wraps any LangChain tool so
      its output is compressed transparently.
    - ``compress_tool_output``: decorator form of ``wrap_tool_with_compression``.
    - ``CompresrExtractor``: ``BaseDocumentCompressor`` for
      ``ContextualCompressionRetriever``. Drop-in for ``LLMChainExtractor``.
"""

from __future__ import annotations

try:
    import langchain_core  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "compresr LangChain integration requires langchain-core. "
        "Install with: pip install compresr[langchain]"
    ) from exc

from .._shared import ErrorPolicy
from .middleware import (
    CompresrPromptMiddleware,
    CompresrSummarizationMiddleware,
    CompresrToolMiddleware,
)
from .retriever import CompresrExtractor
from .wrappers import compress_tool_output, wrap_tool_with_compression

__all__ = [
    "CompresrExtractor",
    "CompresrPromptMiddleware",
    "CompresrSummarizationMiddleware",
    "CompresrToolMiddleware",
    "ErrorPolicy",
    "compress_tool_output",
    "wrap_tool_with_compression",
]
