"""Shared 'compress one tool output' kernel.

Single source of truth used by every integration that needs to compress a
tool's textual output: given a tool name, an output value, and an optional
query, return the (possibly compressed) replacement. Today it backs
``CompresrToolMiddleware``; future facades (Anthropic, OpenAI) and a
server-tool middleware will reuse the same kernel so the eligibility +
threshold + error-policy rule stays in one place.
"""

from __future__ import annotations

from typing import Any, Optional

from .compress import acompress_safe, compress_safe
from .filters import make_filter
from .policy import CompressionPolicy


class ToolOutputCompressor:
    """Compress a single tool output according to a ``CompressionPolicy``.

    The rule is intentionally narrow: if the output is a string and the tool
    name passes the allow/ignore filter, delegate to :func:`compress_safe`
    (or :func:`acompress_safe`). Otherwise return the output unchanged.
    """

    def __init__(self, *, client: Any, policy: CompressionPolicy) -> None:
        self._client = client
        self._policy = policy
        self._eligible = make_filter(
            allow=policy.allow_tools,
            ignore=policy.ignore_tools,
        )

    def process(
        self,
        *,
        tool_name: str,
        output: Any,
        query: Optional[str],
    ) -> Any:
        """Sync path: compress ``output`` if eligible, else return as-is."""
        if not isinstance(output, str) or not self._eligible(tool_name):
            return output
        return compress_safe(
            self._client,
            context=output,
            query=query,
            compression_model_name=self._policy.compression_model_name,
            target_compression_ratio=self._policy.target_compression_ratio,
            coarse=self._policy.coarse,
            min_tokens=self._policy.min_tokens,
            on_error=self._policy.on_error,
            context_label=f"tool:{tool_name or '?'}",
        )

    async def aprocess(
        self,
        *,
        tool_name: str,
        output: Any,
        query: Optional[str],
    ) -> Any:
        """Async path: compress ``output`` if eligible, else return as-is."""
        if not isinstance(output, str) or not self._eligible(tool_name):
            return output
        return await acompress_safe(
            self._client,
            context=output,
            query=query,
            compression_model_name=self._policy.compression_model_name,
            target_compression_ratio=self._policy.target_compression_ratio,
            coarse=self._policy.coarse,
            min_tokens=self._policy.min_tokens,
            on_error=self._policy.on_error,
            context_label=f"tool:{tool_name or '?'}",
        )


__all__ = ["ToolOutputCompressor"]
