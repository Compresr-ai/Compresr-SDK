"""``CompresrMemoryBlock`` — a ``BaseMemoryBlock`` that compresses on
``atruncate`` instead of dropping or LLM-summarizing.

LlamaIndex's 2026 ``Memory`` API flushes from a short-term FIFO into
long-term blocks; each block can implement its own truncation strategy.
Compresr is a natural fit: shrink the buffer in place when token budget
is exceeded, without paying for an LLM summarization call.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .._shared import (
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_RATIO,
    ErrorPolicy,
    acompress_safe,
    build_client,
    estimate_tokens,
)

try:
    from llama_index.core.bridge.pydantic import (  # type: ignore[import-not-found]
        Field,
        PrivateAttr,
    )
    from llama_index.core.llms import ChatMessage  # type: ignore[import-not-found]
    from llama_index.core.memory.memory import BaseMemoryBlock  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CompresrMemoryBlock requires llama-index-core>=0.12 (Memory API). "
        "Install with: pip install compresr[llamaindex]"
    ) from exc


class CompresrMemoryBlock(BaseMemoryBlock[str]):
    """A memory block that aggregates messages into a buffer and compresses
    via Compresr when ``atruncate`` is asked to free tokens.

    Example::

        from llama_index.core.memory import Memory
        from compresr.integrations.llamaindex import CompresrMemoryBlock

        memory = Memory.from_defaults(
            token_limit=8_000,
            memory_blocks=[
                CompresrMemoryBlock(
                    api_key=os.environ["COMPRESR_API_KEY"],
                    target_token=2_000,
                )
            ],
        )
    """

    name: str = "compresr_compressed_history"
    priority: int = 2
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    compression_model: str = Field(default=DEFAULT_MODEL)
    query: Optional[str] = Field(
        default=None,
        description=(
            "Query passed to latte_v1. If unset, the last user line from "
            "the buffer is used (falling back to a generic placeholder)."
        ),
    )
    target_token: Optional[int] = Field(
        default=None,
        description="If set, target output budget in tokens (translates to ratio).",
    )
    target_compression_ratio: float = Field(default=DEFAULT_RATIO)
    min_tokens: int = Field(default=200)
    coarse: Optional[bool] = Field(default=None)
    on_error: ErrorPolicy = Field(default=DEFAULT_POLICY)
    client: Any = Field(default=None, exclude=True)

    _buffer: str = PrivateAttr(default="")
    _client_lazy: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self._client_lazy is None:
            self._client_lazy = build_client(
                api_key=self.api_key,
                base_url=self.base_url,
                caller="CompresrMemoryBlock",
            )
        return self._client_lazy

    async def _aget(self, messages: Optional[List[Any]] = None, **kwargs: Any) -> str:
        return self._buffer

    async def _aput(self, messages: List["ChatMessage"]) -> None:
        for m in messages:
            content = getattr(m, "content", "")
            if isinstance(content, str) and content:
                raw_role = getattr(m, "role", "") or ""
                role = getattr(raw_role, "value", raw_role)
                self._buffer = f"{self._buffer}\n{role}: {content}".strip()

    def _resolve_query(self, content: str) -> str:
        if self.query:
            return self.query
        for line in reversed(content.splitlines()):
            if line.startswith("user:"):
                return line[len("user:") :].strip() or "conversation history"
        return "conversation history"

    async def atruncate(self, content: str, tokens_to_truncate: int) -> Optional[str]:
        if not isinstance(content, str) or not content.strip():
            return content
        if estimate_tokens(content) < self.min_tokens:
            return content

        if self.target_token and self.target_token > 0:
            ratio = max(estimate_tokens(content) / self.target_token, 1.0)
        else:
            current = estimate_tokens(content)
            target = max(self.min_tokens, current - max(1, tokens_to_truncate))
            ratio = max(current / max(1, target), 1.0)

        compressed = await acompress_safe(
            self._get_client(),
            context=content,
            query=self._resolve_query(content),
            compression_model_name=self.compression_model,
            target_compression_ratio=ratio,
            coarse=self.coarse,
            min_tokens=self.min_tokens,
            on_error=self.on_error,
            context_label="memory_block",
        )
        self._buffer = compressed
        return compressed
