"""``CompresrCheckpointSerializer`` — compress long string fields before
they hit the checkpoint store (Postgres/Redis/etc).

LangGraph persists a snapshot of state after each superstep. When state
holds large strings (retrieved docs, agent scratchpads, tool outputs),
Postgres rewrites the whole TOAST row on every update — write amplification
that scales linearly with state size. Compressing the largest text fields
before serialization avoids that without changing graph semantics: the
compressed text persists in storage and flows back into state on resume.

Lossy by design: there is no decompression — what you store is what you
get back. Use ``fields={...}`` to restrict compression to known-large keys
in production.
"""

from __future__ import annotations

from typing import Any, Optional

from .._shared import (
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_RATIO,
    ErrorPolicy,
    build_client,
    compress_safe,
    estimate_tokens,
)

try:
    from langgraph.checkpoint.serde.jsonplus import (  # type: ignore[import-not-found]
        JsonPlusSerializer,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CompresrCheckpointSerializer requires langgraph. "
        "Install with: pip install compresr[langgraph]"
    ) from exc


_SENTINEL_KEY = "__compresr__"


class CompresrCheckpointSerializer(JsonPlusSerializer):
    """Drop-in replacement for ``JsonPlusSerializer`` that compresses
    large string values before encoding to bytes.

    Example::

        from langgraph.checkpoint.postgres import PostgresSaver
        from compresr.integrations.langgraph import CompresrCheckpointSerializer

        saver = PostgresSaver(
            connection_string="postgresql://...",
            serde=CompresrCheckpointSerializer(
                api_key=os.environ["COMPRESR_API_KEY"],
                min_tokens=500,
                fields={"retrieved_text", "scratchpad"},
            ),
        )
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        compression_model: str = DEFAULT_MODEL,
        query: Optional[str] = None,
        target_compression_ratio: float = DEFAULT_RATIO,
        min_tokens: int = 500,
        coarse: Optional[bool] = None,
        fields: Optional[set[str]] = None,
        on_error: ErrorPolicy = DEFAULT_POLICY,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._client = client or build_client(
            api_key=api_key, base_url=base_url, caller="CompresrCheckpointSerializer"
        )
        self._compression_model = compression_model
        self._ratio = target_compression_ratio
        self._min_tokens = max(1, min_tokens)
        self._coarse = coarse
        self._fields = fields
        self._query = query
        self._on_error = on_error

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        transformed = self._compress_in_place(obj, parent_key=None)
        return super().dumps_typed(transformed)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        # Unwrap the sentinel envelope written by ``_wrap_compressed`` so the
        # compressed *string* flows back into state on resume — not the
        # ``{__compresr__: True, "v": ...}`` marker dict. Without this, a
        # field a node wrote as ``str`` round-trips to ``dict`` and breaks
        # any downstream node that treats it as text.
        return self._decompress_in_place(super().loads_typed(data))

    def _decompress_in_place(self, value: Any) -> Any:
        if isinstance(value, dict):
            if value.get(_SENTINEL_KEY) is True and "v" in value:
                return value["v"]
            return {k: self._decompress_in_place(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._decompress_in_place(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._decompress_in_place(v) for v in value)
        return value

    def _compress_in_place(self, value: Any, *, parent_key: Optional[str]) -> Any:
        if isinstance(value, str):
            if self._should_compress(value, parent_key):
                return self._wrap_compressed(value, parent_key)
            return value
        if isinstance(value, dict):
            return {k: self._compress_in_place(v, parent_key=str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self._compress_in_place(v, parent_key=parent_key) for v in value]
        if isinstance(value, tuple):
            return tuple(self._compress_in_place(v, parent_key=parent_key) for v in value)
        return value

    def _should_compress(self, text: str, parent_key: Optional[str]) -> bool:
        if self._fields is not None and parent_key not in self._fields:
            return False
        return estimate_tokens(text) >= self._min_tokens

    def _wrap_compressed(self, text: str, parent_key: Optional[str]) -> Any:
        # latte_v1 requires a query; fall back to the field name (e.g.
        # "retrieved_text") when no explicit query was configured.
        query = self._query or parent_key or "stored content"
        compressed = compress_safe(
            self._client,
            context=text,
            query=query,
            compression_model_name=self._compression_model,
            target_compression_ratio=self._ratio,
            coarse=self._coarse,
            min_tokens=1,  # threshold already applied in _should_compress
            on_error=self._on_error,
            context_label=f"checkpoint:{parent_key or '?'}",
        )
        if compressed == text:
            return text
        return {_SENTINEL_KEY: True, "v": compressed}
