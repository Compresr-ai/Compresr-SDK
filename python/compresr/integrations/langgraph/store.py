"""``CompresrStore`` — wrap a LangGraph ``BaseStore`` and compress long
string values on write.

LangGraph stores (``InMemoryStore``, ``PostgresStore``, ``RedisStore``,
custom implementations) persist arbitrary JSON-like dicts under
``(namespace, key)``. When agents stash long retrieved text, scratchpads,
or chat history fragments, storage size and read-time prompt budgets
balloon fast.

This wrapper composes a real store, intercepts ``put``/``aput``/``batch``,
and rewrites string fields above ``min_tokens`` into compressed form.
Reads and other ops pass straight through. The compression is **lossy**
— the compressed string is what comes back on read, by design.

Use the ``fields`` allowlist in production to restrict the rewrite to
known-large keys (e.g. ``{"retrieved_text", "scratchpad"}``).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .._shared import (
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_RATIO,
    ErrorPolicy,
    acompress_safe,
    build_client,
    compress_safe,
    estimate_tokens,
)

try:
    from langgraph.store.base import BaseStore, PutOp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CompresrStore requires langgraph. Install with: pip install compresr[langgraph]"
    ) from exc


class CompresrStore(BaseStore):
    """Drop-in ``BaseStore`` that compresses long string values on write.

    Example::

        from langgraph.store.memory import InMemoryStore
        from compresr.integrations.langgraph import CompresrStore

        store = CompresrStore(
            InMemoryStore(),
            api_key=os.environ["COMPRESR_API_KEY"],
            fields={"retrieved_text"},
            min_tokens=500,
        )
    """

    def __init__(
        self,
        inner: BaseStore,
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
        self._inner = inner
        self._client = client or build_client(
            api_key=api_key, base_url=base_url, caller="CompresrStore"
        )
        self._compression_model = compression_model
        self._ratio = target_compression_ratio
        self._min_tokens = max(1, min_tokens)
        self._coarse = coarse
        self._fields = fields
        self._query = query
        self._on_error = on_error

    def _should_compress(self, text: str, parent_key: Optional[str]) -> bool:
        if self._fields is not None and parent_key not in self._fields:
            return False
        return estimate_tokens(text) >= self._min_tokens

    def _query_for(self, parent_key: Optional[str]) -> str:
        # latte_v1 requires a query; fall back to the field name when no
        # explicit one was configured.
        return self._query or parent_key or "stored content"

    def _compress(self, text: str, parent_key: Optional[str]) -> str:
        return compress_safe(
            self._client,
            context=text,
            query=self._query_for(parent_key),
            compression_model_name=self._compression_model,
            target_compression_ratio=self._ratio,
            coarse=self._coarse,
            min_tokens=1,
            on_error=self._on_error,
            context_label=f"store:{parent_key or '?'}",
        )

    async def _acompress(self, text: str, parent_key: Optional[str]) -> str:
        return await acompress_safe(
            self._client,
            context=text,
            query=self._query_for(parent_key),
            compression_model_name=self._compression_model,
            target_compression_ratio=self._ratio,
            coarse=self._coarse,
            min_tokens=1,
            on_error=self._on_error,
            context_label=f"store:{parent_key or '?'}",
        )

    def _walk_compress(self, value: Any, parent_key: Optional[str]) -> Any:
        if isinstance(value, str):
            if self._should_compress(value, parent_key):
                return self._compress(value, parent_key)
            return value
        if isinstance(value, dict):
            return {k: self._walk_compress(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self._walk_compress(v, parent_key) for v in value]
        if isinstance(value, tuple):
            return tuple(self._walk_compress(v, parent_key) for v in value)
        return value

    async def _awalk_compress(self, value: Any, parent_key: Optional[str]) -> Any:
        if isinstance(value, str):
            if self._should_compress(value, parent_key):
                return await self._acompress(value, parent_key)
            return value
        if isinstance(value, dict):
            out: dict = {}
            for k, v in value.items():
                out[k] = await self._awalk_compress(v, str(k))
            return out
        if isinstance(value, list):
            return [await self._awalk_compress(v, parent_key) for v in value]
        if isinstance(value, tuple):
            return tuple([await self._awalk_compress(v, parent_key) for v in value])
        return value

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        compressed = self._walk_compress(value, parent_key=None)
        if ttl is None:
            return self._inner.put(namespace, key, compressed, index)
        return self._inner.put(namespace, key, compressed, index, ttl=ttl)

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        compressed = await self._awalk_compress(value, parent_key=None)
        if ttl is None:
            return await self._inner.aput(namespace, key, compressed, index)
        return await self._inner.aput(namespace, key, compressed, index, ttl=ttl)

    def batch(self, ops: Iterable[Any]) -> list:
        ops = list(ops)
        new_ops = [self._rewrite_op(op) for op in ops]
        return self._inner.batch(new_ops)

    async def abatch(self, ops: Iterable[Any]) -> list:
        ops = list(ops)
        new_ops = []
        for op in ops:
            new_ops.append(await self._arewrite_op(op))
        return await self._inner.abatch(new_ops)

    def _rewrite_op(self, op: Any) -> Any:
        if isinstance(op, PutOp) and op.value is not None:
            new_value = self._walk_compress(op.value, parent_key=None)
            return op._replace(value=new_value)
        return op

    async def _arewrite_op(self, op: Any) -> Any:
        if isinstance(op, PutOp) and op.value is not None:
            new_value = await self._awalk_compress(op.value, parent_key=None)
            return op._replace(value=new_value)
        return op

    def get(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None) -> Any:
        return self._inner.get(namespace, key, refresh_ttl=refresh_ttl)

    async def aget(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None
    ) -> Any:
        return await self._inner.aget(namespace, key, refresh_ttl=refresh_ttl)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        return self._inner.delete(namespace, key)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        return await self._inner.adelete(namespace, key)

    def search(self, namespace_prefix: tuple[str, ...], **kwargs: Any) -> list:
        return self._inner.search(namespace_prefix, **kwargs)

    async def asearch(self, namespace_prefix: tuple[str, ...], **kwargs: Any) -> list:
        return await self._inner.asearch(namespace_prefix, **kwargs)

    def list_namespaces(self, **kwargs: Any) -> list:
        return self._inner.list_namespaces(**kwargs)

    async def alist_namespaces(self, **kwargs: Any) -> list:
        return await self._inner.alist_namespaces(**kwargs)
