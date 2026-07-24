"""Wrap a LlamaIndex ``FunctionTool`` so its return value is compressed.

Same unified query API as the LangChain wrapper:

    query: str | None                       # static query
    query_extractor: Callable[[dict], str]  # custom callable
    query_arg: str | None                   # tool arg name to use
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

from .._shared import (
    DEFAULT_MIN_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_RATIO,
    ErrorPolicy,
    acompress_safe,
    build_client,
    compress_safe,
    resolve_query,
)

try:
    from llama_index.core.tools import FunctionTool  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wrap_tool_with_compresr requires llama-index-core. "
        "Install with: pip install compresr[llamaindex]"
    ) from exc


def _resolve_tool_query(
    *,
    static: Optional[str],
    extractor: Optional[Callable[[dict], Optional[str]]],
    arg_key: Optional[str],
    tool_args: dict,
) -> Optional[str]:
    return resolve_query(
        static=static,
        extractor=extractor,
        extractor_arg=tool_args if extractor is not None else None,
        args=tool_args,
        args_key=arg_key,
    )


def wrap_tool_with_compresr(
    tool: FunctionTool,
    *,
    api_key: Optional[str] = None,
    client: Any = None,
    compression_model: str = DEFAULT_MODEL,
    target_compression_ratio: float = DEFAULT_RATIO,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    coarse: Optional[bool] = None,
    query: Optional[str] = None,
    query_extractor: Optional[Callable[[dict], Optional[str]]] = None,
    query_arg: Optional[str] = None,
    on_error: ErrorPolicy = DEFAULT_POLICY,
    base_url: Optional[str] = None,
) -> FunctionTool:
    """Return a new ``FunctionTool`` whose return value is compressed."""
    compresr = client or build_client(
        api_key=api_key, base_url=base_url, caller="wrap_tool_with_compresr"
    )

    metadata = tool.metadata
    original_fn = tool.fn
    original_async = getattr(tool, "async_fn", None)

    def _resolve(args: dict) -> Optional[str]:
        return _resolve_tool_query(
            static=query,
            extractor=query_extractor,
            arg_key=query_arg,
            tool_args=args,
        )

    @wraps(original_fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = original_fn(*args, **kwargs)
        if not isinstance(out, str):
            return out
        return compress_safe(
            compresr,
            context=out,
            query=_resolve(kwargs),
            compression_model_name=compression_model,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
            min_tokens=min_tokens,
            on_error=on_error,
            context_label=f"tool:{metadata.name}",
        )

    async_wrapped: Optional[Callable[..., Any]] = None
    if original_async is not None:

        @wraps(original_async)
        async def _async(*args: Any, **kwargs: Any) -> Any:
            out = await original_async(*args, **kwargs)
            if not isinstance(out, str):
                return out
            return await acompress_safe(
                compresr,
                context=out,
                query=_resolve(kwargs),
                compression_model_name=compression_model,
                target_compression_ratio=target_compression_ratio,
                coarse=coarse,
                min_tokens=min_tokens,
                on_error=on_error,
                context_label=f"tool:{metadata.name}",
            )

        async_wrapped = _async

    return FunctionTool.from_defaults(
        fn=wrapped,
        async_fn=async_wrapped,
        name=metadata.name,
        description=metadata.description,
        fn_schema=metadata.fn_schema,
        return_direct=metadata.return_direct,
    )
