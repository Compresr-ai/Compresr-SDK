"""Tool-output wrappers — add Compresr to any LangChain ``StructuredTool``.

Works without ``create_agent`` middleware, in plain LCEL chains, LangGraph
custom graphs — anywhere a tool is invoked.

Query resolution follows the unified Compresr API:

    query: str | None                       # static query
    query_extractor: Callable[[dict], str]  # custom callable over tool args
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
    CompressionPolicy,
    ErrorPolicy,
    ToolOutputCompressor,
    build_client,
    resolve_query,
)

try:
    from langchain_core.tools import BaseTool, StructuredTool  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wrap_tool_with_compression requires langchain-core. "
        "Install with: pip install compresr[langchain]"
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


def wrap_tool_with_compression(
    tool: BaseTool,
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
) -> StructuredTool:
    """Wrap a ``StructuredTool`` so its string output is compressed transparently.

    Preserves the original ``name``, ``description``, and ``args_schema``.
    Raises ``TypeError`` if ``tool`` is not a ``StructuredTool``.
    """
    if not isinstance(tool, StructuredTool):
        raise TypeError(
            f"wrap_tool_with_compression supports StructuredTool only "
            f"(got {type(tool).__name__}). Use @tool or "
            f"StructuredTool.from_function to wrap your callable first."
        )

    compresr = client or build_client(
        api_key=api_key, base_url=base_url, caller="wrap_tool_with_compression"
    )

    policy = CompressionPolicy(
        target_compression_ratio=target_compression_ratio,
        compression_model_name=compression_model,
        coarse=coarse,
        min_tokens=min_tokens,
        on_error=on_error,
    )
    compressor = ToolOutputCompressor(client=compresr, policy=policy)

    def _resolve(args: dict) -> Optional[str]:
        return _resolve_tool_query(
            static=query,
            extractor=query_extractor,
            arg_key=query_arg,
            tool_args=args,
        )

    def _wrap_sync(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = fn(*args, **kwargs)
            return compressor.process(
                tool_name=tool.name,
                output=out,
                query=_resolve(kwargs),
            )

        return wrapped

    def _wrap_async(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = await fn(*args, **kwargs)
            return await compressor.aprocess(
                tool_name=tool.name,
                output=out,
                query=_resolve(kwargs),
            )

        return wrapped

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_wrap_sync(tool.func) if tool.func else None,
        coroutine=_wrap_async(tool.coroutine) if tool.coroutine else None,
        return_direct=tool.return_direct,
        handle_tool_error=tool.handle_tool_error,
        handle_validation_error=tool.handle_validation_error,
    )


def compress_tool_output(
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
) -> Callable[[BaseTool], StructuredTool]:
    """Decorator form of :func:`wrap_tool_with_compression`."""

    def _decorator(tool: BaseTool) -> StructuredTool:
        return wrap_tool_with_compression(
            tool,
            api_key=api_key,
            client=client,
            compression_model=compression_model,
            target_compression_ratio=target_compression_ratio,
            min_tokens=min_tokens,
            coarse=coarse,
            query=query,
            query_extractor=query_extractor,
            query_arg=query_arg,
            on_error=on_error,
            base_url=base_url,
        )

    return _decorator
