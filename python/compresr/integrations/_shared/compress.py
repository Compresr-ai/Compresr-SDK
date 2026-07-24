"""Compression helpers that wrap the SDK with integration-level concerns:
token-threshold gating, error policy, sync/async correctness.

LangChain middleware often runs inside an asyncio event loop; calling
the sync ``CompressionClient.compress()`` (blocking ``urlopen``) from
there would stall the loop. Use ``acompress_safe`` from async contexts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .errors import DEFAULT_POLICY, ErrorPolicy, apply_error_policy

logger = logging.getLogger("compresr.integrations")


def _build_kwargs(
    *,
    context: str,
    compression_model_name: str,
    query: Optional[str],
    target_compression_ratio: Optional[float],
    coarse: Optional[bool],
) -> dict:
    kw: dict = {
        "context": context,
        "compression_model_name": compression_model_name,
    }
    if query is not None:
        kw["query"] = query
    if target_compression_ratio is not None:
        kw["target_compression_ratio"] = target_compression_ratio
    if coarse is not None:
        kw["coarse"] = coarse
    return kw


def _should_skip(context: str, min_tokens: int) -> bool:
    if not isinstance(context, str) or not context.strip():
        return True
    from .tokens import estimate_tokens

    return estimate_tokens(context) < min_tokens


def compress_safe(
    client: Any,
    *,
    context: str,
    query: Optional[str] = None,
    compression_model_name: str = "latte_v1",
    target_compression_ratio: Optional[float] = None,
    coarse: Optional[bool] = None,
    min_tokens: int = 200,
    on_error: ErrorPolicy = DEFAULT_POLICY,
    context_label: Optional[str] = None,
) -> str:
    if _should_skip(context, min_tokens):
        return context

    kw = _build_kwargs(
        context=context,
        compression_model_name=compression_model_name,
        query=query,
        target_compression_ratio=target_compression_ratio,
        coarse=coarse,
    )

    return apply_error_policy(
        lambda: client.compress(**kw).data.compressed_context,
        fallback=context,
        policy=on_error,
        context={"label": context_label, "model": compression_model_name},
    )


async def acompress_safe(
    client: Any,
    *,
    context: str,
    query: Optional[str] = None,
    compression_model_name: str = "latte_v1",
    target_compression_ratio: Optional[float] = None,
    coarse: Optional[bool] = None,
    min_tokens: int = 200,
    on_error: ErrorPolicy = DEFAULT_POLICY,
    context_label: Optional[str] = None,
) -> str:
    if _should_skip(context, min_tokens):
        return context

    kw = _build_kwargs(
        context=context,
        compression_model_name=compression_model_name,
        query=query,
        target_compression_ratio=target_compression_ratio,
        coarse=coarse,
    )

    async_call = getattr(client, "compress_async", None)
    label = {"label": context_label, "model": compression_model_name}

    if async_call is not None:
        try:
            resp = await async_call(**kw)
            out = resp.data.compressed_context
            return out if isinstance(out, str) else context
        except Exception as exc:  # noqa: BLE001
            if on_error == "raise":
                raise
            logger.warning(
                "compresr async call failed (%s); passthrough. context=%s",
                exc,
                label,
            )
            return context

    return apply_error_policy(
        lambda: client.compress(**kw).data.compressed_context,
        fallback=context,
        policy=on_error,
        context=label,
    )
