"""Tool-output compression kernel: compress via the SDK, cache the original
(with secrets/PII masked), append a recovery footer. Fail-open — any failure
returns the original content.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from . import cache

logger = logging.getLogger(__name__)

# Marks our own footer so the hook never re-compresses an output it produced.
FOOTER_MARKER = "[compresr:recover]"

# Conservative estimate; the exact post-footer size check below is the real gate.
FOOTER_TOKEN_BUDGET = 90

_CHARS_PER_TOKEN = 4

DEFAULT_TOOL_OUTPUT_MODEL = "toc_latte_v2"
_SOURCE_TAG = "integration:hermes"


def count_tokens(s: str) -> int:
    """Cheap, dependency-free token estimate for gating."""
    return (len(s) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def _footer(path: str, base_tok: int, out_tok: int) -> str:
    saved = max(0, base_tok - out_tok)
    pct = int(round(100 * saved / base_tok)) if base_tok else 0
    return (
        f"\n\n{FOOTER_MARKER} Tool output compressed {base_tok}→{out_tok} tokens "
        f"(~{pct}% saved). A copy of the original (with secrets/PII masked) is "
        f"cached at {path} — if you need exact details that were summarized away, "
        f'recover them with read_file("{path}") or search_files.'
    )


def _call_api(
    client: Any,
    content: str,
    query: str,
    tool_name: str,
    model: str,
    target_ratio: float,
) -> Tuple[str, Dict[str, Any]]:
    response = client.compress_tool_output(
        tool_output=content,
        tool_name=tool_name or "unknown",
        query=query,
        compression_model_name=model,
        target_compression_ratio=target_ratio if target_ratio and target_ratio > 0 else None,
        source=_SOURCE_TAG,
    )
    if not response.success or response.data is None:
        raise RuntimeError(f"API error: {response.message or 'no data in response'}")
    compressed = response.data.compressed_output
    if not isinstance(compressed, str):
        raise RuntimeError("API returned a list for a single tool output")
    return compressed, response.data.model_dump()


def compress_with_recovery(
    query: str,
    content: str,
    tool_name: str,
    cache_id: str,
    client: Any,
    model: str = DEFAULT_TOOL_OUTPUT_MODEL,
    task_id: str = "default",
    max_cache_mb: int = 256,
    target_ratio: float = 2.0,
    cache_content: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(output_text, info)``; never raises. On any failure the original
    ``content`` comes back with ``info["shortened"]`` False.

    ``cache_content`` is what gets persisted for recovery (defaults to
    ``content``); it may differ, e.g. a de-numbered copy of line-numbered
    read_file output. The API call and size gate always use ``content``.
    """
    if cache_content is None:
        cache_content = content
    base_tok = count_tokens(content)
    info: Dict[str, Any] = {
        "called_api": False,
        "base_tokens": base_tok,
        "out_tokens": base_tok,
        "shortened": False,
    }

    try:
        compressed, stats = _call_api(client, content, query, tool_name, model, target_ratio)
    except Exception as e:
        logger.warning("compresr: tool-output API failed (%s) — leaving original", e)
        info["error"] = str(e)
        return content, info

    info["called_api"] = True
    info["api_stats"] = stats

    if not compressed or not compressed.strip():
        info["error"] = "empty compressed output"
        return content, info

    body_tok = count_tokens(compressed)
    if body_tok + FOOTER_TOKEN_BUDGET >= base_tok:
        # Benign "no net win", not a failure: error stays unset so the caller
        # doesn't arm its back-off.
        info["skipped_reason"] = "not smaller"
        return content, info

    cache_path = cache.store_original(cache_id, cache_content, task_id, max_cache_mb=max_cache_mb)
    if cache_path is None:
        info["error"] = "cache write failed"
        return content, info

    reported_out_tok = body_tok + FOOTER_TOKEN_BUDGET
    out = compressed + _footer(cache_path, base_tok, reported_out_tok)
    out_tok = count_tokens(out)
    if out_tok >= base_tok:
        # A long cache path can push the footer past its budget; keep the cache
        # entry (content-addressed, pruner reclaims it) and fail open.
        info["skipped_reason"] = "not smaller after footer"
        return content, info
    info.update(
        {
            "shortened": True,
            "out_tokens": out_tok,
            "saved": max(0, base_tok - out_tok),
            "cache_path": cache_path,
        }
    )
    return out, info


__all__ = [
    "FOOTER_MARKER",
    "FOOTER_TOKEN_BUDGET",
    "DEFAULT_TOOL_OUTPUT_MODEL",
    "count_tokens",
    "compress_with_recovery",
]
