"""Shared helpers used across all Compresr integrations.

Pure-Python, zero peer dependencies.
"""

from .client import BATCH_LIMIT, DEFAULT_MIN_TOKENS, DEFAULT_MODEL, DEFAULT_RATIO, build_client
from .compress import acompress_safe, compress_safe
from .errors import DEFAULT_POLICY, ErrorPolicy, apply_error_policy
from .filters import make_filter
from .kernel import ToolOutputCompressor
from .policy import CompressionPolicy
from .query import (
    COMMON_QUERY_KEYS,
    DEFAULT_FALLBACK,
    extract_query_from_args,
    extract_query_from_messages,
    resolve_query,
)
from .tokens import estimate_tokens

__all__ = [
    "BATCH_LIMIT",
    "COMMON_QUERY_KEYS",
    "CompressionPolicy",
    "DEFAULT_FALLBACK",
    "DEFAULT_MIN_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_POLICY",
    "DEFAULT_RATIO",
    "ErrorPolicy",
    "ToolOutputCompressor",
    "acompress_safe",
    "apply_error_policy",
    "build_client",
    "compress_safe",
    "estimate_tokens",
    "extract_query_from_args",
    "extract_query_from_messages",
    "make_filter",
    "resolve_query",
]
