"""Unified error policy for integrations.

Compression should rarely break a user's app. Default is ``passthrough``:
log and return the original/fallback value. ``raise`` is opt-in.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional, TypeVar

T = TypeVar("T")

ErrorPolicy = Literal["raise", "passthrough"]
DEFAULT_POLICY: ErrorPolicy = "passthrough"

logger = logging.getLogger("compresr.integrations")


def apply_error_policy(
    fn: Callable[[], T],
    *,
    fallback: T,
    policy: ErrorPolicy = DEFAULT_POLICY,
    log: Optional[logging.Logger] = None,
    context: Optional[dict[str, Any]] = None,
) -> T:
    """Run ``fn``; on exception, raise (policy='raise') or log+return ``fallback``."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — integration-level safety net
        if policy == "raise":
            raise
        (log or logger).warning(
            "compresr integration call failed (%s); passthrough. context=%s",
            exc,
            context or {},
        )
        return fallback
