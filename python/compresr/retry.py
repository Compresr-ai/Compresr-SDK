"""Bounded retry policy for the SDK transport layer.

Defaults retry ``429`` and ``503`` from the proxy with exponential backoff +
jitter. Pass ``RetryConfig(max_retries=0)`` to opt out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for transient backpressure responses.

    Args:
        max_retries: Maximum number of retry attempts after the initial
            request fails with a retryable status. ``0`` disables retries.
        initial_backoff_s: Backoff before the first retry, in seconds.
        max_backoff_s: Upper bound for any single backoff.
        multiplier: Exponential growth factor between attempts.
        jitter: Fractional symmetric jitter (e.g. ``0.25`` means the actual
            sleep is uniformly sampled from ``[delay * 0.75, delay * 1.25]``).
            Set to ``0`` for deterministic backoff in tests.
        retry_on_status: HTTP status codes that trigger a retry. Defaults
            to ``(429, 503)`` — the two backpressure signals the proxy uses.
        respect_retry_after: When the server returns a ``retry_after`` hint
            (header or body field), use it instead of the computed backoff
            for that attempt.
    """

    max_retries: int = 3
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.25
    retry_on_status: Tuple[int, ...] = field(default_factory=lambda: (429, 503))
    respect_retry_after: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.initial_backoff_s < 0:
            raise ValueError("initial_backoff_s must be >= 0")
        if self.max_backoff_s < 0:
            raise ValueError("max_backoff_s must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be in [0, 1]")


def compute_backoff(
    attempt: int,
    cfg: RetryConfig,
    retry_after: float | None = None,
) -> float:
    """Compute the sleep duration before retry ``attempt`` (0-indexed).

    Honors ``retry_after`` from the server when present and
    ``cfg.respect_retry_after`` is true; otherwise applies exponential
    backoff with optional symmetric jitter, capped at ``max_backoff_s``.
    """
    if retry_after is not None and cfg.respect_retry_after:
        # Server-suggested delay still respects the global cap so a buggy
        # backend can't pin the client into a multi-minute sleep.
        return max(0.0, min(float(retry_after), cfg.max_backoff_s))

    base = cfg.initial_backoff_s * (cfg.multiplier**attempt)
    base = min(base, cfg.max_backoff_s)
    if cfg.jitter == 0:
        return base
    spread = base * cfg.jitter
    return max(0.0, base + random.uniform(-spread, spread))
