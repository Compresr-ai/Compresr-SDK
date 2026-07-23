"""Client construction + integration defaults."""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_MODEL = "latte_v1"
DEFAULT_RATIO: float = 0.5
DEFAULT_MIN_TOKENS: int = 200
BATCH_LIMIT: int = 100


def build_client(
    *,
    api_key: Optional[str],
    base_url: Optional[str] = None,
    caller: str = "Compresr integration",
) -> Any:
    """Construct a ``CompressionClient``. Lazy import to keep this module
    cheap. Raises ``ValueError`` with a caller-aware message when
    ``api_key`` is missing."""
    if not api_key:
        raise ValueError(f"Either `api_key` or `client` is required for {caller}.")
    from compresr import CompressionClient

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return CompressionClient(**kwargs)


__all__ = [
    "BATCH_LIMIT",
    "DEFAULT_MIN_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_RATIO",
    "build_client",
]
