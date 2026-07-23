"""Token estimation.

Prefers ``tiktoken`` when available for accurate counts; falls back to a
4-chars-per-token approximation that's good enough for threshold gating.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _encoding_for(encoding_name: Optional[str], model: Optional[str]):  # type: ignore[no-untyped-def]
    """Return a tiktoken encoder for the given encoding/model, or None.

    Caches by (encoding_name, model). ``None``/``None`` returns the default
    cl100k encoder.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        if model is not None:
            return tiktoken.encoding_for_model(model)
        return tiktoken.get_encoding(encoding_name or _DEFAULT_ENCODING)
    except Exception:
        try:
            return tiktoken.get_encoding(_DEFAULT_ENCODING)
        except Exception:
            return None


def estimate_tokens(text: str, *, model: Optional[str] = None) -> int:
    """Estimate token count for ``text``.

    Uses ``tiktoken`` if importable. Falls back to ``len(text) // 4``
    (a conservative under-estimate that errs toward "compress more often").
    Returns at least 1 for any non-empty string.
    """
    if not text:
        return 0

    encoder = _encoding_for(_DEFAULT_ENCODING, model)
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(text)))
        except Exception:
            pass

    return max(1, len(text) // 4)
