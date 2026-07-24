"""Customer-facing compression policy object.

Bundles the compression knobs (model, ratio, min_tokens, error policy, tool
filters) that future facades (Anthropic/OpenAI) will accept via
``CompressionClient(compression={...})``. Frozen so callers can pass the same
policy to multiple kernels without worrying about hidden mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .client import DEFAULT_MIN_TOKENS, DEFAULT_MODEL, DEFAULT_RATIO
from .errors import DEFAULT_POLICY, ErrorPolicy


@dataclass(frozen=True)
class CompressionPolicy:
    """Immutable bundle of compression knobs shared across integrations."""

    target_compression_ratio: float = DEFAULT_RATIO
    compression_model_name: str = DEFAULT_MODEL
    coarse: Optional[bool] = None
    min_tokens: int = DEFAULT_MIN_TOKENS
    on_error: ErrorPolicy = DEFAULT_POLICY
    allow_tools: Optional[Iterable[str]] = None
    ignore_tools: Optional[Iterable[str]] = None

    def tool_kwargs(self) -> dict:
        """Kwargs to pass to ``CompresrToolMiddleware`` constructor.

        Maps this policy onto the middleware's per-tool compression knobs.
        ``coarse``/``allow_tools``/``ignore_tools`` are only emitted when set
        so the middleware retains its own defaults for unset fields.
        """
        out: dict = {
            "target_compression_ratio": self.target_compression_ratio,
            "compression_model": self.compression_model_name,
            "min_tokens": self.min_tokens,
            "on_error": self.on_error,
        }
        if self.coarse is not None:
            out["coarse"] = self.coarse
        if self.allow_tools is not None:
            out["allow_tools"] = self.allow_tools
        if self.ignore_tools is not None:
            out["ignore_tools"] = self.ignore_tools
        return out


__all__ = ["CompressionPolicy"]
