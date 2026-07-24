"""Normalized result types used by the provider-shape facades.

These shapes are intentionally provider-agnostic — every backend (Anthropic,
OpenAI Responses, Gemini grounding) gets remapped into the same
``NormalizedResult`` by ``_Engine``. Frozen dataclasses keep the shape
immutable for safe downstream rewrapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class CompresrStats:
    """Aggregate compression stats for a single agent run.

    ``by_tool`` maps tool name -> tokens saved.

    NOTE: Currently always zero — middleware-driven aggregation will land
    in a future release. The shape is stable so downstream code can rely
    on the field names today.
    """

    tokens_saved: int = 0
    original_total: int = 0
    compressed_total: int = 0
    by_tool: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """A single citation extracted from a provider response.

    Raw provider data is preserved in ``provider_metadata`` so callers
    can recover anything the normalizer dropped.
    """

    url: str
    title: Optional[str] = None
    cited_text: Optional[str] = None
    provider_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedResult:
    """Provider-agnostic agent invocation result.

    ``raw`` preserves the original (final) ``AIMessage`` so advanced users
    can reach in for fields the normalizer doesn't surface. ``messages``
    holds the full conversation chain returned by the underlying agent so
    callers can walk the trajectory step by step.
    """

    text: str
    content_blocks: list = field(default_factory=list)
    tool_uses: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=dict)
    compresr_stats: CompresrStats = field(default_factory=CompresrStats)
    raw: Any = None
    messages: list = field(default_factory=list)


__all__ = ["NormalizedResult", "Citation", "CompresrStats"]
