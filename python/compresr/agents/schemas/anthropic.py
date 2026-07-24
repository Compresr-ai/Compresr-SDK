"""Anthropic ``Message`` look-alike dataclasses.

These duck-type the Anthropic SDK's response shape (``message.content[0].text``,
``message.usage.input_tokens``, ``message.stop_reason``) so customers can
swap ``compresr.CompressionClient`` in for their existing ``anthropic.Anthropic``
client without installing the ``anthropic`` package or rewriting downstream
parsing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..normalized import CompresrStats, NormalizedResult


@dataclass(frozen=True)
class TextBlock:
    """An assistant text block — mirrors ``anthropic.types.TextBlock``."""

    text: str
    type: str = "text"
    citations: list = field(default_factory=list)  # list[Citation]


@dataclass(frozen=True)
class ToolUseBlock:
    """A tool-use block — mirrors ``anthropic.types.ToolUseBlock``."""

    id: str
    name: str
    input: dict = field(default_factory=dict)
    type: str = "tool_use"


@dataclass(frozen=True)
class Usage:
    """Token usage block — mirrors ``anthropic.types.Usage``."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class AnthropicMessage:
    """Top-level response — mirrors ``anthropic.types.Message``.

    Adds ``messages`` (full agent-loop conversation) and Compresr-specific
    ``compresr`` stats on top of the Anthropic ``Message`` shape. Existing
    Anthropic SDK consumers ignore these extra fields.
    """

    id: str
    role: str  # "assistant"
    model: str
    content: list  # list[TextBlock | ToolUseBlock]
    stop_reason: str
    usage: Usage
    compresr: CompresrStats
    type: str = "message"
    raw: Any = None
    messages: list = field(default_factory=list)


def to_anthropic_message(
    result: NormalizedResult,
    *,
    model: str,
    msg_id: Optional[str] = None,
) -> AnthropicMessage:
    """Translate an engine :class:`NormalizedResult` into an Anthropic ``Message``."""
    blocks: list = []
    if result.text:
        blocks.append(TextBlock(text=result.text, citations=list(result.citations)))
    for tu in result.tool_uses:
        blocks.append(
            ToolUseBlock(
                id=tu.get("id", "") or "",
                name=tu.get("name", "") or "",
                input=dict(tu.get("input", {}) or {}),
            )
        )
    u = result.usage or {}
    return AnthropicMessage(
        id=msg_id or _derive_id(result, "msg_"),
        role="assistant",
        model=model,
        content=blocks,
        stop_reason=result.stop_reason or "end_turn",
        usage=Usage(
            input_tokens=int(u.get("input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cache_read_input_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
        ),
        compresr=result.compresr_stats,
        raw=result.raw,
        messages=list(result.messages),
    )


def _derive_id(result: NormalizedResult, prefix: str) -> str:
    raw = result.raw
    return getattr(raw, "id", None) or f"{prefix}generated"


__all__ = [
    "AnthropicMessage",
    "TextBlock",
    "ToolUseBlock",
    "Usage",
    "to_anthropic_message",
]
