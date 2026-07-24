"""OpenAI ``ChatCompletion`` look-alike dataclasses.

Customers using ``openai.OpenAI().chat.completions.create(...)`` can swap
in ``compresr.CompressionClient(...).chat.completions.create(...)``
without installing ``openai`` or rewriting downstream parsing — the
return shape mirrors ``openai.types.chat.ChatCompletion``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ..normalized import CompresrStats, NormalizedResult


@dataclass(frozen=True)
class FunctionCall:
    """Function-call payload inside a ``ToolCall``."""

    name: str
    arguments: str  # JSON-encoded string, per OpenAI's shape


@dataclass(frozen=True)
class ToolCall:
    """A single tool call — mirrors ``openai.types.ChatCompletionMessageToolCall``."""

    id: str
    function: FunctionCall
    type: str = "function"


@dataclass(frozen=True)
class ChatMessage:
    """Assistant message — mirrors ``openai.types.chat.ChatCompletionMessage``."""

    role: str
    content: Optional[str]
    tool_calls: list = field(default_factory=list)  # list[ToolCall]


@dataclass(frozen=True)
class Choice:
    """Single completion choice — mirrors ``openai.types.chat.chat_completion.Choice``."""

    index: int
    message: ChatMessage
    finish_reason: str


@dataclass(frozen=True)
class UsageOpenAI:
    """Token usage block — mirrors ``openai.types.CompletionUsage``."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ChatCompletion:
    """Top-level response — mirrors ``openai.types.chat.ChatCompletion``.

    Adds ``messages`` (full agent-loop conversation) and Compresr-specific
    ``compresr`` stats on top of the OpenAI ``ChatCompletion`` shape.
    Existing OpenAI SDK consumers ignore these extra fields.
    """

    id: str
    model: str
    choices: list  # list[Choice]
    usage: UsageOpenAI
    compresr: CompresrStats
    object: str = "chat.completion"
    raw: Any = None
    messages: list = field(default_factory=list)


_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "tool_calls",
}


def to_chat_completion(
    result: NormalizedResult,
    *,
    model: str,
    completion_id: Optional[str] = None,
) -> ChatCompletion:
    """Translate an engine :class:`NormalizedResult` into an OpenAI ``ChatCompletion``."""
    tool_calls = [
        ToolCall(
            id=tu.get("id", "") or "",
            function=FunctionCall(
                name=tu.get("name", "") or "",
                arguments=json.dumps(tu.get("input", {}) or {}),
            ),
        )
        for tu in result.tool_uses
    ]
    message = ChatMessage(
        role="assistant",
        content=result.text or None,
        tool_calls=tool_calls,
    )
    finish_reason = _FINISH_REASON_MAP.get(result.stop_reason or "", result.stop_reason or "stop")
    u = result.usage or {}
    prompt_tokens = int(u.get("input_tokens", u.get("prompt_tokens", 0)) or 0)
    completion_tokens = int(u.get("output_tokens", u.get("completion_tokens", 0)) or 0)
    return ChatCompletion(
        id=completion_id or _derive_id(result, "chatcmpl-"),
        model=model,
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
        usage=UsageOpenAI(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        compresr=result.compresr_stats,
        raw=result.raw,
        messages=list(result.messages),
    )


def _derive_id(result: NormalizedResult, prefix: str) -> str:
    raw = result.raw
    return getattr(raw, "id", None) or f"{prefix}generated"


__all__ = [
    "ChatCompletion",
    "ChatMessage",
    "Choice",
    "FunctionCall",
    "ToolCall",
    "UsageOpenAI",
    "to_chat_completion",
]
