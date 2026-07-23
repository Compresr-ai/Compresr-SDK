"""Provider-shaped response dataclasses.

These mirror the Anthropic ``Message`` and OpenAI ``ChatCompletion`` shapes
so customers can swap ``CompressionClient`` in for their existing SDK
without having to install ``anthropic`` or ``openai``.

Re-exported here for both internal use (facades) and customer typing.
"""

from __future__ import annotations

from .anthropic import (
    AnthropicMessage,
    TextBlock,
    ToolUseBlock,
    Usage,
    to_anthropic_message,
)
from .openai import (
    ChatCompletion,
    ChatMessage,
    Choice,
    FunctionCall,
    ToolCall,
    UsageOpenAI,
    to_chat_completion,
)

__all__ = [
    "AnthropicMessage",
    "ChatCompletion",
    "ChatMessage",
    "Choice",
    "FunctionCall",
    "TextBlock",
    "ToolCall",
    "ToolUseBlock",
    "Usage",
    "UsageOpenAI",
    "to_anthropic_message",
    "to_chat_completion",
]
