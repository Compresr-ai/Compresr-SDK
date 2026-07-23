"""Customer-facing facades that mimic the Anthropic and OpenAI SDK shapes.

Each facade wraps a private :class:`compresr.agents.engine._Engine` and
re-emits provider-shaped dataclasses. The native facade returns
:class:`compresr.agents.normalized.NormalizedResult` directly.
"""

from __future__ import annotations

from .anthropic import _Anthropic
from .native import _Native
from .openai import _OpenAI

__all__ = ["_Anthropic", "_Native", "_OpenAI"]
