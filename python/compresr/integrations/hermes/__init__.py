"""Compresr × Hermes agent integration.

Two surfaces, both fail-open and opt-in:

- :class:`CompresrContextEngine` — drop-in replacement for Hermes's built-in
  ``compressor`` context engine (``context.engine: compresr``).
- :class:`ToolOutputCompressor` — ``transform_tool_result`` hook that shrinks
  large tool outputs and caches a copy of the original (secrets/PII masked) for
  recovery.

:func:`register` is the Hermes plugin entry point wiring both.

Requires the Hermes agent runtime (this package is loaded as a Hermes plugin,
inside Hermes's Python process). Importing :mod:`compresr` itself never pulls
Hermes in — resolution is deferred until these names are accessed.
"""

from typing import Any

__all__ = ["CompresrContextEngine", "ToolOutputCompressor", "register"]


def __getattr__(name: str) -> Any:
    if name == "CompresrContextEngine":
        from .engine import CompresrContextEngine as _engine_cls

        return _engine_cls
    if name == "ToolOutputCompressor":
        from .tool_output import ToolOutputCompressor as _hook_cls

        return _hook_cls
    if name == "register":
        from .plugin import register as _fn

        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
