"""LlamaIndex integration for Compresr.

    pip install compresr[llamaindex]

Exports:
    - ``CompresrNodePostprocessor``: ``BaseNodePostprocessor`` that
      compresses retrieved node content. Drop-in for
      ``LongLLMLinguaPostprocessor``.
    - ``wrap_tool_with_compresr``: wrap a ``FunctionTool`` so its return
      value is compressed transparently.
"""

from __future__ import annotations

try:
    import llama_index.core  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "compresr LlamaIndex integration requires llama-index-core. "
        "Install with: pip install compresr[llamaindex]"
    ) from exc

from .._shared import ErrorPolicy
from .postprocessor import CompresrNodePostprocessor
from .wrappers import wrap_tool_with_compresr

try:
    from .memory import CompresrMemoryBlock  # noqa: F401

    _HAS_MEMORY_BLOCK = True
except ImportError:  # llama-index-core older than 0.12 / no Memory API
    _HAS_MEMORY_BLOCK = False

__all__ = [
    "CompresrNodePostprocessor",
    "ErrorPolicy",
    "wrap_tool_with_compresr",
]
if _HAS_MEMORY_BLOCK:
    __all__.append("CompresrMemoryBlock")
