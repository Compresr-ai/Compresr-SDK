"""Compresr agents — provider-shape facades over LangChain.

Public surface is intentionally minimal. Heavy LangChain imports are
deferred via ``__getattr__`` so ``import compresr.agents`` stays cheap
when the user only needs the type names.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "WebSearchTool",
    "CompressionPolicy",
    "NormalizedResult",
    "Citation",
    "CompresrStats",
]


def __getattr__(name: str) -> Any:
    if name == "WebSearchTool":
        from . import tools as _t

        return _t.WebSearchTool
    if name == "CompressionPolicy":
        from compresr.integrations._shared import CompressionPolicy as _P

        return _P
    if name in ("NormalizedResult", "Citation", "CompresrStats"):
        from . import normalized as _n

        return getattr(_n, name)
    raise AttributeError(name)
