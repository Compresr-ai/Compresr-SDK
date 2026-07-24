"""LangGraph integration for Compresr.

    pip install compresr[langgraph]

LangGraph 1.0+ uses ``langchain.agents.create_agent`` with the same
middleware mechanism as LangChain. The two middlewares below are
re-exported from ``compresr.integrations.langchain`` so they're
discoverable here too — import either path you prefer.

Exports:
    - ``CompresrToolMiddleware``: compress each tool output before it
      enters agent state.
    - ``CompresrSummarizationMiddleware``: compress old history into a
      single summary message when token threshold is crossed.
    - ``make_compresr_node``: drop-in compression node for custom
      ``StateGraph`` (the LangGraph-specific helper).
"""

from __future__ import annotations

try:
    import langgraph  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "compresr LangGraph integration requires langgraph. "
        "Install with: pip install compresr[langgraph]"
    ) from exc

from .._shared import ErrorPolicy
from ..langchain import (
    CompresrPromptMiddleware,
    CompresrSummarizationMiddleware,
    CompresrToolMiddleware,
)
from .checkpoint import CompresrCheckpointSerializer
from .handoff import compresr_handoff_tool
from .nodes import make_compresr_node
from .store import CompresrStore

compresr_node = make_compresr_node

__all__ = [
    "CompresrCheckpointSerializer",
    "CompresrPromptMiddleware",
    "CompresrStore",
    "CompresrSummarizationMiddleware",
    "CompresrToolMiddleware",
    "ErrorPolicy",
    "compresr_handoff_tool",
    "compresr_node",
    "make_compresr_node",
]
