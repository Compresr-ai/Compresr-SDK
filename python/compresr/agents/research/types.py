"""Dataclasses returned by :class:`ResearchAgent.run`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

StepKind = Literal["search", "tool_result", "answer", "error"]


@dataclass(frozen=True)
class Citation:
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None


@dataclass(frozen=True)
class Step:
    """One step in the agent trajectory."""

    type: StepKind
    query: Optional[str] = None
    text: Optional[str] = None
    chars: Optional[int] = None
    latency_s: Optional[float] = None


@dataclass(frozen=True)
class ResearchUsage:
    """Token + tool-call counts across the research loop."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0
    search_calls: int = 0


@dataclass(frozen=True)
class ResearchResult:
    """Final research output. ``text`` is the model's raw final response;
    ``answer`` / ``explanation`` / ``confidence`` are parsed from it."""

    answer: str
    explanation: str
    confidence: Optional[float]
    text: str
    citations: list[Citation] = field(default_factory=list)
    trajectory: list[Step] = field(default_factory=list)
    usage: ResearchUsage = field(default_factory=ResearchUsage)
    raw: Any = None
