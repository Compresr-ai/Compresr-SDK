"""Compresr research agent — multi-step web search with per-snippet compression.

Public entry: ``client.research.run(question)`` / ``client.research.search(question)``.
Loop structure adapted from Perplexity ``search_evals`` (MIT,
https://github.com/perplexityai/search_evals).
"""

from .agent import ResearchAgent
from .facade import ResearchFacade
from .parser import parse_research_output
from .prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT
from .types import Citation, ResearchResult, ResearchUsage, Step, StepKind

__all__ = [
    "Citation",
    "DEFAULT_RESEARCH_SYSTEM_PROMPT",
    "ResearchAgent",
    "ResearchFacade",
    "ResearchResult",
    "ResearchUsage",
    "Step",
    "StepKind",
    "parse_research_output",
]
