"""``client.research`` facade — thin wrapper over :class:`ResearchAgent`."""

from __future__ import annotations

import os
from typing import Any, Optional

from .agent import ResearchAgent
from .types import ResearchResult

_SEARCH_PROVIDER_ENV = {
    "tavily": ("TAVILY_API_KEY",),
    "brave": ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"),
}


class ResearchFacade:
    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def run(
        self,
        question: str,
        *,
        search: Any = "tavily",
        max_steps: int = 10,
        model: Optional[str] = None,
        compress_snippets: bool = True,
        compression_model: str = "latte_v1",
        min_compress_tokens: int = 100,
        max_context_tokens: int = 120_000,
        system_prompt: Optional[str] = None,
    ) -> ResearchResult:
        """Multi-step research loop.

        ``search`` is either a provider name (``"tavily"`` | ``"brave"``)
        or a ready-made LangChain ``BaseTool``.
        """
        tool = self._resolve_search_tool(search)
        agent = ResearchAgent(
            engine=self._engine,
            search_tool=tool,
            max_steps=max_steps,
            system_prompt=system_prompt,
            compress_snippets=compress_snippets,
            compression_model=compression_model,
            min_compress_tokens=min_compress_tokens,
            max_context_tokens=max_context_tokens,
        )
        return agent.run(question, model=model)

    def search(
        self,
        question: str,
        *,
        search: Any = "tavily",
        model: Optional[str] = None,
        compress_snippets: bool = True,
        compression_model: str = "latte_v1",
        min_compress_tokens: int = 100,
        max_context_tokens: int = 120_000,
        system_prompt: Optional[str] = None,
    ) -> ResearchResult:
        """Single-shot: one search + forced answer (``max_steps=2``)."""
        return self.run(
            question,
            search=search,
            max_steps=2,
            model=model,
            compress_snippets=compress_snippets,
            compression_model=compression_model,
            min_compress_tokens=min_compress_tokens,
            max_context_tokens=max_context_tokens,
            system_prompt=system_prompt,
        )

    def _resolve_search_tool(self, search: Any) -> Any:
        if not isinstance(search, str):
            return search
        provider = search.lower()
        if provider not in _SEARCH_PROVIDER_ENV:
            raise ValueError(
                f"unsupported search provider {search!r}; expected one of "
                f"{sorted(_SEARCH_PROVIDER_ENV)} or a LangChain BaseTool"
            )
        from ..tools.web_search import WebSearchTool

        api_key: Optional[str] = None
        for env_var in _SEARCH_PROVIDER_ENV[provider]:
            v = os.environ.get(env_var)
            if v:
                api_key = v
                break
        return WebSearchTool(provider=provider, api_key=api_key)
