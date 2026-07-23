"""
Compresr Python SDK

Compress context to reduce LLM costs.

Quick Start - Compression:
    from compresr import CompressionClient, MODELS

    client = CompressionClient(api_key="cmp_...")

    # Query-aware compression (latte_v1, the default model)
    response = client.compress(
        context="Your long context...",
        query="What is the main conclusion?",
    )
    print(response.data.compressed_context)  # str

    # Coarse-grained mode (paragraph-level, faster)
    response = client.compress(
        context="Your long context...",
        query="What is the main conclusion?",
        coarse=True,
    )

Quick Start - Agents (opt-in provider-shape facades):
    from compresr import CompressionClient, WebSearchTool, CompressionPolicy

    client = CompressionClient(
        api_key="cmp_...",
        llm="anthropic:claude-opus-4-8",
        llm_api_key="sk-ant-...",
    )
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Search the web for SDK news."}],
        tools=[WebSearchTool(provider="tavily", max_results=3)],
    )
    print(msg.content[0].text)
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .auth import login, logout
from .clients import CompressionClient
from .config import MODELS
from .retry import RetryConfig

try:
    __version__ = version("compresr")
except PackageNotFoundError:
    # Package not installed (e.g., running from source)
    __version__ = "0.0.0-dev"

__all__ = [
    "CompressionClient",
    "MODELS",
    "RetryConfig",
    "login",
    "logout",
    "WebSearchTool",
    "CompressionPolicy",
    "ResearchResult",
    "ResearchAgent",
    "Citation",
]


def __getattr__(name: str) -> Any:
    """Lazy attribute access — defer langchain imports until used."""
    if name == "WebSearchTool":
        from .agents import tools as _t

        return _t.WebSearchTool
    if name == "CompressionPolicy":
        from .integrations._shared import CompressionPolicy as _P

        return _P
    if name in {"ResearchResult", "ResearchAgent", "Citation"}:
        from .agents import research as _r

        return getattr(_r, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
