"""WebSearchTool — backed by Tavily, Brave, or Amazon Bedrock AgentCore.

Returns a LangChain ``BaseTool`` whose output flows through Compresr's
``CompresrToolMiddleware`` automatically when used with the SDK's engine
(``client.messages.create`` / ``client.chat.completions.create`` /
``client.run``).

The ``agentcore`` provider reaches Amazon Bedrock AgentCore web search via a
Cognito OAuth handshake + MCP streamable-HTTP session; its runtime deps are
optional (``pip install compresr[agentcore]``).

Provider-native server tools (Anthropic ``web_search_20250305``, OpenAI
``web_search_preview``, Gemini ``google_search``) are intentionally NOT
supported here — they execute server-side and return opaque/encrypted
content that Compresr cannot read or compress. Use a real search API
(Tavily or Brave) so the result is plaintext we can compress.
"""

from __future__ import annotations

import os
from typing import Any, Optional


class WebSearchTool:
    """Factory for a LangChain web-search tool. Default provider is Tavily.

    Args:
        provider: ``"tavily"`` (default) or ``"brave"``.
        api_key: Provider API key. Falls back to ``TAVILY_API_KEY`` /
            ``BRAVE_SEARCH_API_KEY`` env vars.
        max_results: How many results to return.
        allowed_domains: Restrict to these domains (Tavily: ``include_domains``).
        blocked_domains: Exclude these domains (Tavily: ``exclude_domains``).
        **extra: Forwarded to the underlying LangChain tool constructor.

    Returns:
        A ``langchain_core.tools.BaseTool`` ready to pass to
        ``client.messages.create(tools=[...])``.

    Prefer the provider-named classmethod factories for discoverability
    and static-typing friendliness::

        WebSearchTool.tavily(api_key=..., max_results=5)
        WebSearchTool.brave(api_key=..., max_results=5)
        WebSearchTool.agentcore(gateway_url=..., ...)
    """

    @classmethod
    def tavily(
        cls,
        *,
        api_key: Optional[str] = None,
        max_results: int = 5,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        **extra: Any,
    ) -> Any:
        """Tavily-backed web search. Returns a LangChain ``BaseTool``.

        Compresr's middleware compresses the tool output automatically.
        """
        return cls(
            provider="tavily",
            api_key=api_key,
            max_results=max_results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            **extra,
        )

    @classmethod
    def brave(
        cls,
        *,
        api_key: Optional[str] = None,
        max_results: int = 5,
        **extra: Any,
    ) -> Any:
        """Brave-backed web search. Returns a LangChain ``BaseTool``."""
        return cls(
            provider="brave",
            api_key=api_key,
            max_results=max_results,
            **extra,
        )

    @classmethod
    def agentcore(
        cls,
        *,
        gateway_url: Optional[str] = None,
        cognito_token_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scope: Optional[str] = None,
        max_results: int = 5,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        **extra: Any,
    ) -> Any:
        """Amazon Bedrock AgentCore web search. Returns a LangChain ``BaseTool``.

        Config resolves from explicit args first, then env vars
        (``AGENTCORE_GATEWAY_MCP_URL`` / ``GATEWAY_MCP_URL``,
        ``AGENTCORE_COGNITO_TOKEN_URL`` / ``COGNITO_TOKEN_URL``,
        ``AGENTCORE_COGNITO_CLIENT_ID`` / ``COGNITO_CLIENT_ID``,
        ``AGENTCORE_COGNITO_CLIENT_SECRET`` / ``COGNITO_CLIENT_SECRET``,
        ``AGENTCORE_COGNITO_SCOPE`` / ``COGNITO_SCOPE``). Requires the
        ``agentcore`` extra: ``pip install compresr[agentcore]``.
        """
        return cls(
            provider="agentcore",
            gateway_url=gateway_url,
            cognito_token_url=cognito_token_url,
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
            max_results=max_results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            **extra,
        )

    def __new__(
        cls,
        *,
        provider: str = "tavily",
        api_key: Optional[str] = None,
        max_results: int = 5,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        gateway_url: Optional[str] = None,
        cognito_token_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scope: Optional[str] = None,
        **extra: Any,
    ) -> Any:
        if provider == "tavily":
            return _build_tavily(
                api_key=api_key,
                max_results=max_results,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                **extra,
            )
        if provider == "brave":
            return _build_brave(
                api_key=api_key,
                max_results=max_results,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                **extra,
            )
        if provider == "agentcore":
            return _build_agentcore(
                gateway_url=gateway_url,
                cognito_token_url=cognito_token_url,
                client_id=client_id,
                client_secret=client_secret,
                scope=scope,
                max_results=max_results,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                **extra,
            )
        raise ValueError(
            f"Unknown web search provider: {provider!r}. Use 'tavily', 'brave', or 'agentcore'."
        )


def _flatten_search_results(out: Any) -> str:
    """Flatten ``{"results": [{title, url, content}, ...]}`` into blank-line
    -separated plain-text blocks.
    """
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        results = out.get("results")
        if isinstance(results, list):
            parts: list[str] = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                title = (r.get("title") or "").strip()
                url = (r.get("url") or "").strip()
                content = (r.get("content") or r.get("snippet") or "").strip()
                block = "\n".join(p for p in (title, url, content) if p)
                if block:
                    parts.append(block)
            if parts:
                return "\n\n".join(parts)
    import json

    return json.dumps(out, ensure_ascii=False, default=str)


def _build_tavily(
    *,
    api_key: Optional[str],
    max_results: int,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    **extra: Any,
) -> Any:
    try:
        from langchain_tavily import TavilySearch
    except ImportError as exc:
        raise ImportError(
            "WebSearchTool(provider='tavily') requires langchain-tavily. "
            "Install with: pip install compresr[agents-tavily]"
        ) from exc
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise ValueError("Tavily requires api_key= or TAVILY_API_KEY env var.")
    kwargs: dict = {"max_results": max_results, "tavily_api_key": key}
    if allowed_domains is not None:
        kwargs["include_domains"] = list(allowed_domains)
    if blocked_domains is not None:
        kwargs["exclude_domains"] = list(blocked_domains)
    kwargs.update(extra)
    base = TavilySearch(**kwargs)

    # Wrap with a StructuredTool so the returned ToolMessage content is plain
    # text (latte_v1 no-ops on JSON), and so an explicit ``query`` schema
    # reaches the LLM (parity with brave).
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _TavilySearchInput(BaseModel):
        query: str = Field(..., description="The search query string.")

    def _run(query: str) -> str:
        return _flatten_search_results(base.invoke({"query": query}))

    return StructuredTool.from_function(
        func=_run,
        name="tavily_search",
        description=getattr(base, "description", None)
        or "Tavily web search. Use to answer questions about current events.",
        args_schema=_TavilySearchInput,
    )


def _build_brave(
    *,
    api_key: Optional[str],
    max_results: int,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    **extra: Any,
) -> Any:
    import warnings as _warnings

    try:
        with _warnings.catch_warnings():
            # langchain-community emits a sunset DeprecationWarning. Brave's
            # canonical home is still langchain-community as of this release;
            # suppress that one specific notice instead of every warning.
            _warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*langchain-community.*sunset.*",
            )
            from langchain_community.tools import BraveSearch
    except ImportError as exc:
        raise ImportError(
            "WebSearchTool(provider='brave') requires langchain-community. "
            "Install with: pip install compresr[agents-brave]"
        ) from exc
    key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY") or os.environ.get("BRAVE_API_KEY")
    if not key:
        raise ValueError("Brave requires api_key= or BRAVE_SEARCH_API_KEY env var.")
    search_kwargs: dict = {"count": max_results}
    # Brave's allowed/blocked domain support is via Goggles — out of scope for v1.
    if allowed_domains or blocked_domains:
        import warnings

        warnings.warn(
            "Brave does not natively support allowed_domains/blocked_domains; "
            "use Tavily for domain filtering.",
            stacklevel=3,
        )
    search_kwargs.update(extra)
    base = BraveSearch.from_api_key(api_key=key, search_kwargs=search_kwargs)

    # ``langchain_community.tools.BraveSearch`` ships with ``args_schema=None``,
    # so when the LLM (especially Anthropic Claude via tool_use) is asked to
    # call it, the tool schema sent in the prompt has no ``query`` field. The
    # model emits an empty args dict, and ``BraveSearch._run(query)`` then
    # raises ``TypeError: missing 1 required positional argument: 'query'``.
    # Wrap with an explicit pydantic schema so the LLM always sees ``query``.
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _BraveSearchInput(BaseModel):
        query: str = Field(..., description="The search query string.")

    return StructuredTool.from_function(
        func=lambda query: base.run(query),
        name="brave_search",
        description=base.description
        or "Brave web search. Use it to answer questions about current events.",
        args_schema=_BraveSearchInput,
    )


# Field -> (explicit-arg resolver, ordered env-var fallbacks). The first env var
# is the AgentCore-namespaced name; the second matches the reference client.
_AGENTCORE_ENV: dict[str, tuple[str, ...]] = {
    "gateway_url": ("AGENTCORE_GATEWAY_MCP_URL", "GATEWAY_MCP_URL"),
    "cognito_token_url": ("AGENTCORE_COGNITO_TOKEN_URL", "COGNITO_TOKEN_URL"),
    "client_id": ("AGENTCORE_COGNITO_CLIENT_ID", "COGNITO_CLIENT_ID"),
    "client_secret": ("AGENTCORE_COGNITO_CLIENT_SECRET", "COGNITO_CLIENT_SECRET"),
    "scope": ("AGENTCORE_COGNITO_SCOPE", "COGNITO_SCOPE"),
}


def _resolve_agentcore_config(values: dict) -> dict:
    """Resolve each AgentCore field via explicit arg → env-var fallbacks.

    Raises ``ValueError`` naming every missing field and its env-var fallbacks.
    Never echoes the resolved secret value.
    """
    resolved: dict = {}
    missing: list[str] = []
    for field, env_names in _AGENTCORE_ENV.items():
        value = values.get(field)
        for env_name in env_names:
            if value:
                break
            value = os.environ.get(env_name)
        if value:
            resolved[field] = value
        else:
            missing.append(f"{field} ({' or '.join(env_names)})")
    if missing:
        raise ValueError(
            "WebSearchTool.agentcore is missing required config: "
            + "; ".join(missing)
            + ". Pass it explicitly or set the env var."
        )
    return resolved


def _build_agentcore(
    *,
    gateway_url: Optional[str],
    cognito_token_url: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
    scope: Optional[str],
    max_results: int,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    **extra: Any,
) -> Any:
    from ._agentcore import AgentCoreClient, AgentCoreConfig

    if allowed_domains or blocked_domains:
        import warnings

        warnings.warn(
            "AgentCore web search does not natively support allowed_domains/"
            "blocked_domains; use Tavily for domain filtering.",
            stacklevel=3,
        )

    resolved = _resolve_agentcore_config(
        {
            "gateway_url": gateway_url,
            "cognito_token_url": cognito_token_url,
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope,
        }
    )
    config = AgentCoreConfig(max_results=max_results, **resolved)
    # One shared client so the cached bearer token is reused across calls.
    client = AgentCoreClient(config)

    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _AgentCoreSearchInput(BaseModel):
        query: str = Field(..., description="The search query string.")

    def _run(query: str) -> str:
        return _flatten_search_results({"results": client.search(query, max_results)})

    tool_kwargs: dict = {
        "func": _run,
        "name": "agentcore_web_search",
        "description": "Amazon Bedrock AgentCore web search. Use to answer "
        "questions about current events.",
        "args_schema": _AgentCoreSearchInput,
    }
    # Unlike tavily/brave, there is no underlying SDK call to forward
    # provider-specific kwargs to — extras are accepted (for signature parity)
    # and passed straight through to StructuredTool.from_function.
    tool_kwargs.update(extra)
    return StructuredTool.from_function(**tool_kwargs)


__all__ = ["WebSearchTool"]
