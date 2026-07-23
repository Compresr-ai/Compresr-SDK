"""AgentCore Web Search client (Amazon Bedrock AgentCore over MCP).

Ported from the standalone ``websearch_client.py`` reference. Wraps the
Cognito client-credentials OAuth handshake plus an MCP streamable-HTTP session
into a synchronous :meth:`AgentCoreClient.search` call that works in plain
scripts and in Jupyter (handles the already-running event-loop case).

The runtime dependencies (``mcp``, ``nest-asyncio``) are **optional** — they are
imported lazily inside the call so a bare ``pip install compresr`` still imports
fine. The error only surfaces when ``WebSearchTool.agentcore(...)`` is actually
invoked without the extra installed::

    pip install compresr[agentcore]

Never logs the Cognito client secret or the bearer token.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional

import httpx

# Guard against an unbounded/misbehaving gateway response before buffering it
# in memory and parsing as JSON.
MAX_RESPONSE_BYTES = 1_000_000

# The gateway also exposes this built-in *tool*-search (semantic search over the
# gateway's own tools). It is NOT web search and must be ignored.
TOOL_SEARCH_META = "x_amz_bedrock_agentcore_search"

TOKEN_TIMEOUT_SECONDS = 30

# AgentCore web search accepts maxResults in the inclusive range 1..25.
MIN_RESULTS = 1
MAX_RESULTS = 25

_MISSING_DEP_MSG = (
    "WebSearchTool.agentcore requires the 'agentcore' extra. "
    "Install: pip install compresr[agentcore]"
)


@dataclass(frozen=True)
class AgentCoreConfig:
    """Immutable connection config for the AgentCore web-search gateway."""

    gateway_url: str
    cognito_token_url: str
    client_id: str
    client_secret: str
    scope: str
    max_results: int = 5


def _clamp_results(value: int) -> int:
    """Clamp a requested result count to AgentCore's inclusive 1..25 range."""
    return max(MIN_RESULTS, min(MAX_RESULTS, value))


def _run(coro: Any) -> Any:
    """Run a coroutine whether or not an event loop is already running (Jupyter)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Inside a running loop (e.g. Jupyter): allow nested execution.
    try:
        import nest_asyncio
    except ImportError as exc:  # pragma: no cover - exercised via search()
        raise ImportError(_MISSING_DEP_MSG) from exc

    nest_asyncio.apply()
    return loop.run_until_complete(coro)


class AgentCoreClient:
    """Synchronous client for AgentCore web search over MCP.

    Mints (and caches) a Cognito bearer token, then opens an MCP
    streamable-HTTP session to call the gateway's web-search tool. A single
    instance is shared by the built tool so the token is reused across calls.
    """

    def __init__(self, config: AgentCoreConfig) -> None:
        self._config = config
        self._token: Optional[str] = None
        self._token_lock = threading.Lock()

    # --- auth ----------------------------------------------------------------

    def token(self, refresh: bool = False) -> str:
        """Return a cached bearer token, minting a fresh one when needed.

        Guarded by a lock (double-checked locking) so concurrent first-use
        calls from multiple threads only mint one token. The client secret
        and the resulting token are never logged.
        """
        if self._token and not refresh:
            return self._token
        with self._token_lock:
            # Re-check after acquiring the lock: another thread may have
            # already minted/refreshed the token while we were waiting.
            if self._token and not refresh:
                return self._token
            try:
                resp = httpx.post(
                    self._config.cognito_token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._config.client_id,
                        "client_secret": self._config.client_secret,
                        "scope": self._config.scope,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=TOKEN_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                self._token = resp.json()["access_token"]
            except httpx.HTTPError as exc:
                # Re-raise sanitized: the original exception's .request (and
                # this frame's locals) carry the client_secret, which we do
                # not want chained into error-monitoring tools (e.g. Sentry).
                status = getattr(getattr(exc, "response", None), "status_code", None)
                detail = f" (HTTP {status})" if status is not None else ""
                raise RuntimeError(f"AgentCore Cognito token request failed{detail}.") from None
            return self._token

    # --- tool selection ------------------------------------------------------

    @staticmethod
    def _pick_web_search_tool(names: list[str]) -> str:
        """Pick the web-search tool, ignoring the semantic tool-search meta tool."""
        candidates = [n for n in names if n != TOOL_SEARCH_META]
        for name in candidates:
            if "websearch" in name.lower().replace("-", "").replace("_", ""):
                return name
        if not candidates:
            raise RuntimeError(f"No web-search tool found among {names}")
        return candidates[0]

    # --- search --------------------------------------------------------------

    async def _search_async(self, query: str, max_results: int, *, refresh: bool) -> dict:
        # Lazy import so a bare install imports fine; only fails on actual use.
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise ImportError(_MISSING_DEP_MSG) from exc

        headers = {"Authorization": f"Bearer {self.token(refresh=refresh)}"}
        async with streamablehttp_client(self._config.gateway_url, headers=headers) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = [t.name for t in (await session.list_tools()).tools]
                tool = self._pick_web_search_tool(names)
                result = await session.call_tool(tool, {"query": query, "maxResults": max_results})
                return _parse_tool_result(result)

    def search(self, query: str, max_results: Optional[int] = None) -> list[dict]:
        """Run a web search and return normalized result dicts.

        Each result is ``{"title", "url", "content", "published_date"}`` — the
        ``content`` key matches the SDK's ``_flatten_search_results`` flattener.
        Re-mints the bearer token once on an MCP 401 and retries.
        """
        count = _clamp_results(self._config.max_results if max_results is None else max_results)
        try:
            payload = _run(self._search_async(query, count, refresh=False))
        except Exception as exc:  # noqa: BLE001 - inspect for a 401 to retry once
            if _is_unauthorized(exc):
                payload = _run(self._search_async(query, count, refresh=True))
            else:
                raise
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError(
                "AgentCore web-search returned an unexpected 'results' shape "
                f"(expected a list, got {type(results).__name__})."
            )
        out: list[dict] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("text", "") or "",
                    "published_date": item.get("publishedDate"),
                }
            )
        return out


def _parse_tool_result(result: Any) -> dict:
    """Validate and parse an MCP ``CallToolResult`` into a JSON dict.

    Defensive against everything a misbehaving/compromised gateway could
    return: a tool-level error (``isError=True`` with plain-text content,
    NOT JSON), an oversized body, malformed JSON, or a JSON value that isn't
    an object. Raises ``RuntimeError`` with a clear, secret-free message in
    every case instead of letting a bare ``JSONDecodeError``/``AttributeError``
    propagate.
    """
    text = "".join(getattr(b, "text", "") for b in result.content)
    if getattr(result, "isError", False):
        # Gateway tool-level errors (bad query, throttling, auth/scope error)
        # come back as plain text, NOT JSON. Surface them directly instead of
        # failing on json.loads with a confusing JSONDecodeError. The text may
        # itself contain "401" — keep it intact so the unauthorized-retry
        # heuristic in _is_unauthorized still matches.
        raise RuntimeError(f"AgentCore web-search tool error: {text}")
    if len(text.encode("utf-8", errors="ignore")) > MAX_RESPONSE_BYTES:
        raise RuntimeError(
            f"AgentCore web-search response exceeded the {MAX_RESPONSE_BYTES} byte size guard."
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"AgentCore web-search returned malformed JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "AgentCore web-search returned an unexpected JSON shape "
            f"(expected an object, got {type(payload).__name__})."
        )
    return payload


def _is_unauthorized(exc: BaseException) -> bool:
    """Best-effort detection of an MCP/HTTP 401 to trigger a single token re-mint."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 401:
        return True
    return "401" in str(exc)


__all__ = ["AgentCoreConfig", "AgentCoreClient"]
