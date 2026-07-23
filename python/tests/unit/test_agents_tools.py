"""Unit tests for compresr.agents.tools — WebSearchTool."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("langchain_core")

from compresr.agents.tools import WebSearchTool  # noqa: E402

# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    def test_default_provider_is_tavily(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        tool = WebSearchTool()
        # Returned object is a langchain BaseTool (TavilySearch). Duck-type check.
        assert hasattr(tool, "name") or hasattr(tool, "invoke")

    def test_tavily_with_explicit_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = WebSearchTool(api_key="tvly-test", max_results=3)
        assert hasattr(tool, "invoke")
        # New contract: the key is used internally; the env stays untouched
        # so the SDK doesn't leak state into the caller's process.
        assert "TAVILY_API_KEY" not in os.environ

    def test_tavily_max_results_passed(self, monkeypatch):
        """``max_results=`` must reach the underlying TavilySearch constructor."""
        seen_kwargs: dict = {}

        class _StubTavily:
            def __init__(self, **kw):
                seen_kwargs.update(kw)

            def invoke(self, _):  # pragma: no cover - not exercised here
                return {"results": []}

        import sys
        from types import ModuleType

        fake_mod = ModuleType("langchain_tavily")
        fake_mod.TavilySearch = _StubTavily  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_tavily", fake_mod)

        WebSearchTool(api_key="tvly-test", max_results=7)
        assert seen_kwargs.get("max_results") == 7

    def test_tavily_include_exclude_domains(self, monkeypatch):
        """``allowed_domains`` / ``blocked_domains`` map onto Tavily's
        ``include_domains`` / ``exclude_domains`` kwargs."""
        seen_kwargs: dict = {}

        class _StubTavily:
            def __init__(self, **kw):
                seen_kwargs.update(kw)

            def invoke(self, _):  # pragma: no cover - not exercised here
                return {"results": []}

        import sys
        from types import ModuleType

        fake_mod = ModuleType("langchain_tavily")
        fake_mod.TavilySearch = _StubTavily  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_tavily", fake_mod)

        WebSearchTool(
            api_key="tvly-test",
            allowed_domains=["nytimes.com"],
            blocked_domains=["example.com"],
        )
        assert seen_kwargs.get("include_domains") == ["nytimes.com"]
        assert seen_kwargs.get("exclude_domains") == ["example.com"]

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown web search provider"):
            WebSearchTool(provider="duckduckgo")

    def test_brave_requires_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Brave requires"):
            WebSearchTool(provider="brave")

    def test_brave_with_explicit_key(self):
        # langchain-community may or may not be installed in the venv; allow either path.
        try:
            tool = WebSearchTool(provider="brave", api_key="brave-test-key", max_results=4)
        except ImportError:
            pytest.skip("langchain-community not installed")
        assert hasattr(tool, "invoke") or hasattr(tool, "run")

    def test_tavily_classmethod_factory(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = WebSearchTool.tavily(api_key="tvly-test", max_results=3)
        # Equivalent to passing provider="tavily" explicitly.
        assert hasattr(tool, "invoke")
        # Env stays clean — the key is passed directly to the underlying tool.
        assert "TAVILY_API_KEY" not in os.environ
        # Behaves the same as the string-provider form.
        reference = WebSearchTool(provider="tavily", api_key="tvly-test", max_results=3)
        assert type(tool) is type(reference)

    def test_tavily_requires_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Tavily requires"):
            WebSearchTool(provider="tavily")

    def test_brave_classmethod_factory_requires_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Brave requires"):
            WebSearchTool.brave()

    def test_brave_classmethod_with_key(self):
        try:
            tool = WebSearchTool.brave(api_key="brave-test", max_results=4)
        except ImportError:
            pytest.skip("langchain-community not installed")
        assert hasattr(tool, "invoke") or hasattr(tool, "run")

    def test_brave_has_query_args_schema(self):
        """Regression: langchain_community.BraveSearch ships with args_schema=None,
        so the LLM can't tell it needs to emit ``query`` and ``_run`` errors with
        ``missing 1 required positional argument: 'query'``. Our wrapper must
        attach an explicit Pydantic schema."""
        try:
            tool = WebSearchTool.brave(api_key="brave-test", max_results=4)
        except ImportError:
            pytest.skip("langchain-community not installed")
        assert tool.args_schema is not None, "Brave tool must declare an args_schema"
        fields = getattr(tool.args_schema, "model_fields", None) or getattr(
            tool.args_schema, "__fields__", {}
        )
        assert "query" in fields, f"Brave schema missing 'query' field: {list(fields)}"

    def test_tavily_has_query_args_schema(self, monkeypatch):
        """Parity with brave: an explicit query schema so the LLM emits a
        well-formed tool call."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        tool = WebSearchTool.tavily(max_results=3)
        assert tool.args_schema is not None
        fields = getattr(tool.args_schema, "model_fields", None) or getattr(
            tool.args_schema, "__fields__", {}
        )
        assert "query" in fields

    def test_tavily_invoke_returns_plain_text_blocks(self, monkeypatch):
        """latte_v1 no-ops on JSON-shaped tool output — the Tavily wrapper
        must flatten ``{results: [...]}`` into plain text before the middleware
        sees it."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

        class _StubTavily:
            def __init__(self, *_a, **_k):
                pass

            def invoke(self, _args):
                return {
                    "query": "x",
                    "results": [
                        {
                            "title": "SAS 9.4 cohort",
                            "url": "https://example.org/a",
                            "content": "All analyses used SAS 9.4 (Cary, NC).",
                        },
                        {
                            "title": "BRCA1 mutation study",
                            "url": "https://example.org/b",
                            "content": "Logistic regression with SAS 9.1.",
                        },
                    ],
                }

        import compresr.agents.tools.web_search as ws

        monkeypatch.setattr(
            ws,
            "_build_tavily",
            (
                ws._build_tavily.__wrapped__
                if hasattr(ws._build_tavily, "__wrapped__")
                else ws._build_tavily
            ),
        )
        # Patch the import target inside _build_tavily so the wrapper builds a
        # StructuredTool around our stub instead of hitting the real Tavily SDK.
        import sys
        from types import ModuleType

        fake_mod = ModuleType("langchain_tavily")
        fake_mod.TavilySearch = _StubTavily  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_tavily", fake_mod)

        tool = WebSearchTool.tavily(api_key="tvly-test", max_results=2)
        out = tool.invoke({"query": "anything"})

        # Plain text (not JSON), with the title/url/content blocks visible.
        assert isinstance(out, str)
        assert not out.lstrip().startswith("{"), "Output must be plain text, not JSON"
        assert "SAS 9.4 cohort" in out
        assert "https://example.org/a" in out
        assert "All analyses used SAS 9.4" in out
        assert "BRCA1 mutation study" in out
        # Two blocks, blank-line separated.
        assert "\n\n" in out

    def test_tavily_invoke_handles_unknown_shape(self, monkeypatch):
        """Fallback: dict without ``results`` is still stringified deterministically."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

        class _StubWeirdTavily:
            def __init__(self, *_a, **_k):
                pass

            def invoke(self, _args):
                # No "results" key — fallback path.
                return {"error": "rate-limited", "code": 429}

        import sys
        from types import ModuleType

        fake_mod = ModuleType("langchain_tavily")
        fake_mod.TavilySearch = _StubWeirdTavily  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_tavily", fake_mod)

        tool = WebSearchTool.tavily(api_key="tvly-test", max_results=2)
        out = tool.invoke({"query": "anything"})

        assert isinstance(out, str)
        # Fallback path stringifies the dict so the LLM still sees something.
        assert "rate-limited" in out


# ---------------------------------------------------------------------------
# WebSearchTool — AgentCore (Amazon Bedrock AgentCore web search)
# ---------------------------------------------------------------------------

_AGENTCORE_ENV_VARS = (
    "AGENTCORE_GATEWAY_MCP_URL",
    "GATEWAY_MCP_URL",
    "AGENTCORE_COGNITO_TOKEN_URL",
    "COGNITO_TOKEN_URL",
    "AGENTCORE_COGNITO_CLIENT_ID",
    "COGNITO_CLIENT_ID",
    "AGENTCORE_COGNITO_CLIENT_SECRET",
    "COGNITO_CLIENT_SECRET",
    "AGENTCORE_COGNITO_SCOPE",
    "COGNITO_SCOPE",
)


def _clear_agentcore_env(monkeypatch):
    for name in _AGENTCORE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


_AGENTCORE_ARGS = dict(
    gateway_url="https://gw.example/mcp",
    cognito_token_url="https://auth.example/oauth2/token",
    client_id="client-123",
    client_secret="secret-xyz",
    scope="gateway/invoke",
)


class _StubAgentCoreClient:
    """Records construction + search calls; never touches the network."""

    instances: list = []

    def __init__(self, config):
        self.config = config
        self.searches: list = []
        _StubAgentCoreClient.instances.append(self)

    def search(self, query, max_results=None):
        self.searches.append((query, max_results))
        return [
            {
                "title": "SAS 9.4 cohort",
                "url": "https://example.org/a",
                "content": "All analyses used SAS 9.4 (Cary, NC).",
                "published_date": "2024-01-01",
            },
            {
                "title": "BRCA1 mutation study",
                "url": "https://example.org/b",
                "content": "Logistic regression with SAS 9.1.",
                "published_date": None,
            },
        ]


class TestWebSearchToolAgentCore:
    def test_missing_config_raises_valueerror_naming_keys(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        with pytest.raises(ValueError) as exc:
            WebSearchTool.agentcore()
        msg = str(exc.value)
        assert "missing required config" in msg
        # Names each missing field and its env-var fallbacks.
        assert "gateway_url" in msg
        assert "AGENTCORE_GATEWAY_MCP_URL" in msg
        assert "GATEWAY_MCP_URL" in msg
        assert "client_secret" in msg

    def test_unknown_provider_message_lists_agentcore(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        with pytest.raises(ValueError, match="agentcore"):
            WebSearchTool(provider="duckduckgo")

    def test_explicit_args_build_tool_named_agentcore_web_search(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        tool = WebSearchTool.agentcore(**_AGENTCORE_ARGS, max_results=5)
        assert tool.name == "agentcore_web_search"
        fields = getattr(tool.args_schema, "model_fields", None) or getattr(
            tool.args_schema, "__fields__", {}
        )
        assert "query" in fields

    def test_config_from_env_vars(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        monkeypatch.setenv("AGENTCORE_GATEWAY_MCP_URL", "https://gw.example/mcp")
        monkeypatch.setenv("AGENTCORE_COGNITO_TOKEN_URL", "https://auth.example/token")
        monkeypatch.setenv("AGENTCORE_COGNITO_CLIENT_ID", "client-env")
        monkeypatch.setenv("AGENTCORE_COGNITO_CLIENT_SECRET", "secret-env")
        monkeypatch.setenv("AGENTCORE_COGNITO_SCOPE", "gateway/invoke")
        tool = WebSearchTool.agentcore()
        assert tool.name == "agentcore_web_search"

    def test_config_from_bare_env_fallbacks(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        # Only the non-prefixed fallback names are set.
        monkeypatch.setenv("GATEWAY_MCP_URL", "https://gw.example/mcp")
        monkeypatch.setenv("COGNITO_TOKEN_URL", "https://auth.example/token")
        monkeypatch.setenv("COGNITO_CLIENT_ID", "client-bare")
        monkeypatch.setenv("COGNITO_CLIENT_SECRET", "secret-bare")
        monkeypatch.setenv("COGNITO_SCOPE", "gateway/invoke")
        tool = WebSearchTool.agentcore()
        assert tool.name == "agentcore_web_search"

    def test_run_flattens_results_to_plaintext(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        _StubAgentCoreClient.instances = []
        import compresr.agents.tools._agentcore as ac

        monkeypatch.setattr(ac, "AgentCoreClient", _StubAgentCoreClient)

        tool = WebSearchTool.agentcore(**_AGENTCORE_ARGS, max_results=2)
        out = tool.invoke({"query": "anything"})

        assert isinstance(out, str)
        assert not out.lstrip().startswith("{"), "Output must be plain text, not JSON"
        assert "SAS 9.4 cohort" in out
        assert "https://example.org/a" in out
        assert "All analyses used SAS 9.4" in out
        assert "BRCA1 mutation study" in out
        # Two blocks, blank-line separated.
        assert "\n\n" in out

    def test_max_results_reaches_client(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        _StubAgentCoreClient.instances = []
        import compresr.agents.tools._agentcore as ac

        monkeypatch.setattr(ac, "AgentCoreClient", _StubAgentCoreClient)

        tool = WebSearchTool.agentcore(**_AGENTCORE_ARGS, max_results=9)
        tool.invoke({"query": "anything"})

        assert len(_StubAgentCoreClient.instances) == 1
        client = _StubAgentCoreClient.instances[0]
        assert client.config.max_results == 9
        assert client.searches == [("anything", 9)]

    def test_max_results_clamped_to_1_25(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        from compresr.agents.tools._agentcore import AgentCoreClient, AgentCoreConfig

        seen: list = []

        async def _fake_search_async(self, query, max_results, *, refresh):
            seen.append(max_results)
            return {"results": []}

        monkeypatch.setattr(AgentCoreClient, "_search_async", _fake_search_async)

        high = AgentCoreClient(AgentCoreConfig(**_AGENTCORE_ARGS, max_results=100))
        high.search("q")
        assert seen[-1] == 25

        low = AgentCoreClient(AgentCoreConfig(**_AGENTCORE_ARGS, max_results=0))
        low.search("q")
        assert seen[-1] == 1

    def test_search_retries_once_on_401_with_refreshed_token(self, monkeypatch):
        """Hard-won fact (IMPLEMENTATION_GUIDE.md P1): a 401 on the first
        ``_search_async`` call must trigger exactly one token re-mint
        (``token(refresh=True)``) and a retried call that succeeds."""
        _clear_agentcore_env(monkeypatch)
        from compresr.agents.tools._agentcore import AgentCoreClient, AgentCoreConfig

        calls: list[bool] = []
        token_calls: list[bool] = []

        class _Unauthorized(Exception):
            class _Response:
                status_code = 401

            response = _Response()

        async def _fake_search_async(self, query, max_results, *, refresh):
            calls.append(refresh)
            # Mirror the real implementation: every search attempt mints/
            # reuses a bearer token via self.token(refresh=...).
            self.token(refresh=refresh)
            if len(calls) == 1:
                raise _Unauthorized("401 Unauthorized")
            return {
                "results": [
                    {
                        "title": "Recovered after refresh",
                        "url": "https://example.org/c",
                        "text": "Worked on retry.",
                        "publishedDate": "2024-02-02",
                    }
                ]
            }

        def _fake_token(self, refresh: bool = False) -> str:
            token_calls.append(refresh)
            return "fake-token"

        monkeypatch.setattr(AgentCoreClient, "_search_async", _fake_search_async)
        monkeypatch.setattr(AgentCoreClient, "token", _fake_token)

        client = AgentCoreClient(AgentCoreConfig(**_AGENTCORE_ARGS, max_results=5))
        out = client.search("q")

        # Exactly one retry: the first call failed (refresh=False), the
        # second succeeded (refresh=True).
        assert calls == [False, True]
        # token(refresh=True) was actually exercised on the retry.
        assert token_calls == [False, True]
        assert out == [
            {
                "title": "Recovered after refresh",
                "url": "https://example.org/c",
                "content": "Worked on retry.",
                "published_date": "2024-02-02",
            }
        ]

    def test_parse_tool_result_surfaces_gateway_tool_error(self):
        """isError=True means plain-text content, NOT JSON — must not be fed
        to json.loads (would raise a confusing JSONDecodeError)."""
        from compresr.agents.tools._agentcore import _parse_tool_result

        class _Block:
            text = "Tool error: bad query"

        class _Result:
            content = [_Block()]
            isError = True

        with pytest.raises(RuntimeError, match="Tool error: bad query"):
            _parse_tool_result(_Result())

    def test_parse_tool_result_error_text_preserves_401_for_retry_heuristic(self):
        from compresr.agents.tools._agentcore import _is_unauthorized, _parse_tool_result

        class _Block:
            text = "401 Unauthorized: token expired"

        class _Result:
            content = [_Block()]
            isError = True

        with pytest.raises(RuntimeError) as exc:
            _parse_tool_result(_Result())
        assert _is_unauthorized(exc.value)

    def test_parse_tool_result_malformed_json_raises_clear_runtimeerror(self):
        from compresr.agents.tools._agentcore import _parse_tool_result

        class _Block:
            text = "not json at all {"

        class _Result:
            content = [_Block()]
            isError = False

        with pytest.raises(RuntimeError, match="malformed JSON"):
            _parse_tool_result(_Result())

    def test_parse_tool_result_rejects_non_object_json(self):
        from compresr.agents.tools._agentcore import _parse_tool_result

        class _Block:
            text = "[1, 2, 3]"

        class _Result:
            content = [_Block()]
            isError = False

        with pytest.raises(RuntimeError, match="unexpected JSON shape"):
            _parse_tool_result(_Result())

    def test_parse_tool_result_oversized_response_rejected(self, monkeypatch):
        import compresr.agents.tools._agentcore as ac

        monkeypatch.setattr(ac, "MAX_RESPONSE_BYTES", 10)

        class _Block:
            text = '{"results": []}'  # > 10 bytes

        class _Result:
            content = [_Block()]
            isError = False

        with pytest.raises(RuntimeError, match="size guard"):
            ac._parse_tool_result(_Result())

    def test_search_skips_non_dict_items_in_results(self, monkeypatch):
        """Defensive parsing: a malformed gateway payload with non-dict rows
        in 'results' must not crash with AttributeError."""
        _clear_agentcore_env(monkeypatch)
        from compresr.agents.tools._agentcore import AgentCoreClient, AgentCoreConfig

        async def _fake_search_async(self, query, max_results, *, refresh):
            return {"results": ["not-a-dict", {"title": "ok", "url": "u", "text": "c"}, 42]}

        monkeypatch.setattr(AgentCoreClient, "_search_async", _fake_search_async)
        client = AgentCoreClient(AgentCoreConfig(**_AGENTCORE_ARGS, max_results=5))
        out = client.search("q")
        assert len(out) == 1
        assert out[0]["title"] == "ok"

    def test_token_error_does_not_leak_secret_in_message(self, monkeypatch):
        """The Cognito client_secret must never appear in a raised error message."""
        from compresr.agents.tools._agentcore import AgentCoreClient, AgentCoreConfig

        class _FakeResponse:
            status_code = 401

            def raise_for_status(self):
                import httpx

                raise httpx.HTTPStatusError(
                    "401", request=None, response=self  # type: ignore[arg-type]
                )

        def _fake_post(*_a, **_k):
            return _FakeResponse()

        import compresr.agents.tools._agentcore as ac

        monkeypatch.setattr(ac.httpx, "post", _fake_post)

        client = AgentCoreClient(AgentCoreConfig(**_AGENTCORE_ARGS, max_results=5))
        with pytest.raises(RuntimeError) as exc:
            client.token()
        assert _AGENTCORE_ARGS["client_secret"] not in str(exc.value)
        assert "401" in str(exc.value)

    def test_mcp_not_installed_raises_import_error_with_hint(self, monkeypatch):
        _clear_agentcore_env(monkeypatch)
        import sys

        from compresr.agents.tools._agentcore import AgentCoreClient, AgentCoreConfig

        # Simulate the `agentcore` extra not being installed.
        monkeypatch.setitem(sys.modules, "mcp", None)
        monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", None)

        client = AgentCoreClient(AgentCoreConfig(**_AGENTCORE_ARGS, max_results=5))
        with pytest.raises(ImportError, match=r"compresr\[agentcore\]"):
            client.search("q")
