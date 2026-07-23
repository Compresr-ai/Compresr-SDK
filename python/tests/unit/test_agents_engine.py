"""Unit tests for compresr.agents.engine._Engine.

LangChain's ``init_chat_model`` and ``create_agent`` are patched so no
live API calls happen.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain")

from langchain_core.messages import AIMessage  # noqa: E402

from compresr.agents.engine import _Engine  # noqa: E402
from compresr.integrations._shared import CompressionPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# Construction / parsing
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_parses_llm_string(self, fake_client):
        with patch("compresr.agents.engine.init_chat_model") as m:
            m.return_value = MagicMock()
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-opus-4-8",
                llm_api_key="sk-test",
            )
        assert e.provider == "anthropic"
        assert e.model_name == "claude-opus-4-8"
        assert e.default_model_name == "claude-opus-4-8"

    def test_parses_provider_only_llm_string(self, fake_client):
        """Provider-only is the new preferred form — model lives at the call site."""
        e = _Engine(
            compresr_client=fake_client,
            llm="anthropic",
            llm_api_key="sk-test",
        )
        assert e.provider == "anthropic"
        assert e.default_model_name is None
        # Back-compat alias mirrors default_model_name
        assert e.model_name is None

    def test_rejects_empty_llm(self, fake_client):
        with pytest.raises(ValueError, match="provider"):
            _Engine(compresr_client=fake_client, llm="")

    def test_accepts_slash_separator_for_ts_sdk_parity(self, fake_client):
        with patch("compresr.agents.engine.init_chat_model") as m:
            m.return_value = MagicMock()
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic/claude-opus-4-8",
                llm_api_key="sk-test",
            )
        assert e.provider == "anthropic"
        assert e.model_name == "claude-opus-4-8"

    def test_uses_default_policy_when_none_given(self, fake_client):
        with patch("compresr.agents.engine.init_chat_model"):
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
        assert e._policy.compression_model_name == "latte_v1"

    def test_accepts_custom_policy(self, fake_client):
        policy = CompressionPolicy(
            target_compression_ratio=0.3,
            compression_model_name="espresso_v1",
            min_tokens=500,
        )
        with patch("compresr.agents.engine.init_chat_model"):
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-opus-4-8",
                policy=policy,
            )
        assert e._policy is policy
        assert e._policy.compression_model_name == "espresso_v1"

    def test_openai_attempts_responses_output_version(self, fake_client):
        """Chat-model construction is deferred to run(); verify the kwargs there."""
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="openai:gpt-4.1",
                llm_api_key="sk-x",
            )
            # Deferred — nothing initialized yet.
            assert ic.call_count == 0
            e.run(messages=[{"role": "user", "content": "hi"}])
        assert ic.call_count == 1
        call_kwargs = ic.call_args.kwargs
        assert call_kwargs.get("output_version") == "responses/v1"
        assert call_kwargs.get("api_key") == "sk-x"

    def test_openai_falls_back_when_output_version_rejected(self, fake_client):
        chat = MagicMock(name="chat")

        def side_effect(*_a, **kw):
            if "output_version" in kw:
                raise TypeError("unexpected keyword argument 'output_version'")
            return chat

        with (
            patch("compresr.agents.engine.init_chat_model", side_effect=side_effect) as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="openai:gpt-4.1",
                llm_api_key="sk-x",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        # The cached chat model came from the fallback path. Cache is keyed
        # by (model_name, sorted_kwargs_tuple). With max_tokens=4096 default,
        # the key includes that kwarg.
        cache_keys = list(e._chat_models.keys())
        assert any(k[0] == "gpt-4.1" for k in cache_keys), cache_keys
        assert all(v is chat for v in e._chat_models.values())
        # Two calls: first with output_version, second without.
        assert ic.call_count == 2


# ---------------------------------------------------------------------------
# run() — middleware wiring + normalization
# ---------------------------------------------------------------------------


class TestEngineRun:
    def test_run_calls_create_agent_with_middleware(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            chat = MagicMock()
            bound_chat = MagicMock(name="bound_chat")
            chat.bind = MagicMock(return_value=bound_chat)
            ic.return_value = chat
            agent = MagicMock()
            ai_msg = AIMessage(
                content="hello",
                response_metadata={"stop_reason": "end_turn"},
            )
            agent.invoke.return_value = {"messages": [ai_msg]}
            ca.return_value = agent

            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            out = e.run(messages=[{"role": "user", "content": "hi"}], tools=[])

            ca.assert_called_once()
            kw = ca.call_args.kwargs
            # max_tokens=4096 (the default) is baked into init_chat_model so
            # the freshly-built chat reaches create_agent. We never use `.bind`
            # because LangChain's bind_tools strips a prior bind's kwargs.
            assert kw["model"] is chat
            chat.bind.assert_not_called()
            assert any("CompresrToolMiddleware" in type(m).__name__ for m in kw["middleware"])
            assert out.text == "hello"
            assert out.stop_reason == "end_turn"
            assert out.raw is ai_msg

    def test_http_client_passthrough_to_constructor(self, fake_client):
        """``http_client``/``http_async_client`` reach the chat-model constructor
        (init_chat_model), not ``.bind`` — needed for corporate-proxy / custom-CA
        setups where the provider SDK needs a custom httpx client."""
        # Force the native-field path so the pre-native shim doesn't run against
        # the MagicMock chat (the shim is covered separately by
        # ``test_inject_anthropic_http_client_seeds_cached_clients``).
        native_ca = MagicMock()
        native_ca.model_fields = {"http_client": None, "http_async_client": None}
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
            patch("langchain_anthropic.ChatAnthropic", native_ca),
        ):
            chat = MagicMock()
            ic.return_value = chat
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            sync_client = object()
            async_client = object()

            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            e.run(
                messages=[{"role": "user", "content": "hi"}],
                http_client=sync_client,
                http_async_client=async_client,
            )

            kwargs = ic.call_args.kwargs
            assert kwargs["http_client"] is sync_client
            assert kwargs["http_async_client"] is async_client
            chat.bind.assert_not_called()

    def test_llm_http_client_injected_from_constructor(self, fake_client):
        """``llm_http_client`` set on the engine is baked into every chat-model
        build (set-once ergonomics, like the provider SDKs)."""
        native_ca = MagicMock()
        native_ca.model_fields = {"http_client": None, "http_async_client": None}
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
            patch("langchain_anthropic.ChatAnthropic", native_ca),
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            sync_client = object()
            async_client = object()

            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-opus-4-8",
                llm_http_client=sync_client,
                llm_http_async_client=async_client,
            )
            e.run(messages=[{"role": "user", "content": "hi"}])

            kwargs = ic.call_args.kwargs
            assert kwargs["http_client"] is sync_client
            assert kwargs["http_async_client"] is async_client

    def test_per_call_http_client_overrides_constructor(self, fake_client):
        """A per-call ``run(http_client=...)`` wins over the engine-level client."""
        native_ca = MagicMock()
        native_ca.model_fields = {"http_client": None, "http_async_client": None}
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
            patch("langchain_anthropic.ChatAnthropic", native_ca),
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            engine_client = object()
            call_client = object()

            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-opus-4-8",
                llm_http_client=engine_client,
            )
            e.run(
                messages=[{"role": "user", "content": "hi"}],
                http_client=call_client,
            )

            assert ic.call_args.kwargs["http_client"] is call_client

    def test_inject_anthropic_http_client_seeds_cached_clients(self):
        """The stock-version shim swaps the cached anthropic SDK clients for ones
        built around the caller's httpx clients, reusing ``_client_params``."""
        import anthropic
        import httpx

        class _FakeChat:
            _client_params = {
                "api_key": "sk-x",
                "base_url": "https://api.anthropic.com",
                "max_retries": 2,
            }

        chat = _FakeChat()
        sync_c = httpx.Client()
        async_c = httpx.AsyncClient()

        _Engine._inject_anthropic_http_client(chat, sync_c, async_c)

        assert isinstance(chat.__dict__["_client"], anthropic.Anthropic)
        assert chat.__dict__["_client"]._client is sync_c
        assert isinstance(chat.__dict__["_async_client"], anthropic.AsyncClient)
        assert chat.__dict__["_async_client"]._client is async_c

    def test_tools_pass_through(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            fake_tool = MagicMock()
            e.run(messages=[{"role": "user", "content": "hi"}], tools=[fake_tool])
            tools_passed = ca.call_args.kwargs["tools"]
            assert tools_passed == [fake_tool]

    def test_run_normalizes_tool_calls(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ai_msg = AIMessage(
                content="",
                tool_calls=[{"id": "t1", "name": "search", "args": {"q": "x"}}],
            )
            ca.return_value = MagicMock(invoke=lambda *_a, **_k: {"messages": [ai_msg]})
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            out = e.run(messages=[{"role": "user", "content": "hi"}], tools=[])
            assert out.tool_uses == [{"id": "t1", "name": "search", "input": {"q": "x"}}]

    def test_run_passes_system_prompt_to_create_agent(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            e.run(
                messages=[{"role": "user", "content": "hi"}],
                system="You are a helpful assistant.",
            )
            assert ca.call_args.kwargs.get("system_prompt") == "You are a helpful assistant."

    def test_run_normalizes_list_content_blocks(self, fake_client):
        """Anthropic-style list content with a text block should join to text."""
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ai_msg = AIMessage(
                content=[
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world."},
                ],
            )
            ca.return_value = MagicMock(invoke=lambda *_a, **_k: {"messages": [ai_msg]})
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            out = e.run(messages=[{"role": "user", "content": "hi"}])
            assert out.text == "Hello world."

    def test_run_extracts_anthropic_citation(self, fake_client):
        """Anthropic text block with embedded `citations` should flow into NormalizedResult.citations."""
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ai_msg = AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Per the docs",
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "url": "https://example.com/a",
                                "title": "Example A",
                                "cited_text": "the answer is 42",
                            }
                        ],
                    }
                ],
            )
            ca.return_value = MagicMock(invoke=lambda *_a, **_k: {"messages": [ai_msg]})
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            out = e.run(messages=[{"role": "user", "content": "hi"}])
            assert len(out.citations) == 1
            cit = out.citations[0]
            assert cit.url == "https://example.com/a"
            assert cit.title == "Example A"
            assert cit.cited_text == "the answer is 42"
            assert cit.provider_metadata["type"] == "web_search_result_location"

    def test_run_falls_back_stop_reason(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ai_msg = AIMessage(content="ok")
            ca.return_value = MagicMock(invoke=lambda *_a, **_k: {"messages": [ai_msg]})
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            out = e.run(messages=[{"role": "user", "content": "hi"}])
            assert out.stop_reason == "end_turn"

    def test_run_uses_finish_reason_when_present(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ai_msg = AIMessage(
                content="ok",
                response_metadata={"finish_reason": "stop"},
            )
            ca.return_value = MagicMock(invoke=lambda *_a, **_k: {"messages": [ai_msg]})
            e = _Engine(compresr_client=fake_client, llm="openai:gpt-4.1")
            out = e.run(messages=[{"role": "user", "content": "hi"}])
            assert out.stop_reason == "stop"

    def test_run_uses_call_site_model_override(self, fake_client):
        """A ``model=`` kwarg on run() always wins over the constructor default."""
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-haiku-4-5",
            )
            e.run(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-foo",
            )
        # Even with a default of claude-haiku-4-5, the call-site override wins.
        assert ic.call_args.args[0] == "anthropic:claude-foo"

    def test_run_raises_without_any_model(self, fake_client):
        """Provider-only client + no call-site model -> clear CompresrError."""
        from compresr.exceptions import CompresrError

        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent"),
        ):
            e = _Engine(compresr_client=fake_client, llm="anthropic")
            with pytest.raises(CompresrError, match="model is required"):
                e.run(messages=[{"role": "user", "content": "hi"}])

    def test_chat_model_is_cached(self, fake_client):
        """Calling run() twice with the same model triggers only one init_chat_model call."""
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-haiku-4-5",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
            e.run(messages=[{"role": "user", "content": "hi again"}])
        assert ic.call_count == 1

    def test_default_compresr_stats_are_zero(self, fake_client):
        """Wave 2A leaves stats at defaults until middleware plumbing lands."""
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            out = e.run(messages=[{"role": "user", "content": "hi"}])
            assert out.compresr_stats.tokens_saved == 0
            assert out.compresr_stats.by_tool == {}


# ---------------------------------------------------------------------------
# run() — per-call LLM kwargs flow through chat.bind(...)
# ---------------------------------------------------------------------------


class TestRunBindsLLMKwargs:
    """Verify that LLM-level kwargs reach the chat model via ``.bind(...)``.

    The cached chat model must NOT be mutated — every call gets a fresh
    ``RunnableBinding`` so concurrent / interleaved calls with different
    knobs don't pollute one another.
    """

    def _engine_with_mock_chat(self, fake_client, *, llm="anthropic:claude-haiku-4-5"):
        """Build an _Engine with a MagicMock chat that supports .bind()."""
        chat = MagicMock(name="chat")
        bound_chat = MagicMock(name="bound_chat")
        chat.bind = MagicMock(return_value=bound_chat)
        return chat, bound_chat, llm

    # NOTE: LangChain's `bind_tools(...)` strips a prior `chat.bind(...)`'s
    # kwargs. To make per-call knobs (max_tokens, temperature, …) survive
    # through create_agent, the engine now bakes them into the chat-model
    # CONSTRUCTOR via `init_chat_model(...)` and caches per (model, kwargs).
    # These tests verify the kwargs reach init_chat_model, not `.bind`.

    def test_run_passes_temperature_and_top_p_to_init(self, fake_client):
        chat, _bound, llm = self._engine_with_mock_chat(fake_client)
        with (
            patch("compresr.agents.engine.init_chat_model", return_value=chat) as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(compresr_client=fake_client, llm=llm)
            e.run(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.5,
                top_p=0.9,
            )
        init_kwargs = ic.call_args.kwargs
        assert init_kwargs["temperature"] == 0.5
        assert init_kwargs["top_p"] == 0.9
        # The freshly-built chat (not a `.bind()` wrapper) reaches create_agent.
        assert ca.call_args.kwargs["model"] is chat
        chat.bind.assert_not_called()

    def test_run_passes_max_tokens_to_init(self, fake_client):
        chat, _bound, llm = self._engine_with_mock_chat(fake_client)
        with (
            patch("compresr.agents.engine.init_chat_model", return_value=chat) as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(compresr_client=fake_client, llm=llm)
            e.run(messages=[{"role": "user", "content": "hi"}], max_tokens=42)
        assert ic.call_args.kwargs["max_tokens"] == 42

    def test_gemini_max_tokens_aliases_to_max_output_tokens(self, fake_client):
        chat, _bound, _ = self._engine_with_mock_chat(fake_client)
        with (
            patch("compresr.agents.engine.init_chat_model", return_value=chat) as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="google_genai:gemini-2.5-flash",
            )
            e.run(messages=[{"role": "user", "content": "hi"}], max_tokens=42)
        init_kwargs = ic.call_args.kwargs
        # Gemini's chat-model constructor expects max_output_tokens.
        assert init_kwargs.get("max_output_tokens") == 42
        assert "max_tokens" not in init_kwargs

    def test_unknown_kwarg_does_not_reach_init(self, fake_client):
        """Only ``_LLM_BIND_KWARGS`` flow to ``init_chat_model`` — unknown stay out."""
        chat, _bound, llm = self._engine_with_mock_chat(fake_client)
        with (
            patch("compresr.agents.engine.init_chat_model", return_value=chat) as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(compresr_client=fake_client, llm=llm)
            e.run(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.1,
                widget=True,
            )
        init_kwargs = ic.call_args.kwargs
        assert "temperature" in init_kwargs
        assert "widget" not in init_kwargs

    def test_cache_keyed_by_kwargs(self, fake_client):
        """Same kwargs hit cache; different kwargs build fresh chat models."""
        chat, _bound, llm = self._engine_with_mock_chat(fake_client)
        with (
            patch("compresr.agents.engine.init_chat_model", return_value=chat) as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(compresr_client=fake_client, llm=llm)
            e.run(messages=[{"role": "user", "content": "hi"}], temperature=0.5)
            e.run(messages=[{"role": "user", "content": "hi"}], temperature=0.5)  # cache hit
            e.run(messages=[{"role": "user", "content": "hi"}])  # different kwargs → miss
        # 2 init_chat_model invocations: one for {temperature: 0.5}, one for {} default.
        assert ic.call_count == 2
        # And we never use post-hoc .bind(...) — that would re-introduce the
        # bind_tools-strips-kwargs bug.
        chat.bind.assert_not_called()

    def test_run_uses_unbound_chat_when_max_tokens_none(self, fake_client):
        """``max_tokens=None`` with no other knobs → no extras reach init."""
        chat, _bound, llm = self._engine_with_mock_chat(fake_client)
        with (
            patch("compresr.agents.engine.init_chat_model", return_value=chat) as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(compresr_client=fake_client, llm=llm)
            e.run(messages=[{"role": "user", "content": "hi"}], max_tokens=None)
        init_kwargs = ic.call_args.kwargs
        for kw in ("max_tokens", "temperature", "top_p"):
            assert kw not in init_kwargs
        assert ca.call_args.kwargs["model"] is chat
        chat.bind.assert_not_called()


# ---------------------------------------------------------------------------
# Policy → middleware kwargs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Usage aggregation — every AIMessage in the conversation contributes
# ---------------------------------------------------------------------------


class TestUsageAggregation:
    """The engine sums ``usage_metadata`` across every AIMessage in the
    agent-loop conversation. A multi-turn loop (tool use, ReAct, …) produces
    one AIMessage per turn; reading only the final one under-counts the run
    by N-1 LLM round-trips.
    """

    def _run_with_messages(self, fake_client, messages):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(invoke=lambda *_a, **_k: {"messages": messages})
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")
            return e.run(messages=[{"role": "user", "content": "hi"}])

    @staticmethod
    def _um(input_tokens: int, output_tokens: int, **extra) -> dict:
        """Build a minimal valid ``usage_metadata`` dict.

        LangChain's Pydantic model rejects ``usage_metadata`` without
        ``total_tokens`` (when other fields are present), so all helpers
        compute it from input + output.
        """
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            **extra,
        }

    def test_sums_input_and_output_tokens_across_three_ai_messages(self, fake_client):
        msgs = [
            AIMessage(content="step 1", usage_metadata=self._um(100, 10)),
            AIMessage(content="step 2", usage_metadata=self._um(200, 20)),
            AIMessage(content="final", usage_metadata=self._um(300, 30)),
        ]
        out = self._run_with_messages(fake_client, msgs)
        assert out.usage["input_tokens"] == 600
        assert out.usage["output_tokens"] == 60
        assert out.usage["ai_message_count"] == 3

    def test_aggregates_total_tokens_when_present(self, fake_client):
        msgs = [
            AIMessage(content="a", usage_metadata=self._um(10, 2)),
            AIMessage(content="b", usage_metadata=self._um(30, 4)),
        ]
        out = self._run_with_messages(fake_client, msgs)
        assert out.usage["total_tokens"] == 46

    def test_flattens_input_token_details_cache_fields(self, fake_client):
        """LangChain reports cache numbers nested under ``input_token_details`` —
        the aggregator promotes them onto top-level ``cache_read_input_tokens`` /
        ``cache_creation_input_tokens`` for downstream consumers."""
        msgs = [
            AIMessage(
                content="a",
                usage_metadata=self._um(
                    100,
                    10,
                    input_token_details={"cache_read": 50, "cache_creation": 20},
                ),
            ),
            AIMessage(
                content="b",
                usage_metadata=self._um(
                    200,
                    20,
                    input_token_details={"cache_read": 70, "cache_creation": 30},
                ),
            ),
        ]
        out = self._run_with_messages(fake_client, msgs)
        assert out.usage["cache_read_input_tokens"] == 120
        assert out.usage["cache_creation_input_tokens"] == 50

    def test_preserves_top_level_cache_tokens(self, fake_client):
        """Newer LangChain versions surface cache_* at the top level — those
        are read directly (no add-on from nested)."""
        msgs = [
            AIMessage(
                content="a",
                usage_metadata=self._um(
                    10,
                    2,
                    cache_read_input_tokens=5,
                    cache_creation_input_tokens=3,
                ),
            ),
        ]
        out = self._run_with_messages(fake_client, msgs)
        assert out.usage["cache_read_input_tokens"] == 5
        assert out.usage["cache_creation_input_tokens"] == 3

    def test_top_level_cache_wins_over_nested(self, fake_client):
        """If an AIMessage somehow carries BOTH the new (top-level) and the
        old (nested ``input_token_details``) cache fields, the top-level
        numbers win — adding both would double-count what is conceptually the
        same cache hit."""
        msgs = [
            AIMessage(
                content="a",
                usage_metadata=self._um(
                    100,
                    5,
                    cache_read_input_tokens=10,
                    cache_creation_input_tokens=4,
                    input_token_details={"cache_read": 15, "cache_creation": 6},
                ),
            ),
        ]
        out = self._run_with_messages(fake_client, msgs)
        # Top-level numbers — nested values are ignored when top-level is set.
        assert out.usage["cache_read_input_tokens"] == 10
        assert out.usage["cache_creation_input_tokens"] == 4

    def test_skips_non_ai_messages(self, fake_client):
        """Tool / human messages don't carry billable usage — they shouldn't
        contribute to the totals and shouldn't bump ``ai_message_count``."""
        from langchain_core.messages import HumanMessage, ToolMessage

        msgs = [
            HumanMessage(content="ask"),
            AIMessage(content="step", usage_metadata=self._um(100, 5)),
            ToolMessage(content="search-result", tool_call_id="t1"),
            AIMessage(content="final", usage_metadata=self._um(200, 7)),
        ]
        out = self._run_with_messages(fake_client, msgs)
        assert out.usage["input_tokens"] == 300
        assert out.usage["output_tokens"] == 12
        assert out.usage["ai_message_count"] == 2

    def test_ai_message_without_usage_metadata_does_not_count(self, fake_client):
        """An AIMessage with no usage block (mock / corner case) shouldn't
        crash and shouldn't bump ai_message_count."""
        msgs = [
            AIMessage(content="usage-less"),
            AIMessage(content="real", usage_metadata=self._um(50, 1)),
        ]
        out = self._run_with_messages(fake_client, msgs)
        assert out.usage["input_tokens"] == 50
        assert out.usage["output_tokens"] == 1
        assert out.usage["ai_message_count"] == 1

    def test_returns_zeros_when_no_usage_metadata_anywhere(self, fake_client):
        msgs = [AIMessage(content="hi")]
        out = self._run_with_messages(fake_client, msgs)
        assert out.usage["input_tokens"] == 0
        assert out.usage["output_tokens"] == 0
        assert out.usage["ai_message_count"] == 0

    def test_carries_provider_specific_extras(self, fake_client):
        """Provider-specific numeric extras (e.g. reasoning_tokens) survive."""
        msgs = [
            AIMessage(content="a", usage_metadata=self._um(10, 1, reasoning_tokens=5)),
            AIMessage(content="b", usage_metadata=self._um(20, 2, reasoning_tokens=7)),
        ]
        out = self._run_with_messages(fake_client, msgs)
        assert out.usage["reasoning_tokens"] == 12

    def test_messages_field_propagated_to_normalized_result(self, fake_client):
        """The full conversation chain is exposed on ``NormalizedResult.messages``
        so downstream callers (e.g. benchmark trajectories) can walk it."""
        msgs = [
            AIMessage(content="step 1"),
            AIMessage(content="step 2"),
            AIMessage(content="final"),
        ]
        out = self._run_with_messages(fake_client, msgs)
        assert len(out.messages) == 3
        # raw still points at the LAST AIMessage for back-compat.
        assert out.raw is msgs[-1]

    def test_messages_field_present_when_empty(self, fake_client):
        """Even when the agent returns an empty conversation, messages is the
        empty list (never None) so callers can iterate without a guard."""
        out = self._run_with_messages(fake_client, [])
        assert out.messages == []
        assert out.raw is None


# ---------------------------------------------------------------------------
# Policy → middleware kwargs
# ---------------------------------------------------------------------------


class TestPolicyToolKwargs:
    def test_tool_kwargs_includes_required_fields(self):
        p = CompressionPolicy()
        kw = p.tool_kwargs()
        assert kw["target_compression_ratio"] == p.target_compression_ratio
        assert kw["compression_model"] == p.compression_model_name
        assert kw["min_tokens"] == p.min_tokens
        assert kw["on_error"] == p.on_error
        # Optional fields stay absent when not set so the middleware default
        # is preserved.
        assert "coarse" not in kw
        assert "allow_tools" not in kw
        assert "ignore_tools" not in kw

    def test_tool_kwargs_emits_optional_fields_when_set(self):
        p = CompressionPolicy(
            coarse=True,
            allow_tools=["a"],
            ignore_tools=["b"],
        )
        kw = p.tool_kwargs()
        assert kw["coarse"] is True
        assert kw["allow_tools"] == ["a"]
        assert kw["ignore_tools"] == ["b"]


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------


class _FakeAnthropicCacheMW:
    """Stand-in for langchain_anthropic.middleware.AnthropicPromptCachingMiddleware."""

    def __init__(self, *, ttl, min_messages_to_cache, unsupported_model_behavior):
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior


def _patch_cache_middleware():
    import sys
    import types

    fake_module = types.ModuleType("langchain_anthropic.middleware")
    fake_module.AnthropicPromptCachingMiddleware = _FakeAnthropicCacheMW
    parent = sys.modules.setdefault("langchain_anthropic", types.ModuleType("langchain_anthropic"))
    parent.middleware = fake_module  # type: ignore[attr-defined]
    return patch.dict(sys.modules, {"langchain_anthropic.middleware": fake_module})


class TestPromptCaching:
    def test_anthropic_cache_middleware_appended_after_compresr(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
            _patch_cache_middleware(),
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-sonnet-4-6",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        mw_list = ca.call_args.kwargs["middleware"]
        names = [type(m).__name__ for m in mw_list]
        assert "CompresrToolMiddleware" in names
        assert "_FakeAnthropicCacheMW" in names
        assert names.index("_FakeAnthropicCacheMW") > names.index("CompresrToolMiddleware")
        cache_mw = next(m for m in mw_list if type(m).__name__ == "_FakeAnthropicCacheMW")
        assert cache_mw.ttl == "5m"
        assert cache_mw.min_messages_to_cache == 2
        assert cache_mw.unsupported_model_behavior == "ignore"

    def test_prompt_cache_can_be_disabled(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
            _patch_cache_middleware(),
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-sonnet-4-6",
                enable_prompt_cache=False,
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        mw_list = ca.call_args.kwargs["middleware"]
        names = [type(m).__name__ for m in mw_list]
        assert "_FakeAnthropicCacheMW" not in names

    def test_prompt_cache_ttl_pass_through(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
            _patch_cache_middleware(),
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-sonnet-4-6",
                prompt_cache_ttl="1h",
                prompt_cache_min_messages=5,
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        cache_mw = next(
            m
            for m in ca.call_args.kwargs["middleware"]
            if type(m).__name__ == "_FakeAnthropicCacheMW"
        )
        assert cache_mw.ttl == "1h"
        assert cache_mw.min_messages_to_cache == 5

    def test_openai_does_not_get_anthropic_middleware(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
            _patch_cache_middleware(),
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="openai:gpt-4.1",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        names = [type(m).__name__ for m in ca.call_args.kwargs["middleware"]]
        assert "_FakeAnthropicCacheMW" not in names

    def test_degrades_silently_when_middleware_import_fails(self, fake_client):
        import sys

        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
            patch.dict(sys.modules, {"langchain_anthropic.middleware": None}),
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-sonnet-4-6",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        names = [type(m).__name__ for m in ca.call_args.kwargs["middleware"]]
        assert "CompresrToolMiddleware" in names
        # No crash, no cache mw — just Compresr.
        assert len(names) == 1

    def test_openai_prompt_cache_key_passes_through_model_kwargs(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="openai:gpt-4.1",
                llm_api_key="sk-x",
                openai_prompt_cache_key="tenant-42",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = ic.call_args.kwargs
        mk = call_kwargs.get("model_kwargs") or {}
        assert mk.get("prompt_cache_key") == "tenant-42"
        # default ttl="5m" -> no retention override
        assert "prompt_cache_retention" not in mk

    def test_openai_1h_ttl_maps_to_24h_retention(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="openai:gpt-4.1",
                llm_api_key="sk-x",
                prompt_cache_ttl="1h",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        mk = ic.call_args.kwargs.get("model_kwargs") or {}
        assert mk.get("prompt_cache_retention") == "24h"

    def test_openai_cache_kwargs_skipped_when_caching_disabled(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="openai:gpt-4.1",
                llm_api_key="sk-x",
                enable_prompt_cache=False,
                openai_prompt_cache_key="tenant-42",
                prompt_cache_ttl="1h",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        mk = ic.call_args.kwargs.get("model_kwargs") or {}
        assert "prompt_cache_key" not in mk
        assert "prompt_cache_retention" not in mk

    def test_anthropic_provider_ignores_openai_cache_key(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model") as ic,
            patch("compresr.agents.engine.create_agent") as ca,
            _patch_cache_middleware(),
        ):
            ic.return_value = MagicMock()
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-sonnet-4-6",
                llm_api_key="sk-ant",
                openai_prompt_cache_key="should-be-ignored",
            )
            e.run(messages=[{"role": "user", "content": "hi"}])
        mk = ic.call_args.kwargs.get("model_kwargs") or {}
        assert "prompt_cache_key" not in mk

    def test_tools_are_sorted_for_cache_stability(self, fake_client):
        with (
            patch("compresr.agents.engine.init_chat_model"),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="ok")]}
            )
            e = _Engine(
                compresr_client=fake_client,
                llm="anthropic:claude-sonnet-4-6",
                enable_prompt_cache=False,
            )
            t_zebra = MagicMock(spec=["name"])
            t_zebra.name = "zebra"
            t_alpha = MagicMock(spec=["name"])
            t_alpha.name = "alpha"
            t_middle = MagicMock(spec=["name"])
            t_middle.name = "middle"
            e.run(
                messages=[{"role": "user", "content": "hi"}],
                tools=[t_zebra, t_alpha, t_middle],
            )
        tools_passed = ca.call_args.kwargs["tools"]
        assert [getattr(t, "name") for t in tools_passed] == ["alpha", "middle", "zebra"]


class TestCacheTokenAggregation:
    """Cache-token aggregation across provider surfaces."""

    def test_aggregates_openai_cached_tokens_from_details(self):
        from compresr.agents.engine import _aggregate_usage

        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
                "input_token_details": {"cached_tokens": 800},
            },
        )
        out = _aggregate_usage([msg])
        assert out["cache_read_input_tokens"] == 800

    def test_aggregates_openai_priority_and_flex_cache_read(self):
        from compresr.agents.engine import _aggregate_usage

        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
                "input_token_details": {
                    "priority_cache_read": 300,
                    "flex_cache_read": 200,
                },
            },
        )
        out = _aggregate_usage([msg])
        assert out["cache_read_input_tokens"] == 500

    def test_top_level_anthropic_cache_keys_take_precedence_over_details(self):
        from compresr.agents.engine import _aggregate_usage

        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "cache_read_input_tokens": 5000,
                "cache_creation_input_tokens": 100,
                "input_token_details": {"cached_tokens": 9999},
            },
        )
        out = _aggregate_usage([msg])
        assert out["cache_read_input_tokens"] == 5000
        assert out["cache_creation_input_tokens"] == 100

    def test_langchain_anthropic_subtracts_cache_from_input_tokens(self):
        """``langchain-anthropic`` 1.x inflates ``input_tokens`` to fresh + read
        + create. Aggregate must return fresh-only so accounting can apply the
        cache multipliers separately without double-counting.
        """
        from compresr.agents.engine import _aggregate_usage

        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 197_694,
                "output_tokens": 2_128,
                "total_tokens": 199_822,
                "input_token_details": {
                    "cache_read": 159_233,
                    "cache_creation": 0,
                    "ephemeral_5m_input_tokens": 30_000,
                    "ephemeral_1h_input_tokens": 0,
                },
            },
        )
        out = _aggregate_usage([msg])
        # Fresh = 197_694 - 159_233 - 30_000 = 8_461
        assert out["input_tokens"] == 8_461
        assert out["cache_read_input_tokens"] == 159_233
        assert out["cache_creation_input_tokens"] == 30_000

    def test_aggregates_gemini_cached_content_token_count(self):
        """Gemini surfaces cache reads under ``cached_content_token_count``.

        See langchain-google #989 — when present in ``input_token_details``
        it should fold into ``cache_read_input_tokens``.
        """
        from compresr.agents.engine import _aggregate_usage

        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 1500,
                "output_tokens": 80,
                "total_tokens": 1580,
                "input_token_details": {"cached_content_token_count": 1200},
            },
        )
        out = _aggregate_usage([msg])
        assert out["cache_read_input_tokens"] == 1200

    def test_anthropic_5m_and_1h_ephemeral_writes_combine(self):
        """Both TTL buckets should sum into cache_creation_input_tokens."""
        from compresr.agents.engine import _aggregate_usage

        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 12_000,  # fresh + 5m + 1h
                "output_tokens": 100,
                "total_tokens": 12_100,
                "input_token_details": {
                    "cache_read": 0,
                    "cache_creation": 0,
                    "ephemeral_5m_input_tokens": 2_000,
                    "ephemeral_1h_input_tokens": 1_500,
                },
            },
        )
        out = _aggregate_usage([msg])
        assert out["cache_creation_input_tokens"] == 3_500
        # Fresh = 12000 - 0 - 3500 = 8500
        assert out["input_tokens"] == 8_500


# ---------------------------------------------------------------------------
# Solo-WebSearchTool auto-route into the research loop
# ---------------------------------------------------------------------------


class TestIsSoloWebSearch:
    """``_is_solo_web_search`` matches exactly one tool whose ``.name``
    is one of the Compresr-built web-search tool names."""

    def test_detects_tavily(self):
        from compresr.agents.engine import _is_solo_web_search

        tool = MagicMock(name="tavily_search")
        tool.name = "tavily_search"
        assert _is_solo_web_search([tool]) is True

    def test_detects_brave(self):
        from compresr.agents.engine import _is_solo_web_search

        tool = MagicMock()
        tool.name = "brave_search"
        assert _is_solo_web_search([tool]) is True

    def test_rejects_empty(self):
        from compresr.agents.engine import _is_solo_web_search

        assert _is_solo_web_search([]) is False
        assert _is_solo_web_search(()) is False

    def test_rejects_multiple_tools(self):
        from compresr.agents.engine import _is_solo_web_search

        a = MagicMock()
        a.name = "tavily_search"
        b = MagicMock()
        b.name = "calc"
        assert _is_solo_web_search([a, b]) is False

    def test_rejects_unknown_tool_name(self):
        from compresr.agents.engine import _is_solo_web_search

        tool = MagicMock()
        tool.name = "calc"
        assert _is_solo_web_search([tool]) is False

    def test_rejects_tool_with_no_name(self):
        from compresr.agents.engine import _is_solo_web_search

        # Anything without a string ``.name`` shouldn't trigger the auto-route.
        tool = object()
        assert _is_solo_web_search([tool]) is False


class TestExtractLastUserText:
    def test_dict_form(self):
        from compresr.agents.engine import _extract_last_user_text

        msgs = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert _extract_last_user_text(msgs) == "second"

    def test_handles_list_content_blocks(self):
        from compresr.agents.engine import _extract_last_user_text

        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
            }
        ]
        assert _extract_last_user_text(msgs) == "hello world"

    def test_empty(self):
        from compresr.agents.engine import _extract_last_user_text

        assert _extract_last_user_text([]) == ""


# ----- Routing tests -----------------------------------------------------------------
# Patterns mirror ``test_agents_research.py``: a ``_FakeChat`` whose ``bind_tools``
# pops scripted ``AIMessage`` responses lets us verify the loop's behavior without
# any live LangChain or provider call.


class _FakeRouteBound:
    def __init__(self, parent: "_FakeRouteChat", tool_choice: str) -> None:
        self._parent = parent
        self._parent.bind_history.append({"tool_choice": tool_choice})

    def invoke(self, messages: list):
        self._parent.invoke_history.append(list(messages))
        if not self._parent.scripted:
            raise AssertionError("FakeChat ran out of scripted responses")
        return self._parent.scripted.pop(0)


class _FakeRouteChat:
    def __init__(self, scripted: list) -> None:
        self.scripted = list(scripted)
        self.bind_history: list[dict] = []
        self.invoke_history: list[list] = []

    def bind_tools(self, tools: list, tool_choice: str = "auto") -> _FakeRouteBound:
        return _FakeRouteBound(self, tool_choice)


def _route_tool() -> MagicMock:
    """A bare ``tavily_search``-named tool that records its invoke calls.

    The fake snippet is >400 chars so it clears the default
    ``min_compress_tokens=100`` threshold and actually triggers compression.
    """
    t = MagicMock()
    t.name = "tavily_search"
    t.invoke.return_value = (
        "Compresr is a YC W26-batch startup building intelligent LLM context "
        "compression. Source: https://example.com/compresr-yc-w26\n\n"
        + ("paragraph filler content. " * 30)
    )
    return t


def _route_engine(fake_client, chat: _FakeRouteChat):
    """Construct an ``_Engine`` whose chat-model cache is pre-seeded with ``chat``
    so neither ``init_chat_model`` nor ``create_agent`` is ever called."""
    e = _Engine.__new__(_Engine)
    e._compresr_client = fake_client
    e._provider = "anthropic"
    e._default_model_name = "claude-sonnet-4-6"
    from compresr.integrations._shared import CompressionPolicy

    e._policy = CompressionPolicy()
    e._llm_api_key = "sk-ant-test"
    e._enable_prompt_cache = False
    e._prompt_cache_ttl = "5m"
    e._prompt_cache_min_messages = 2
    e._openai_prompt_cache_key = None
    e._chat_models = {}
    # Both the no-extra-kwargs path and a likely max_tokens binding get the same fake.
    e._chat_models[("claude-sonnet-4-6", ())] = chat
    e._chat_models[("claude-sonnet-4-6", (("max_tokens", 4096),))] = chat
    return e


class TestSoloWebSearchAutoRoute:
    """End-to-end: ``_Engine.run`` with ``tools=[<tavily_search>]`` skips
    ``create_agent`` and drives the research loop instead."""

    def _scripted_research_messages(self):
        tool_call = {"id": "t0", "name": "tavily_search", "args": {"query": "yc 2026"}}
        return [
            AIMessage(content="searching…", tool_calls=[tool_call]),
            AIMessage(
                content=(
                    "Explanation: looked it up.\n"
                    "Exact Answer: YC W26\n"
                    "Confidence: 90\n"
                    "Citations: https://ycombinator.com"
                ),
            ),
        ]

    def test_routes_through_research_when_solo_web_search(self, fake_client):
        """``create_agent`` must never be called when the auto-route fires."""
        chat = _FakeRouteChat(self._scripted_research_messages())
        engine = _route_engine(fake_client, chat)

        with patch("compresr.agents.engine.create_agent") as ca:
            out = engine.run(
                messages=[{"role": "user", "content": "what batch is compresr in?"}],
                tools=[_route_tool()],
            )

        ca.assert_not_called()
        assert "YC W26" in out.text

    def test_forces_tool_choice_none_on_final_step(self, fake_client):
        """The hallmark research-loop behavior — final step must be ``tool_choice=none``.

        Script 10 tool-calling responses so the loop actually exhausts its
        default ``max_steps=10`` budget and reaches the forced-commit step.
        """
        tool_call = {"id": "t0", "name": "tavily_search", "args": {"query": "q"}}
        scripted = [AIMessage(content="searching", tool_calls=[tool_call]) for _ in range(10)]
        chat = _FakeRouteChat(scripted)
        engine = _route_engine(fake_client, chat)

        engine.run(
            messages=[{"role": "user", "content": "hi"}],
            tools=[_route_tool()],
        )

        choices = [b["tool_choice"] for b in chat.bind_history]
        assert len(choices) == 10
        assert choices[-1] == "none"
        assert all(c == "auto" for c in choices[:-1])

    def test_compresses_each_snippet_with_live_query(self, fake_client):
        chat = _FakeRouteChat(self._scripted_research_messages())
        engine = _route_engine(fake_client, chat)

        engine.run(
            messages=[{"role": "user", "content": "what batch is compresr in?"}],
            tools=[_route_tool()],
        )

        # Exactly one tool call → exactly one compression call.
        assert len(fake_client.calls) == 1
        # Live query is the tool call's ``query`` arg, not the user's message.
        assert fake_client.calls[0]["query"] == "yc 2026"
        assert fake_client.calls[0]["compression_model_name"] == "latte_v1"

    def test_uses_research_system_prompt_when_caller_omits_system(self, fake_client):
        from compresr.agents.research.prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

        chat = _FakeRouteChat(self._scripted_research_messages())
        engine = _route_engine(fake_client, chat)

        engine.run(
            messages=[{"role": "user", "content": "q?"}],
            tools=[_route_tool()],
        )

        # The first invoke call's first message is the SystemMessage from the loop.
        first_invoke = chat.invoke_history[0]
        assert first_invoke and first_invoke[0].content == DEFAULT_RESEARCH_SYSTEM_PROMPT

    def test_caller_system_overrides_research_default(self, fake_client):
        chat = _FakeRouteChat(self._scripted_research_messages())
        engine = _route_engine(fake_client, chat)

        engine.run(
            messages=[{"role": "user", "content": "q?"}],
            tools=[_route_tool()],
            system="You are a finance analyst.",
        )

        first_invoke = chat.invoke_history[0]
        assert first_invoke[0].content == "You are a finance analyst."

    def test_uses_last_user_message_as_question(self, fake_client):
        chat = _FakeRouteChat(self._scripted_research_messages())
        engine = _route_engine(fake_client, chat)

        engine.run(
            messages=[
                {"role": "user", "content": "first ask"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "the real ask"},
            ],
            tools=[_route_tool()],
        )

        # HumanMessage is index 1 (after the SystemMessage).
        first_invoke = chat.invoke_history[0]
        assert first_invoke[1].content == "the real ask"

    def test_falls_back_to_normal_path_when_multiple_tools(self, fake_client):
        """Multi-tool calls keep the existing ``create_agent`` path."""
        chat = MagicMock()
        with (
            patch("compresr.agents.engine.init_chat_model", return_value=chat),
            patch("compresr.agents.engine.create_agent") as ca,
        ):
            ca.return_value = MagicMock(
                invoke=lambda *_a, **_k: {"messages": [AIMessage(content="standard path")]}
            )
            e = _Engine(compresr_client=fake_client, llm="anthropic:claude-opus-4-8")

            a = MagicMock()
            a.name = "tavily_search"
            b = MagicMock()
            b.name = "calculator"
            out = e.run(messages=[{"role": "user", "content": "hi"}], tools=[a, b])

        ca.assert_called_once()
        assert out.text == "standard path"
