"""Unit tests for the research agent (port of Perplexity ``search_evals``).

The agent is exercised with patched ``init_chat_model`` / ``create_agent``
so no live API calls happen. We don't go through the LangGraph agent loop —
this agent runs its own ReAct loop — so we mock the chat model's
``bind_tools(...).invoke(messages)`` chain.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool as lc_tool  # noqa: E402

from compresr.agents.engine import _Engine  # noqa: E402
from compresr.agents.research import (  # noqa: E402
    Citation,
    ResearchAgent,
    ResearchResult,
    parse_research_output,
)
from compresr.agents.research.agent import _estimate_tokens  # noqa: E402

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseResearchOutput:
    def test_parses_all_four_fields(self):
        text = (
            "Explanation: I searched a lot.\n"
            "Exact Answer: SAS 9.1\n"
            "Confidence: 62%\n"
            "Citations: https://example.com/a, https://example.com/b"
        )
        out = parse_research_output(text)
        assert out.answer == "SAS 9.1"
        assert out.explanation == "I searched a lot."
        assert out.confidence == 0.62
        assert out.citation_urls == ["https://example.com/a", "https://example.com/b"]

    def test_confidence_percent_no_sign(self):
        assert parse_research_output("Confidence: 80").confidence == 0.80

    def test_confidence_fraction(self):
        assert parse_research_output("Confidence: 0.42").confidence == 0.42

    def test_missing_fields_default_safely(self):
        out = parse_research_output("just a free-form answer")
        assert out.answer == ""
        assert out.explanation == ""
        assert out.confidence is None
        assert out.citation_urls == []

    def test_dedupes_citation_urls(self):
        text = "Citations: https://a.com, https://a.com, https://b.com"
        out = parse_research_output("Confidence: 50\n" + text)
        assert out.citation_urls == ["https://a.com", "https://b.com"]

    def test_multiline_explanation(self):
        text = "Explanation: line 1\nline 2\nline 3\n" "Exact Answer: yes\n" "Confidence: 90"
        out = parse_research_output(text)
        assert "line 1" in out.explanation and "line 3" in out.explanation
        assert out.answer == "yes"

    def test_empty_string(self):
        out = parse_research_output("")
        assert out.answer == "" and out.confidence is None


# ---------------------------------------------------------------------------
# ResearchAgent — loop control
# ---------------------------------------------------------------------------


def _ai(content: str = "", tool_calls: list[dict] | None = None) -> AIMessage:
    """Build an AIMessage that mimics LangChain's shape."""
    if tool_calls:
        return AIMessage(content=content, tool_calls=tool_calls)
    return AIMessage(content=content)


class _FakeBound:
    """Stand-in for ``chat.bind_tools(...)``. Records ``tool_choice`` and pops
    the next scripted response."""

    def __init__(self, parent: "_FakeChat", tool_choice: str) -> None:
        self._parent = parent
        self._parent.bind_history.append({"tool_choice": tool_choice})

    def invoke(self, messages: list) -> AIMessage:
        self._parent.invoke_history.append(list(messages))
        if not self._parent.scripted:
            raise AssertionError("FakeChat ran out of scripted responses")
        return self._parent.scripted.pop(0)


class _FakeChat:
    def __init__(self, scripted: list[AIMessage]) -> None:
        self.scripted = list(scripted)
        self.bind_history: list[dict] = []
        self.invoke_history: list[list] = []

    def bind_tools(self, tools: list, tool_choice: str = "auto") -> _FakeBound:
        return _FakeBound(self, tool_choice)


def _make_engine_with_chat(chat: _FakeChat, fake_client: Any) -> _Engine:
    e = _Engine.__new__(_Engine)
    e._compresr_client = fake_client
    e._provider = "anthropic"
    e._default_model_name = "claude-sonnet-4-6"
    from compresr.integrations._shared import CompressionPolicy

    e._policy = CompressionPolicy()
    e._llm_api_key = "sk-ant-test"
    e._enable_prompt_cache = True
    e._prompt_cache_ttl = "5m"
    e._prompt_cache_min_messages = 2
    e._openai_prompt_cache_key = None
    e._chat_models = {}
    # Pre-seed the chat-model cache so the agent never tries to call init_chat_model.
    e._chat_models[("claude-sonnet-4-6", ())] = chat
    return e


@lc_tool
def fake_search(query: str) -> str:
    """Search the web (test fake)."""
    return f"raw-result for {query} https://example.com/{query}"


class TestResearchAgentLoop:
    def test_returns_final_answer_when_no_tool_calls(self, fake_client):
        chat = _FakeChat(
            [
                _ai(
                    "Explanation: trivial\n"
                    "Exact Answer: 42\n"
                    "Confidence: 99\n"
                    "Citations: https://a.com",
                ),
            ]
        )
        engine = _make_engine_with_chat(chat, fake_client)
        agent = ResearchAgent(
            engine=engine, search_tool=fake_search, max_steps=4, compress_snippets=False
        )

        result = agent.run("what is the answer?")
        assert isinstance(result, ResearchResult)
        assert result.answer == "42"
        assert result.confidence == 0.99
        assert result.citations == [Citation(url="https://a.com")]
        # First call must NOT force tool_choice="none" (it's not the last step).
        assert chat.bind_history[0]["tool_choice"] == "auto"

    def test_forces_tool_choice_none_on_last_step(self, fake_client):
        # The agent calls a tool on each step until the last, where the model
        # is forced to commit. We script 3 tool-using AIs then one final text.
        tool_call = {"id": "t0", "name": "fake_search", "args": {"query": "q"}}
        chat = _FakeChat(
            [
                _ai("thinking", tool_calls=[tool_call]),
                _ai("thinking", tool_calls=[tool_call]),
                _ai("thinking", tool_calls=[tool_call]),
                _ai(
                    "Explanation: done\nExact Answer: final\nConfidence: 50",
                ),
            ]
        )
        engine = _make_engine_with_chat(chat, fake_client)
        agent = ResearchAgent(
            engine=engine, search_tool=fake_search, max_steps=4, compress_snippets=False
        )

        agent.run("question?")
        tool_choices = [b["tool_choice"] for b in chat.bind_history]
        assert tool_choices == ["auto", "auto", "auto", "none"]

    def test_calls_compress_per_tool_result(self, fake_client):
        tool_call = {"id": "t0", "name": "fake_search", "args": {"query": "longq"}}
        chat = _FakeChat(
            [
                _ai("call tool", tool_calls=[tool_call]),
                _ai("Explanation: e\nExact Answer: a\nConfidence: 10"),
            ]
        )
        engine = _make_engine_with_chat(chat, fake_client)
        # min_compress_tokens=1 to ensure our tiny fake snippet qualifies
        agent = ResearchAgent(
            engine=engine,
            search_tool=fake_search,
            max_steps=2,
            compress_snippets=True,
            min_compress_tokens=1,
        )
        agent.run("hi")
        # FakeCompressionClient records every call.compress(); we expect exactly one.
        assert len(fake_client.calls) == 1
        assert fake_client.calls[0]["query"] == "longq"

    def test_skips_compress_when_disabled(self, fake_client):
        tool_call = {"id": "t0", "name": "fake_search", "args": {"query": "q"}}
        chat = _FakeChat(
            [
                _ai("", tool_calls=[tool_call]),
                _ai("Explanation: x\nExact Answer: y\nConfidence: 1"),
            ]
        )
        engine = _make_engine_with_chat(chat, fake_client)
        agent = ResearchAgent(
            engine=engine, search_tool=fake_search, max_steps=2, compress_snippets=False
        )
        agent.run("hi")
        assert len(fake_client.calls) == 0

    def test_skips_compress_for_tiny_snippets(self, fake_client):
        tool_call = {"id": "t0", "name": "fake_search", "args": {"query": "q"}}
        chat = _FakeChat(
            [
                _ai("", tool_calls=[tool_call]),
                _ai("Explanation: x\nExact Answer: y\nConfidence: 1"),
            ]
        )
        engine = _make_engine_with_chat(chat, fake_client)
        agent = ResearchAgent(
            engine=engine,
            search_tool=fake_search,
            max_steps=2,
            compress_snippets=True,
            min_compress_tokens=10_000,
        )
        agent.run("hi")
        assert len(fake_client.calls) == 0

    def test_message_order_matches_react_protocol(self, fake_client):
        tool_call = {"id": "t0", "name": "fake_search", "args": {"query": "q"}}
        chat = _FakeChat(
            [
                _ai("calling", tool_calls=[tool_call]),
                _ai("Exact Answer: done\nConfidence: 90"),
            ]
        )
        engine = _make_engine_with_chat(chat, fake_client)
        agent = ResearchAgent(
            engine=engine, search_tool=fake_search, max_steps=2, compress_snippets=False
        )
        agent.run("hi")
        # Final invoke sees: system, human, AI(tool_call), tool_result
        final_messages = chat.invoke_history[-1]
        roles = [type(m).__name__ for m in final_messages]
        assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "ToolMessage"]
        assert "Search the web" in final_messages[0].content  # system prompt

    def test_collects_citations_from_text_and_tool_output(self, fake_client):
        tool_call = {"id": "t0", "name": "fake_search", "args": {"query": "q"}}
        chat = _FakeChat(
            [
                _ai("", tool_calls=[tool_call]),
                _ai(
                    "Explanation: e\n"
                    "Exact Answer: a\n"
                    "Confidence: 1\n"
                    "Citations: https://parsed.example.com",
                ),
            ]
        )
        engine = _make_engine_with_chat(chat, fake_client)
        agent = ResearchAgent(
            engine=engine, search_tool=fake_search, max_steps=2, compress_snippets=False
        )
        result = agent.run("hi")
        urls = [c.url for c in result.citations]
        assert "https://parsed.example.com" in urls
        # fake_search returns a URL too — should appear after the explicit one.
        assert any(u.startswith("https://example.com/") for u in urls)

    def test_truncates_when_context_too_big(self, fake_client):
        big = "x" * (200_000 * 4)  # ~200k tokens via the char/4 estimator

        @lc_tool
        def huge_search(query: str) -> str:
            """Returns a giant blob."""
            return big

        tool_call = {"id": "t0", "name": "huge_search", "args": {"query": "q"}}
        chat = _FakeChat(
            [
                _ai("", tool_calls=[tool_call]),
                _ai("", tool_calls=[tool_call]),
                _ai("Exact Answer: x\nConfidence: 1"),
            ]
        )
        engine = _make_engine_with_chat(chat, fake_client)
        agent = ResearchAgent(
            engine=engine,
            search_tool=huge_search,
            max_steps=3,
            compress_snippets=False,
            max_context_tokens=10_000,
        )
        agent.run("hi")
        # On the second LLM call, the agent should have truncated the oldest pair.
        # We can't easily inspect messages by reference, but the third call
        # must not have ballooned to 400k+ tokens.
        third_call_messages = chat.invoke_history[-1]
        total = sum(_estimate_tokens(str(m.content)) for m in third_call_messages)
        assert total < 100_000


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class TestApplyCacheControl:
    """Manual cache_control stamping (the bypass-path mirror of
    AnthropicPromptCachingMiddleware) for the research agent."""

    def _agent(self, fake_client: Any, **engine_overrides: Any) -> ResearchAgent:
        chat = _FakeChat([_ai("done")])
        engine = _make_engine_with_chat(chat, fake_client)
        for k, v in engine_overrides.items():
            setattr(engine, k, v)
        return ResearchAgent(
            engine=engine, search_tool=fake_search, max_steps=1, compress_snippets=False
        )

    def test_anthropic_stamps_cache_control_on_string_content(self, fake_client):
        agent = self._agent(fake_client)
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        patched = agent._apply_cache_control(msgs)
        last = patched[-1]
        assert isinstance(last.content, list)
        assert last.content[-1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
        assert last.content[-1]["text"] == "hi"

    def test_non_anthropic_provider_noop(self, fake_client):
        agent = self._agent(fake_client, _provider="openai")
        msgs = [HumanMessage(content="hi"), HumanMessage(content="more")]
        patched = agent._apply_cache_control(msgs)
        assert patched == msgs  # same identity
        assert isinstance(patched[-1].content, str)

    def test_disabled_when_enable_prompt_cache_false(self, fake_client):
        agent = self._agent(fake_client, _enable_prompt_cache=False)
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert agent._apply_cache_control(msgs) == msgs

    def test_skipped_below_min_messages(self, fake_client):
        agent = self._agent(fake_client, _prompt_cache_min_messages=5)
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert agent._apply_cache_control(msgs) == msgs

    def test_preserves_other_content_blocks(self, fake_client):
        agent = self._agent(fake_client)
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ]
        )
        patched = agent._apply_cache_control([SystemMessage(content="s"), msg])
        last = patched[-1]
        assert len(last.content) == 2
        assert last.content[0] == {"type": "text", "text": "a"}
        assert last.content[1]["text"] == "b"
        assert last.content[1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


class TestResearchFacade:
    def test_property_raises_when_no_llm(self, fake_client):
        from compresr import CompressionClient
        from compresr.exceptions import CompresrError

        c = CompressionClient(api_key="cmp_test")
        with pytest.raises(CompresrError, match="research requires an LLM provider"):
            _ = c.research

    def test_resolve_search_tool_rejects_unknown(self):
        from compresr.agents.research.facade import ResearchFacade

        r = ResearchFacade(engine=MagicMock())
        with pytest.raises(ValueError, match="unsupported search provider"):
            r._resolve_search_tool("duckduckgo-foo")

    def test_resolve_search_tool_passes_through_basetool(self):
        from compresr.agents.research.facade import ResearchFacade

        r = ResearchFacade(engine=MagicMock())
        out = r._resolve_search_tool(fake_search)
        assert out is fake_search
