"""Unit tests for the customer-facing provider-shape facades (Wave 2B).

Each test stubs ``_Engine`` with a ``MagicMock`` so we can verify the
remapping logic without involving langchain or any real LLM call.
"""

from unittest.mock import MagicMock

import pytest

from compresr.agents.facades.anthropic import _Anthropic
from compresr.agents.facades.native import _Native
from compresr.agents.facades.openai import _OpenAI
from compresr.agents.normalized import Citation, CompresrStats, NormalizedResult


def _result(
    *,
    text: str = "hi",
    tool_uses=(),
    citations=(),
    stop_reason: str = "end_turn",
    usage=None,
    raw=None,
    messages=(),
) -> NormalizedResult:
    return NormalizedResult(
        text=text,
        content_blocks=[],
        tool_uses=list(tool_uses),
        citations=list(citations),
        stop_reason=stop_reason,
        usage=usage or {"input_tokens": 10, "output_tokens": 5},
        compresr_stats=CompresrStats(),
        raw=raw,
        messages=list(messages),
    )


class TestAnthropicFacade:
    def test_create_translates_to_anthropic_message(self):
        engine = MagicMock()
        engine.run.return_value = _result(text="Hello!")
        facade = _Anthropic(engine)
        out = facade.messages.create(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
        assert out.role == "assistant"
        assert out.type == "message"
        assert out.model == "claude-opus-4-8"
        assert out.content[0].type == "text"
        assert out.content[0].text == "Hello!"
        assert out.stop_reason == "end_turn"
        assert out.usage.input_tokens == 10
        assert out.usage.output_tokens == 5
        engine.run.assert_called_once()

    def test_tool_use_becomes_block(self):
        engine = MagicMock()
        engine.run.return_value = _result(
            text="",
            tool_uses=[{"id": "t1", "name": "search", "input": {"q": "x"}}],
            stop_reason="tool_use",
        )
        facade = _Anthropic(engine)
        out = facade.messages.create(model="m", messages=[], max_tokens=10)
        assert any(getattr(b, "type", "") == "tool_use" for b in out.content)

    def test_passes_system_through(self):
        engine = MagicMock()
        engine.run.return_value = _result()
        _Anthropic(engine).messages.create(model="m", messages=[], max_tokens=10, system="be brief")
        assert engine.run.call_args.kwargs["system"] == "be brief"

    def test_text_block_carries_citations(self):
        engine = MagicMock()
        engine.run.return_value = _result(
            text="cited", citations=[Citation(url="https://example.com")]
        )
        out = _Anthropic(engine).messages.create(model="m", messages=[], max_tokens=10)
        text_block = out.content[0]
        assert text_block.citations[0].url == "https://example.com"

    def test_carries_full_message_chain(self):
        """``AnthropicMessage.messages`` exposes the full agent-loop
        conversation so consumers (e.g. benchmark trajectories) can walk it."""
        engine = MagicMock()
        engine.run.return_value = _result(messages=["m1", "m2", "m3"])
        out = _Anthropic(engine).messages.create(model="m", messages=[], max_tokens=10)
        assert list(out.messages) == ["m1", "m2", "m3"]

    def test_usage_reflects_aggregate_token_counts(self):
        """The Anthropic Usage block reads from the engine's already-aggregated
        ``NormalizedResult.usage`` dict, so a multi-turn run surfaces summed
        input/output tokens (not just the final turn)."""
        engine = MagicMock()
        engine.run.return_value = _result(
            usage={
                "input_tokens": 600,
                "output_tokens": 60,
                "cache_read_input_tokens": 120,
                "cache_creation_input_tokens": 50,
                "ai_message_count": 3,
            }
        )
        out = _Anthropic(engine).messages.create(model="m", messages=[], max_tokens=10)
        assert out.usage.input_tokens == 600
        assert out.usage.output_tokens == 60
        assert out.usage.cache_read_input_tokens == 120
        assert out.usage.cache_creation_input_tokens == 50


class TestOpenAIFacade:
    def test_create_translates_to_chat_completion(self):
        engine = MagicMock()
        engine.run.return_value = _result(text="Hello!")
        out = _OpenAI(engine).chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert out.object == "chat.completion"
        assert out.model == "gpt-4o"
        assert out.choices[0].index == 0
        assert out.choices[0].message.role == "assistant"
        assert out.choices[0].message.content == "Hello!"
        assert out.choices[0].finish_reason == "stop"
        assert out.usage.prompt_tokens == 10
        assert out.usage.completion_tokens == 5
        assert out.usage.total_tokens == 15

    def test_tool_uses_become_tool_calls(self):
        engine = MagicMock()
        engine.run.return_value = _result(
            text="",
            tool_uses=[{"id": "t1", "name": "search", "input": {"q": "x"}}],
            stop_reason="tool_use",
        )
        out = _OpenAI(engine).chat.completions.create(model="m", messages=[])
        tc = out.choices[0].message.tool_calls[0]
        assert tc.id == "t1"
        assert tc.function.name == "search"
        import json

        assert json.loads(tc.function.arguments) == {"q": "x"}
        assert out.choices[0].finish_reason == "tool_calls"

    def test_finish_reason_max_tokens_maps_to_length(self):
        engine = MagicMock()
        engine.run.return_value = _result(stop_reason="max_tokens")
        out = _OpenAI(engine).chat.completions.create(model="m", messages=[])
        assert out.choices[0].finish_reason == "length"

    def test_empty_text_becomes_none_content(self):
        engine = MagicMock()
        engine.run.return_value = _result(text="", stop_reason="tool_use")
        out = _OpenAI(engine).chat.completions.create(model="m", messages=[])
        assert out.choices[0].message.content is None

    def test_carries_full_message_chain(self):
        engine = MagicMock()
        engine.run.return_value = _result(messages=["m1", "m2"])
        out = _OpenAI(engine).chat.completions.create(model="m", messages=[])
        assert list(out.messages) == ["m1", "m2"]

    def test_usage_reflects_aggregate_token_counts(self):
        engine = MagicMock()
        engine.run.return_value = _result(
            usage={"input_tokens": 500, "output_tokens": 40, "ai_message_count": 4}
        )
        out = _OpenAI(engine).chat.completions.create(model="m", messages=[])
        assert out.usage.prompt_tokens == 500
        assert out.usage.completion_tokens == 40
        assert out.usage.total_tokens == 540


class TestNativeFacade:
    def test_run_wraps_prompt_in_messages_list(self):
        engine = MagicMock()
        engine.run.return_value = _result()
        _Native(engine)(prompt="hello world")
        assert engine.run.call_args.kwargs["messages"] == [
            {"role": "user", "content": "hello world"}
        ]

    @pytest.mark.asyncio
    async def test_arun_wraps_prompt_in_messages_list(self):
        engine = MagicMock()

        async def _arun(**kw):
            return _result()

        engine.arun = MagicMock(side_effect=_arun)
        await _Native(engine).arun(prompt="async hi")
        assert engine.arun.call_args.kwargs["messages"] == [{"role": "user", "content": "async hi"}]

    def test_passes_through_messages_and_usage(self):
        """The native facade returns the NormalizedResult unchanged, so the
        message chain and aggregated usage already on ``result`` reach the
        caller without remapping."""
        engine = MagicMock()
        engine.run.return_value = _result(
            messages=["m1", "m2", "m3"],
            usage={"input_tokens": 600, "output_tokens": 60, "ai_message_count": 3},
        )
        out = _Native(engine)(prompt="hi")
        assert list(out.messages) == ["m1", "m2", "m3"]
        assert out.usage["input_tokens"] == 600
        assert out.usage["ai_message_count"] == 3


# ---------------------------------------------------------------------------
# Per-call LLM kwarg forwarding (max_tokens, temperature, top_p, ...)
#
# Each facade must surface a typed argument for the most common LLM knobs
# AND forward arbitrary extra kwargs through to ``engine.run`` so they
# reach ``chat.bind(...)``. Customers passing ``temperature=0.7`` should
# actually constrain the model — not have the value silently dropped.
# ---------------------------------------------------------------------------


class TestFacadeForwardsLLMKwargs:
    def test_anthropic_facade_forwards_temperature_and_top_p(self):
        engine = MagicMock()
        engine.run.return_value = _result()
        _Anthropic(engine).messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[],
            temperature=0.7,
            top_p=0.9,
        )
        run_kwargs = engine.run.call_args.kwargs
        assert run_kwargs["temperature"] == 0.7
        assert run_kwargs["top_p"] == 0.9
        assert run_kwargs["max_tokens"] == 512

    def test_anthropic_facade_drops_unset_optional_kwargs(self):
        """``temperature=None`` shouldn't pollute the engine call."""
        engine = MagicMock()
        engine.run.return_value = _result()
        _Anthropic(engine).messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[],
        )
        run_kwargs = engine.run.call_args.kwargs
        assert "temperature" not in run_kwargs
        assert "top_p" not in run_kwargs
        assert "top_k" not in run_kwargs
        assert "stop_sequences" not in run_kwargs

    def test_anthropic_facade_forwards_stop_sequences_and_top_k(self):
        engine = MagicMock()
        engine.run.return_value = _result()
        _Anthropic(engine).messages.create(
            model="m",
            max_tokens=10,
            messages=[],
            top_k=40,
            stop_sequences=["STOP"],
        )
        run_kwargs = engine.run.call_args.kwargs
        assert run_kwargs["top_k"] == 40
        assert run_kwargs["stop_sequences"] == ["STOP"]

    def test_openai_facade_forwards_presence_and_frequency_penalty(self):
        engine = MagicMock()
        engine.run.return_value = _result()
        _OpenAI(engine).chat.completions.create(
            model="gpt-4o-mini",
            messages=[],
            presence_penalty=0.4,
            frequency_penalty=0.2,
        )
        run_kwargs = engine.run.call_args.kwargs
        assert run_kwargs["presence_penalty"] == 0.4
        assert run_kwargs["frequency_penalty"] == 0.2

    def test_openai_facade_forwards_seed_and_stop(self):
        engine = MagicMock()
        engine.run.return_value = _result()
        _OpenAI(engine).chat.completions.create(
            model="gpt-4o-mini",
            messages=[],
            seed=42,
            stop=["END"],
            logprobs=True,
            top_logprobs=3,
        )
        run_kwargs = engine.run.call_args.kwargs
        assert run_kwargs["seed"] == 42
        assert run_kwargs["stop"] == ["END"]
        assert run_kwargs["logprobs"] is True
        assert run_kwargs["top_logprobs"] == 3

    def test_native_forwards_top_p_and_temperature(self):
        engine = MagicMock()
        engine.run.return_value = _result()
        _Native(engine)(prompt="hi", temperature=0.3, top_p=0.95)
        run_kwargs = engine.run.call_args.kwargs
        assert run_kwargs["temperature"] == 0.3
        assert run_kwargs["top_p"] == 0.95

    def test_native_drops_unset_optional_kwargs(self):
        engine = MagicMock()
        engine.run.return_value = _result()
        _Native(engine)(prompt="hi")
        run_kwargs = engine.run.call_args.kwargs
        assert "temperature" not in run_kwargs
        assert "top_p" not in run_kwargs

    def test_unknown_kwarg_passes_through_to_engine(self):
        """Custom provider-specific kwargs still flow via **kw to engine.run."""
        engine = MagicMock()
        engine.run.return_value = _result()
        _Anthropic(engine).messages.create(
            model="m",
            max_tokens=10,
            messages=[],
            metadata={"user_id": "abc"},  # arbitrary kw
        )
        run_kwargs = engine.run.call_args.kwargs
        assert run_kwargs["metadata"] == {"user_id": "abc"}
