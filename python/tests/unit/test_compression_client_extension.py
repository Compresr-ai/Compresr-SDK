"""Unit tests for the ``CompressionClient`` Wave 2B extensions.

Covers the opt-in ``llm=`` constructor path, the lazy facade properties,
and the top-level re-exports added to ``compresr/__init__.py``.
"""

from unittest.mock import MagicMock, patch

import pytest

from compresr import CompressionClient
from compresr.exceptions import CompresrError


class TestNoLLM:
    def test_compress_still_works_without_llm(self):
        # Construct client; .compress() path uses the existing HTTPClient stack
        c = CompressionClient(api_key="cmp_x")
        assert c._engine is None  # internal but useful sanity check

    def test_messages_raises_clear_error(self):
        c = CompressionClient(api_key="cmp_x")
        with pytest.raises(CompresrError, match="messages.create"):
            _ = c.messages

    def test_chat_raises_clear_error(self):
        c = CompressionClient(api_key="cmp_x")
        with pytest.raises(CompresrError, match="chat.completions.create"):
            _ = c.chat

    def test_run_raises_clear_error(self):
        c = CompressionClient(api_key="cmp_x")
        with pytest.raises(CompresrError, match="run"):
            c.run(prompt="hi")


class TestWithLLM:
    @patch("compresr.agents.engine.init_chat_model")
    def test_engine_constructed_with_llm(self, ic):
        ic.return_value = MagicMock()
        c = CompressionClient(
            api_key="cmp_x",
            llm="anthropic:claude-opus-4-8",
            llm_api_key="sk-test",
        )
        assert c._engine is not None
        assert c._engine.provider == "anthropic"

    @patch("compresr.agents.engine.init_chat_model")
    def test_messages_is_lazy(self, ic):
        ic.return_value = MagicMock()
        c = CompressionClient(
            api_key="cmp_x",
            llm="anthropic:claude-opus-4-8",
            llm_api_key="sk-test",
        )
        assert not hasattr(c, "_anthropic_facade")
        _ = c.messages
        assert hasattr(c, "_anthropic_facade")

    @patch("compresr.agents.engine.init_chat_model")
    def test_chat_is_lazy(self, ic):
        ic.return_value = MagicMock()
        c = CompressionClient(
            api_key="cmp_x",
            llm="openai:gpt-4o",
            llm_api_key="sk-test",
        )
        assert not hasattr(c, "_openai_facade")
        _ = c.chat
        assert hasattr(c, "_openai_facade")

    @patch("compresr.agents.engine.init_chat_model")
    def test_messages_create_delegates_to_engine(self, ic):
        ic.return_value = MagicMock()
        c = CompressionClient(
            api_key="cmp_x",
            llm="anthropic:claude-opus-4-8",
            llm_api_key="sk-test",
        )
        with patch.object(c._engine, "run") as run:
            from compresr.agents.normalized import CompresrStats, NormalizedResult

            run.return_value = NormalizedResult(text="ok", compresr_stats=CompresrStats())
            out = c.messages.create(
                model="claude-opus-4-8",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            assert out.content[0].text == "ok"
            run.assert_called_once()

    @patch("compresr.agents.engine.init_chat_model")
    def test_provider_only_llm_works(self, ic):
        """``llm='anthropic'`` -> model lives at the call site (Anthropic-style)."""
        ic.return_value = MagicMock()
        c = CompressionClient(
            api_key="cmp_x",
            llm="anthropic",
            llm_api_key="sk-test",
        )
        assert c._engine.provider == "anthropic"
        assert c._engine.default_model_name is None

        with patch.object(c._engine, "run") as run:
            from compresr.agents.normalized import CompresrStats, NormalizedResult

            run.return_value = NormalizedResult(text="ok", compresr_stats=CompresrStats())
            out = c.messages.create(
                model="claude-haiku-4-5",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            run.assert_called_once()
            # The facade forwarded the call-site model to the engine.
            assert run.call_args.kwargs["model"] == "claude-haiku-4-5"
            # And the Anthropic response envelope echoes that model.
            assert out.model == "claude-haiku-4-5"

    @patch("compresr.agents.engine.init_chat_model")
    def test_compression_dict_becomes_policy(self, ic):
        ic.return_value = MagicMock()
        c = CompressionClient(
            api_key="cmp_x",
            llm="anthropic:claude-opus-4-8",
            compression={"target_compression_ratio": 0.7, "min_tokens": 1000},
        )
        assert c._engine._policy.target_compression_ratio == 0.7
        assert c._engine._policy.min_tokens == 1000


class TestTopLevelExports:
    def test_websearchtool_importable_from_top_level(self):
        from compresr import WebSearchTool

        assert WebSearchTool is not None

    def test_compressionpolicy_importable_from_top_level(self):
        from compresr import CompressionPolicy

        assert CompressionPolicy is not None
