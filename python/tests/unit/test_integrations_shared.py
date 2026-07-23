"""Pure-Python unit tests for the shared integration helpers.

No peer-dependency imports — runs in any environment.
"""

from __future__ import annotations

import pytest

from compresr.integrations._shared import (
    COMMON_QUERY_KEYS,
    apply_error_policy,
    estimate_tokens,
    extract_query_from_args,
    extract_query_from_messages,
    make_filter,
    resolve_query,
)

# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_short_string_returns_at_least_one(self):
        assert estimate_tokens("hi") >= 1

    def test_longer_strings_have_higher_counts(self):
        short = estimate_tokens("hello world")
        long_ = estimate_tokens("hello world " * 100)
        assert long_ > short


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


class TestMakeFilter:
    def test_default_allows_all(self):
        f = make_filter()
        assert f("anything") is True
        assert f(None) is True

    def test_allow_list(self):
        f = make_filter(allow={"a", "b"})
        assert f("a") is True
        assert f("b") is True
        assert f("c") is False
        assert f(None) is False

    def test_ignore_list(self):
        f = make_filter(ignore={"x"})
        assert f("a") is True
        assert f("x") is False

    def test_both_raises(self):
        with pytest.raises(ValueError):
            make_filter(allow={"a"}, ignore={"b"})


# ---------------------------------------------------------------------------
# query — args
# ---------------------------------------------------------------------------


class TestExtractQueryFromArgs:
    def test_preferred_key_wins(self):
        args = {"query": "search this", "url": "https://x"}
        assert extract_query_from_args(args, preferred_key="url") == "https://x"

    def test_preferred_missing_returns_none(self):
        # Explicit key + missing => None (no silent fallback)
        assert extract_query_from_args({"a": "b"}, preferred_key="missing") is None

    def test_common_keys_priority(self):
        # `query` should beat `q`
        args = {"q": "short", "query": "long"}
        assert extract_query_from_args(args) == "long"

    def test_no_match_returns_none(self):
        assert extract_query_from_args({"unrelated": "value"}) is None

    def test_empty_args(self):
        assert extract_query_from_args({}) is None
        assert extract_query_from_args(None) is None  # type: ignore[arg-type]

    def test_common_query_keys_constant_is_tuple(self):
        assert isinstance(COMMON_QUERY_KEYS, tuple)
        assert "query" in COMMON_QUERY_KEYS
        assert "question" in COMMON_QUERY_KEYS


# ---------------------------------------------------------------------------
# query — messages
# ---------------------------------------------------------------------------


class TestExtractQueryFromMessages:
    def test_tool_call_args_win(self):
        messages = [
            {"role": "user", "content": "old question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "abc", "args": {"query": "actual intent"}, "name": "search"}],
            },
        ]
        out = extract_query_from_messages(messages, current_tool_call_id="abc")
        assert out == "actual intent"

    def test_falls_back_to_last_human(self):
        messages = [
            {"role": "user", "content": "what's up"},
            {"role": "assistant", "content": "hi"},
        ]
        assert extract_query_from_messages(messages) == "what's up"

    def test_fallback_when_no_messages(self):
        assert extract_query_from_messages([], fallback="X") == "X"


# ---------------------------------------------------------------------------
# resolve_query — the union point
# ---------------------------------------------------------------------------


class TestResolveQuery:
    def test_static_wins_over_everything(self):
        out = resolve_query(
            static="STATIC",
            args={"query": "from_args"},
            messages=[{"role": "user", "content": "from_msg"}],
        )
        assert out == "STATIC"

    def test_extractor_wins_over_args(self):
        out = resolve_query(
            extractor=lambda a: "EXTRACTED",
            extractor_arg={},
            args={"query": "ARGS"},
        )
        assert out == "EXTRACTED"

    def test_args_with_key(self):
        assert resolve_query(args={"q": "yes"}, args_key="q") == "yes"

    def test_args_smart_pick(self):
        assert resolve_query(args={"question": "smart"}) == "smart"

    def test_messages_fallback(self):
        out = resolve_query(messages=[{"role": "user", "content": "from history"}])
        assert out == "from history"

    def test_final_fallback(self):
        assert resolve_query(fallback="FB") == "FB"

    def test_extractor_exception_is_swallowed(self):
        def boom(_):
            raise RuntimeError("nope")

        out = resolve_query(
            extractor=boom,
            extractor_arg={},
            args={"query": "fallback_to_args"},
        )
        assert out == "fallback_to_args"


# ---------------------------------------------------------------------------
# error policy
# ---------------------------------------------------------------------------


class TestApplyErrorPolicy:
    def test_passthrough_on_failure(self):
        out = apply_error_policy(lambda: 1 / 0, fallback="ok")
        assert out == "ok"

    def test_raise_on_failure(self):
        with pytest.raises(ZeroDivisionError):
            apply_error_policy(lambda: 1 / 0, fallback="ok", policy="raise")

    def test_success_returns_value(self):
        assert apply_error_policy(lambda: 42, fallback=0) == 42
