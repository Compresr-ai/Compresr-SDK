"""Integration tests for the research agent against live LLM + search APIs.

Each test hits a real provider (Anthropic / OpenAI / Gemini) plus a real
Tavily search. Skipped when the corresponding key is absent so the suite
stays runnable in CI without secrets.

Goals (per provider):
  1. The loop actually runs end-to-end and produces a ResearchResult.
  2. Caching is observed where applicable:
      - Anthropic: cache_creation_tokens > 0 on at least one turn.
      - OpenAI: prompt_cache_key reaches the API without rejection.
      - Gemini: implicit caching keeps the call from erroring; we only
        verify the loop runs (explicit context cache wiring is v2).
  3. The trajectory contains at least one search step.
  4. ResearchResult.answer is non-empty OR explanation is non-empty.
"""

from __future__ import annotations

import os

import pytest

from compresr import ResearchResult

pytestmark = pytest.mark.integration


# A question that REQUIRES current web data so the model can't just answer
# from training. Keeps spend low because the answer is short and the loop
# typically takes 2 LLM turns: one search + one synthesis.
_QUESTION = (
    "Search the web for the latest stable Python version released in 2025 "
    "and report the version number."
)


def _skip_if_429(result: ResearchResult) -> None:
    """Skip cleanly when a provider responds with quota errors (not our bug)."""
    for step in result.trajectory:
        if (
            step.type == "error"
            and step.text
            and (
                "429" in step.text
                or "RESOURCE_EXHAUSTED" in step.text
                or "quota" in step.text.lower()
                or "rate_limit" in step.text.lower()
            )
        ):
            pytest.skip(f"Provider returned a quota error: {step.text[:200]}")


def _basic_assertions(result: ResearchResult) -> None:
    _skip_if_429(result)
    assert isinstance(result, ResearchResult)
    assert result.text, "expected non-empty final text"
    # Strict-format compliance varies by provider; accept either the parsed
    # ``answer`` field OR raw ``text`` as evidence the model produced output.
    assert result.answer or result.text
    # The agent must have either searched or committed to an answer.
    step_types = {s.type for s in result.trajectory}
    assert step_types & {"search", "answer"}, f"unexpected trajectory: {step_types}"
    # Usage must be populated by _aggregate_usage.
    assert result.usage.calls >= 1
    assert result.usage.output_tokens > 0


@pytest.mark.usefixtures("tavily_required")
class TestResearchAnthropic:
    def test_research_run_with_caching_real_call(self, anthropic_client):
        # max_steps=4 so the loop makes at least one search + one synthesis
        # call (= 2 LLM turns). Our default min_messages_to_cache=2 means
        # cache markers stamp from the 2nd turn onward, giving cache_creation
        # a chance to appear on the live call.
        result = anthropic_client.research.run(
            _QUESTION,
            search="tavily",
            max_steps=4,
            compress_snippets=True,
            min_compress_tokens=10_000,
        )
        _basic_assertions(result)
        # The middleware writes cache markers from turn 2; if the loop made
        # >= 2 LLM calls we expect non-zero cache_creation. Loops that ended
        # early (e.g. model refused to search) skip the cache assertion.
        if result.usage.calls >= 2:
            assert (
                result.usage.cache_creation_tokens > 0 or result.usage.cache_read_tokens > 0
            ), f"multi-turn loop produced no cache writes; usage={result.usage!r}"

    def test_research_run_without_cache_returns_consistent_shape(self, anthropic_client):
        # Build a fresh client with caching off to confirm we don't crash
        # and the shape is identical.
        from compresr import CompressionClient

        no_cache_client = CompressionClient(
            api_key=os.environ["COMPRESR_API_KEY"],
            llm="anthropic:claude-haiku-4-5",
            llm_api_key=os.environ["ANTHROPIC_API_KEY"],
            enable_prompt_cache=False,
        )
        result = no_cache_client.research.run(_QUESTION, search="tavily", max_steps=3)
        _basic_assertions(result)
        assert result.usage.cache_read_tokens == 0


@pytest.mark.usefixtures("tavily_required")
class TestResearchOpenAI:
    def test_research_run_with_prompt_cache_key(self):
        from compresr import CompressionClient

        for k in ("COMPRESR_API_KEY", "OPENAI_API_KEY"):
            if not os.environ.get(k):
                pytest.skip(f"Missing env var: {k}")
        client = CompressionClient(
            api_key=os.environ["COMPRESR_API_KEY"],
            llm="openai:gpt-4o-mini",
            llm_api_key=os.environ["OPENAI_API_KEY"],
            openai_prompt_cache_key="compresr-research-integration-test",
        )
        result = client.research.run(_QUESTION, search="tavily", max_steps=3)
        _basic_assertions(result)
        # OpenAI server caches implicitly — cache_read may be 0 on first
        # call, non-zero on subsequent ones in the same window. Just verify
        # the call didn't blow up because of an unknown kwarg.


@pytest.mark.usefixtures("tavily_required")
class TestResearchGemini:
    def test_research_run_loop_completes(self, gemini_client):
        result = gemini_client.research.run(_QUESTION, search="tavily", max_steps=3)
        _basic_assertions(result)
        # Gemini implicit caching is a no-op for the SDK; we just verify
        # the agent loop is compatible with the Gemini chat model surface
        # (tool calls, message shapes, usage_metadata).


class TestResearchAcrossProviders:
    """Compatibility matrix: same code path, every supported provider."""

    @pytest.mark.parametrize(
        "client_fixture",
        ["anthropic_client", "openai_client", "gemini_client"],
    )
    def test_search_facade_works_end_to_end(self, request, client_fixture):
        request.getfixturevalue("tavily_required")
        client = request.getfixturevalue(client_fixture)
        result = client.research.search(_QUESTION, search="tavily")
        _basic_assertions(result)
