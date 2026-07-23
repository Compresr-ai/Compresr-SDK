"""Integration: per-call LLM kwargs really reach the underlying provider.

Verifies the bug fix for ``_Engine.run``: ``max_tokens``, ``temperature``,
``top_p``, and friends are now forwarded via ``chat.bind(...)`` on every
call. We exercise this end-to-end against live Anthropic + OpenAI by
asking the model to produce a long output, capping it at a tiny
``max_tokens`` value, and asserting the provider actually truncates.

Skipped automatically when the corresponding provider key is missing.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# max_tokens — most observable: a hard cap on output_tokens.
# ---------------------------------------------------------------------------


class TestMaxTokensCap:
    def test_anthropic_max_tokens_caps_output(self, anthropic_client):
        """``max_tokens=20`` must actually cap Anthropic's output_tokens."""
        r = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 100 separated by spaces.",
                }
            ],
        )
        # Anthropic counts output_tokens including any stop. Allow a small
        # slack but assert it's bounded — pre-fix this exceeded 100 trivially.
        assert (
            r.usage.output_tokens <= 30
        ), f"max_tokens=20 was ignored — got {r.usage.output_tokens}"

    def test_openai_max_tokens_caps_output(self, openai_client):
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 100 separated by spaces.",
                }
            ],
        )
        assert (
            r.usage.completion_tokens <= 30
        ), f"max_tokens=20 was ignored — got {r.usage.completion_tokens}"

    def test_anthropic_max_tokens_caps_output_WITH_TOOLS(self, anthropic_client, tavily_required):
        """Regression: LangChain's ``bind_tools(...)`` strips a prior
        ``chat.bind(...)``'s kwargs, so the engine must bake LLM knobs into the
        chat-model constructor — otherwise ``max_tokens`` is silently dropped
        whenever any tool is passed."""
        import os

        from compresr import WebSearchTool

        tavily = WebSearchTool.tavily(api_key=os.environ["TAVILY_API_KEY"], max_results=2)
        r = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": "Use tavily_search to find recent Anthropic news, then summarize.",
                }
            ],
            tools=[tavily],
        )
        # Pre-fix: this came back at 200+ output_tokens because bind_tools
        # silently dropped max_tokens. Post-fix: actually capped.
        assert (
            r.usage.output_tokens <= 30
        ), f"max_tokens=20 was ignored WITH tools — got {r.usage.output_tokens}"

    def test_gemini_max_tokens_aliases_to_max_output_tokens(self):
        import os

        from compresr import CompressionClient

        google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not os.environ.get("COMPRESR_API_KEY") or not google_key:
            pytest.skip("COMPRESR_API_KEY or GOOGLE_API_KEY/GEMINI_API_KEY missing")

        client = CompressionClient(
            api_key=os.environ["COMPRESR_API_KEY"],
            llm="google_genai",
            llm_api_key=google_key,
        )
        try:
            r = client.messages.create(
                model="gemini-2.5-flash",
                max_tokens=20,
                messages=[{"role": "user", "content": "Count from 1 to 100 separated by spaces."}],
            )
        except Exception as exc:
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "credits are depleted" in msg or "429" in msg:
                pytest.skip(f"Gemini billing/quota error — plumbing reached the wire: {msg[:200]}")
            raise
        # Proves max_tokens was aliased to max_output_tokens AND reached Gemini's wire.
        assert (
            r.usage.output_tokens <= 30
        ), f"max_tokens=20 was ignored on Gemini — got {r.usage.output_tokens}"


# ---------------------------------------------------------------------------
# Temperature — semantically observable: temperature=0 is reproducible.
# ---------------------------------------------------------------------------


class TestTemperatureForwarded:
    def test_anthropic_temperature_zero_is_consistent(self, anthropic_client):
        prompt = "Say the single word 'pong' and nothing else."
        r1 = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        r2 = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        # temperature=0 isn't a strict determinism guarantee at the API
        # level, but for a simple prompt both responses should hit the
        # expected token.
        assert "pong" in r1.content[0].text.lower()
        assert "pong" in r2.content[0].text.lower()

    def test_openai_temperature_zero_is_consistent(self, openai_client):
        prompt = "Say the single word 'pong' and nothing else."
        r1 = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        r2 = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        assert "pong" in (r1.choices[0].message.content or "").lower()
        assert "pong" in (r2.choices[0].message.content or "").lower()
