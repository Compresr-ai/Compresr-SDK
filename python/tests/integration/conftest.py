"""Fixtures for live integration tests.

These tests hit a real Compresr backend and are skipped when no API key
is available. The mocked ``FakeCompressionClient`` fixtures live in the
top-level ``tests/conftest.py`` so unit tests can use them too.
"""

from __future__ import annotations

import os

import pytest

from compresr import CompressionClient

# ---------------------------------------------------------------------------
# Live API key + client
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_client(admin_api_key):
    """Create CompressionClient with ADMIN key."""
    if not admin_api_key:
        pytest.skip("Admin API key not available")
    return CompressionClient(api_key=admin_api_key)


@pytest.fixture
def skip_if_no_admin_key(admin_api_key):
    """Skip test if no admin API key available."""
    if not admin_api_key:
        pytest.skip("Admin API key not configured")


@pytest.fixture(scope="session")
def live_client():
    """Real ``CompressionClient`` hitting the configured backend.

    Session-scoped: we probe reachability once. Skips the entire live
    suite when:
        - No ``COMPRESR_API_KEY`` is set, OR
        - The configured ``COMPRESR_BASE_URL`` doesn't answer a small
          ping compress.

    Prevents silent passthroughs (compression "fails open" by default)
    from masquerading as failing assertions.
    """
    api_key = os.getenv("COMPRESR_API_KEY") or os.getenv("COMPRESSION_SERVICE_ADMIN_KEY")
    if not api_key:
        pytest.skip("COMPRESR_API_KEY not set — skipping live integration tests")

    kwargs = {"api_key": api_key}
    base_url = os.environ.get("COMPRESR_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    client = CompressionClient(**kwargs)

    try:
        client.compress(
            context="ping " * 60,
            query="ping",
            target_compression_ratio=0.5,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Compresr backend at {kwargs.get('base_url','default')} unreachable: {exc}")

    return client


# ---------------------------------------------------------------------------
# Shared content used by every live integration test
# ---------------------------------------------------------------------------

LIVE_LONG_TEXT = (
    "The Transformer architecture introduced by Vaswani et al. in 2017 in "
    "'Attention Is All You Need' replaced recurrent layers with multi-head "
    "self-attention. This enabled massive parallelization on GPUs and "
    "removed the sequential bottleneck of LSTMs and GRUs. The encoder-decoder "
    "design with stacked attention blocks became the foundation for BERT, "
    "GPT, T5, and most modern large language models. "
) * 8

LIVE_QUERY = "What replaced recurrence in the Transformer?"


@pytest.fixture
def live_long_text() -> str:
    return LIVE_LONG_TEXT


@pytest.fixture
def live_query() -> str:
    return LIVE_QUERY


# ---------------------------------------------------------------------------
# Wave 2.5 agents — live LLM fixtures
# ---------------------------------------------------------------------------


def _require(*keys: str) -> None:
    """Skip the test if any of the listed env vars are missing."""
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")


@pytest.fixture
def anthropic_client():
    """Compresr client wired to Anthropic Claude Haiku."""
    _require("COMPRESR_API_KEY", "ANTHROPIC_API_KEY")
    return CompressionClient(
        api_key=os.environ["COMPRESR_API_KEY"],
        llm="anthropic:claude-haiku-4-5",
        llm_api_key=os.environ["ANTHROPIC_API_KEY"],
    )


@pytest.fixture
def openai_client():
    """Compresr client wired to OpenAI gpt-4o-mini."""
    _require("COMPRESR_API_KEY", "OPENAI_API_KEY")
    return CompressionClient(
        api_key=os.environ["COMPRESR_API_KEY"],
        llm="openai:gpt-4o-mini",
        llm_api_key=os.environ["OPENAI_API_KEY"],
    )


@pytest.fixture
def gemini_client():
    """Compresr client wired to Google Gemini."""
    _require("COMPRESR_API_KEY", "GEMINI_API_KEY")
    return CompressionClient(
        api_key=os.environ["COMPRESR_API_KEY"],
        llm="google_genai:gemini-2.5-flash",
        llm_api_key=os.environ["GEMINI_API_KEY"],
    )


@pytest.fixture
def tavily_required():
    """Gate tests that need a live Tavily key."""
    _require("TAVILY_API_KEY")


@pytest.fixture
def compress_spy(monkeypatch):
    """Wrap ``CompressionClient.compress`` (and async variant) on every
    instance so we can count compression backend calls without breaking
    real behavior. Returns a list that grows as ``compress()`` fires.
    """
    calls: list = []
    original = CompressionClient.compress
    original_async = CompressionClient.compress_async

    def spy(self, **kw):
        calls.append(dict(kw))
        return original(self, **kw)

    async def spy_async(self, **kw):
        calls.append(dict(kw))
        return await original_async(self, **kw)

    monkeypatch.setattr(CompressionClient, "compress", spy)
    monkeypatch.setattr(CompressionClient, "compress_async", spy_async)
    return calls
