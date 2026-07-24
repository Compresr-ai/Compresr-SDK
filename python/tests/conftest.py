"""
Pytest Configuration for Compresr SDK Tests

Usage:
    pytest                      # Run tests against localhost (default)
    pytest --prod               # Run tests against production API
    pytest -v                   # Verbose output

Set COMPRESR_BASE_URL to test against a different environment.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file from SDK root
env_file = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_file)


def pytest_addoption(parser):
    """Add --prod command line option."""
    parser.addoption(
        "--prod",
        action="store_true",
        default=False,
        help="Run tests against production API (api.compresr.ai)",
    )


def pytest_configure(config):
    """Configure pytest."""
    use_prod = config.getoption("--prod", default=False)

    if use_prod:
        os.environ["COMPRESR_BASE_URL"] = "https://api.compresr.ai"
    elif "COMPRESR_BASE_URL" not in os.environ:
        os.environ["COMPRESR_BASE_URL"] = "http://localhost:8000"

    base_url = os.environ["COMPRESR_BASE_URL"]
    env_name = "PRODUCTION" if "api.compresr.ai" in base_url else "LOCAL"

    print(f"\n{'='*50}")
    print(f"Testing against: {base_url}")
    print(f"Environment: {env_name}")
    print(f"{'='*50}\n")


@pytest.fixture
def admin_api_key():
    """Get admin API key."""
    return os.getenv("COMPRESR_API_KEY") or os.getenv("COMPRESSION_SERVICE_ADMIN_KEY")


@pytest.fixture
def user_api_key():
    """Get user API key (rate limited)."""
    return os.getenv("COMPRESSION_SERVICE_USER_KEY")


# ---------------------------------------------------------------------------
# In-process fake client for unit tests of integration code.
#
# Lives at the top-level conftest so it's available to both ``tests/unit``
# (mocked integration tests) and ``tests/integration`` (where the fixture
# isn't used directly but its types may be re-exported).
# ---------------------------------------------------------------------------

from typing import Any, List, Optional  # noqa: E402


class _Call(dict):
    """A recorded call — dict + dotted-attribute access for ergonomics."""

    def __getattr__(self, key: str) -> Any:  # pragma: no cover
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _half(text: str) -> str:
    return text[: max(1, len(text) // 2)] + "<<C>>"


class FakeCompressionClient:
    """In-process mock of ``CompressionClient`` used by unit-level
    integration tests. Records calls; returns deterministic "compressed"
    output (first half + ``<<C>>`` marker)."""

    def __init__(self, *, raise_on_call: bool = False) -> None:
        self.calls: List[_Call] = []
        self.batch_calls: List[_Call] = []
        self._raise = raise_on_call

    def compress(self, **kw: Any):
        from compresr.schemas import CompressResponse, CompressResult

        self.calls.append(_Call(kw))
        if self._raise:
            raise RuntimeError("forced failure")
        ctx = kw["context"]
        out = _half(ctx)
        return CompressResponse(
            success=True,
            data=CompressResult(
                original_context=ctx,
                compressed_context=out,
                original_tokens=max(1, len(ctx) // 4),
                compressed_tokens=max(1, len(out) // 4),
                actual_compression_ratio=0.5,
                tokens_saved=max(0, (len(ctx) - len(out)) // 4),
                duration_ms=1,
            ),
        )

    async def compress_async(self, **kw: Any):
        return self.compress(**kw)

    def compress_batch(
        self,
        contexts: List[str],
        queries: Optional[Any] = None,
        compression_model_name: str = "latte_v1",
        **kw: Any,
    ):
        from compresr.schemas import (
            CompressBatchItemResult,
            CompressBatchResponse,
            CompressBatchResult,
        )

        self.batch_calls.append(
            _Call(
                contexts=list(contexts),
                queries=queries,
                compression_model_name=compression_model_name,
                **kw,
            )
        )
        if self._raise:
            raise RuntimeError("forced failure")

        items = []
        total_in = 0
        total_out = 0
        for ctx in contexts:
            out = _half(ctx)
            in_t = max(1, len(ctx) // 4)
            out_t = max(1, len(out) // 4)
            total_in += in_t
            total_out += out_t
            items.append(
                CompressBatchItemResult(
                    original_context=ctx,
                    compressed_context=out,
                    original_tokens=in_t,
                    compressed_tokens=out_t,
                    actual_compression_ratio=0.5,
                    tokens_saved=in_t - out_t,
                    duration_ms=1,
                )
            )

        return CompressBatchResponse(
            success=True,
            data=CompressBatchResult(
                results=items,
                total_original_tokens=total_in,
                total_compressed_tokens=total_out,
                total_tokens_saved=total_in - total_out,
                average_compression_ratio=0.5,
                count=len(items),
            ),
        )

    async def compress_batch_async(self, *a: Any, **kw: Any):
        return self.compress_batch(*a, **kw)


@pytest.fixture
def fake_client() -> FakeCompressionClient:
    return FakeCompressionClient()


@pytest.fixture
def failing_client() -> FakeCompressionClient:
    return FakeCompressionClient(raise_on_call=True)
