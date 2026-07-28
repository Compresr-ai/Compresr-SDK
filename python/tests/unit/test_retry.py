"""Unit tests for the SDK transport-layer retry policy."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.error import HTTPError

import pytest

from compresr import CompressionClient, RetryConfig
from compresr.exceptions import ServiceUnavailableError
from compresr.retry import compute_backoff


def _http_error(status: int, body: Dict[str, Any], retry_after: Optional[str] = None) -> HTTPError:
    import json as _json

    headers = {"Retry-After": retry_after} if retry_after else {}
    return HTTPError(
        url="https://api.compresr.ai/x",
        code=status,
        msg="err",
        hdrs=headers,  # type: ignore[arg-type]
        fp=io.BytesIO(_json.dumps(body).encode()),
    )


SUCCESS_BODY: Dict[str, Any] = {
    "success": True,
    "data": {
        "compressed_context": "compressed",
        "original_tokens": 100,
        "compressed_tokens": 50,
        "actual_compression_ratio": 0.5,
        "tokens_saved": 50,
        "duration_ms": 10,
    },
}


class TestRetryConfig:
    def test_defaults(self) -> None:
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.retry_on_status == (429, 503)
        assert cfg.respect_retry_after is True

    def test_user_override(self) -> None:
        cfg = RetryConfig(max_retries=5, initial_backoff_s=0.1, retry_on_status=(503,))
        assert cfg.max_retries == 5
        assert cfg.initial_backoff_s == 0.1
        assert cfg.retry_on_status == (503,)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            RetryConfig(max_retries=-1)
        with pytest.raises(ValueError):
            RetryConfig(initial_backoff_s=-0.1)
        with pytest.raises(ValueError):
            RetryConfig(multiplier=0.5)
        with pytest.raises(ValueError):
            RetryConfig(jitter=2.0)


class TestComputeBackoff:
    def test_exponential_growth(self) -> None:
        cfg = RetryConfig(initial_backoff_s=1.0, multiplier=2.0, jitter=0.0, max_backoff_s=100.0)
        assert compute_backoff(0, cfg) == 1.0
        assert compute_backoff(1, cfg) == 2.0
        assert compute_backoff(2, cfg) == 4.0
        assert compute_backoff(3, cfg) == 8.0

    def test_capped_at_max(self) -> None:
        cfg = RetryConfig(initial_backoff_s=1.0, multiplier=10.0, jitter=0.0, max_backoff_s=5.0)
        assert compute_backoff(5, cfg) == 5.0

    def test_jitter_bounded(self) -> None:
        cfg = RetryConfig(initial_backoff_s=1.0, multiplier=1.0, jitter=0.5, max_backoff_s=100.0)
        for _ in range(50):
            v = compute_backoff(0, cfg)
            assert 0.5 <= v <= 1.5

    def test_honors_retry_after(self) -> None:
        cfg = RetryConfig(initial_backoff_s=1.0, jitter=0.0, max_backoff_s=100.0)
        assert compute_backoff(0, cfg, retry_after=7.5) == 7.5

    def test_retry_after_capped(self) -> None:
        cfg = RetryConfig(initial_backoff_s=1.0, jitter=0.0, max_backoff_s=10.0)
        assert compute_backoff(0, cfg, retry_after=120) == 10.0

    def test_retry_after_ignored_when_disabled(self) -> None:
        cfg = RetryConfig(
            initial_backoff_s=2.0,
            jitter=0.0,
            max_backoff_s=100.0,
            respect_retry_after=False,
        )
        assert compute_backoff(0, cfg, retry_after=60) == 2.0


def _make_client(**kwargs: Any) -> CompressionClient:
    return CompressionClient(api_key="cmp_test", base_url="https://api.compresr.ai", **kwargs)


class TestSyncRetry:
    def test_retries_on_503_then_succeeds(self) -> None:
        client = _make_client(
            retry_config=RetryConfig(max_retries=3, initial_backoff_s=0.01, jitter=0.0),
        )
        responses: List[Any] = [
            _http_error(503, {"error": "busy", "code": "service_unavailable"}),
            _http_error(503, {"error": "busy", "code": "service_unavailable"}),
            MagicMock(
                read=lambda *a: b'{"success": true, "data": '
                b'{"compressed_context": "c", "original_tokens": 100, '
                b'"compressed_tokens": 50, "actual_compression_ratio": 0.5, '
                b'"tokens_saved": 50, "duration_ms": 1}}'
            ),
        ]
        calls = {"n": 0}

        def fake_urlopen(*_: Any, **__: Any) -> Any:
            calls["n"] += 1
            r = responses.pop(0)
            if isinstance(r, HTTPError):
                raise r
            cm = MagicMock()
            cm.__enter__ = lambda *_a: r
            cm.__exit__ = lambda *_a: False
            return cm

        with (
            patch("compresr.services.proxy.urlopen", side_effect=fake_urlopen),
            patch("compresr.services.proxy.time.sleep") as sleep_mock,
        ):
            result = client.compress(context="x", query="y")
        assert result.data.compressed_context == "c"
        assert calls["n"] == 3
        assert sleep_mock.call_count == 2

    def test_retries_on_429(self) -> None:
        client = _make_client(
            retry_config=RetryConfig(max_retries=2, initial_backoff_s=0.0, jitter=0.0),
        )
        responses: List[Any] = [
            _http_error(429, {"error": "rate", "code": "rate_limit_exceeded"}),
            MagicMock(
                read=lambda *a: b'{"success": true, "data": '
                b'{"compressed_context": "c", "original_tokens": 100, '
                b'"compressed_tokens": 50, "actual_compression_ratio": 0.5, '
                b'"tokens_saved": 50, "duration_ms": 1}}'
            ),
        ]

        def fake_urlopen(*_: Any, **__: Any) -> Any:
            r = responses.pop(0)
            if isinstance(r, HTTPError):
                raise r
            cm = MagicMock()
            cm.__enter__ = lambda *_a: r
            cm.__exit__ = lambda *_a: False
            return cm

        with (
            patch("compresr.services.proxy.urlopen", side_effect=fake_urlopen),
            patch("compresr.services.proxy.time.sleep"),
        ):
            result = client.compress(context="x", query="y")
        assert result.data.compressed_context == "c"

    def test_does_not_retry_on_4xx(self) -> None:
        client = _make_client(retry_config=RetryConfig(max_retries=3))
        calls = {"n": 0}

        def fake_urlopen(*_: Any, **__: Any) -> Any:
            calls["n"] += 1
            raise _http_error(400, {"error": "bad", "detail": "nope"})

        with patch("compresr.services.proxy.urlopen", side_effect=fake_urlopen):
            with pytest.raises(Exception):
                client.compress(context="x", query="y")
        assert calls["n"] == 1

    def test_exhausts_retries_and_raises(self) -> None:
        client = _make_client(
            retry_config=RetryConfig(max_retries=2, initial_backoff_s=0.0, jitter=0.0),
        )
        calls = {"n": 0}

        def fake_urlopen(*_: Any, **__: Any) -> Any:
            calls["n"] += 1
            raise _http_error(503, {"error": "busy", "code": "service_unavailable"})

        with (
            patch("compresr.services.proxy.urlopen", side_effect=fake_urlopen),
            patch("compresr.services.proxy.time.sleep"),
        ):
            with pytest.raises(ServiceUnavailableError):
                client.compress(context="x", query="y")
        assert calls["n"] == 3  # initial + 2 retries

    def test_disabled_with_max_retries_zero(self) -> None:
        client = _make_client(retry_config=RetryConfig(max_retries=0))
        calls = {"n": 0}

        def fake_urlopen(*_: Any, **__: Any) -> Any:
            calls["n"] += 1
            raise _http_error(503, {"error": "busy", "code": "service_unavailable"})

        with patch("compresr.services.proxy.urlopen", side_effect=fake_urlopen):
            with pytest.raises(ServiceUnavailableError):
                client.compress(context="x", query="y")
        assert calls["n"] == 1

    def test_respects_retry_after_header(self) -> None:
        client = _make_client(
            retry_config=RetryConfig(max_retries=1, initial_backoff_s=99.0, jitter=0.0),
        )
        responses: List[Any] = [
            _http_error(503, {"error": "busy", "code": "service_unavailable"}, retry_after="0.1"),
            MagicMock(
                read=lambda *a: b'{"success": true, "data": '
                b'{"compressed_context": "c", "original_tokens": 100, '
                b'"compressed_tokens": 50, "actual_compression_ratio": 0.5, '
                b'"tokens_saved": 50, "duration_ms": 1}}'
            ),
        ]

        def fake_urlopen(*_: Any, **__: Any) -> Any:
            r = responses.pop(0)
            if isinstance(r, HTTPError):
                raise r
            cm = MagicMock()
            cm.__enter__ = lambda *_a: r
            cm.__exit__ = lambda *_a: False
            return cm

        with (
            patch("compresr.services.proxy.urlopen", side_effect=fake_urlopen),
            patch("compresr.services.proxy.time.sleep") as sleep_mock,
        ):
            client.compress(context="x", query="y")
        # Server hint (0.1s) wins over the initial_backoff_s=99 default.
        assert sleep_mock.call_args.args[0] == pytest.approx(0.1)


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_retries_on_503_then_succeeds(self) -> None:
        client = _make_client(
            retry_config=RetryConfig(max_retries=2, initial_backoff_s=0.0, jitter=0.0),
        )

        call_count = {"n": 0}

        def make_response(status: int, body: Dict[str, Any]) -> Any:
            resp = MagicMock()
            resp.status_code = status
            resp.json = MagicMock(return_value=body)
            resp.headers = {}
            return resp

        async def fake_request(method: str, url: str, **kw: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] < 3:
                return make_response(503, {"error": "busy", "code": "service_unavailable"})
            return make_response(200, SUCCESS_BODY)

        client._async_client.request = AsyncMock(side_effect=fake_request)  # type: ignore[method-assign]
        with patch("compresr.services.proxy.asyncio.sleep", new=AsyncMock()):
            result = await client.compress_async(context="x", query="y")
        assert result.data.compressed_context == "compressed"
        assert call_count["n"] == 3
        await client.aclose()

    @pytest.mark.asyncio
    async def test_async_does_not_retry_on_4xx(self) -> None:
        client = _make_client(retry_config=RetryConfig(max_retries=3))
        call_count = {"n": 0}

        async def fake_request(method: str, url: str, **kw: Any) -> Any:
            call_count["n"] += 1
            resp = MagicMock()
            resp.status_code = 400
            resp.json = MagicMock(return_value={"error": "bad"})
            resp.headers = {}
            return resp

        client._async_client.request = AsyncMock(side_effect=fake_request)  # type: ignore[method-assign]
        with pytest.raises(Exception):
            await client.compress_async(context="x", query="y")
        assert call_count["n"] == 1
        await client.aclose()
