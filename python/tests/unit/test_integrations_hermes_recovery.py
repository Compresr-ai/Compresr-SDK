"""Unit tests for the compress_with_recovery kernel (fake Hermes runtime)."""

from typing import Any, Dict, List, Optional

import pytest

from compresr.schemas import CompressToolOutputResponse, CompressToolOutputResult

LONG_CONTENT = "needle line\n" + ("filler line with some text\n" * 300)


def _response(compressed: str = "short summary", **overrides: Any) -> CompressToolOutputResponse:
    fields: Dict[str, Any] = dict(
        compressed_output=compressed,
        original_tokens=900,
        compressed_tokens=30,
        compression_ratio=30.0,
        tool_name="grep",
        duration_ms=50,
    )
    fields.update(overrides)
    return CompressToolOutputResponse(success=True, data=CompressToolOutputResult(**fields))


class FakeClient:
    def __init__(
        self,
        response: Optional[CompressToolOutputResponse] = None,
        error: Optional[Exception] = None,
    ):
        self.response = response or _response()
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def compress_tool_output(self, **kwargs: Any) -> CompressToolOutputResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def recovery(hermes_env):
    return hermes_env.load("recovery")


@pytest.fixture
def cache_mod(hermes_env):
    return hermes_env.load("cache")


class TestCompressWithRecovery:
    def test_success_appends_footer_and_caches_original(self, recovery, cache_mod):
        client = FakeClient()
        out, info = recovery.compress_with_recovery(
            query="find needle",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="abc123",
            client=client,
        )
        assert info["shortened"] is True
        assert recovery.FOOTER_MARKER in out
        assert out.startswith("short summary")
        cached = cache_mod.cache_file_path("abc123")
        assert cached.read_text() == LONG_CONTENT
        assert info["cache_path"] == str(cached.resolve())
        assert info["saved"] > 0

    def test_cache_content_differs_from_api_content(self, recovery, cache_mod):
        out, info = recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            cache_content="denumbered original",
            tool_name="read_file",
            cache_id="def456",
            client=FakeClient(),
        )
        assert info["shortened"] is True
        assert cache_mod.cache_file_path("def456").read_text() == "denumbered original"

    def test_api_call_arguments(self, recovery):
        client = FakeClient()
        recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="a1",
            client=client,
            model="toc_latte_v1",
            target_ratio=3.0,
        )
        call = client.calls[0]
        assert call["tool_output"] == LONG_CONTENT
        assert call["tool_name"] == "grep"
        assert call["compression_model_name"] == "toc_latte_v1"
        assert call["target_compression_ratio"] == 3.0
        assert call["source"] == "integration:hermes"

    def test_zero_ratio_sent_as_none(self, recovery):
        client = FakeClient()
        recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="a2",
            client=client,
            target_ratio=0.0,
        )
        assert client.calls[0]["target_compression_ratio"] is None

    def test_api_error_fails_open(self, recovery):
        out, info = recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="a3",
            client=FakeClient(error=RuntimeError("api down")),
        )
        assert out == LONG_CONTENT
        assert info["called_api"] is False
        assert "api down" in info["error"]
        assert info["shortened"] is False

    def test_unsuccessful_response_fails_open(self, recovery):
        resp = CompressToolOutputResponse(success=False, message="quota", data=None)
        out, info = recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="a4",
            client=FakeClient(response=resp),
        )
        assert out == LONG_CONTENT
        assert "quota" in info["error"]

    def test_list_output_fails_open(self, recovery):
        out, info = recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="a5",
            client=FakeClient(response=_response(compressed=["a", "b"])),
        )
        assert out == LONG_CONTENT
        assert "list" in info["error"]

    def test_whitespace_output_fails_open(self, recovery):
        out, info = recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="a6",
            client=FakeClient(response=_response(compressed="   \n  ")),
        )
        assert out == LONG_CONTENT
        assert info["error"] == "empty compressed output"

    def test_no_net_win_skips_without_error(self, recovery, cache_mod):
        out, info = recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="a7",
            client=FakeClient(response=_response(compressed=LONG_CONTENT[:-10])),
        )
        assert out == LONG_CONTENT
        assert info["skipped_reason"] == "not smaller"
        assert "error" not in info
        assert not cache_mod.cache_file_path("a7").exists()

    def test_cache_write_failure_fails_open(self, recovery, monkeypatch):
        monkeypatch.setattr(
            "compresr.integrations.hermes.cache.store_original",
            lambda *a, **k: None,
        )
        out, info = recovery.compress_with_recovery(
            query="q",
            content=LONG_CONTENT,
            tool_name="grep",
            cache_id="a8",
            client=FakeClient(),
        )
        assert out == LONG_CONTENT
        assert info["error"] == "cache write failed"

    def test_count_tokens_rounds_up(self, recovery):
        assert recovery.count_tokens("") == 0
        assert recovery.count_tokens("abcd") == 1
        assert recovery.count_tokens("abcde") == 2
