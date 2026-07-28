"""Unit tests for the ToolOutputCompressor hook (fake Hermes runtime)."""

import json
from typing import Any, Dict, List, Optional

import pytest

from compresr.schemas import CompressToolOutputResponse, CompressToolOutputResult

API_KEY = "cmp_" + "t" * 32
BIG = "relevant fact\n" + ("filler content for the tool output\n" * 400)


def _response(compressed: str = "tiny summary") -> CompressToolOutputResponse:
    return CompressToolOutputResponse(
        success=True,
        data=CompressToolOutputResult(
            compressed_output=compressed,
            original_tokens=3000,
            compressed_tokens=10,
            compression_ratio=300.0,
            tool_name="grep",
            duration_ms=40,
        ),
    )


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
def make_hook(hermes_env, monkeypatch):
    def _make(
        api_key: str = API_KEY,
        enabled: bool = True,
        client: Optional[FakeClient] = None,
        **env: str,
    ):
        if api_key:
            monkeypatch.setenv("COMPRESR_API_KEY", api_key)
        if enabled:
            monkeypatch.setenv("COMPRESR_TOOL_OUTPUT_ENABLED", "1")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        mod = hermes_env.load("tool_output")
        hook = mod.ToolOutputCompressor()
        hook._client = client or FakeClient()
        return hook

    return _make


class TestGating:
    def test_inactive_without_key(self, make_hook):
        hook = make_hook(api_key="")
        assert hook.active is False
        assert hook.on_transform_tool_result(tool_name="grep", result=BIG) is None

    def test_inactive_when_disabled(self, make_hook):
        hook = make_hook(enabled=False)
        assert hook.active is False
        assert hook.on_transform_tool_result(tool_name="grep", result=BIG) is None

    def test_non_string_result_ignored(self, make_hook):
        hook = make_hook()
        assert hook.on_transform_tool_result(tool_name="grep", result={"a": 1}) is None

    def test_error_status_ignored(self, make_hook):
        hook = make_hook()
        assert hook.on_transform_tool_result(tool_name="grep", result=BIG, status="error") is None

    def test_already_compressed_output_ignored(self, make_hook, hermes_env):
        recovery = hermes_env.load("recovery")
        hook = make_hook()
        marked = BIG + f"\n{recovery.FOOTER_MARKER} cached at /x"
        assert hook.on_transform_tool_result(tool_name="grep", result=marked) is None

    def test_small_output_ignored(self, make_hook):
        hook = make_hook()
        client = hook._client
        assert hook.on_transform_tool_result(tool_name="grep", result="short") is None
        assert client.calls == []

    def test_recovery_read_passthrough_counts(self, make_hook):
        hook = make_hook()
        args = {"file_path": "/root/.hermes/cache/compresr/tool-output/abc"}
        assert hook.on_transform_tool_result(tool_name="read_file", args=args, result=BIG) is None
        assert hook.recoveries == 1

    def test_cooldown_suppresses_calls(self, make_hook):
        hook = make_hook(client=FakeClient(error=RuntimeError("down")))
        assert hook.on_transform_tool_result(tool_name="grep", result=BIG) is None
        assert hook.errors == 1
        hook._client = FakeClient()
        assert hook.on_transform_tool_result(tool_name="grep", result=BIG) is None
        assert hook._client.calls == []

    def test_unrecoverable_long_line_ignored(self, make_hook, hermes_env):
        hermes_env.reinstall(max_line_length=100)
        hook = make_hook()
        long_line = "x" * 200 + "\n" + BIG
        assert hook.on_transform_tool_result(tool_name="grep", result=long_line) is None
        assert hook._client.calls == []


class TestQueryDerivation:
    def test_query_arg_used(self, make_hook):
        hook = make_hook()
        q = hook._derive_query("search_files", {"pattern": "TODO items"})
        assert q == "search_files: TODO items"

    def test_path_arg_used(self, make_hook):
        hook = make_hook()
        q = hook._derive_query("read_file", {"file_path": "/src/main.py"})
        assert "/src/main.py" in q

    def test_fallback_to_tool_name(self, make_hook):
        hook = make_hook()
        assert "terminal" in hook._derive_query("terminal", {})

    def test_fallback_without_tool_name(self, make_hook):
        hook = make_hook()
        assert hook._derive_query("", None).startswith("Preserve the facts")


class TestCompression:
    def test_success_returns_compressed_with_footer(self, make_hook, hermes_env):
        recovery = hermes_env.load("recovery")
        hook = make_hook()
        out = hook.on_transform_tool_result(
            tool_name="grep", args={"pattern": "needle"}, result=BIG
        )
        assert out is not None
        assert out.startswith("tiny summary")
        assert recovery.FOOTER_MARKER in out
        assert hook.calls == 1
        assert hook.tokens_saved > 0
        assert hook._client.calls[0]["query"] == "grep: needle"

    def test_json_envelope_spliced(self, make_hook):
        hook = make_hook()
        envelope = json.dumps({"content": BIG, "path": "/f"})
        out = hook.on_transform_tool_result(tool_name="read_file", result=envelope)
        assert out is not None
        parsed = json.loads(out)
        assert parsed["path"] == "/f"
        assert parsed["content"].startswith("tiny summary")

    def test_guttered_read_file_cached_denumbered(self, make_hook, hermes_env):
        cache_mod = hermes_env.load("cache")
        hook = make_hook()
        inner = "\n".join(f"{i}|line {i} of the file body content here" for i in range(400))
        envelope = json.dumps({"content": inner})
        out = hook.on_transform_tool_result(tool_name="read_file", result=envelope)
        assert out is not None
        cached_files = [p for p in cache_mod.get_cache_root().iterdir() if p.is_file()]
        assert len(cached_files) == 1
        assert cached_files[0].read_text().startswith("line 0")

    def test_api_failure_leaves_original(self, make_hook):
        hook = make_hook(client=FakeClient(error=RuntimeError("boom")))
        assert hook.on_transform_tool_result(tool_name="grep", result=BIG) is None
        assert hook.errors == 1
        assert hook.calls == 0

    def test_no_net_win_leaves_original_without_error(self, make_hook):
        hook = make_hook(client=FakeClient(response=_response(compressed=BIG[:-5])))
        assert hook.on_transform_tool_result(tool_name="grep", result=BIG) is None
        assert hook.errors == 0
        assert hook._cooldown_until == 0.0

    def test_redaction_applied_to_outbound(self, make_hook, hermes_env):
        hermes_env.reinstall(redact=lambda s: s.replace("SECRET", "[REDACTED]"))
        hook = make_hook()
        out = hook.on_transform_tool_result(
            tool_name="grep", args={"pattern": "SECRET"}, result="SECRET " + BIG
        )
        assert out is not None
        call = hook._client.calls[0]
        assert "SECRET" not in call["query"]
        assert "SECRET" not in call["tool_output"]


class TestClientConstruction:
    def test_client_built_eagerly_with_key(self, hermes_env, monkeypatch):
        from compresr import CompressionClient

        monkeypatch.setenv("COMPRESR_API_KEY", API_KEY)
        mod = hermes_env.load("tool_output")
        hook = mod.ToolOutputCompressor()
        assert isinstance(hook._client, CompressionClient)

    def test_no_client_without_key(self, hermes_env):
        mod = hermes_env.load("tool_output")
        hook = mod.ToolOutputCompressor()
        assert hook._client is None

    def test_explicit_zero_target_ratio_passed_through(self, make_hook):
        hook = make_hook(COMPRESR_TOOL_OUTPUT_TARGET_RATIO="0")
        hook.on_transform_tool_result(tool_name="grep", result=BIG)
        assert hook._client.calls[0]["target_compression_ratio"] is None


class TestStatus:
    def test_status_shape(self, make_hook):
        hook = make_hook()
        status = hook.get_status()
        assert status["active"] is True
        assert status["model"] == "toc_latte_v2"
        assert status["calls"] == 0


def test_fallback_redactor_redacts_secrets_and_pii():
    """When agent.redact is unavailable, the built-in fallback must not pass
    secrets/PII through unchanged (Greptile P1: no identity fallback)."""
    from compresr.integrations.hermes.tool_output import _redact

    out = _redact("token sk_live_ABCD1234EFGH5678 user a.b@corp.com key cmp_abcdef0123456789xy")
    assert "sk_live_ABCD1234EFGH5678" not in out
    assert "a.b@corp.com" not in out
    assert "cmp_abcdef0123456789xy" not in out
    assert "[REDACTED]" in out


def test_fallback_redactor_covers_pem_google_and_conn_strings():
    """Fallback must also mask credential formats outside the finite key list:
    PEM private keys, Google API keys/OAuth tokens, and connection-string creds
    (Greptile P1: fallback still leaks secrets)."""
    from compresr.integrations.hermes.tool_output import _redact

    pem = "-----BEGIN RSA PRIVATE KEY-----\n" "MIIEowIBAAKCAQEAoat3\n-----END RSA PRIVATE KEY-----"
    assert "PRIVATE KEY" not in _redact(pem).replace("[REDACTED]", "")
    assert "AIza" not in _redact("key AIzaabcdefghijklmnopqrstuvwxyz012345678 end")
    assert "ya29." not in _redact("tok ya29.a0AfH6SMBxyz1234567890abcdefg end")
    masked = _redact("postgres://admin:s3cr3tpw@db.internal:5432/app")
    assert "s3cr3tpw" not in masked
    assert "db.internal" in masked  # host preserved, only user:pass masked
    # Password-only URI (empty username), e.g. Redis (Greptile follow-up P1).
    redis = _redact("redis://:topsecret@cache.internal:6379/0")
    assert "topsecret" not in redis
    assert "cache.internal" in redis
