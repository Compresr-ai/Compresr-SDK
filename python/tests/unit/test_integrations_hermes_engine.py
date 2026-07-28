"""Unit tests for CompresrContextEngine (fake Hermes runtime)."""

from typing import Any, Dict, List, Optional

import pytest

from compresr.schemas import CompressResponse, CompressResult

API_KEY = "cmp_" + "e" * 32


def _result(compressed: str = "compressed body", **overrides: Any) -> CompressResult:
    fields: Dict[str, Any] = dict(
        compressed_context=compressed,
        original_tokens=1000,
        compressed_tokens=200,
        actual_compression_ratio=5.0,
        tokens_saved=800,
        duration_ms=120,
    )
    fields.update(overrides)
    return CompressResult(**fields)


class FakeClient:
    def __init__(
        self, response: Optional[CompressResponse] = None, error: Optional[Exception] = None
    ):
        self.response = response or CompressResponse(success=True, data=_result())
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def compress(self, **kwargs: Any) -> CompressResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def make_engine(hermes_env, monkeypatch):
    def _make(api_key: str = API_KEY, client: Optional[FakeClient] = None, **kwargs: Any):
        if api_key:
            monkeypatch.setenv("COMPRESR_API_KEY", api_key)
        else:
            monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
        engine_mod = hermes_env.load("engine")
        engine = engine_mod.CompresrContextEngine(**kwargs)
        if client is not None:
            engine._compresr_client = client
        return engine

    return _make


TURNS = [{"role": "user", "content": "long conversation " * 50}]


class TestConstruction:
    def test_defaults(self, make_engine):
        engine = make_engine()
        assert engine.name == "compresr"
        assert engine.compresr_model == "latte_v2"
        assert engine.compresr_timeout == 60
        assert engine.abort_on_summary_failure is True
        assert engine.is_available() is True

    def test_unavailable_without_key(self, make_engine):
        engine = make_engine(api_key="")
        assert engine.is_available() is False

    def test_env_overrides(self, make_engine, monkeypatch):
        monkeypatch.setenv("COMPRESR_MODEL", "latte_v1")
        monkeypatch.setenv("COMPRESR_TIMEOUT", "15")
        engine = make_engine()
        assert engine.compresr_model == "latte_v1"
        assert engine.compresr_timeout == 15

    def test_malformed_timeout_falls_back(self, make_engine, monkeypatch):
        monkeypatch.setenv("COMPRESR_TIMEOUT", "soon")
        engine = make_engine()
        assert engine.compresr_timeout == 60

    def test_update_model_passthrough(self, make_engine):
        engine = make_engine()
        engine.update_model("claude-x", 100_000)
        assert engine.update_model_calls[-1][0] == "claude-x"
        assert engine.context_length == 100_000


class TestRatioMapping:
    def test_override_wins(self, make_engine, monkeypatch):
        monkeypatch.setenv("COMPRESR_TARGET_RATIO", "7.5")
        engine = make_engine()
        assert engine._target_compression_ratio() == 7.5

    def test_keep_fraction_maps_to_nx(self, make_engine):
        engine = make_engine(summary_target_ratio=0.2)
        assert engine._target_compression_ratio() == 5.0

    def test_tiny_keep_fraction_clamped(self, make_engine):
        engine = make_engine(summary_target_ratio=0.001)
        assert engine._target_compression_ratio() == 100.0


class TestGenerateSummary:
    def test_success_returns_prefixed_summary(self, make_engine):
        client = FakeClient()
        engine = make_engine(client=client)
        out = engine._generate_summary(TURNS)
        assert out is not None and out.endswith("compressed body")
        assert engine._previous_summary == "compressed body"
        assert engine.compresr_calls == 1
        assert engine.compresr_tokens_saved == 800
        assert engine.compresr_last_duration_ms == 120

    def test_request_payload(self, make_engine):
        client = FakeClient()
        engine = make_engine(client=client)
        engine._generate_summary(TURNS, focus_topic="the fix")
        call = client.calls[0]
        assert call["query"] == "the fix"
        assert call["compression_model_name"] == "latte_v2"
        assert call["coarse"] is None
        assert call["source"] == "integration:hermes"

    def test_memory_context_kwarg_accepted_and_folded(self, make_engine):
        client = FakeClient()
        engine = make_engine(client=client)
        out = engine._generate_summary(TURNS, memory_context="remember the deploy key rotation")
        assert out is not None
        context = client.calls[0]["context"]
        assert "<memory-provider-context>" in context
        assert "deploy key rotation" in context

    def test_empty_memory_context_adds_nothing(self, make_engine):
        client = FakeClient()
        engine = make_engine(client=client)
        engine._generate_summary(TURNS, memory_context="")
        assert "<memory-provider-context>" not in client.calls[0]["context"]

    def test_client_built_eagerly_with_key(self, make_engine):
        from compresr import CompressionClient

        engine = make_engine()
        assert isinstance(engine._compresr_client, CompressionClient)

    def test_deepcopy_shares_client_and_isolates_state(self, make_engine):
        import copy

        from compresr import CompressionClient

        engine = make_engine()
        clone = copy.deepcopy(engine)
        assert isinstance(engine._compresr_client, CompressionClient)
        assert clone._compresr_client is engine._compresr_client
        clone.compresr_calls += 5
        assert engine.compresr_calls == 0

    def test_no_client_without_key(self, make_engine):
        engine = make_engine(api_key="")
        assert engine._compresr_client is None

    def test_coarse_forwarded_for_latte_v1(self, make_engine, monkeypatch):
        monkeypatch.setenv("COMPRESR_MODEL", "latte_v1")
        monkeypatch.setenv("COMPRESR_COARSE", "true")
        client = FakeClient()
        engine = make_engine(client=client)
        engine._generate_summary(TURNS)
        assert client.calls[0]["coarse"] is True

    def test_prior_summary_folded_into_context(self, make_engine):
        client = FakeClient()
        engine = make_engine(client=client)
        engine._previous_summary = "earlier facts"
        engine._generate_summary(TURNS)
        context = client.calls[0]["context"]
        assert "[PRIOR CONTEXT SUMMARY]" in context
        assert "earlier facts" in context

    def test_api_error_returns_none_and_records_cooldown(self, make_engine):
        client = FakeClient(error=RuntimeError("boom"))
        engine = make_engine(client=client)
        assert engine._generate_summary(TURNS) is None
        assert engine.compresr_errors == 1
        assert engine._summary_failure_cooldown_until > 0
        assert "boom" in engine._last_summary_error

    def test_empty_compressed_returns_none_and_records_cooldown(self, make_engine):
        client = FakeClient(response=CompressResponse(success=True, data=_result("   ")))
        engine = make_engine(client=client)
        assert engine._generate_summary(TURNS) is None
        assert engine.compresr_errors == 1

    def test_unsuccessful_response_treated_as_error(self, make_engine):
        client = FakeClient(response=CompressResponse(success=False, message="quota", data=None))
        engine = make_engine(client=client)
        assert engine._generate_summary(TURNS) is None
        assert engine.compresr_errors == 1

    def test_cooldown_skips_api_call(self, make_engine):
        client = FakeClient()
        engine = make_engine(client=client)
        engine._record_compression_failure_cooldown(30.0, "x")
        assert engine._generate_summary(TURNS) is None
        assert client.calls == []

    def test_empty_serialized_context_returns_none(self, make_engine):
        client = FakeClient()
        engine = make_engine(client=client)
        assert engine._generate_summary([{"role": "user", "content": "  "}]) is None
        assert client.calls == []

    def test_no_key_fails_without_calling_api(self, make_engine):
        client = FakeClient()
        engine = make_engine(api_key="", client=client)
        assert engine._generate_summary(TURNS) is None
        assert client.calls == []
        assert engine.compresr_errors == 1


class TestStatus:
    def test_status_extends_base(self, make_engine):
        engine = make_engine()
        status = engine.get_status()
        assert status["engine"] == "compresr"
        assert status["compresr_model"] == "latte_v2"
        assert status["compresr_calls"] == 0
