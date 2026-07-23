"""Unit tests for SDK configuration."""

import pytest

from compresr import CompressionClient
from compresr.config import API_CONFIG, APIConfig, Endpoints, Headers, StatusCodes


class TestAPIConfig:
    def test_default_values(self):
        config = APIConfig()
        assert config.API_KEY_PREFIX == "cmp_"
        assert config.DEFAULT_TIMEOUT == 300

    def test_no_stream_timeout_attr(self):
        config = APIConfig()
        assert not hasattr(config, "STREAM_TIMEOUT")

    def test_frozen_dataclass(self):
        config = APIConfig()
        with pytest.raises(Exception):
            config.DEFAULT_TIMEOUT = 999


class TestEndpoints:
    def test_question_specific_paths(self):
        endpoints = Endpoints()
        assert endpoints.COMPRESS == "/api/compress/question-specific/"
        assert endpoints.COMPRESS_STREAM == "/api/compress/question-specific/stream"
        assert endpoints.COMPRESS_BATCH == "/api/compress/question-specific/batch"

    def test_endpoints_start_with_api(self):
        endpoints = Endpoints()
        for name, value in endpoints.__dict__.items():
            if not name.startswith("_"):
                assert value.startswith("/api/")

    def test_frozen_dataclass(self):
        endpoints = Endpoints()
        with pytest.raises(Exception):
            endpoints.COMPRESS = "/new/path"


class TestHeaders:
    def test_header_names(self):
        headers = Headers()
        assert headers.API_KEY == "X-API-Key"
        assert headers.CONTENT_TYPE == "Content-Type"
        assert headers.ACCEPT == "Accept"
        assert headers.JSON == "application/json"
        assert headers.SSE == "text/event-stream"


class TestStatusCodes:
    def test_success_codes(self):
        codes = StatusCodes()
        assert codes.OK == 200

    def test_client_error_codes(self):
        codes = StatusCodes()
        assert codes.BAD_REQUEST == 400
        assert codes.UNAUTHORIZED == 401
        assert codes.FORBIDDEN == 403
        assert codes.NOT_FOUND == 404
        assert codes.VALIDATION_ERROR == 422
        assert codes.RATE_LIMITED == 429

    def test_server_error_codes(self):
        codes = StatusCodes()
        assert codes.SERVER_ERROR == 500

    def test_all_codes_are_valid_http(self):
        codes = StatusCodes()
        for name, value in codes.__dict__.items():
            if not name.startswith("_"):
                assert isinstance(value, int)
                assert 100 <= value < 600


class TestClientTimeoutWiring:
    """The 5-minute default must reach the HTTP transport, and a
    per-call override must replace it. Regression guard against the
    previous bug where ``STREAM_TIMEOUT`` was set in config but never
    referenced, leaving the streaming path on the default."""

    def test_default_timeout_is_five_minutes(self):
        assert API_CONFIG.DEFAULT_TIMEOUT == 300

    def test_client_uses_default_timeout(self):
        client = CompressionClient(api_key="cmp_test_key")
        assert client._timeout == API_CONFIG.DEFAULT_TIMEOUT

    def test_client_honours_explicit_timeout(self):
        client = CompressionClient(api_key="cmp_test_key", timeout=42)
        assert client._timeout == 42
