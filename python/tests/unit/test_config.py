"""
Unit Tests for SDK Configuration

Tests for configuration classes and constants.
"""

import pytest

from compresr.config import (
    AGNOSTIC_ENDPOINT_MODELS,
    ALLOWED_COMPRESSION_MODELS,
    ALLOWED_FILTER_MODELS,
    APIConfig,
    Endpoints,
    Headers,
    QS_ENDPOINT_MODELS,
    QUERY_REQUIRED_MODELS,
    StatusCodes,
)


class TestAPIConfig:
    """Test APIConfig dataclass."""

    def test_default_values(self):
        """Test API config has correct defaults."""
        config = APIConfig()
        assert config.API_KEY_PREFIX == "cmp_"
        assert config.DEFAULT_TIMEOUT == 60
        assert config.STREAM_TIMEOUT == 300

    def test_frozen_dataclass(self):
        """Test that APIConfig is frozen (immutable)."""
        config = APIConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.DEFAULT_TIMEOUT = 999


class TestEndpoints:
    """Test Endpoints configuration."""

    def test_endpoint_paths_agnostic(self):
        """Test agnostic endpoint paths are correct."""
        endpoints = Endpoints()
        assert endpoints.COMPRESS_AGNOSTIC == "/api/compress/question-agnostic/"
        assert endpoints.COMPRESS_AGNOSTIC_STREAM == "/api/compress/question-agnostic/stream"

    def test_endpoint_paths_question_specific(self):
        """Test query-specific endpoint paths are correct."""
        endpoints = Endpoints()
        assert endpoints.COMPRESS_QS == "/api/compress/question-specific/"
        assert endpoints.COMPRESS_QS_STREAM == "/api/compress/question-specific/stream"

    def test_endpoints_start_with_api(self):
        """Test all endpoints start with /api/."""
        endpoints = Endpoints()
        for name, value in endpoints.__dict__.items():
            if not name.startswith("_"):
                assert value.startswith("/api/"), f"{name} endpoint doesn't start with /api/"

    def test_frozen_dataclass(self):
        """Test that Endpoints is frozen."""
        endpoints = Endpoints()
        with pytest.raises(Exception):
            endpoints.COMPRESS_AGNOSTIC = "/new/path"


class TestModelGroups:
    """Test model group constants."""

    def test_compression_models(self):
        """Test allowed compression models."""
        assert "espresso_v1" in ALLOWED_COMPRESSION_MODELS
        assert "latte_v1" in ALLOWED_COMPRESSION_MODELS
        assert len(ALLOWED_COMPRESSION_MODELS) == 2

    def test_filter_models(self):
        """Test allowed filter models."""
        assert "coldbrew_v1" in ALLOWED_FILTER_MODELS
        assert len(ALLOWED_FILTER_MODELS) == 1

    def test_query_required_models(self):
        """Test query required models."""
        assert "latte_v1" in QUERY_REQUIRED_MODELS
        assert "coldbrew_v1" in QUERY_REQUIRED_MODELS
        assert "espresso_v1" not in QUERY_REQUIRED_MODELS

    def test_agnostic_endpoint_models(self):
        """Test agnostic endpoint models."""
        assert "espresso_v1" in AGNOSTIC_ENDPOINT_MODELS
        assert "latte_v1" not in AGNOSTIC_ENDPOINT_MODELS

    def test_qs_endpoint_models(self):
        """Test QS endpoint models."""
        assert "latte_v1" in QS_ENDPOINT_MODELS
        assert "coldbrew_v1" in QS_ENDPOINT_MODELS
        assert "espresso_v1" not in QS_ENDPOINT_MODELS


class TestHeaders:
    """Test Headers configuration."""

    def test_header_names(self):
        """Test header names are correct."""
        headers = Headers()
        assert headers.API_KEY == "X-API-Key"
        assert headers.CONTENT_TYPE == "Content-Type"
        assert headers.ACCEPT == "Accept"
        assert headers.JSON == "application/json"
        assert headers.SSE == "text/event-stream"

    def test_json_content_type(self):
        """Test JSON content type is correct."""
        headers = Headers()
        assert headers.JSON == "application/json"

    def test_sse_content_type(self):
        """Test SSE content type for streaming."""
        headers = Headers()
        assert headers.SSE == "text/event-stream"


class TestStatusCodes:
    """Test StatusCodes configuration."""

    def test_success_codes(self):
        """Test success status codes."""
        codes = StatusCodes()
        assert codes.OK == 200

    def test_client_error_codes(self):
        """Test client error codes."""
        codes = StatusCodes()
        assert codes.BAD_REQUEST == 400
        assert codes.UNAUTHORIZED == 401
        assert codes.FORBIDDEN == 403
        assert codes.NOT_FOUND == 404
        assert codes.VALIDATION_ERROR == 422
        assert codes.RATE_LIMITED == 429

    def test_server_error_codes(self):
        """Test server error codes."""
        codes = StatusCodes()
        assert codes.SERVER_ERROR == 500

    def test_all_codes_are_integers(self):
        """Test all status codes are integers."""
        codes = StatusCodes()
        for name, value in codes.__dict__.items():
            if not name.startswith("_"):
                assert isinstance(value, int), f"{name} is not an integer"
                assert 100 <= value < 600, f"{name} is not a valid HTTP status code"
