"""
Unit Tests for SDK Schemas

Tests for request/response schema validation and serialization.
"""

import pytest
from pydantic import ValidationError

from compresr.schemas import (
    CompressRequest,
    CompressResponse,
    CompressResult,
    StreamChunk,
)


class TestCompressRequest:
    """Test CompressRequest schema validation."""

    def test_valid_request(self):
        """Test valid compress request."""
        req = CompressRequest(
            context="Test context", compression_model_name="espresso_v1"
        )
        assert req.context == "Test context"
        assert req.compression_model_name == "espresso_v1"
        assert req.target_compression_ratio is None

    def test_request_with_ratio(self):
        """Test request with compression ratio."""
        req = CompressRequest(
            context="Test context",
            compression_model_name="espresso_v1",
            target_compression_ratio=0.5,
        )
        assert req.target_compression_ratio == 0.5

    def test_empty_context_fails(self):
        """Test that empty context fails validation at SDK level."""
        with pytest.raises(ValidationError):
            CompressRequest(context="", compression_model_name="espresso_v1")

    def test_invalid_ratio_high_fails(self):
        """Test that ratio > 1.0 fails validation."""
        with pytest.raises(ValidationError):
            CompressRequest(
                context="Test",
                compression_model_name="espresso_v1",
                target_compression_ratio=1.5,
            )

    def test_invalid_ratio_low_fails(self):
        """Test that ratio < 0 fails validation."""
        with pytest.raises(ValidationError):
            CompressRequest(
                context="Test",
                compression_model_name="espresso_v1",
                target_compression_ratio=-0.1,
            )

    def test_missing_model_name_fails(self):
        """Test that missing model name fails."""
        with pytest.raises(ValidationError):
            CompressRequest(context="Test")


class TestCompressResult:
    """Test CompressResult schema."""

    def test_valid_result(self):
        """Test valid compressed result."""
        result = CompressResult(
            original_context="Original text",
            compressed_context="Compressed text",
            original_tokens=100,
            compressed_tokens=50,
            actual_compression_ratio=0.5,
            tokens_saved=50,
            duration_ms=100,
        )
        assert result.compressed_context == "Compressed text"
        assert result.original_context == "Original text"
        assert result.original_tokens == 100
        assert result.compressed_tokens == 50
        assert result.actual_compression_ratio == 0.5
        assert result.tokens_saved == 50
        assert result.duration_ms == 100

    def test_result_with_optional_fields(self):
        """Test result with optional target ratio."""
        result = CompressResult(
            original_context="Original",
            compressed_context="Compressed",
            original_tokens=100,
            compressed_tokens=50,
            actual_compression_ratio=0.5,
            tokens_saved=50,
            duration_ms=100,
            target_compression_ratio=0.5,
        )
        assert result.target_compression_ratio == 0.5


class TestCompressResponse:
    """Test CompressResponse schema."""

    def test_valid_response(self):
        """Test valid compress response."""
        response = CompressResponse(
            success=True,
            data=CompressResult(
                original_context="Original",
                compressed_context="Compressed",
                original_tokens=100,
                compressed_tokens=50,
                actual_compression_ratio=0.5,
                tokens_saved=50,
                duration_ms=100,
            ),
        )
        assert response.success is True
        assert response.data.compressed_tokens == 50

    def test_error_response(self):
        """Test error response."""
        response = CompressResponse(success=False, message="Compression failed")
        assert response.success is False
        assert response.message == "Compression failed"
        assert response.data is None


class TestStreamChunk:
    """Test StreamChunk schema."""

    def test_content_chunk(self):
        """Test stream chunk with content."""
        chunk = StreamChunk(content="Partial text", done=False)
        assert chunk.content == "Partial text"
        assert chunk.done is False

    def test_done_chunk(self):
        """Test final stream chunk."""
        chunk = StreamChunk(content="Final text", done=True)
        assert chunk.done is True
        assert chunk.content == "Final text"

    def test_chunk_serialization(self):
        """Test chunk can be serialized to dict."""
        chunk = StreamChunk(content="Test", done=False)
        data = chunk.model_dump()
        assert data["content"] == "Test"
        assert data["done"] is False


# =============================================================================
# Hard Edge Case Tests
# =============================================================================


class TestUnionTypeHandling:
    """Test Union[str, List[str]] type handling in schemas."""

    def test_compress_request_with_list_context(self):
        """Test CompressRequest accepts list of strings."""
        req = CompressRequest(
            context=["Context 1", "Context 2", "Context 3"],
            compression_model_name="espresso_v1",
        )
        assert isinstance(req.context, list)
        assert len(req.context) == 3

    def test_compress_request_single_string(self):
        """Test CompressRequest with single string context."""
        req = CompressRequest(
            context="Single context string",
            compression_model_name="espresso_v1",
        )
        assert isinstance(req.context, str)

    def test_compress_result_with_list_compressed_context(self):
        """Test CompressResult with list compressed_context."""
        result = CompressResult(
            original_context=["A", "B"],
            compressed_context=["A compressed", "B compressed"],
            original_tokens=100,
            compressed_tokens=50,
            actual_compression_ratio=0.5,
            tokens_saved=50,
            duration_ms=100,
        )
        assert isinstance(result.compressed_context, list)
        assert len(result.compressed_context) == 2

    def test_compress_result_string_compressed_context(self):
        """Test CompressResult with string compressed_context."""
        result = CompressResult(
            original_context="Original",
            compressed_context="Compressed",
            original_tokens=100,
            compressed_tokens=50,
            actual_compression_ratio=0.5,
            tokens_saved=50,
            duration_ms=100,
        )
        assert isinstance(result.compressed_context, str)


class TestBoundaryValidations:
    """Test boundary value validations."""

    def test_compression_ratio_at_minimum_boundary(self):
        """Test compression ratio at minimum allowed value (0.1)."""
        req = CompressRequest(
            context="Test",
            compression_model_name="espresso_v1",
            target_compression_ratio=0.1,
        )
        assert req.target_compression_ratio == 0.1

    def test_compression_ratio_at_maximum_boundary(self):
        """Test compression ratio at maximum allowed value (0.9)."""
        req = CompressRequest(
            context="Test",
            compression_model_name="espresso_v1",
            target_compression_ratio=0.9,
        )
        assert req.target_compression_ratio == 0.9

    def test_compression_ratio_below_minimum_fails(self):
        """Test compression ratio below 0.1 fails."""
        with pytest.raises(ValidationError):
            CompressRequest(
                context="Test",
                compression_model_name="espresso_v1",
                target_compression_ratio=0.05,
            )

    def test_compression_ratio_above_maximum_fails(self):
        """Test compression ratio above 0.9 fails."""
        with pytest.raises(ValidationError):
            CompressRequest(
                context="Test",
                compression_model_name="espresso_v1",
                target_compression_ratio=0.95,
            )

    def test_query_min_length_validation(self):
        """Test query minimum length validation (min_length=1)."""
        req = CompressRequest(
            context="Test",
            compression_model_name="latte_v1",
            query="Q",  # Single char should pass
        )
        assert req.query == "Q"


class TestComplexSerialization:
    """Test complex serialization scenarios."""

    def test_response_full_round_trip(self):
        """Test full response serialization/deserialization."""
        result = CompressResult(
            original_context="Original long text here",
            compressed_context="Compressed text",
            original_tokens=150,
            compressed_tokens=75,
            actual_compression_ratio=0.5,
            tokens_saved=75,
            duration_ms=250,
            target_compression_ratio=0.5,
        )
        response = CompressResponse(success=True, data=result)

        # Serialize
        data = response.model_dump()

        # Deserialize
        reconstructed = CompressResponse(**data)

        assert reconstructed.success == response.success
        assert reconstructed.data.compressed_tokens == response.data.compressed_tokens
        assert reconstructed.data.original_context == response.data.original_context
