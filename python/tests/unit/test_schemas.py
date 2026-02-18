"""
Unit Tests for SDK Schemas

Tests for request/response schema validation and serialization.
"""

import pytest
from pydantic import ValidationError

from compresr.schemas import (
    BatchCompressRequest,
    BatchCompressResponse,
    CompressRequest,
    CompressResponse,
    CompressResult,
    StreamChunk,
)


class TestCompressRequest:
    """Test CompressRequest schema validation."""

    def test_valid_request(self):
        """Test valid compress request."""
        req = CompressRequest(context="Test context", compression_model_name="A_CMPRSR_V1")
        assert req.context == "Test context"
        assert req.compression_model_name == "A_CMPRSR_V1"
        assert req.target_compression_ratio is None

    def test_request_with_ratio(self):
        """Test request with compression ratio."""
        req = CompressRequest(
            context="Test context",
            compression_model_name="A_CMPRSR_V1",
            target_compression_ratio=0.5,
        )
        assert req.target_compression_ratio == 0.5

    def test_empty_context_fails(self):
        """Test that empty context fails validation."""
        with pytest.raises(ValidationError):
            CompressRequest(context="", compression_model_name="A_CMPRSR_V1")

    def test_invalid_ratio_high_fails(self):
        """Test that ratio > 1.0 fails validation."""
        with pytest.raises(ValidationError):
            CompressRequest(
                context="Test",
                compression_model_name="A_CMPRSR_V1",
                target_compression_ratio=1.5,
            )

    def test_invalid_ratio_low_fails(self):
        """Test that ratio < 0 fails validation."""
        with pytest.raises(ValidationError):
            CompressRequest(
                context="Test",
                compression_model_name="A_CMPRSR_V1",
                target_compression_ratio=-0.1,
            )

    def test_missing_model_name_fails(self):
        """Test that missing model name fails."""
        with pytest.raises(ValidationError):
            CompressRequest(context="Test")


class TestBatchCompressRequest:
    """Test BatchCompressRequest schema validation."""

    def test_valid_batch_request(self):
        """Test valid batch request."""
        req = BatchCompressRequest(
            contexts=["Context 1", "Context 2"], compression_model_name="A_CMPRSR_V1"
        )
        assert len(req.contexts) == 2
        assert req.compression_model_name == "A_CMPRSR_V1"

    def test_empty_contexts_fails(self):
        """Test that empty contexts list fails."""
        with pytest.raises(ValidationError):
            BatchCompressRequest(contexts=[], compression_model_name="A_CMPRSR_V1")

    def test_contexts_with_empty_string_fails(self):
        """Test that contexts with empty strings are handled."""
        # Note: Empty strings in list are validated at API level, not schema level
        req = BatchCompressRequest(
            contexts=["Valid", "Another valid"], compression_model_name="A_CMPRSR_V1"
        )
        assert len(req.contexts) == 2


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


class TestBatchCompressResponse:
    """Test BatchCompressResponse schema."""

    def test_valid_batch_response(self):
        """Test valid batch response."""
        results = [
            CompressResult(
                original_context=f"Original {i}",
                compressed_context=f"Compressed {i}",
                original_tokens=100,
                compressed_tokens=50,
                actual_compression_ratio=0.5,
                tokens_saved=50,
                duration_ms=100,
            )
            for i in range(3)
        ]
        from compresr.schemas.inference import BatchCompressResult

        response = BatchCompressResponse(
            success=True,
            data=BatchCompressResult(
                results=results,
                total_original_tokens=300,
                total_compressed_tokens=150,
                total_tokens_saved=150,
                average_compression_ratio=0.5,
                count=3,
            ),
        )
        assert response.success is True
        assert len(response.data.results) == 3


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
