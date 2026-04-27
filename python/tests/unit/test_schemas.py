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
        req = CompressRequest(context="Test context", compression_model_name="espresso_v1")
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

    def test_high_ratio_passes(self):
        """Test that high ratios (e.g., 60x) pass SDK validation - backend enforces upper bound."""
        req = CompressRequest(
            context="Test",
            compression_model_name="espresso_v1",
            target_compression_ratio=60.0,
        )
        assert req.target_compression_ratio == 60.0

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
# Batch Compression Schema Tests
# =============================================================================


class TestCompressBatchInput:
    """Test CompressBatchInput schema."""

    def test_valid_input(self):
        """Test valid batch input."""
        from compresr.schemas import CompressBatchInput

        inp = CompressBatchInput(context="Test context", query="Test query?")
        assert inp.context == "Test context"
        assert inp.query == "Test query?"

    def test_empty_context_fails(self):
        """Test empty context fails validation."""
        from compresr.schemas import CompressBatchInput

        with pytest.raises(ValidationError):
            CompressBatchInput(context="", query="Test query?")

    def test_empty_query_fails(self):
        """Test empty query fails validation."""
        from compresr.schemas import CompressBatchInput

        with pytest.raises(ValidationError):
            CompressBatchInput(context="Test context", query="")


class TestCompressBatchRequest:
    """Test CompressBatchRequest schema."""

    def test_valid_request(self):
        """Test valid batch request."""
        from compresr.schemas import CompressBatchInput, CompressBatchRequest

        inputs = [
            CompressBatchInput(context="Context 1", query="Query 1"),
            CompressBatchInput(context="Context 2", query="Query 2"),
        ]
        req = CompressBatchRequest(
            inputs=inputs,
            compression_model_name="latte_v1",
        )
        assert len(req.inputs) == 2
        assert req.compression_model_name == "latte_v1"
        assert req.source == "sdk:python"

    def test_request_with_options(self):
        """Test batch request with optional parameters."""
        from compresr.schemas import CompressBatchInput, CompressBatchRequest

        inputs = [CompressBatchInput(context="C1", query="Q1")]
        req = CompressBatchRequest(
            inputs=inputs,
            compression_model_name="latte_v1",
            target_compression_ratio=0.5,
            coarse=True,
        )
        assert req.target_compression_ratio == 0.5
        assert req.coarse is True

    def test_empty_inputs_fails(self):
        """Test empty inputs list fails."""
        from compresr.schemas import CompressBatchRequest

        with pytest.raises(ValidationError):
            CompressBatchRequest(
                inputs=[],
                compression_model_name="latte_v1",
            )

    def test_max_inputs_limit(self):
        """Test maximum 100 inputs limit."""
        from compresr.schemas import CompressBatchInput, CompressBatchRequest

        inputs = [CompressBatchInput(context=f"C{i}", query=f"Q{i}") for i in range(101)]
        with pytest.raises(ValidationError):
            CompressBatchRequest(
                inputs=inputs,
                compression_model_name="latte_v1",
            )


class TestCompressBatchItemResult:
    """Test CompressBatchItemResult schema."""

    def test_valid_item_result(self):
        """Test valid batch item result."""
        from compresr.schemas import CompressBatchItemResult

        result = CompressBatchItemResult(
            original_context="Original",
            compressed_context="Compressed",
            original_tokens=100,
            compressed_tokens=50,
            actual_compression_ratio=0.5,
            tokens_saved=50,
            duration_ms=100,
        )
        assert result.original_context == "Original"
        assert result.compressed_context == "Compressed"
        assert result.tokens_saved == 50


class TestCompressBatchResult:
    """Test CompressBatchResult schema."""

    def test_valid_batch_result(self):
        """Test valid batch result with items."""
        from compresr.schemas import CompressBatchItemResult, CompressBatchResult

        items = [
            CompressBatchItemResult(
                original_context="Orig1",
                compressed_context="Comp1",
                original_tokens=100,
                compressed_tokens=50,
                actual_compression_ratio=0.5,
                tokens_saved=50,
                duration_ms=50,
            ),
            CompressBatchItemResult(
                original_context="Orig2",
                compressed_context="Comp2",
                original_tokens=80,
                compressed_tokens=40,
                actual_compression_ratio=0.5,
                tokens_saved=40,
                duration_ms=40,
            ),
        ]
        result = CompressBatchResult(
            results=items,
            total_original_tokens=180,
            total_compressed_tokens=90,
            total_tokens_saved=90,
            average_compression_ratio=0.5,
            count=2,
        )
        assert result.count == 2
        assert result.total_original_tokens == 180
        assert result.total_compressed_tokens == 90
        assert len(result.results) == 2

    def test_empty_batch_result_defaults(self):
        """Test batch result with default values."""
        from compresr.schemas import CompressBatchResult

        result = CompressBatchResult()
        assert result.count == 0
        assert result.total_original_tokens == 0
        assert result.results == []


class TestCompressBatchResponse:
    """Test CompressBatchResponse schema."""

    def test_valid_response(self):
        """Test valid batch response."""
        from compresr.schemas import (
            CompressBatchItemResult,
            CompressBatchResponse,
            CompressBatchResult,
        )

        data = CompressBatchResult(
            results=[
                CompressBatchItemResult(
                    original_context="O",
                    compressed_context="C",
                    original_tokens=10,
                    compressed_tokens=5,
                    actual_compression_ratio=0.5,
                    tokens_saved=5,
                    duration_ms=10,
                )
            ],
            total_original_tokens=10,
            total_compressed_tokens=5,
            total_tokens_saved=5,
            average_compression_ratio=0.5,
            count=1,
        )
        response = CompressBatchResponse(success=True, data=data)
        assert response.success is True
        assert response.data.count == 1

    def test_error_response(self):
        """Test batch error response."""
        from compresr.schemas import CompressBatchResponse

        response = CompressBatchResponse(success=False, message="Batch failed")
        assert response.success is False
        assert response.message == "Batch failed"
        assert response.data is None


# =============================================================================
# Hard Edge Case Tests
# =============================================================================


class TestContextTypeHandling:
    """Test context type handling in schemas (context is now str only)."""

    def test_compress_request_rejects_list_context(self):
        """Test CompressRequest rejects list of strings (use batch endpoint)."""
        with pytest.raises(ValidationError):
            CompressRequest(
                context=["Context 1", "Context 2", "Context 3"],  # type: ignore[arg-type]
                compression_model_name="espresso_v1",
            )

    def test_compress_request_single_string(self):
        """Test CompressRequest with single string context."""
        req = CompressRequest(
            context="Single context string",
            compression_model_name="espresso_v1",
        )
        assert isinstance(req.context, str)

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
        """Test compression ratio at minimum allowed value (0.0)."""
        req = CompressRequest(
            context="Test",
            compression_model_name="espresso_v1",
            target_compression_ratio=0.0,
        )
        assert req.target_compression_ratio == 0.0

    def test_compression_ratio_high_value_allowed(self):
        """Test high compression ratios are allowed (backend enforces 200 max)."""
        req = CompressRequest(
            context="Test",
            compression_model_name="espresso_v1",
            target_compression_ratio=100.0,
        )
        assert req.target_compression_ratio == 100.0

    def test_compression_ratio_negative_fails(self):
        """Test negative compression ratio fails."""
        with pytest.raises(ValidationError):
            CompressRequest(
                context="Test",
                compression_model_name="espresso_v1",
                target_compression_ratio=-0.1,
            )

    def test_compression_ratio_very_high_passes(self):
        """Test very high ratios pass SDK - backend returns error for >200."""
        req = CompressRequest(
            context="Test",
            compression_model_name="espresso_v1",
            target_compression_ratio=150.0,
        )
        assert req.target_compression_ratio == 150.0

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
