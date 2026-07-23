"""Unit tests for SDK schemas."""

import pytest
from pydantic import ValidationError

from compresr.schemas import (
    CompressBatchInput,
    CompressBatchItemResult,
    CompressBatchRequest,
    CompressBatchResponse,
    CompressBatchResult,
    CompressRequest,
    CompressResponse,
    CompressResult,
    StreamChunk,
)


class TestCompressRequest:
    def test_valid_request_with_query(self):
        req = CompressRequest(context="Test context", query="What?")
        assert req.context == "Test context"
        assert req.query == "What?"
        assert req.compression_model_name == "latte_v1"

    def test_query_is_optional(self):
        # Optional client-side; backend validates per model.
        req = CompressRequest(context="Test context")
        assert req.query is None

    def test_request_with_ratio(self):
        req = CompressRequest(context="Test", query="q", target_compression_ratio=0.5)
        assert req.target_compression_ratio == 0.5

    def test_empty_context_fails(self):
        with pytest.raises(ValidationError):
            CompressRequest(context="", query="q")

    def test_empty_query_fails(self):
        # If supplied, query can't be empty.
        with pytest.raises(ValidationError):
            CompressRequest(context="Test", query="")

    def test_arbitrary_model_name_passes(self):
        # SDK is permissive — backend validates model names.
        req = CompressRequest(context="Test", query="q", compression_model_name="some_future_v3")
        assert req.compression_model_name == "some_future_v3"

    def test_high_ratio_passes(self):
        req = CompressRequest(context="Test", query="q", target_compression_ratio=60.0)
        assert req.target_compression_ratio == 60.0

    def test_invalid_ratio_low_fails(self):
        with pytest.raises(ValidationError):
            CompressRequest(context="Test", query="q", target_compression_ratio=-0.1)


class TestCompressResult:
    def test_valid_result(self):
        result = CompressResult(
            original_context="Original",
            compressed_context="Compressed",
            original_tokens=100,
            compressed_tokens=50,
            actual_compression_ratio=0.5,
            tokens_saved=50,
            duration_ms=100,
        )
        assert result.compressed_context == "Compressed"
        assert result.tokens_saved == 50

    def test_result_without_original_context(self):
        result = CompressResult(
            compressed_context="Compressed",
            original_tokens=100,
            compressed_tokens=50,
            actual_compression_ratio=0.5,
            tokens_saved=50,
            duration_ms=100,
        )
        assert result.original_context is None


class TestCompressResponse:
    def test_valid_response(self):
        response = CompressResponse(
            success=True,
            data=CompressResult(
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
        response = CompressResponse(success=False, message="Compression failed")
        assert response.success is False
        assert response.data is None


class TestStreamChunk:
    def test_content_chunk(self):
        chunk = StreamChunk(content="Partial", done=False)
        assert chunk.content == "Partial"
        assert chunk.done is False

    def test_done_chunk(self):
        chunk = StreamChunk(content="", done=True)
        assert chunk.done is True

    def test_chunk_serialization(self):
        chunk = StreamChunk(content="Test", done=False)
        data = chunk.model_dump()
        assert data["content"] == "Test"
        assert data["done"] is False


class TestCompressBatchInput:
    def test_valid_input(self):
        inp = CompressBatchInput(context="Test", query="Q?")
        assert inp.query == "Q?"

    def test_query_is_optional(self):
        inp = CompressBatchInput(context="Test")
        assert inp.query is None

    def test_empty_context_fails(self):
        with pytest.raises(ValidationError):
            CompressBatchInput(context="", query="Q")

    def test_empty_query_fails(self):
        with pytest.raises(ValidationError):
            CompressBatchInput(context="Test", query="")


class TestCompressBatchRequest:
    def test_valid_request(self):
        inputs = [
            CompressBatchInput(context="C1", query="Q1"),
            CompressBatchInput(context="C2", query="Q2"),
        ]
        req = CompressBatchRequest(inputs=inputs)
        assert len(req.inputs) == 2
        assert req.compression_model_name == "latte_v1"
        assert req.source == "sdk:python"

    def test_request_with_options(self):
        req = CompressBatchRequest(
            inputs=[CompressBatchInput(context="C", query="Q")],
            target_compression_ratio=0.5,
            coarse=True,
        )
        assert req.target_compression_ratio == 0.5
        assert req.coarse is True

    def test_empty_inputs_fails(self):
        with pytest.raises(ValidationError):
            CompressBatchRequest(inputs=[])

    def test_max_inputs_limit(self):
        inputs = [CompressBatchInput(context=f"C{i}", query=f"Q{i}") for i in range(101)]
        with pytest.raises(ValidationError):
            CompressBatchRequest(inputs=inputs)


class TestCompressBatchItemResult:
    def test_valid_item_result(self):
        result = CompressBatchItemResult(
            original_context="O",
            compressed_context="C",
            original_tokens=100,
            compressed_tokens=50,
            actual_compression_ratio=0.5,
            tokens_saved=50,
            duration_ms=100,
        )
        assert result.tokens_saved == 50


class TestCompressBatchResult:
    def test_valid_batch_result(self):
        items = [
            CompressBatchItemResult(
                compressed_context="C1",
                original_tokens=100,
                compressed_tokens=50,
                actual_compression_ratio=0.5,
                tokens_saved=50,
                duration_ms=50,
            )
        ]
        result = CompressBatchResult(
            results=items,
            total_original_tokens=100,
            total_compressed_tokens=50,
            total_tokens_saved=50,
            average_compression_ratio=0.5,
            count=1,
        )
        assert result.count == 1
        assert len(result.results) == 1

    def test_empty_defaults(self):
        result = CompressBatchResult()
        assert result.count == 0
        assert result.results == []


class TestCompressBatchResponse:
    def test_valid_response(self):
        data = CompressBatchResult(
            results=[
                CompressBatchItemResult(
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
        response = CompressBatchResponse(success=False, message="Batch failed")
        assert response.success is False
        assert response.data is None


class TestLatteV2DynamicFields:
    """3 latte_v2-only knobs: dynamic + dynamic_min_ratio + dynamic_max_ratio.

    SDK is permissive — backend validates that they're only honored on v2
    models. Defaults are all None so unset keys never reach the wire.
    """

    def test_dynamic_fields_default_to_none(self):
        req = CompressRequest(context="t", query="q")
        assert req.dynamic is None
        assert req.dynamic_min_ratio is None
        assert req.dynamic_max_ratio is None

    def test_dynamic_fields_explicit(self):
        req = CompressRequest(
            context="t",
            query="q",
            compression_model_name="latte_v2",
            dynamic=True,
            dynamic_min_ratio=2.0,
            dynamic_max_ratio=8.0,
        )
        assert req.dynamic is True
        assert req.dynamic_min_ratio == 2.0
        assert req.dynamic_max_ratio == 8.0

    def test_unset_dynamic_fields_excluded_from_wire(self):
        """model_dump(exclude_none=True) is what hits the network — no noise."""
        req = CompressRequest(context="t", query="q")
        payload = req.model_dump(exclude_none=True)
        assert "dynamic" not in payload
        assert "dynamic_min_ratio" not in payload
        assert "dynamic_max_ratio" not in payload

    def test_batch_request_carries_dynamic_fields(self):
        req = CompressBatchRequest(
            inputs=[CompressBatchInput(context="c", query="q")],
            compression_model_name="latte_v2",
            dynamic=True,
            dynamic_min_ratio=1.5,
            dynamic_max_ratio=10.0,
        )
        assert req.dynamic is True
        assert req.dynamic_min_ratio == 1.5
        assert req.dynamic_max_ratio == 10.0

    def test_internal_knobs_not_on_schema(self):
        """aggregation and include_tokens are intentionally absent — they're
        internal/debug knobs gated server-side, not exposed to SDK callers."""
        fields = set(CompressRequest.model_fields.keys())
        assert "aggregation" not in fields
        assert "include_tokens" not in fields


class TestBoundaryValidations:
    def test_ratio_minimum_zero(self):
        req = CompressRequest(context="Test", query="q", target_compression_ratio=0.0)
        assert req.target_compression_ratio == 0.0

    def test_ratio_high_value(self):
        req = CompressRequest(context="Test", query="q", target_compression_ratio=100.0)
        assert req.target_compression_ratio == 100.0

    def test_ratio_negative_fails(self):
        with pytest.raises(ValidationError):
            CompressRequest(context="Test", query="q", target_compression_ratio=-0.1)

    def test_query_single_char(self):
        req = CompressRequest(context="Test", query="Q")
        assert req.query == "Q"


class TestSerialization:
    def test_response_round_trip(self):
        result = CompressResult(
            original_context="Original",
            compressed_context="Compressed",
            original_tokens=150,
            compressed_tokens=75,
            actual_compression_ratio=0.5,
            tokens_saved=75,
            duration_ms=250,
        )
        response = CompressResponse(success=True, data=result)
        data = response.model_dump()
        reconstructed = CompressResponse(**data)
        assert reconstructed.success is True
        assert reconstructed.data.compressed_tokens == 75
