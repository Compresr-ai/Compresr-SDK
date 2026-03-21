"""
Unit Tests for Search Schemas

Tests for agentic search request/response schema validation and serialization.
"""

import pytest
from pydantic import ValidationError

from compresr.schemas.agentic_search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)


class TestSearchRequest:
    """Test SearchRequest schema validation."""

    def test_valid_request(self):
        """Test valid search request."""
        req = SearchRequest(
            query="What is machine learning?",
            index_name="my-knowledge-base",
            compression_model_name="macchiato_v1",
        )
        assert req.query == "What is machine learning?"
        assert req.index_name == "my-knowledge-base"
        assert req.compression_model_name == "macchiato_v1"
        assert req.max_time_s == 4.5  # default

    def test_request_with_custom_max_time(self):
        """Test request with custom max_time_s."""
        req = SearchRequest(
            query="Test query",
            index_name="test-index",
            compression_model_name="macchiato_v1",
            max_time_s=10.0,
        )
        assert req.max_time_s == 10.0

    def test_empty_query_fails(self):
        """Test that empty query fails validation."""
        with pytest.raises(ValidationError):
            SearchRequest(
                query="",
                index_name="test-index",
                compression_model_name="macchiato_v1",
            )

    def test_whitespace_query_fails(self):
        """Test that whitespace-only query fails validation."""
        with pytest.raises(ValidationError):
            SearchRequest(
                query="   ",
                index_name="test-index",
                compression_model_name="macchiato_v1",
            )

    def test_empty_index_name_fails(self):
        """Test that empty index_name fails validation."""
        with pytest.raises(ValidationError):
            SearchRequest(
                query="What is ML?",
                index_name="",
                compression_model_name="macchiato_v1",
            )

    def test_whitespace_index_name_fails(self):
        """Test that whitespace-only index_name fails validation."""
        with pytest.raises(ValidationError):
            SearchRequest(
                query="What is ML?",
                index_name="   ",
                compression_model_name="macchiato_v1",
            )

    def test_missing_query_fails(self):
        """Test that missing query fails."""
        with pytest.raises(ValidationError):
            SearchRequest(
                index_name="test-index",
                compression_model_name="macchiato_v1",
            )

    def test_missing_index_name_fails(self):
        """Test that missing index_name fails."""
        with pytest.raises(ValidationError):
            SearchRequest(
                query="What is ML?",
                compression_model_name="macchiato_v1",
            )

    def test_max_time_too_low_fails(self):
        """Test that max_time_s below 0.1 fails."""
        with pytest.raises(ValidationError):
            SearchRequest(
                query="Test",
                index_name="test-index",
                compression_model_name="macchiato_v1",
                max_time_s=0.05,
            )

    def test_max_time_too_high_fails(self):
        """Test that max_time_s above 30 fails."""
        with pytest.raises(ValidationError):
            SearchRequest(
                query="Test",
                index_name="test-index",
                compression_model_name="macchiato_v1",
                max_time_s=31.0,
            )

    def test_default_source(self):
        """Test default source is sdk:python."""
        req = SearchRequest(
            query="Test",
            index_name="test-index",
            compression_model_name="macchiato_v1",
        )
        assert req.source == "sdk:python"

    def test_serialization(self):
        """Test request serialization to dict."""
        req = SearchRequest(
            query="What is ML?",
            index_name="kb-1",
            compression_model_name="macchiato_v1",
            max_time_s=5.0,
        )
        data = req.model_dump()
        assert data["query"] == "What is ML?"
        assert data["index_name"] == "kb-1"
        assert data["compression_model_name"] == "macchiato_v1"
        assert data["max_time_s"] == 5.0


class TestSearchResult:
    """Test SearchResult schema."""

    def test_valid_result(self):
        """Test valid search result."""
        result = SearchResult(
            chunks=["Chunk 1 about ML", "Chunk 2 about neural nets"],
            chunks_count=2,
            original_tokens=5000,
            compressed_tokens=800,
            compression_ratio=0.16,
            duration_ms=2300,
            index_name="my-kb",
        )
        assert len(result.chunks) == 2
        assert result.chunks_count == 2
        assert result.original_tokens == 5000
        assert result.compressed_tokens == 800
        assert result.compression_ratio == 0.16
        assert result.duration_ms == 2300
        assert result.index_name == "my-kb"

    def test_empty_chunks(self):
        """Test result with empty chunks list."""
        result = SearchResult(
            chunks=[],
            chunks_count=0,
            original_tokens=5000,
            compressed_tokens=0,
            compression_ratio=0.0,
            duration_ms=1000,
            index_name="my-kb",
        )
        assert result.chunks == []
        assert result.chunks_count == 0

    def test_default_duration(self):
        """Test default duration_ms is 0."""
        result = SearchResult(
            chunks=["test"],
            chunks_count=1,
            original_tokens=100,
            compressed_tokens=50,
            compression_ratio=0.5,
            index_name="test-kb",
        )
        assert result.duration_ms == 0


class TestSearchResponse:
    """Test SearchResponse schema."""

    def test_valid_response(self):
        """Test valid search response."""
        response = SearchResponse(
            success=True,
            data=SearchResult(
                chunks=["Relevant chunk"],
                chunks_count=1,
                original_tokens=1000,
                compressed_tokens=200,
                compression_ratio=0.2,
                duration_ms=500,
                index_name="test-kb",
            ),
        )
        assert response.success is True
        assert response.data.chunks_count == 1

    def test_error_response(self):
        """Test error response."""
        response = SearchResponse(success=False, message="Index not found")
        assert response.success is False
        assert response.message == "Index not found"
        assert response.data is None

    def test_response_round_trip(self):
        """Test full response serialization/deserialization."""
        result = SearchResult(
            chunks=["Chunk A", "Chunk B"],
            chunks_count=2,
            original_tokens=2000,
            compressed_tokens=400,
            compression_ratio=0.2,
            duration_ms=1500,
            index_name="round-trip-kb",
        )
        response = SearchResponse(success=True, data=result)

        data = response.model_dump()
        reconstructed = SearchResponse(**data)

        assert reconstructed.success == response.success
        assert reconstructed.data.chunks == response.data.chunks
        assert reconstructed.data.index_name == response.data.index_name
