"""
Integration Tests for SearchClient / macchiato_v1 Model

Tests for macchiato_v1 model which performs agentic search over
pre-indexed knowledge bases.

Run with:
    pytest tests/integration/test_search_client.py -v
"""

import pytest

from compresr import SearchClient
from compresr.exceptions import ValidationError

MACCHIATO_MODEL = "macchiato_v1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def search_client(admin_api_key):
    """Create SearchClient with ADMIN key."""
    if not admin_api_key:
        pytest.skip("Admin API key not available")
    return SearchClient(api_key=admin_api_key)


# =============================================================================
# Search Model Validation Tests
# =============================================================================


class TestSearchModelValidation:
    """Tests for search model validation."""

    def test_invalid_model_raises(self, admin_api_key):
        """Test that an invalid model name raises ValidationError."""
        if not admin_api_key:
            pytest.skip("Admin API key not available")
        client = SearchClient(api_key=admin_api_key)
        with pytest.raises(ValidationError):
            client.search(
                query="What is machine learning?",
                index_name="test-index",
                compression_model_name="invalid_model",
            )

    def test_macchiato_model_accepted(self, admin_api_key):
        """Test that macchiato_v1 is accepted by SearchClient."""
        if not admin_api_key:
            pytest.skip("Admin API key not available")
        client = SearchClient(api_key=admin_api_key)
        # Should not raise ValidationError for model name
        # (may fail for other reasons like missing index, but model is valid)
        try:
            client.search(
                query="Test query",
                index_name="test-index",
                compression_model_name=MACCHIATO_MODEL,
            )
        except ValidationError as e:
            # Model validation error would say "not valid for SearchClient"
            assert "not valid for SearchClient" not in str(e)


# =============================================================================
# Search Tests
# =============================================================================


class TestSearchClient:
    """Tests for SearchClient search operations."""

    def test_search_requires_query(self, admin_api_key):
        """Test that empty query raises ValidationError."""
        if not admin_api_key:
            pytest.skip("Admin API key not available")
        client = SearchClient(api_key=admin_api_key)
        with pytest.raises(ValidationError):
            client.search(
                query="",
                index_name="test-index",
            )

    def test_search_requires_index_name(self, admin_api_key):
        """Test that empty index_name raises ValidationError."""
        if not admin_api_key:
            pytest.skip("Admin API key not available")
        client = SearchClient(api_key=admin_api_key)
        with pytest.raises(ValidationError):
            client.search(
                query="What is machine learning?",
                index_name="",
            )

    def test_search_succeeds(self, search_client):
        """Test that a valid search request succeeds."""
        response = search_client.search(
            query="What is machine learning?",
            index_name="test-knowledge-base",
            compression_model_name=MACCHIATO_MODEL,
        )

        assert response.success is True
        assert response.data is not None
        assert response.data.chunks is not None
        assert isinstance(response.data.chunks, list)
        assert response.data.chunks_count >= 0
        assert response.data.index_name == "test-knowledge-base"

    def test_search_returns_metrics(self, search_client):
        """Test that search response includes metrics."""
        response = search_client.search(
            query="What are neural networks?",
            index_name="test-knowledge-base",
        )

        assert response.success is True
        assert response.data.duration_ms >= 0
        assert response.data.compression_ratio >= 0

    def test_search_with_custom_max_time(self, search_client):
        """Test search with custom max_time_s parameter."""
        response = search_client.search(
            query="What is deep learning?",
            index_name="test-knowledge-base",
            max_time_s=2.0,
        )

        assert response.success is True


# =============================================================================
# Async Tests
# =============================================================================


class TestSearchAsyncClient:
    """Tests for SearchClient async operations."""

    @pytest.mark.asyncio
    async def test_search_async(self, search_client):
        """Test macchiato_v1 async search."""
        response = await search_client.search_async(
            query="What is machine learning?",
            index_name="test-knowledge-base",
            compression_model_name=MACCHIATO_MODEL,
        )

        assert response.success is True
        assert response.data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
