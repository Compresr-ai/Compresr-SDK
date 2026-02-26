"""
Integration Tests for FilterClient / coldbrew_v1 Model

Tests for coldbrew_v1 model which is a chunk-level filter and does NOT support compression_ratio.

Run with:
    pytest tests/integration/test_sat_models.py -v
"""

import pytest

from compresr import FilterClient

COLDBREW_MODEL = "coldbrew_v1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def filter_client(admin_api_key):
    """Create FilterClient with ADMIN key."""
    if not admin_api_key:
        pytest.skip("Admin API key not available")
    return FilterClient(api_key=admin_api_key)


# =============================================================================
# Filter Model Tests
# =============================================================================


class TestFilterModelValidation:
    """Tests for filter model validation."""

    def test_coldbrew_filter_succeeds(self, filter_client):
        """Test that coldbrew_v1 works for filtering."""
        response = filter_client.filter(
            chunks=[
                "Machine learning is a powerful tool for data analysis.",
                "Deep learning enables complex pattern recognition.",
                "Neural networks are inspired by biological systems.",
            ],
            query="What is machine learning?",
            compression_model_name=COLDBREW_MODEL,
        )

        assert response.success is True
        assert response.data.compressed_context is not None

    def test_coldbrew_returns_filtered_content(self, filter_client):
        """Test that coldbrew_v1 filters content based on query."""
        chunks = [
            "Python is a programming language created by Guido van Rossum.",
            "JavaScript was created by Brendan Eich for web browsers.",
            "Java was developed by James Gosling at Sun Microsystems.",
            "C++ was designed by Bjarne Stroustrup at Bell Labs.",
        ]
        query = "Who created Python?"

        response = filter_client.filter(
            chunks=chunks,
            query=query,
            compression_model_name=COLDBREW_MODEL,
        )

        assert response.success is True


class TestFilterAsyncCompression:
    """Tests for filter model in async mode."""

    @pytest.mark.asyncio
    async def test_coldbrew_async(self, filter_client):
        """Test coldbrew_v1 async filtering."""
        response = await filter_client.filter_async(
            chunks=["Async test chunk for coldbrew_v1 model validation."],
            query="What is this test?",
            compression_model_name=COLDBREW_MODEL,
        )

        assert response.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
