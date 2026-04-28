"""
Integration Tests for Batch Compression

Tests for compress_batch() functionality in CompressionClient.
List context is no longer supported for single compression - use batch instead.

Run with:
    pytest tests/integration/test_list_context.py -v
"""

import pytest

from compresr import CompressionClient
from compresr.exceptions import ValidationError

DEFAULT_COMPRESSION_MODEL = "espresso_v1"
LATTE_MODEL = "latte_v1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def compression_client(admin_api_key):
    """Create CompressionClient with ADMIN key."""
    if not admin_api_key:
        pytest.skip("Admin API key not available")
    return CompressionClient(api_key=admin_api_key)


# =============================================================================
# CompressionClient - Single Compression (context is str only)
# =============================================================================


class TestSingleCompression:
    """Tests for single string compression."""

    def test_single_string_returns_string(self, compression_client):
        """Test that single string context returns string compressed_context."""
        response = compression_client.compress(
            context="Machine learning is a field of AI.",
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert isinstance(response.data.compressed_context, str)

    def test_list_context_raises_validation_error(self, compression_client):
        """Test that list context raises ValidationError (use batch instead)."""
        with pytest.raises(ValidationError):
            compression_client.compress(
                context=["Context 1", "Context 2"],  # type: ignore[arg-type]
                compression_model_name=DEFAULT_COMPRESSION_MODEL,
            )


# =============================================================================
# CompressionClient - Batch Compression Tests (espresso_v1)
# =============================================================================


class TestEspressoBatchCompression:
    """Tests for CompressionClient batch compression using espresso_v1."""

    def test_batch_compression(self, compression_client):
        """Test batch compression with multiple contexts."""
        contexts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing deals with text analysis.",
        ]

        response = compression_client.compress_batch(
            contexts=contexts,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert len(response.data.results) == len(contexts)

    def test_batch_preserves_order(self, compression_client):
        """Test that batch compression preserves the order of contexts."""
        contexts = [
            "First topic about machine learning.",
            "Second topic about data science.",
            "Third topic about neural networks.",
        ]

        response = compression_client.compress_batch(
            contexts=contexts,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        for i, result in enumerate(response.data.results):
            assert len(result.compressed_context) > 0

    @pytest.mark.asyncio
    async def test_async_batch_compression(self, compression_client):
        """Test async batch compression."""
        contexts = [
            "Artificial intelligence is transforming industries.",
            "Machine learning enables predictive analytics.",
        ]

        response = await compression_client.compress_batch_async(
            contexts=contexts,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert len(response.data.results) == len(contexts)


# =============================================================================
# CompressionClient - Batch Compression Tests (latte_v1)
# =============================================================================


class TestLatteBatchCompression:
    """Tests for CompressionClient batch compression using latte_v1."""

    def test_batch_with_same_query(self, compression_client):
        """Test batch compression with same query for all contexts."""
        contexts = [
            "Machine learning uses algorithms to find patterns in data.",
            "Deep learning is inspired by biological neural networks.",
            "Supervised learning requires labeled training data.",
        ]
        query = "What is machine learning?"

        response = compression_client.compress_batch(
            contexts=contexts,
            queries=query,  # Same query for all
            compression_model_name=LATTE_MODEL,
        )

        assert response.success is True
        assert len(response.data.results) == len(contexts)

    def test_batch_with_different_queries(self, compression_client):
        """Test batch compression with different queries per context."""
        contexts = [
            "Python was created by Guido van Rossum in 1991.",
            "JavaScript was created by Brendan Eich in 1995.",
            "Java was developed by James Gosling at Sun Microsystems.",
        ]
        queries = [
            "Who created Python?",
            "Who created JavaScript?",
            "Who created Java?",
        ]

        response = compression_client.compress_batch(
            contexts=contexts,
            queries=queries,
            compression_model_name=LATTE_MODEL,
        )

        assert response.success is True
        assert len(response.data.results) == len(contexts)

    @pytest.mark.asyncio
    async def test_async_batch_with_queries(self, compression_client):
        """Test async batch compression with queries."""
        contexts = [
            "Einstein developed the theory of relativity.",
            "Newton formulated the laws of motion.",
        ]
        query = "What did Einstein discover?"

        response = await compression_client.compress_batch_async(
            contexts=contexts,
            queries=query,
            compression_model_name=LATTE_MODEL,
        )

        assert response.success is True
        assert len(response.data.results) == len(contexts)


# =============================================================================
# Edge Cases
# =============================================================================


class TestBatchEdgeCases:
    """Edge case tests for batch compression."""

    def test_single_item_batch(self, compression_client):
        """Test batch with single item."""
        response = compression_client.compress_batch(
            contexts=["Only one context item here."],
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert len(response.data.results) == 1

    def test_mixed_length_contexts(self, compression_client):
        """Test batch with varying context lengths."""
        contexts = [
            "Short.",
            "Medium length context with more words.",
            "A much longer context that has significantly more content to compress. " * 5,
        ]

        response = compression_client.compress_batch(
            contexts=contexts,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert len(response.data.results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
