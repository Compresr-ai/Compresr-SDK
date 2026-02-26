"""
Integration Tests for List Context Support

Tests for context: Union[str, List[str]] support in CompressionClient.
Output type matches input type.

Run with:
    pytest tests/integration/test_list_context.py -v
"""

import pytest

from compresr import CompressionClient

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
# CompressionClient - List Context Tests (espresso_v1)
# =============================================================================


class TestEspressoListContext:
    """Tests for CompressionClient with List[str] context using espresso_v1."""

    def test_single_string_returns_string(self, compression_client):
        """Test that single string context returns string compressed_context."""
        response = compression_client.compress(
            context="Machine learning is a field of AI.",
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert isinstance(response.data.compressed_context, str)

    def test_list_context_returns_list(self, compression_client):
        """Test that list context returns list compressed_context."""
        contexts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing deals with text analysis.",
        ]

        response = compression_client.compress(
            context=contexts,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert isinstance(response.data.compressed_context, list)
        assert len(response.data.compressed_context) == len(contexts)

    def test_list_context_preserves_order(self, compression_client):
        """Test that list compression preserves the order of contexts."""
        contexts = [
            "First topic about machine learning.",
            "Second topic about data science.",
            "Third topic about neural networks.",
        ]

        response = compression_client.compress(
            context=contexts,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        for i, compressed in enumerate(response.data.compressed_context):
            assert len(compressed) > 0

    @pytest.mark.asyncio
    async def test_async_list_context(self, compression_client):
        """Test async compression with list context."""
        contexts = [
            "Artificial intelligence is transforming industries.",
            "Machine learning enables predictive analytics.",
        ]

        response = await compression_client.compress_async(
            context=contexts,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert isinstance(response.data.compressed_context, list)
        assert len(response.data.compressed_context) == len(contexts)


# =============================================================================
# CompressionClient - List Context Tests (latte_v1)
# =============================================================================


class TestLatteListContext:
    """Tests for CompressionClient with List[str] context using latte_v1."""

    def test_single_string_returns_string(self, compression_client):
        """Test that single string context returns string compressed_context."""
        response = compression_client.compress(
            context="Machine learning is a field of AI that enables learning from data.",
            query="What is machine learning?",
            compression_model_name=LATTE_MODEL,
        )

        assert response.success is True
        assert isinstance(response.data.compressed_context, str)

    def test_list_context_returns_list(self, compression_client):
        """Test that list context returns list compressed_context."""
        contexts = [
            "Machine learning uses algorithms to find patterns in data.",
            "Deep learning is inspired by biological neural networks.",
            "Supervised learning requires labeled training data.",
        ]
        query = "What is machine learning?"

        response = compression_client.compress(
            context=contexts,
            query=query,
            compression_model_name=LATTE_MODEL,
        )

        assert response.success is True
        assert isinstance(response.data.compressed_context, list)
        assert len(response.data.compressed_context) == len(contexts)

    def test_list_context_with_same_query(self, compression_client):
        """Test that all contexts are filtered using the same query."""
        contexts = [
            "Python was created by Guido van Rossum in 1991.",
            "JavaScript was created by Brendan Eich in 1995.",
            "Java was developed by James Gosling at Sun Microsystems.",
        ]
        query = "Who created Python?"

        response = compression_client.compress(
            context=contexts,
            query=query,
            compression_model_name=LATTE_MODEL,
        )

        assert response.success is True
        assert len(response.data.compressed_context[0]) > 0

    @pytest.mark.asyncio
    async def test_async_list_context(self, compression_client):
        """Test async compression with list context."""
        contexts = [
            "Einstein developed the theory of relativity.",
            "Newton formulated the laws of motion.",
        ]
        query = "What did Einstein discover?"

        response = await compression_client.compress_async(
            context=contexts,
            query=query,
            compression_model_name=LATTE_MODEL,
        )

        assert response.success is True
        assert isinstance(response.data.compressed_context, list)
        assert len(response.data.compressed_context) == len(contexts)


# =============================================================================
# Edge Cases
# =============================================================================


class TestListContextEdgeCases:
    """Edge case tests for list context."""

    def test_single_item_list(self, compression_client):
        """Test list with single item."""
        response = compression_client.compress(
            context=["Only one context item here."],
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert isinstance(response.data.compressed_context, list)
        assert len(response.data.compressed_context) == 1

    def test_mixed_length_contexts(self, compression_client):
        """Test list with varying context lengths."""
        contexts = [
            "Short.",
            "Medium length context with more words.",
            "A much longer context that has significantly more content to compress. " * 5,
        ]

        response = compression_client.compress(
            context=contexts,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert len(response.data.compressed_context) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
