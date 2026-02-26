"""
Integration Tests for FilterClient (Chunk-Level Filtering)

Tests for the FilterClient which handles coarse-grained chunk selection
based on query relevance using coldbrew_v1.

Run with:
    pytest tests/integration/test_qs_compression_client.py -v
"""

import pytest

from compresr import FilterClient
from compresr.schemas import CompressResponse, StreamChunk

DEFAULT_FILTER_MODEL = "coldbrew_v1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def admin_client(admin_api_key):
    """Create FilterClient with ADMIN key."""
    if not admin_api_key:
        pytest.skip("Admin API key not available")
    return FilterClient(api_key=admin_api_key)


@pytest.fixture
def user_client(user_api_key):
    """Create FilterClient with USER key (rate limited)."""
    if not user_api_key:
        pytest.skip("User API key not available")
    return FilterClient(api_key=user_api_key)


# =============================================================================
# Basic Filter Tests
# =============================================================================


class TestBasicFilter:
    """Basic filter tests."""

    def test_basic_filter(self, admin_client):
        """Test basic sync filtering."""
        chunks = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing deals with text and speech.",
            "Computer vision handles image and video analysis.",
            "Reinforcement learning trains agents through rewards and penalties.",
        ]
        query = "What is machine learning?"

        response = admin_client.filter(
            chunks=chunks,
            query=query,
            compression_model_name=DEFAULT_FILTER_MODEL,
        )

        assert response is not None
        assert isinstance(response, CompressResponse)
        assert response.success is True
        assert response.data is not None
        assert response.data.original_tokens > 0
        assert response.data.compressed_tokens > 0
        assert response.data.compressed_context is not None

    def test_filter_returns_metrics(self, admin_client):
        """Test that filtering returns all expected metrics."""
        response = admin_client.filter(
            chunks=[
                "Einstein's theory of relativity describes space, time, and gravity.",
                "Newton's laws of motion explain classical mechanics.",
                "Quantum mechanics governs subatomic particles.",
            ],
            query="What did Einstein discover?",
            compression_model_name=DEFAULT_FILTER_MODEL,
        )

        assert response.data.original_tokens is not None
        assert response.data.compressed_tokens is not None
        assert response.data.actual_compression_ratio is not None

    def test_filter_relevance(self, admin_client):
        """Test that filtering focuses on query-relevant content."""
        chunks = [
            "Python is a programming language created by Guido van Rossum in 1991.",
            "JavaScript was created by Brendan Eich for Netscape in 1995.",
            "Java was developed by James Gosling at Sun Microsystems in 1995.",
        ]
        query = "Who created Python and when?"

        response = admin_client.filter(
            chunks=chunks,
            query=query,
            compression_model_name=DEFAULT_FILTER_MODEL,
        )

        assert response.success is True
        assert response.data.compressed_tokens < response.data.original_tokens


class TestAsyncFilter:
    """Tests for async filtering."""

    @pytest.mark.asyncio
    async def test_async_filter(self, admin_client):
        """Test basic async filtering."""
        response = await admin_client.filter_async(
            chunks=[
                "The capital of France is Paris.",
                "London is the capital of UK.",
                "Berlin is the capital of Germany.",
            ],
            query="What is the capital of France?",
            compression_model_name=DEFAULT_FILTER_MODEL,
        )

        assert response.success is True
        assert response.data.compressed_context is not None


class TestStreamingFilter:
    """Tests for streaming filtering."""

    def test_streaming_filter(self, admin_client):
        """Test streaming filtering yields chunks.

        Note: Streaming only supports single string input, not batch/list.
        """
        chunks = []
        content = ""

        # Streaming requires single string input, not list
        context = "Artificial intelligence is transforming many industries. Healthcare uses AI for diagnosis. Finance uses AI for fraud detection. Transportation uses AI for autonomous vehicles."

        for chunk in admin_client.filter_stream(
            chunks=context,
            query="How is AI used in healthcare?",
            compression_model_name=DEFAULT_FILTER_MODEL,
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)
            if chunk.content:
                content += chunk.content

        assert len(chunks) > 0
        assert chunks[-1].done is True


class TestFilterTokenCounting:
    """Tests for filter token counting."""

    def test_filter_returns_token_counts(self, admin_client):
        """Test that filtering returns valid token counts."""
        response = admin_client.filter(
            chunks=[
                "Quantum computing uses qubits instead of classical bits.",
                "It can solve certain problems exponentially faster.",
                "Cryptography may be affected by quantum computers.",
                "Error correction is a major challenge in quantum systems.",
            ],
            query="What problems can quantum computing solve?",
            compression_model_name=DEFAULT_FILTER_MODEL,
        )

        assert response.data.original_tokens > 0
        assert response.data.compressed_tokens > 0


class TestFilterResponseStructure:
    """Tests for filter response structure."""

    def test_filter_response_structure(self, admin_client):
        """Test filter response has all expected fields."""
        response = admin_client.filter(
            chunks=["Test context for validation of response structure."],
            query="What is this test about?",
            compression_model_name=DEFAULT_FILTER_MODEL,
        )

        assert hasattr(response, "success")
        assert hasattr(response, "data")
        assert response.data is not None

        data = response.data
        assert hasattr(data, "original_tokens")
        assert hasattr(data, "compressed_tokens")
        assert hasattr(data, "compressed_context")
        assert hasattr(data, "actual_compression_ratio")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestFilterErrorHandling:
    """Tests for filter error handling."""

    def test_empty_chunks_error(self, admin_client):
        """Test that empty chunks list raises error."""
        with pytest.raises(Exception):
            admin_client.filter(
                chunks=[],
                query="What is this about?",
                compression_model_name=DEFAULT_FILTER_MODEL,
            )

    def test_empty_query_error(self, admin_client):
        """Test that empty query raises error."""
        with pytest.raises(Exception):
            admin_client.filter(
                chunks=["Valid chunk for testing."],
                query="",
                compression_model_name=DEFAULT_FILTER_MODEL,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
