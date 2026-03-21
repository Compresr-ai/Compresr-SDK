"""
Integration Tests for CompressionClient

Tests for the CompressionClient which handles token-level context compression.
Supports espresso_v1 (agnostic) and latte_v1 (query-specific).

Run with:
    pytest tests/integration/test_compression_client.py -v
"""

import pytest

from compresr import CompressionClient
from compresr.schemas import CompressResponse, StreamChunk

DEFAULT_COMPRESSION_MODEL = "espresso_v1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def admin_client(admin_api_key):
    """Create CompressionClient with ADMIN key."""
    if not admin_api_key:
        pytest.skip("Admin API key not available")
    return CompressionClient(api_key=admin_api_key)


@pytest.fixture
def user_client(user_api_key):
    """Create CompressionClient with USER key (rate limited)."""
    if not user_api_key:
        pytest.skip("User API key not available")
    return CompressionClient(api_key=user_api_key)


# =============================================================================
# Basic Compression Tests
# =============================================================================


class TestBasicCompression:
    """Basic compression tests."""

    def test_basic_compression(self, admin_client):
        """Test basic sync compression."""
        response = admin_client.compress(
            context="Explain machine learning and its applications in modern technology.",
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response is not None
        assert isinstance(response, CompressResponse)
        assert response.success is True
        assert response.data is not None
        assert response.data.original_tokens > 0
        assert response.data.compressed_tokens > 0
        assert response.data.compressed_context is not None
        assert len(response.data.compressed_context) > 0

    def test_compression_returns_metrics(self, admin_client):
        """Test that compression returns all expected metrics."""
        response = admin_client.compress(
            context="Explain the theory of relativity and its implications.",
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.data.original_tokens is not None
        assert response.data.compressed_tokens is not None
        assert response.data.actual_compression_ratio is not None


class TestLatteCompression:
    """Tests for latte_v1 (query-specific compression)."""

    def test_latte_compression_with_query(self, admin_client):
        """Test latte_v1 compression with query."""
        context = """
        Machine learning is a subset of artificial intelligence that enables
        systems to learn from data. Deep learning uses neural networks with
        many layers. Natural language processing deals with text and speech.
        """
        response = admin_client.compress(
            context=context,
            query="What is machine learning?",
            compression_model_name="latte_v1",
        )

        assert response.success is True
        assert response.data.compressed_context is not None

    def test_latte_with_ratio(self, admin_client):
        """Test latte_v1 compression with compression ratio."""
        response = admin_client.compress(
            context="Climate change affects weather. Renewable energy helps. Electric vehicles reduce emissions.",
            query="What are the effects of climate change?",
            compression_model_name="latte_v1",
            target_compression_ratio=0.5,
        )

        assert response.success is True
        assert response.data.actual_compression_ratio is not None


class TestCoarseCompression:
    """Tests for coarse (paragraph-level) compression."""

    def test_coarse_true(self, admin_client):
        """Test latte_v1 with coarse=True (paragraph-level)."""
        context = """
        Machine learning is a subset of artificial intelligence that enables
        systems to learn from data. Deep learning uses neural networks with
        many layers. Natural language processing deals with text and speech.
        """
        response = admin_client.compress(
            context=context,
            query="What is machine learning?",
            compression_model_name="latte_v1",
            coarse=True,
        )

        assert response.success is True
        assert response.data.compressed_context is not None

    def test_coarse_false(self, admin_client):
        """Test latte_v1 with coarse=False (token-level)."""
        context = """
        Machine learning is a subset of artificial intelligence that enables
        systems to learn from data. Deep learning uses neural networks with
        many layers. Natural language processing deals with text and speech.
        """
        response = admin_client.compress(
            context=context,
            query="What is machine learning?",
            compression_model_name="latte_v1",
            coarse=False,
        )

        assert response.success is True
        assert response.data.compressed_context is not None

    def test_coarse_with_ratio(self, admin_client):
        """Test coarse compression with target ratio."""
        response = admin_client.compress(
            context="Climate change affects weather. Renewable energy helps. Electric vehicles reduce emissions.",
            query="What are the effects of climate change?",
            compression_model_name="latte_v1",
            target_compression_ratio=0.5,
            coarse=True,
        )

        assert response.success is True
        assert response.data.actual_compression_ratio is not None

    @pytest.mark.asyncio
    async def test_async_coarse(self, admin_client):
        """Test async compression with coarse=True."""
        response = await admin_client.compress_async(
            context="AI is transforming industries. Healthcare uses AI. Finance uses AI.",
            query="How is AI used?",
            compression_model_name="latte_v1",
            coarse=True,
        )

        assert response.success is True
        assert response.data.compressed_context is not None


class TestBatchCompression:
    """Tests for batch compression with compress_batch()."""

    def test_batch_with_single_query(self, admin_client):
        """Test batch compression with same query for all contexts."""
        contexts = [
            "Machine learning enables systems to learn from data automatically.",
            "Deep learning uses neural networks with multiple hidden layers.",
            "Natural language processing helps computers understand human text.",
        ]
        response = admin_client.compress_batch(
            contexts=contexts,
            queries="What is this technology about?",
            compression_model_name="latte_v1",
        )

        assert response.success is True
        assert response.data is not None
        assert response.data.count == 3
        assert len(response.data.results) == 3
        assert response.data.total_original_tokens > 0
        assert response.data.total_compressed_tokens > 0
        assert response.data.average_compression_ratio >= 0

    def test_batch_with_different_queries(self, admin_client):
        """Test batch compression with different query per context."""
        contexts = [
            "Machine learning enables systems to learn from data automatically.",
            "Deep learning uses neural networks with multiple hidden layers.",
            "Natural language processing helps computers understand human text.",
        ]
        queries = [
            "What is machine learning?",
            "What is deep learning?",
            "What is NLP?",
        ]
        response = admin_client.compress_batch(
            contexts=contexts,
            queries=queries,
            compression_model_name="latte_v1",
        )

        assert response.success is True
        assert response.data.count == 3
        for result in response.data.results:
            assert result.compressed_context is not None
            assert result.original_tokens > 0
            assert result.compressed_tokens > 0

    def test_batch_with_coarse(self, admin_client):
        """Test batch compression with coarse=True."""
        contexts = [
            "AI is transforming healthcare with diagnostic tools.",
            "Finance uses AI for fraud detection and trading.",
        ]
        response = admin_client.compress_batch(
            contexts=contexts,
            queries="How is AI used?",
            compression_model_name="latte_v1",
            coarse=True,
        )

        assert response.success is True
        assert response.data.count == 2

    def test_batch_with_ratio(self, admin_client):
        """Test batch compression with target compression ratio."""
        contexts = [
            "Machine learning is a subset of AI that enables systems to learn.",
            "Deep learning uses neural networks with many layers for patterns.",
        ]
        response = admin_client.compress_batch(
            contexts=contexts,
            queries="Explain this technology.",
            compression_model_name="latte_v1",
            target_compression_ratio=0.5,
        )

        assert response.success is True
        assert response.data.average_compression_ratio is not None

    def test_batch_result_structure(self, admin_client):
        """Test batch response has all expected fields."""
        contexts = ["AI enables automation.", "ML learns from data."]
        response = admin_client.compress_batch(
            contexts=contexts,
            queries="What is this?",
            compression_model_name="latte_v1",
        )

        assert response.data.count == 2
        assert response.data.total_original_tokens > 0
        assert response.data.total_compressed_tokens > 0
        assert response.data.total_tokens_saved >= 0
        assert response.data.average_compression_ratio >= 0

        for result in response.data.results:
            assert hasattr(result, "original_context")
            assert hasattr(result, "compressed_context")
            assert hasattr(result, "original_tokens")
            assert hasattr(result, "compressed_tokens")
            assert hasattr(result, "actual_compression_ratio")
            assert hasattr(result, "tokens_saved")
            assert hasattr(result, "duration_ms")

    @pytest.mark.asyncio
    async def test_async_batch_single_query(self, admin_client):
        """Test async batch compression with single query."""
        contexts = ["AI runs on data.", "ML models learn patterns."]
        response = await admin_client.compress_batch_async(
            contexts=contexts,
            queries="What is this about?",
            compression_model_name="latte_v1",
        )

        assert response.success is True
        assert response.data.count == 2

    @pytest.mark.asyncio
    async def test_async_batch_different_queries(self, admin_client):
        """Test async batch compression with different queries."""
        contexts = ["AI automates tasks.", "ML predicts outcomes."]
        queries = ["What does AI do?", "What does ML do?"]
        response = await admin_client.compress_batch_async(
            contexts=contexts,
            queries=queries,
            compression_model_name="latte_v1",
        )

        assert response.success is True
        assert response.data.count == 2

    def test_batch_mismatched_queries_raises_error(self, admin_client):
        """Test that mismatched queries length raises ValidationError."""
        from compresr.exceptions import ValidationError

        contexts = ["Context 1", "Context 2", "Context 3"]
        queries = ["Query 1", "Query 2"]  # Only 2, but 3 contexts

        with pytest.raises(ValidationError) as exc_info:
            admin_client.compress_batch(
                contexts=contexts,
                queries=queries,
                compression_model_name="latte_v1",
            )

        assert "must match" in str(exc_info.value).lower()

    def test_batch_invalid_model_raises_error(self, admin_client):
        """Test that non-query models raise ValidationError for batch."""
        from compresr.exceptions import ValidationError

        contexts = ["Context 1", "Context 2"]

        with pytest.raises(ValidationError) as exc_info:
            admin_client.compress_batch(
                contexts=contexts,
                queries="test query",
                compression_model_name="espresso_v1",  # Not supported for batch
            )

        assert "query-specific" in str(exc_info.value).lower()


class TestCompressionRatio:
    """Tests for compression ratio control."""

    def test_compression_with_ratio(self, admin_client):
        """Test compression with specified ratio."""
        context = """
        Explain machine learning including supervised learning,
        unsupervised learning, and reinforcement learning.
        Cover the history, applications, and future trends.
        """

        response = admin_client.compress(
            context=context,
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
            target_compression_ratio=0.5,
        )

        assert response.success is True
        assert response.data.actual_compression_ratio is not None


class TestAsyncCompression:
    """Tests for async compression."""

    @pytest.mark.asyncio
    async def test_async_compression(self, admin_client):
        """Test basic async compression."""
        response = await admin_client.compress_async(
            context="What is artificial intelligence?",
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        )

        assert response.success is True
        assert response.data.compressed_context is not None


class TestStreamingCompression:
    """Tests for streaming compression."""

    def test_streaming_compression(self, admin_client):
        """Test streaming compression yields chunks."""
        chunks = []
        content = ""

        for chunk in admin_client.compress_stream(
            context="Write a detailed explanation of blockchain technology.",
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)
            if chunk.content:
                content += chunk.content

        assert len(chunks) > 0
        assert chunks[-1].done is True


class TestTokenCounting:
    """Tests for token counting."""

    def test_compressed_fewer_tokens(self, admin_client):
        """Test that compressed text has fewer tokens."""
        response = admin_client.compress(
            context="Explain quantum computing and cryptography applications.",
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
            target_compression_ratio=0.5,
        )

        assert response.data.compressed_tokens <= response.data.original_tokens


class TestResponseStructure:
    """Tests for response structure."""

    def test_response_structure(self, admin_client):
        """Test response has all expected fields."""
        response = admin_client.compress(
            context="Test context for validation.",
            compression_model_name=DEFAULT_COMPRESSION_MODEL,
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


class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_context_error(self, admin_client):
        """Test that empty context raises error."""
        with pytest.raises(Exception):
            admin_client.compress(
                context="",
                compression_model_name=DEFAULT_COMPRESSION_MODEL,
            )

    def test_invalid_compression_ratio_high(self, admin_client):
        """Test that compression ratio > 200 raises error (backend limit)."""
        with pytest.raises(Exception):
            admin_client.compress(
                context="Test context",
                compression_model_name=DEFAULT_COMPRESSION_MODEL,
                target_compression_ratio=250.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
