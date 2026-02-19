"""
Integration Tests for QSCompressionClient (Question-Specific Compression)

Tests for the QSCompressionClient which handles question-specific
context compression - optimizing context based on a given question.

Run with:
    pytest tests/integration/test_qs_compression_client.py -v
"""

import pytest

from compresr import QSCompressionClient
from compresr.schemas import CompressResponse, StreamChunk

DEFAULT_QS_MODEL = "QS_CMPRSR_V1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def admin_client(admin_api_key):
    """Create QSCompressionClient with ADMIN key."""
    if not admin_api_key:
        pytest.skip("Admin API key not available")
    return QSCompressionClient(api_key=admin_api_key)


@pytest.fixture
def user_client(user_api_key):
    """Create QSCompressionClient with USER key (rate limited)."""
    if not user_api_key:
        pytest.skip("User API key not available")
    return QSCompressionClient(api_key=user_api_key)


# =============================================================================
# Basic Question-Specific Compression Tests
# =============================================================================


class TestBasicQSCompression:
    """Basic question-specific compression tests."""

    def test_basic_qs_compression(self, admin_client):
        """Test basic sync question-specific compression."""
        context = """
        Machine learning is a subset of artificial intelligence that enables 
        systems to learn from data. Deep learning uses neural networks with 
        many layers. Natural language processing deals with text and speech.
        Computer vision handles image and video analysis. Reinforcement 
        learning trains agents through rewards and penalties.
        """
        question = "What is machine learning?"

        response = admin_client.compress(
            context=context,
            question=question,
            compression_model_name=DEFAULT_QS_MODEL,
        )

        assert response is not None
        assert isinstance(response, CompressResponse)
        assert response.success is True
        assert response.data is not None
        assert response.data.original_tokens > 0
        assert response.data.compressed_tokens > 0
        assert response.data.compressed_context is not None
        assert len(response.data.compressed_context) > 0

    def test_qs_compression_returns_metrics(self, admin_client):
        """Test that QS compression returns all expected metrics."""
        response = admin_client.compress(
            context="Einstein's theory of relativity describes space, time, and gravity. "
            "Newton's laws of motion explain classical mechanics. "
            "Quantum mechanics governs subatomic particles.",
            question="What did Einstein discover?",
            compression_model_name=DEFAULT_QS_MODEL,
        )

        assert response.data.original_tokens is not None
        assert response.data.compressed_tokens is not None
        assert response.data.actual_compression_ratio is not None

    def test_question_relevance(self, admin_client):
        """Test that compression focuses on question-relevant content."""
        context = """
        Python is a programming language created by Guido van Rossum in 1991.
        It is known for its simple syntax and readability.
        JavaScript was created by Brendan Eich for Netscape in 1995.
        It is primarily used for web development.
        Java was developed by James Gosling at Sun Microsystems in 1995.
        It follows the principle of "write once, run anywhere".
        """
        question = "Who created Python and when?"

        response = admin_client.compress(
            context=context,
            question=question,
            compression_model_name=DEFAULT_QS_MODEL,
        )

        # Verify response is successful and compression occurred
        assert response.success is True
        # The compression should preserve question-relevant content
        assert response.data.compressed_tokens < response.data.original_tokens


class TestQSCompressionRatio:
    """Tests for QS compression ratio control."""

    def test_qs_compression_with_ratio(self, admin_client):
        """Test QS compression with specified ratio."""
        context = """
        Climate change is affecting global weather patterns significantly.
        Rising sea levels threaten coastal communities worldwide.
        Deforestation contributes to carbon dioxide emissions.
        Renewable energy sources include solar, wind, and hydroelectric power.
        Electric vehicles are becoming more popular and affordable.
        """
        question = "What are the effects of climate change?"

        response = admin_client.compress(
            context=context,
            question=question,
            compression_model_name=DEFAULT_QS_MODEL,
            target_compression_ratio=0.5,
        )

        assert response.success is True
        assert response.data.actual_compression_ratio is not None


class TestAsyncQSCompression:
    """Tests for async QS compression."""

    @pytest.mark.asyncio
    async def test_async_qs_compression(self, admin_client):
        """Test basic async QS compression."""
        response = await admin_client.compress_async(
            context="The capital of France is Paris. London is the capital of UK. "
            "Berlin is the capital of Germany. Rome is the capital of Italy.",
            question="What is the capital of France?",
            compression_model_name=DEFAULT_QS_MODEL,
        )

        assert response.success is True
        assert response.data.compressed_context is not None


class TestBatchQSCompression:
    """Tests for batch QS compression."""

    def test_batch_qs_compression(self, admin_client):
        """Test batch QS compression with inputs containing context and question."""
        inputs = [
            {
                "context": "Machine learning uses data to train models. Deep learning uses neural networks.",
                "question": "What is ML?",
            },
            {
                "context": "The solar system has eight planets. Mars is called the Red Planet.",
                "question": "How many planets?",
            },
            {
                "context": "Shakespeare wrote Hamlet and Macbeth. He lived in Stratford-upon-Avon.",
                "question": "What did Shakespeare write?",
            },
        ]

        response = admin_client.compress_batch(
            inputs=inputs,
            compression_model_name=DEFAULT_QS_MODEL,
        )

        assert response.success is True
        assert len(response.data.results) == len(inputs)


class TestStreamingQSCompression:
    """Tests for streaming QS compression."""

    def test_streaming_qs_compression(self, admin_client):
        """Test streaming QS compression yields chunks."""
        chunks = []
        content = ""

        for chunk in admin_client.compress_stream(
            context="Artificial intelligence is transforming many industries. "
            "Healthcare uses AI for diagnosis. Finance uses AI for fraud detection. "
            "Transportation uses AI for autonomous vehicles.",
            question="How is AI used in healthcare?",
            compression_model_name=DEFAULT_QS_MODEL,
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)
            if chunk.content:
                content += chunk.content

        assert len(chunks) > 0
        assert chunks[-1].done is True


class TestQSTokenCounting:
    """Tests for QS token counting."""

    def test_qs_compression_returns_token_counts(self, admin_client):
        """Test that QS compression returns valid token counts.

        Note: QS models may not always reduce tokens - they focus on
        preserving question-relevant information, not minimizing size.
        """
        response = admin_client.compress(
            context="Quantum computing uses qubits instead of classical bits. "
            "It can solve certain problems exponentially faster. "
            "Cryptography may be affected by quantum computers. "
            "Error correction is a major challenge in quantum systems.",
            question="What problems can quantum computing solve?",
            compression_model_name=DEFAULT_QS_MODEL,
            target_compression_ratio=0.5,
        )

        # QS models return valid token counts (may not always reduce)
        assert response.data.original_tokens > 0
        assert response.data.compressed_tokens > 0


class TestQSResponseStructure:
    """Tests for QS response structure."""

    def test_qs_response_structure(self, admin_client):
        """Test QS response has all expected fields."""
        response = admin_client.compress(
            context="Test context for validation of response structure.",
            question="What is this test about?",
            compression_model_name=DEFAULT_QS_MODEL,
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


class TestQSErrorHandling:
    """Tests for QS error handling."""

    def test_empty_context_error(self, admin_client):
        """Test that empty context raises error."""
        with pytest.raises(Exception):
            admin_client.compress(
                context="",
                question="What is this about?",
                compression_model_name=DEFAULT_QS_MODEL,
            )

    def test_empty_question_error(self, admin_client):
        """Test that empty question raises error."""
        with pytest.raises(Exception):
            admin_client.compress(
                context="Valid context for testing.",
                question="",
                compression_model_name=DEFAULT_QS_MODEL,
            )

    def test_invalid_compression_ratio_high(self, admin_client):
        """Test that compression ratio > 0.9 raises error."""
        with pytest.raises(Exception):
            admin_client.compress(
                context="Test context",
                question="Test question?",
                compression_model_name=DEFAULT_QS_MODEL,
                target_compression_ratio=1.5,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
