"""
Integration Tests for Model Operations

Tests model listing, validation, and different model types.
"""

import pytest

from compresr.exceptions import CompresrError, ValidationError


class TestModelValidation:
    """Test model validation and selection."""

    def test_default_model_compression(self, admin_client):
        """Test compression with default model."""
        response = admin_client.compress(
            context="Test compression with default model settings.",
            compression_model_name="espresso_v1",
        )
        assert response.success is True
        assert response.data.compressed_context is not None

    def test_explicit_model_name(self, admin_client):
        """Test compression with explicit model name."""
        response = admin_client.compress(
            context="Testing explicit model name parameter in compression.",
            compression_model_name="espresso_v1",
        )
        assert response.success is True
        assert response.data.original_tokens > 0

    def test_invalid_model_name_raises_error(self, admin_client):
        """Test that invalid model name raises appropriate error."""
        with pytest.raises((ValidationError, CompresrError)):
            admin_client.compress(
                context="Test with invalid model name",
                compression_model_name="non_existent_model_xyz_123",
            )


class TestCompressionRatios:
    """Test different compression ratio settings."""

    @pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
    def test_different_ratios(self, admin_client, ratio):
        """Test compression with different target ratios."""
        context = "Test compression with various target compression ratios to verify ratio control."

        response = admin_client.compress(
            context=context,
            compression_model_name="espresso_v1",
            target_compression_ratio=ratio,
        )

        assert response.success is True
        assert response.data.actual_compression_ratio is not None
        assert 0 < response.data.actual_compression_ratio < 1

    def test_aggressive_compression(self, admin_client):
        """Test high compression ratio (70% reduction)."""
        response = admin_client.compress(
            context="This is a longer context that should be compressed aggressively " * 5,
            compression_model_name="espresso_v1",
            target_compression_ratio=0.7,
        )

        assert response.success is True
        assert response.data.compressed_tokens < response.data.original_tokens

    def test_conservative_compression(self, admin_client):
        """Test low compression ratio (30% reduction)."""
        response = admin_client.compress(
            context="This context will be compressed conservatively to retain more information.",
            compression_model_name="espresso_v1",
            target_compression_ratio=0.3,
        )

        assert response.success is True
        assert response.data.compressed_tokens > 0


class TestContextSizes:
    """Test compression with different context sizes."""

    def test_short_context(self, admin_client):
        """Test compression with short context."""
        response = admin_client.compress(
            context="Short text.", compression_model_name="espresso_v1"
        )
        assert response.success is True

    def test_medium_context(self, admin_client):
        """Test compression with medium context."""
        context = "This is a medium-sized context. " * 20
        response = admin_client.compress(
            context=context, compression_model_name="espresso_v1"
        )
        assert response.success is True
        assert response.data.original_tokens > 50

    def test_long_context(self, admin_client):
        """Test compression with long context."""
        context = "This is a long context for testing compression efficiency. " * 100
        response = admin_client.compress(
            context=context,
            compression_model_name="espresso_v1",
            target_compression_ratio=0.5,
        )
        assert response.success is True
        assert response.data.original_tokens > 500
        assert response.data.tokens_saved >= 0


class TestStreamingCompression:
    """Test streaming compression functionality."""

    def test_stream_yields_chunks(self, admin_client):
        """Test that streaming yields multiple chunks."""
        chunks = list(
            admin_client.compress_stream(
                context="Stream this context and verify chunks are received progressively.",
                compression_model_name="espresso_v1",
            )
        )

        assert len(chunks) > 0
        assert chunks[-1].done is True

    def test_stream_completion(self, admin_client):
        """Test stream completes with done flag."""
        done_received = False

        for chunk in admin_client.compress_stream(
            context="Test streaming completion with done flag verification.",
            compression_model_name="espresso_v1",
        ):
            if chunk.done:
                done_received = True

        assert done_received is True

    def test_stream_content_accumulation(self, admin_client):
        """Test accumulating streamed content."""
        full_content = ""

        for chunk in admin_client.compress_stream(
            context="Accumulate all streamed chunks to reconstruct the full compressed context.",
            compression_model_name="espresso_v1",
        ):
            if chunk.content:
                full_content += chunk.content

        assert len(full_content) > 0
