"""
Integration Tests for Error Handling

Tests how the SDK handles various error scenarios from the API.
"""

import pytest

from compresr import CompressionClient
from compresr.exceptions import AuthenticationError, CompresrError, ValidationError


@pytest.fixture
def client_with_invalid_key():
    """Create client with invalid API key."""
    return CompressionClient(api_key="cmp_invalid_key_12345")


class TestAuthenticationErrors:
    """Test authentication error handling."""

    def test_invalid_api_key(self, client_with_invalid_key):
        """Test that invalid API key raises AuthenticationError."""
        with pytest.raises(AuthenticationError):
            client_with_invalid_key.compress(
                context="Test context", compression_model_name="A_CMPRSR_V1"
            )

    def test_missing_api_key(self):
        """Test that missing API key raises AuthenticationError."""
        with pytest.raises(AuthenticationError):
            CompressionClient(api_key="")


class TestValidationErrors:
    """Test validation error handling."""

    def test_empty_context(self, admin_client):
        """Test that empty context raises ValidationError."""
        with pytest.raises(ValidationError):
            admin_client.compress(context="", compression_model_name="A_CMPRSR_V1")

    def test_invalid_compression_ratio_high(self, admin_client):
        """Test that ratio > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            admin_client.compress(
                context="Test context",
                compression_model_name="A_CMPRSR_V1",
                target_compression_ratio=1.5,
            )

    def test_invalid_compression_ratio_negative(self, admin_client):
        """Test that negative ratio raises ValidationError."""
        with pytest.raises(ValidationError):
            admin_client.compress(
                context="Test context",
                compression_model_name="A_CMPRSR_V1",
                target_compression_ratio=-0.5,
            )

    def test_invalid_model_name(self, admin_client):
        """Test that invalid model name is handled."""
        with pytest.raises((ValidationError, CompresrError)):
            admin_client.compress(
                context="Test context", compression_model_name="invalid_model_xyz"
            )


class TestBatchErrors:
    """Test batch-specific error handling."""

    def test_batch_empty_contexts(self, admin_client):
        """Test that empty contexts list raises error."""
        with pytest.raises(ValidationError):
            admin_client.compress_batch(contexts=[], compression_model_name="A_CMPRSR_V1")

    def test_batch_with_empty_string(self, admin_client):
        """Test batch with empty string in contexts."""
        with pytest.raises(ValidationError):
            admin_client.compress_batch(
                contexts=["Valid context", ""], compression_model_name="A_CMPRSR_V1"
            )


class TestAsyncErrors:
    """Test async error handling."""

    @pytest.mark.asyncio
    async def test_async_invalid_key(self, client_with_invalid_key):
        """Test async compression with invalid key."""
        with pytest.raises(AuthenticationError):
            await client_with_invalid_key.compress_async(
                context="Test context", compression_model_name="A_CMPRSR_V1"
            )

    @pytest.mark.asyncio
    async def test_async_validation_error(self, admin_client):
        """Test async compression with validation error."""
        with pytest.raises(ValidationError):
            await admin_client.compress_async(context="", compression_model_name="A_CMPRSR_V1")


class TestConnectionErrors:
    """Test connection and network error handling."""

    def test_timeout_handling(self, admin_client):
        """Test that timeout is respected."""
        # Create client with very short timeout
        client = CompressionClient(
            api_key=admin_client._api_key,
            timeout=0.001,  # 1ms - should timeout
        )

        with pytest.raises(CompresrError):
            client.compress(context="Long context " * 100, compression_model_name="A_CMPRSR_V1")
