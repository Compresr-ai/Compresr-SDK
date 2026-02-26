"""
Integration Tests for Error Handling

Tests how the SDK handles various error scenarios from the API.
"""

import pytest

from compresr import CompressionClient, FilterClient
from compresr.exceptions import AuthenticationError, CompresrError, ServerError, ValidationError


@pytest.fixture
def client_with_invalid_key():
    """Create client with invalid API key."""
    return CompressionClient(api_key="cmp_invalid_key_12345")


@pytest.fixture
def filter_client_with_invalid_key():
    """Create filter client with invalid API key."""
    return FilterClient(api_key="cmp_invalid_key_12345")


class TestAuthenticationErrors:
    """Test authentication error handling."""

    def test_invalid_api_key(self, client_with_invalid_key):
        """Test that invalid API key raises AuthenticationError."""
        with pytest.raises(AuthenticationError):
            client_with_invalid_key.compress(
                context="Test context", compression_model_name="espresso_v1"
            )

    def test_missing_api_key(self):
        """Test that missing API key raises AuthenticationError."""
        with pytest.raises(AuthenticationError):
            CompressionClient(api_key="")

    def test_filter_invalid_api_key(self, filter_client_with_invalid_key):
        """Test that invalid API key raises AuthenticationError for FilterClient."""
        with pytest.raises(AuthenticationError):
            filter_client_with_invalid_key.filter(
                chunks=["Test chunk"],
                query="Test query?",
                compression_model_name="coldbrew_v1",
            )


class TestValidationErrors:
    """Test validation error handling."""

    def test_empty_context(self, admin_client):
        """Test that empty context raises error (backend validation)."""
        with pytest.raises((ValidationError, ServerError, CompresrError)):
            admin_client.compress(context="", compression_model_name="espresso_v1")

    def test_invalid_compression_ratio_high(self, admin_client):
        """Test that ratio > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            admin_client.compress(
                context="Test context",
                compression_model_name="espresso_v1",
                target_compression_ratio=1.5,
            )

    def test_invalid_compression_ratio_negative(self, admin_client):
        """Test that negative ratio raises ValidationError."""
        with pytest.raises(ValidationError):
            admin_client.compress(
                context="Test context",
                compression_model_name="espresso_v1",
                target_compression_ratio=-0.5,
            )

    def test_invalid_model_name(self, admin_client):
        """Test that invalid model name is handled."""
        with pytest.raises((ValidationError, CompresrError)):
            admin_client.compress(
                context="Test context", compression_model_name="invalid_model_xyz"
            )


class TestFilterErrors:
    """Test filter-specific error handling."""

    def test_filter_empty_chunks(self, admin_api_key):
        """Test that empty chunks list raises error."""
        if not admin_api_key:
            pytest.skip("Admin API key not available")
        client = FilterClient(api_key=admin_api_key)
        with pytest.raises(ValidationError):
            client.filter(chunks=[], query="Test?", compression_model_name="coldbrew_v1")

    def test_filter_missing_api_key(self):
        """Test that missing API key raises AuthenticationError for FilterClient."""
        with pytest.raises(AuthenticationError):
            FilterClient(api_key="")


class TestAsyncErrors:
    """Test async error handling."""

    @pytest.mark.asyncio
    async def test_async_invalid_key(self, client_with_invalid_key):
        """Test async compression with invalid key."""
        with pytest.raises(AuthenticationError):
            await client_with_invalid_key.compress_async(
                context="Test context", compression_model_name="espresso_v1"
            )

    @pytest.mark.asyncio
    async def test_async_validation_error(self, admin_client):
        """Test async compression with validation error (backend validation)."""
        with pytest.raises((ValidationError, ServerError, CompresrError)):
            await admin_client.compress_async(context="", compression_model_name="espresso_v1")


class TestConnectionErrors:
    """Test connection and network error handling."""

    def test_timeout_handling(self, admin_client):
        """Test that timeout is respected."""
        client = CompressionClient(
            api_key=admin_client._api_key,
            timeout=0.001,  # 1ms - should timeout
        )

        with pytest.raises(CompresrError):
            client.compress(context="Long context " * 100, compression_model_name="espresso_v1")
