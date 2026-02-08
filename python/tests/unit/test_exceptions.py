"""
Unit Tests for SDK Exceptions

Tests for custom exception classes.
"""

import pytest

from compresr.exceptions import AuthenticationError, CompresrError, RateLimitError, ValidationError


class TestCompresrError:
    """Test base CompresrError exception."""

    def test_create_error(self):
        """Test creating a base error."""
        error = CompresrError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.response_data == {} or error.response_data is None

    def test_error_with_response_data(self):
        """Test error with response data."""
        response_data = {"error": "details", "code": 500}
        error = CompresrError("Error occurred", response_data=response_data)
        assert error.response_data == response_data
        assert error.message == "Error occurred"

    def test_error_is_exception(self):
        """Test that CompresrError is an Exception."""
        error = CompresrError("Test")
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self):
        """Test that error can be raised and caught."""
        with pytest.raises(CompresrError) as exc_info:
            raise CompresrError("Test error")
        assert "Test error" in str(exc_info.value)


class TestAuthenticationError:
    """Test AuthenticationError exception."""

    def test_create_auth_error(self):
        """Test creating authentication error."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, CompresrError)

    def test_auth_error_inherits_from_base(self):
        """Test AuthenticationError inherits from CompresrError."""
        error = AuthenticationError("Unauthorized")
        assert isinstance(error, CompresrError)
        assert isinstance(error, Exception)

    def test_auth_error_with_response_data(self):
        """Test auth error with response data."""
        response = {"error": "Invalid token", "status": 401}
        error = AuthenticationError("Auth failed", response_data=response)
        assert error.response_data["status"] == 401


class TestRateLimitError:
    """Test RateLimitError exception."""

    def test_create_rate_limit_error(self):
        """Test creating rate limit error."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, CompresrError)

    def test_rate_limit_error_inherits(self):
        """Test RateLimitError inheritance."""
        error = RateLimitError("Too many requests")
        assert isinstance(error, CompresrError)
        assert isinstance(error, Exception)

    def test_rate_limit_with_retry_after(self):
        """Test rate limit error with retry-after data."""
        response = {"retry_after": 60, "limit": 100}
        error = RateLimitError("Rate limited", response_data=response)
        assert error.response_data["retry_after"] == 60


class TestValidationError:
    """Test ValidationError exception."""

    def test_create_validation_error(self):
        """Test creating validation error."""
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert isinstance(error, CompresrError)

    def test_validation_error_inherits(self):
        """Test ValidationError inheritance."""
        error = ValidationError("Validation failed")
        assert isinstance(error, CompresrError)
        assert isinstance(error, Exception)

    def test_validation_with_details(self):
        """Test validation error with field details."""
        response = {
            "errors": [
                {"field": "context", "message": "Cannot be empty"},
                {"field": "ratio", "message": "Must be between 0 and 1"},
            ]
        }
        error = ValidationError("Validation failed", response_data=response)
        assert len(error.response_data["errors"]) == 2
        assert error.response_data["errors"][0]["field"] == "context"


class TestExceptionHierarchy:
    """Test exception hierarchy and catching."""

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Auth failed")

    def test_catch_base_exception(self):
        """Test catching via base CompresrError."""
        with pytest.raises(CompresrError):
            raise RateLimitError("Rate limited")

    def test_catch_as_generic_exception(self):
        """Test catching as generic Exception."""
        with pytest.raises(Exception):
            raise ValidationError("Invalid")

    def test_exception_chain(self):
        """Test exception can be chained."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise CompresrError("Wrapped error") from e
        except CompresrError as ce:
            assert ce.__cause__ is not None
            assert isinstance(ce.__cause__, ValueError)
