"""
Compresr SDK Exceptions

Exception classes and response models for API error handling.
"""

from typing import Optional

from pydantic import BaseModel

# =============================================================================
# Response Models (for documentation)
# =============================================================================


class ErrorResponse(BaseModel):
    """Generic error response."""

    success: bool = False
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    """Validation error - invalid input."""

    success: bool = False
    error: str
    code: str = "validation_error"
    field: Optional[str] = None


class AuthenticationErrorResponse(BaseModel):
    """Authentication error - invalid/missing API key."""

    success: bool = False
    error: str = "Authentication failed"
    code: str = "authentication_error"


class RateLimitErrorResponse(BaseModel):
    """Rate limit error - too many requests."""

    success: bool = False
    error: str = "Rate limit exceeded"
    code: str = "rate_limit_exceeded"
    retry_after: Optional[int] = None


class ScopeErrorResponse(BaseModel):
    """Scope error - API key lacks permission."""

    success: bool = False
    error: str = "Insufficient permissions"
    code: str = "scope_error"


class ServerErrorResponse(BaseModel):
    """Server error - internal error."""

    success: bool = False
    error: str = "Internal server error"
    code: str = "server_error"


class NotFoundErrorResponse(BaseModel):
    """Not found error - resource doesn't exist."""

    success: bool = False
    error: str = "Resource not found"
    code: str = "not_found"


class ConnectionErrorResponse(BaseModel):
    """Connection error - failed to connect."""

    success: bool = False
    error: str = "Connection failed"
    code: str = "connection_error"


# =============================================================================
# Exception Classes
# =============================================================================


class CompresrError(Exception):
    """Base exception for all Compresr errors."""

    def __init__(
        self, message: str, response_data: Optional[dict] = None, code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.response_data = response_data or {}
        self.code = code


class AuthenticationError(CompresrError):
    """Invalid or missing API key."""

    def __init__(
        self, message: str = "Authentication failed", response_data: Optional[dict] = None
    ):
        super().__init__(message, response_data, "authentication_error")


class RateLimitError(CompresrError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "rate_limit_exceeded")
        self.retry_after = retry_after


class ValidationError(CompresrError):
    """Request validation failed."""

    def __init__(
        self, message: str, field: Optional[str] = None, response_data: Optional[dict] = None
    ):
        super().__init__(message, response_data, "validation_error")
        self.field = field


class ScopeError(CompresrError):
    """API key lacks required permissions."""

    def __init__(
        self,
        message: str,
        required_scope: Optional[str] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "scope_error")
        self.required_scope = required_scope


class ServerError(CompresrError):
    """Internal server error."""

    def __init__(
        self, message: str = "Internal server error", response_data: Optional[dict] = None
    ):
        super().__init__(message, response_data, "server_error")


class NotFoundError(CompresrError):
    """Resource not found."""

    def __init__(
        self, message: str, resource: Optional[str] = None, response_data: Optional[dict] = None
    ):
        super().__init__(message, response_data, "not_found")
        self.resource = resource


class ConnectionError(CompresrError):
    """Connection to service failed."""

    def __init__(
        self, message: str, service: Optional[str] = None, response_data: Optional[dict] = None
    ):
        super().__init__(message, response_data, "connection_error")
        self.service = service
