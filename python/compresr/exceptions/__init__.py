"""
Compresr SDK Exceptions

Exact copies of exceptions from backend.
Single source of truth maintained in backend.
"""

from .exceptions import (  # Response models; Exception classes
    AuthenticationError,
    AuthenticationErrorResponse,
    CompresrError,
    ConnectionError,
    ConnectionErrorResponse,
    ErrorResponse,
    NotFoundError,
    NotFoundErrorResponse,
    RateLimitError,
    RateLimitErrorResponse,
    ScopeError,
    ScopeErrorResponse,
    ServerError,
    ServerErrorResponse,
    ValidationError,
    ValidationErrorResponse,
)

__all__ = [
    # Response models
    "ErrorResponse",
    "ValidationErrorResponse",
    "AuthenticationErrorResponse",
    "RateLimitErrorResponse",
    "ScopeErrorResponse",
    "ServerErrorResponse",
    "NotFoundErrorResponse",
    "ConnectionErrorResponse",
    # Exception classes
    "CompresrError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ScopeError",
    "ServerError",
    "NotFoundError",
    "ConnectionError",
]
