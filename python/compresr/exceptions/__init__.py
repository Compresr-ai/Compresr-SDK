"""
Compresr SDK Exceptions

Exact copies of exceptions from backend.
Single source of truth maintained in backend.
"""

from .exceptions import (
    ApiKeyBudgetError,
    AuthenticationError,
    AuthenticationErrorResponse,
    BudgetLimitError,
    CompresrConnectionError,
    CompresrError,
    CompresrTimeoutError,
    ConnectionErrorResponse,
    ContentPolicyError,
    ContextWindowExceededError,
    DailyLimitError,
    ErrorResponse,
    InsufficientCreditsError,
    ModelNotFoundError,
    NotFoundError,
    NotFoundErrorResponse,
    RateLimitError,
    RateLimitErrorResponse,
    ScopeError,
    ScopeErrorResponse,
    ServerError,
    ServerErrorResponse,
    ServiceUnavailableError,
    TargetAuthenticationError,
    ValidationError,
    ValidationErrorResponse,
)

# Deprecated: shadows builtin, will be removed in 3.0
ConnectionError = CompresrConnectionError
# Deprecated: shadows builtin, will be removed in 3.0
TimeoutError = CompresrTimeoutError

__all__ = [
    "ErrorResponse",
    "ValidationErrorResponse",
    "AuthenticationErrorResponse",
    "RateLimitErrorResponse",
    "ScopeErrorResponse",
    "ServerErrorResponse",
    "NotFoundErrorResponse",
    "ConnectionErrorResponse",
    "CompresrError",
    "AuthenticationError",
    "TargetAuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ScopeError",
    "ServerError",
    "NotFoundError",
    "CompresrConnectionError",
    "InsufficientCreditsError",
    "BudgetLimitError",
    "DailyLimitError",
    "ApiKeyBudgetError",
    "ModelNotFoundError",
    "ContextWindowExceededError",
    "ContentPolicyError",
    "CompresrTimeoutError",
    "ServiceUnavailableError",
]
