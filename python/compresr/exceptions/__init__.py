"""
Compresr SDK Exceptions

Exact copies of exceptions from backend.
Single source of truth maintained in backend.
"""

from .exceptions import (  # Response models; Exception classes
    ApiKeyBudgetError,
    AuthenticationError,
    AuthenticationErrorResponse,
    BudgetLimitError,
    CompresrError,
    ConnectionError,
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
    TimeoutError,
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
    "TargetAuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ScopeError",
    "ServerError",
    "NotFoundError",
    "ConnectionError",
    # Budget & Credits
    "InsufficientCreditsError",
    "BudgetLimitError",
    "DailyLimitError",
    "ApiKeyBudgetError",
    # Model & Input
    "ModelNotFoundError",
    "ContextWindowExceededError",
    "ContentPolicyError",
    # Service
    "TimeoutError",
    "ServiceUnavailableError",
]
