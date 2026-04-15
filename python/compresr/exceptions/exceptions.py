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


# =============================================================================
# Budget & Credits Errors
# =============================================================================


class InsufficientCreditsError(CompresrError):
    """User has insufficient credits to complete the request."""

    def __init__(
        self,
        message: str = "Insufficient credits",
        credits_required: Optional[float] = None,
        credits_remaining: Optional[float] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "insufficient_credits")
        self.credits_required = credits_required
        self.credits_remaining = credits_remaining


class BudgetLimitError(CompresrError):
    """User's budget limit has been reached."""

    def __init__(
        self,
        message: str = "Budget limit reached",
        current_budget: Optional[float] = None,
        budget_used: Optional[float] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "budget_limit_reached")
        self.current_budget = current_budget
        self.budget_used = budget_used


class DailyLimitError(CompresrError):
    """Daily request limit exceeded."""

    def __init__(
        self,
        message: str = "Daily limit exceeded",
        daily_limit: Optional[int] = None,
        requests_used: Optional[int] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "daily_limit_exceeded")
        self.daily_limit = daily_limit
        self.requests_used = requests_used


class ApiKeyBudgetError(CompresrError):
    """Per-API-key budget limit exceeded."""

    def __init__(
        self,
        message: str = "API key budget exceeded",
        api_key_budget: Optional[float] = None,
        api_key_used: Optional[float] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "api_key_budget_exceeded")
        self.api_key_budget = api_key_budget
        self.api_key_used = api_key_used


# =============================================================================
# Model & Input Errors
# =============================================================================


class ModelNotFoundError(CompresrError):
    """Requested model does not exist."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        available_models: Optional[list] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "model_not_found")
        self.model_name = model_name
        self.available_models = available_models or []


class ContextWindowExceededError(CompresrError):
    """Input exceeds model's context window size."""

    def __init__(
        self,
        message: str = "Context window exceeded",
        max_tokens: Optional[int] = None,
        actual_tokens: Optional[int] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "context_window_exceeded")
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


class ContentPolicyError(CompresrError):
    """Content violates provider's content policy."""

    def __init__(
        self,
        message: str = "Content policy violation",
        provider: Optional[str] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "content_policy_violation")
        self.provider = provider


# =============================================================================
# Service Errors
# =============================================================================


class TimeoutError(CompresrError):
    """Request timed out."""

    def __init__(
        self,
        message: str = "Request timed out",
        timeout_seconds: Optional[int] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "timeout")
        self.timeout_seconds = timeout_seconds


class ServiceUnavailableError(CompresrError):
    """Service is temporarily unavailable."""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        service: Optional[str] = None,
        retry_after: Optional[int] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "service_unavailable")
        self.service = service
        self.retry_after = retry_after


class TargetAuthenticationError(CompresrError):
    """User's target LLM API key is invalid."""

    def __init__(
        self,
        message: str = "Invalid target API key",
        provider: Optional[str] = None,
        response_data: Optional[dict] = None,
    ):
        super().__init__(message, response_data, "target_authentication_error")
        self.provider = provider
