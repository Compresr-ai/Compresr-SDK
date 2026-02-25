"""
Compresr SDK Schemas

Exact copies of schemas from backend.
Single source of truth maintained in backend.
"""

from ..exceptions import (  # Response models; Exception classes
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

# Import local schemas
from .base import BaseResponse, MessageResponse
from .inference import (  # Streaming; Compression
    CompressRequest,
    CompressResponse,
    CompressResult,
    StreamChunk,
)
from .tool_discovery import (
    DeferredTool,
    ToolDiscoverySearchRequest,
    ToolDiscoverySearchResponse,
)
from .usage import MoneyBalanceResponse, MoneyBalanceResult

__all__ = [
    # Base
    "BaseResponse",
    "MessageResponse",
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
    # Streaming
    "StreamChunk",
    # Compression
    "CompressRequest",
    "CompressResponse",
    "CompressResult",
    # Tool Discovery
    "DeferredTool",
    "ToolDiscoverySearchRequest",
    "ToolDiscoverySearchResponse",
    # Usage
    "MoneyBalanceResponse",
    "MoneyBalanceResult",
]
