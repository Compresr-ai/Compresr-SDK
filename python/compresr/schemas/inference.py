"""
Inference Schemas

Schemas for compression endpoints.
Matches backend schemas exactly - backend is single source of truth.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .base import BaseResponse


# Compression ratio constants
# SDK only validates non-negative; backend enforces full range (0-200)
class CompressionConfig:
    MIN_RATIO = 0.0  # Only validate non-negative
    DEFAULT_RATIO = 0.5  # Remove 50%


# =============================================================================
# Streaming
# =============================================================================


class StreamChunk(BaseModel):
    """A chunk of streamed response."""

    content: str
    done: bool = False
    error: Optional[str] = None


# =============================================================================
# Compression Requests
# =============================================================================


class CompressRequest(BaseModel):
    """Request to compress a context.

    context can be:
    - str: Single context -> returns single compressed_context
    - List[str]: Multiple contexts -> returns list of compressed_context
    """

    context: Union[str, List[str]] = Field(
        ..., description="Context text to compress - single string or list of strings"
    )
    compression_model_name: str = Field(..., description="Compression model (e.g., 'espresso_v1')")
    query: Optional[str] = Field(
        None,
        min_length=1,
        description="Query for query-specific models (required for latte_v1, coldbrew_v1)",
    )
    target_compression_ratio: Optional[float] = Field(
        None,
        ge=0.0,  # Only validate non-negative; backend enforces upper bound (200)
        description="Target compression ratio: 0-1 (strength) or >1 for Nx factor (e.g., 60=60x). Max 200.",
    )
    source: str = Field(default="sdk:python", description="Source of request for analytics")

    @field_validator("context")
    @classmethod
    def validate_context_not_empty(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        """Validate context is not empty."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("context must not be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("context list must not be empty")
            for i, item in enumerate(v):
                if not item.strip():
                    raise ValueError(f"context[{i}] must not be empty")
        return v


# =============================================================================
# Compression Results
# =============================================================================


class CompressResult(BaseModel):
    """Compression result with metrics.

    compressed_context type matches input context type:
    - If context was str -> compressed_context is str
    - If context was List[str] -> compressed_context is List[str]
    """

    model_config = {"from_attributes": True, "protected_namespaces": ()}

    original_context: Union[str, List[str]]
    compressed_context: Union[str, List[str]]
    original_tokens: int
    compressed_tokens: int
    actual_compression_ratio: float
    tokens_saved: int
    duration_ms: int
    target_compression_ratio: Optional[float] = None


# =============================================================================
# Responses
# =============================================================================


class CompressResponse(BaseResponse):
    """Response for single compression."""

    data: Optional[CompressResult] = None
