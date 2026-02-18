"""
Inference Schemas

Schemas for compression endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .base import BaseResponse


# Compression ratio constants
class CompressionConfig:
    MIN_RATIO = 0.1  # Remove minimum 10%
    MAX_RATIO = 0.9  # Remove maximum 90%
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
    """Request to compress a context."""

    context: str = Field(..., min_length=1, description="Context text to compress")
    compression_model_name: str = Field(..., description="Compression model (e.g., 'A_CMPRSR_V1')")
    question: Optional[str] = Field(None, min_length=1, description="Question for question-specific models")
    target_compression_ratio: Optional[float] = Field(
        None, ge=CompressionConfig.MIN_RATIO, le=CompressionConfig.MAX_RATIO
    )
    source: str = Field(default="sdk:python", description="Source of request for analytics")


class BatchCompressRequest(BaseModel):
    """Request to compress multiple contexts."""

    contexts: List[str] = Field(..., min_length=1, max_length=100)
    compression_model_name: str = Field(..., description="Compression model (e.g., 'A_CMPRSR_V1')")
    question: Optional[str] = Field(None, min_length=1, description="Question for question-specific models")
    target_compression_ratio: Optional[float] = Field(
        None, ge=CompressionConfig.MIN_RATIO, le=CompressionConfig.MAX_RATIO
    )
    source: str = Field(default="sdk:python", description="Source of request for analytics")

    @field_validator("contexts")
    @classmethod
    def validate_contexts(cls, v: list[str]) -> list[str]:
        """Ensure no empty strings in contexts list."""
        if any(not ctx.strip() for ctx in v):
            raise ValueError("All contexts must be non-empty strings")
        return v


# =============================================================================
# Compression Results
# =============================================================================


class CompressResult(BaseModel):
    """Compression result with metrics."""

    model_config = {"from_attributes": True, "protected_namespaces": ()}

    original_context: str
    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    actual_compression_ratio: float
    tokens_saved: int
    duration_ms: int
    target_compression_ratio: Optional[float] = None


class BatchCompressResult(BaseModel):
    """Results for batch compression."""

    results: List[CompressResult] = []
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_tokens_saved: int = 0
    average_compression_ratio: float = 0.0
    count: int = 0


# =============================================================================
# Responses
# =============================================================================


class CompressResponse(BaseResponse):
    """Response for single compression."""

    data: Optional[CompressResult] = None


class BatchCompressResponse(BaseResponse):
    """Response for batch compression."""

    data: Optional[BatchCompressResult] = None
