"""
Inference Schemas

Schemas for compression endpoints.
Matches backend schemas exactly - backend is single source of truth.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

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
    """Request to compress a single context.

    For multiple contexts, use the /batch endpoint.
    """

    context: str = Field(..., min_length=1, description="Context text to compress (single string)")
    compression_model_name: str = Field(..., description="Compression model (e.g., 'espresso_v1')")
    query: Optional[str] = Field(
        None,
        min_length=1,
        description="Query for query-specific models (required for latte_v1)",
    )
    target_compression_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        description="Target compression ratio: 0-1 (strength) or >1 for Nx factor (e.g., 60=60x). Max 200.",
    )
    coarse: Optional[bool] = Field(
        None,
        description="Coarse-grained (paragraph-level) compression. True=faster, False=token-level (default). Only for latte_v1.",
    )
    source: str = Field(default="sdk:python", description="Source of request for analytics")


# =============================================================================
# Compression Results
# =============================================================================


class CompressResult(BaseModel):
    """Compression result with metrics for single context."""

    model_config = {"from_attributes": True, "protected_namespaces": ()}

    original_context: str
    compressed_context: str
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


# =============================================================================
# Batch Compression
# =============================================================================
# Agnostic Batch Compression
# =============================================================================


class AgnosticBatchInput(BaseModel):
    """A single input in an agnostic batch compression request."""

    context: str = Field(..., min_length=1, description="Context text to compress")


class AgnosticBatchRequest(BaseModel):
    """Batch request for question-agnostic compression (no query required)."""

    inputs: List[AgnosticBatchInput] = Field(
        ..., min_length=1, max_length=100, description="List of contexts to compress"
    )
    compression_model_name: str = Field(..., description="Compression model to use")
    target_compression_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        description="Target compression ratio: 0-1 (strength) or >1 for Nx factor",
    )
    source: str = Field(default="sdk:python", description="Source of request")


# =============================================================================
# Query-Specific Batch Compression
# =============================================================================


class CompressBatchInput(BaseModel):
    """A single input in a query-specific batch compression request."""

    context: str = Field(..., min_length=1, description="Context text to compress")
    query: str = Field(..., min_length=1, description="Query for this context (required)")


class CompressBatchRequest(BaseModel):
    """Batch request for question-specific compression."""

    inputs: List[CompressBatchInput] = Field(
        ..., min_length=1, max_length=100, description="List of context+query pairs"
    )
    compression_model_name: str = Field(..., description="Compression model to use")
    target_compression_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        description="Target compression ratio: 0-1 (strength) or >1 for Nx factor",
    )
    coarse: Optional[bool] = Field(
        None,
        description="Coarse-grained (paragraph-level) compression. Only for latte_v1.",
    )
    source: str = Field(default="sdk:python", description="Source of request")


class CompressBatchItemResult(BaseModel):
    """Result for a single item in batch compression."""

    model_config = {"from_attributes": True, "protected_namespaces": ()}

    original_context: str
    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    actual_compression_ratio: float
    tokens_saved: int
    duration_ms: int


class CompressBatchResult(BaseModel):
    """Batch compression result with aggregated metrics."""

    model_config = {"from_attributes": True, "protected_namespaces": ()}

    results: List[CompressBatchItemResult] = Field(default_factory=list)
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_tokens_saved: int = 0
    average_compression_ratio: float = 0.0
    count: int = 0


class CompressBatchResponse(BaseResponse):
    """Response for batch compression."""

    data: Optional[CompressBatchResult] = None
