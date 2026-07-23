"""Compression schemas. Mirrors backend types — backend is source of truth."""

from typing import List, Optional

from pydantic import BaseModel, Field

from .base import BaseResponse


class CompressionConfig:
    MIN_RATIO = 0.0
    DEFAULT_RATIO = 0.5


class StreamChunk(BaseModel):
    content: str
    done: bool = False
    error: Optional[str] = None


class CompressRequest(BaseModel):
    """Single-context compression request.

    ``query`` is optional client-side; the backend decides whether the
    chosen ``compression_model_name`` requires it. ``latte_v1`` does."""

    context: str = Field(..., min_length=1, description="Context text to compress")
    query: Optional[str] = Field(
        None,
        min_length=1,
        description="Query expressing the LLM's intent (required by latte_v1)",
    )
    compression_model_name: str = Field(
        default="latte_v1",
        description="Compression model name — backend validates",
    )
    target_compression_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        description="0-1 (fraction to remove) or >1 for Nx factor (e.g. 60 = 60x). Max 200.",
    )
    coarse: Optional[bool] = Field(
        None,
        description="Paragraph-level (True, default, faster) vs token-level (False, fine-grained). Omit to use backend default (True).",
    )
    heuristic_chunking: Optional[bool] = Field(
        None,
        description="Use heuristic chunking for better structure preservation.",
    )
    disable_placeholders: Optional[bool] = Field(
        None,
        description="Disable placeholder tokens in compressed output.",
    )
    # latte_v2-only knobs. Backend validates that they're only sent to v2
    # models; on latte_v1 the backend returns 422 with a clear message.
    dynamic: Optional[bool] = Field(
        None,
        description=(
            "latte_v2 only. Use adaptive (Kneedle elbow) selection instead "
            "of a fixed ratio. When True, target_compression_ratio is ignored."
        ),
    )
    dynamic_min_ratio: Optional[float] = Field(
        None,
        description="latte_v2 only. Floor on adaptive compression (server default 1.5).",
    )
    dynamic_max_ratio: Optional[float] = Field(
        None,
        description="latte_v2 only. Ceiling on adaptive compression (server default 10.0).",
    )
    source: str = Field(default="sdk:python", description="Source tag for analytics")


class CompressResult(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    original_context: Optional[str] = None
    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    actual_compression_ratio: float
    tokens_saved: int
    duration_ms: int
    target_compression_ratio: Optional[float] = None


class CompressResponse(BaseResponse):
    data: Optional[CompressResult] = None


class CompressBatchInput(BaseModel):
    context: str = Field(..., min_length=1)
    query: Optional[str] = Field(None, min_length=1)


class CompressBatchRequest(BaseModel):
    inputs: List[CompressBatchInput] = Field(..., min_length=1, max_length=100)
    compression_model_name: str = Field(default="latte_v1")
    target_compression_ratio: Optional[float] = Field(None, ge=0.0)
    coarse: Optional[bool] = None
    heuristic_chunking: Optional[bool] = None
    disable_placeholders: Optional[bool] = None
    # latte_v2-only knobs (shared across the whole batch).
    dynamic: Optional[bool] = None
    dynamic_min_ratio: Optional[float] = None
    dynamic_max_ratio: Optional[float] = None
    source: str = Field(default="sdk:python")


class CompressBatchItemResult(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    original_context: Optional[str] = None
    compressed_context: str
    original_tokens: int
    compressed_tokens: int
    actual_compression_ratio: float
    tokens_saved: int
    duration_ms: int


class CompressBatchResult(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    results: List[CompressBatchItemResult] = Field(default_factory=list)
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_tokens_saved: int = 0
    average_compression_ratio: float = 0.0
    count: int = 0


class CompressBatchResponse(BaseResponse):
    data: Optional[CompressBatchResult] = None
