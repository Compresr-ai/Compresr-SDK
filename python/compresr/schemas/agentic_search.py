"""
Agentic Search Schemas

Schemas for agentic search endpoints (macchiato_v1).
Matches backend schemas exactly - backend is single source of truth.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .base import BaseResponse


class SearchRequest(BaseModel):
    """Request for agentic search over a pre-indexed knowledge base.

    query: Natural language search question (REQUIRED)
    index_name: Name of pre-built index to search (REQUIRED)
    max_time_s: Maximum search time in seconds (optional, default 4.5)
    """

    query: str = Field(..., min_length=1, description="Natural language search question")
    index_name: str = Field(..., min_length=1, description="Name of pre-built index to search")
    max_time_s: float = Field(
        4.5,
        ge=0.1,
        le=30.0,
        description="Maximum search time in seconds (default: 4.5)",
    )
    compression_model_name: str = Field(..., description="Search model (e.g., 'macchiato_v1')")
    source: str = Field(default="sdk:python", description="Source of request for analytics")

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("index_name")
    @classmethod
    def validate_index_name_not_empty(cls, v: str) -> str:
        """Validate index_name is not empty."""
        if not v.strip():
            raise ValueError("index_name must not be empty")
        return v


class SearchResult(BaseModel):
    """Result of agentic search with metrics."""

    model_config = {"from_attributes": True, "protected_namespaces": ()}

    chunks: List[str] = Field(..., description="Relevant chunks from the knowledge base")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of returned chunks")
    chunks_count: int = Field(..., description="Number of chunks returned")
    original_tokens: int = Field(..., description="Total tokens in indexed content")
    compressed_tokens: int = Field(..., description="Tokens in returned chunks")
    compression_ratio: float = Field(..., description="Compression ratio achieved")
    duration_ms: int = Field(0, description="Processing time in milliseconds")
    index_name: str = Field(..., description="Name of the index searched")
    cost: Optional[Dict[str, Any]] = Field(None, description="Cost info from Opie")


class SearchResponse(BaseResponse):
    """Response for agentic search."""

    data: Optional[SearchResult] = None


__all__ = [
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
]
