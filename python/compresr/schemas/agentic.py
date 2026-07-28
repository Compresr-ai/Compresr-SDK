"""Agentic compression schemas (tool-output). Mirrors backend types — backend
is source of truth (``app/schemas/compression_agentic.py``)."""

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from .base import BaseResponse


class CompressToolOutputRequest(BaseModel):
    """Tool-output compression request.

    ``tool_output`` accepts a single string or a list of strings; the response
    ``compressed_output`` mirrors the input type. ``query`` is optional
    client-side; the backend decides whether the chosen model requires it
    (``toc_latte_*`` models do)."""

    tool_output: Union[str, List[str]] = Field(
        ..., description="Tool output to compress — single string or list of strings"
    )
    query: Optional[str] = Field(
        None,
        description="Query expressing the tool call's intent (required by toc_latte models)",
    )
    tool_name: str = Field(
        ..., min_length=1, description="Name of the tool that produced the output"
    )
    compression_model_name: str = Field(
        default="toc_latte_v2",
        description="Compression model name — backend validates",
    )
    target_compression_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=200.0,
        description="0-1 (fraction to remove) or >1 for Nx factor (e.g. 5 = 5x). Max 200.",
    )
    source: str = Field(default="sdk:python", description="Source tag for analytics")


class CompressToolOutputResult(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    compressed_output: Union[str, List[str]]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tool_name: str = ""
    duration_ms: int = 0


class CompressToolOutputResponse(BaseResponse):
    data: Optional[CompressToolOutputResult] = None
