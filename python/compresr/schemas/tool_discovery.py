"""Schemas for tool discovery search endpoint."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DeferredTool(BaseModel):
    """Candidate tool metadata sent by Gateway."""

    name: Optional[str] = Field(None, description="Unique tool name")
    description: Optional[str] = Field(None, description="Short tool description")
    definition: Optional[Dict[str, Any]] = Field(None, description="Original full tool schema")


class ToolDiscoverySearchRequest(BaseModel):
    """Request payload for deferred tool selection."""

    pattern: str = Field(..., description="Model-provided search pattern")
    top_k: int = Field(5, description="Maximum number of tools requested")
    always_keep: List[str] = Field(default_factory=list, description="Hints to keep")
    tools: List[DeferredTool] = Field(default_factory=list, description="Deferred candidate tools")


class ToolDiscoverySearchResponse(BaseModel):
    """Response payload with selected tool names."""

    selected_names: List[str] = Field(default_factory=list)


__all__ = [
    "DeferredTool",
    "ToolDiscoverySearchRequest",
    "ToolDiscoverySearchResponse",
]
