"""
Base Schemas - Common response patterns.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict


class BaseResponse(BaseModel):
    """Base response with success flag."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = True
    message: Optional[str] = None


class MessageResponse(BaseModel):
    """Simple message response."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = True
    message: str
