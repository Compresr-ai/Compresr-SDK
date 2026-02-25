"""
Usage Schemas

Balance and usage tracking for SDK.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .base import BaseResponse

# =============================================================================
# Balance Schemas
# =============================================================================


class MoneyBalanceResult(BaseModel):
    """Current balance in monetary terms."""

    model_config = ConfigDict(from_attributes=True)

    # Money Tracking
    total_uploaded_money: float = Field(0.0, description="Total money uploaded to account (USD)")
    total_used_money: float = Field(0.0, description="Total money consumed via API usage (USD)")
    current_remaining_credits: float = Field(0.0, description="Remaining balance (USD)")
    percentage_used: float = Field(0.0, description="Percentage of uploaded money used")

    # Budget Management
    monthly_budget: float = Field(200.0, description="User's monthly budget limit (USD)")
    budget_used_this_month: float = Field(0.0, description="Money used this month (USD)")
    budget_remaining_this_month: float = Field(
        200.0, description="Budget remaining this month (USD)"
    )

    # Account info
    tier: str = Field(default="free", description="Current subscription tier")
    is_admin: bool = Field(False, description="Whether user is admin (infinite usage)")
    account_created_at: Optional[str] = Field(None, description="Account creation date (ISO)")


class MoneyBalanceResponse(BaseResponse):
    """Response for money balance endpoint."""

    result: Optional[MoneyBalanceResult] = None
