"""Trade proposal, risk assessment, and portfolio decision contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TradeAction = Literal["buy", "sell", "hold"]
OrderType = Literal["market", "limit"]
FinalAction = Literal["buy", "sell", "hold", "reject"]


class TradeProposal(BaseModel):
    """What the Trader agent wants to do, before risk/portfolio review."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    action: TradeAction
    quantity: float = Field(ge=0.0)
    order_type: OrderType
    limit_price: float | None = None
    rationale: str
    time_horizon: str


class RiskAssessment(BaseModel):
    """Independent risk review of one TradeProposal."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    approved: bool
    max_position_size_pct: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)
    adjusted_quantity: float = Field(ge=0.0)
    rationale: str


class PortfolioDecision(BaseModel):
    """Final, cross-ticker allocation decision made by the Portfolio Manager."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    final_action: FinalAction
    final_quantity: float = Field(ge=0.0)
    approved_by: str
    rationale: str
    decided_at: datetime
