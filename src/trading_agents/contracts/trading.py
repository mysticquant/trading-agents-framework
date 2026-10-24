"""Trade proposal, risk assessment, and portfolio decision contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TradeAction = Literal["buy", "sell", "hold"]
OrderType = Literal["market", "limit"]
FinalAction = Literal["buy", "sell", "hold", "reject"]


class ExecutionCosts(BaseModel):
    """Per-trade frictions the broker charges on a fill. Agents that size buys
    against available cash use this so a trade trimmed to "all remaining cash"
    is still affordable once slippage and commission are added. Defaults
    match `PaperBrokerAdapter`'s defaults."""

    model_config = ConfigDict(frozen=True)

    commission_bps: float = Field(default=5.0, ge=0.0)
    slippage_bps: float = Field(default=5.0, ge=0.0)

    def buy_cost_per_share(self, price: float) -> float:
        return price * (1 + self.slippage_bps / 10_000) * (1 + self.commission_bps / 10_000)

    def max_buy_quantity(self, cash: float, price: float) -> float:
        """Largest quantity whose all-in cost fits in `cash`. Shaved by a
        tiny relative margin so floating-point rounding can't push the
        broker's own cost computation a hair over `cash`."""
        if price <= 0 or cash <= 0:
            return 0.0
        return cash / self.buy_cost_per_share(price) * (1 - 1e-9)


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
