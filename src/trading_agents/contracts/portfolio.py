"""Portfolio state contract, shared by live/paper trading and backtesting."""

from datetime import datetime

from pydantic import BaseModel, Field

from trading_agents.contracts.broker import Position


class PortfolioState(BaseModel):
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)
    equity: float
    realized_pnl: float
    as_of: datetime
