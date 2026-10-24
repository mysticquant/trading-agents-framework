"""Backtest result contracts."""

from datetime import date, datetime

from pydantic import BaseModel, Field

from trading_agents.contracts.broker import Order
from trading_agents.contracts.portfolio import PortfolioState


class EquityCurvePoint(BaseModel):
    as_of: datetime
    equity: float
    cash: float


class BacktestResult(BaseModel):
    start_date: date
    end_date: date
    ticker_universe: list[str]
    equity_curve: list[EquityCurvePoint] = Field(default_factory=list)
    trade_log: list[Order] = Field(default_factory=list)
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    final_portfolio_state: PortfolioState
