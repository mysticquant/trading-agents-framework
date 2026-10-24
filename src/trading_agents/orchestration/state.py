"""Shared state that flows through one decision cycle. Plain (mutable)
dataclasses, not pydantic models — this is internal orchestration
bookkeeping filled in incrementally as the pipeline runs, not a wire
contract.
"""

from dataclasses import dataclass, field
from datetime import datetime

from trading_agents.contracts.analysis import AnalystReport, ResearchDebateResult
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.contracts.screening import StockScreenResult
from trading_agents.contracts.trading import PortfolioDecision, RiskAssessment, TradeProposal


@dataclass
class TickerPipelineState:
    ticker: str
    analyst_reports: list[AnalystReport] = field(default_factory=list)
    research: ResearchDebateResult | None = None
    trade_proposal: TradeProposal | None = None
    risk_assessment: RiskAssessment | None = None
    portfolio_decision: PortfolioDecision | None = None
    # Set when any stage of this ticker's pipeline raised; the ticker is
    # then left out of the Portfolio Manager step for this cycle.
    error: str | None = None


@dataclass
class DecisionCycleState:
    cycle_id: str
    as_of: datetime
    ticker_universe: list[str]
    portfolio_state_snapshot: PortfolioState
    screen_results: list[StockScreenResult] = field(default_factory=list)
    per_ticker: dict[str, TickerPipelineState] = field(default_factory=dict)
