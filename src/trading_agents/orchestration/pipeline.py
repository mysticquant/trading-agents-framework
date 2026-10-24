"""Hand-rolled async orchestrator for one decision cycle.

The pipeline is a fixed, acyclic DAG known at compile time: Screener (once,
over the whole universe) -> per shortlisted ticker, the four Analysts in
parallel -> Researcher -> Trader -> RiskManager -> PortfolioManager (the
single sequential join point at the end). A graph framework (e.g.
LangGraph) was considered and intentionally not used: this DAG has no
cycles or conditional branching, so plain `asyncio.gather` over a shared
state object gives full concurrency control with no added dependency.
Revisit that choice only if a future version needs cyclic re-debate or
human-in-the-loop branching.

`run_cycle` is the single entry point used by both the backtest runner and
(later) a live/paper trading loop.
"""

import asyncio
import uuid
from datetime import datetime

from trading_agents.agents.analysts.base import AnalystInput
from trading_agents.agents.analysts.fundamentals import FundamentalsAnalystAgent
from trading_agents.agents.analysts.news import NewsAnalystAgent
from trading_agents.agents.analysts.sentiment import SentimentAnalystAgent
from trading_agents.agents.analysts.technical import TechnicalAnalystAgent, TechnicalAnalystInput
from trading_agents.agents.portfolio_manager import (
    PortfolioManagerAgent,
    PortfolioManagerInput,
    ProposalAssessmentPair,
)
from trading_agents.agents.researcher import ResearcherAgent, ResearcherInput
from trading_agents.agents.risk_manager import RiskConfig, RiskManagerAgent, RiskManagerInput
from trading_agents.agents.screener import (
    ScreenerAgent,
    ScreenerInput,
    compute_prefilter_candidates,
)
from trading_agents.agents.trader import TraderAgent, TraderInput
from trading_agents.contracts.market import MarketBar
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.indicators.technical import bars_to_frame
from trading_agents.orchestration.state import DecisionCycleState, TickerPipelineState


class DecisionCycleOrchestrator:
    def __init__(
        self,
        *,
        screener: ScreenerAgent,
        fundamentals_analyst: FundamentalsAnalystAgent,
        sentiment_analyst: SentimentAnalystAgent,
        news_analyst: NewsAnalystAgent,
        technical_analyst: TechnicalAnalystAgent,
        researcher: ResearcherAgent,
        trader: TraderAgent,
        risk_manager: RiskManagerAgent,
        portfolio_manager: PortfolioManagerAgent,
        risk_config: RiskConfig | None = None,
        screener_top_n: int = 10,
        max_concurrent_tickers: int = 5,
    ) -> None:
        self.screener = screener
        self.fundamentals_analyst = fundamentals_analyst
        self.sentiment_analyst = sentiment_analyst
        self.news_analyst = news_analyst
        self.technical_analyst = technical_analyst
        self.researcher = researcher
        self.trader = trader
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.risk_config = risk_config or RiskConfig()
        self.screener_top_n = screener_top_n
        self._ticker_semaphore = asyncio.Semaphore(max_concurrent_tickers)

    async def run_cycle(
        self,
        *,
        as_of: datetime,
        universe: list[str],
        market_data: dict[str, list[MarketBar]],
        portfolio_state: PortfolioState,
        fundamentals_evidence: dict[str, dict[str, object]] | None = None,
        sentiment_evidence: dict[str, dict[str, object]] | None = None,
        news_evidence: dict[str, dict[str, object]] | None = None,
    ) -> DecisionCycleState:
        """Evidence dicts are optional and default to empty per ticker —
        Phase 1 has no dedicated fundamentals/sentiment/news data source
        yet (see `agents/analysts/base.py`); callers that have one can pass
        it here without any other change to this method."""
        fundamentals_evidence = fundamentals_evidence or {}
        sentiment_evidence = sentiment_evidence or {}
        news_evidence = news_evidence or {}

        state = DecisionCycleState(
            cycle_id=str(uuid.uuid4()),
            as_of=as_of,
            ticker_universe=universe,
            portfolio_state_snapshot=portfolio_state,
        )

        bars_by_ticker = {
            ticker: bars_to_frame(bars) for ticker, bars in market_data.items() if bars
        }
        candidates = compute_prefilter_candidates(bars_by_ticker, top_n=self.screener_top_n)
        state.screen_results = await self.screener.run(
            ScreenerInput(candidates=candidates, as_of=as_of)
        )
        # Guard against the LLM naming a ticker outside what it was given:
        # only candidates that passed the deterministic pre-filter (and
        # therefore have enough history for the technical analyst) proceed.
        valid_tickers = {candidate.ticker for candidate in candidates}
        shortlist = [
            result.ticker for result in state.screen_results if result.ticker in valid_tickers
        ]

        ticker_states = await asyncio.gather(
            *(
                self._run_ticker_pipeline(
                    ticker=ticker,
                    as_of=as_of,
                    bars=market_data.get(ticker, []),
                    portfolio_state=portfolio_state,
                    fundamentals_evidence=fundamentals_evidence.get(ticker, {}),
                    sentiment_evidence=sentiment_evidence.get(ticker, {}),
                    news_evidence=news_evidence.get(ticker, {}),
                )
                for ticker in shortlist
            )
        )
        for ticker_state in ticker_states:
            state.per_ticker[ticker_state.ticker] = ticker_state

        pairs = [
            ProposalAssessmentPair(proposal=ts.trade_proposal, assessment=ts.risk_assessment)
            for ts in ticker_states
            if ts.trade_proposal is not None and ts.risk_assessment is not None
        ]
        current_prices = {
            ticker: market_data[ticker][-1].close for ticker in shortlist if market_data.get(ticker)
        }
        decisions = await self.portfolio_manager.run(
            PortfolioManagerInput(
                pairs=pairs,
                portfolio_state=portfolio_state,
                current_prices=current_prices,
                as_of=as_of,
            )
        )
        for decision in decisions:
            state.per_ticker[decision.ticker].portfolio_decision = decision

        return state

    async def _run_ticker_pipeline(
        self,
        *,
        ticker: str,
        as_of: datetime,
        bars: list[MarketBar],
        portfolio_state: PortfolioState,
        fundamentals_evidence: dict[str, object],
        sentiment_evidence: dict[str, object],
        news_evidence: dict[str, object],
    ) -> TickerPipelineState:
        async with self._ticker_semaphore:
            ticker_state = TickerPipelineState(ticker=ticker)

            (
                fundamentals_report,
                sentiment_report,
                news_report,
                technical_report,
            ) = await asyncio.gather(
                self.fundamentals_analyst.run(
                    AnalystInput(ticker=ticker, as_of=as_of, evidence=fundamentals_evidence)
                ),
                self.sentiment_analyst.run(
                    AnalystInput(ticker=ticker, as_of=as_of, evidence=sentiment_evidence)
                ),
                self.news_analyst.run(
                    AnalystInput(ticker=ticker, as_of=as_of, evidence=news_evidence)
                ),
                self.technical_analyst.run(
                    TechnicalAnalystInput(ticker=ticker, as_of=as_of, bars=bars)
                ),
            )
            ticker_state.analyst_reports = [
                fundamentals_report,
                sentiment_report,
                news_report,
                technical_report,
            ]

            research = await self.researcher.run(
                ResearcherInput(ticker=ticker, analyst_reports=ticker_state.analyst_reports)
            )
            ticker_state.research = research

            current_price = bars[-1].close if bars else 0.0
            current_position = portfolio_state.positions.get(ticker)

            trade_proposal = await self.trader.run(
                TraderInput(
                    ticker=ticker,
                    research=research,
                    portfolio_state=portfolio_state,
                    current_position=current_position,
                    current_price=current_price,
                )
            )
            ticker_state.trade_proposal = trade_proposal

            risk_assessment = await self.risk_manager.run(
                RiskManagerInput(
                    proposal=trade_proposal,
                    portfolio_state=portfolio_state,
                    current_price=current_price,
                    risk_config=self.risk_config,
                )
            )
            ticker_state.risk_assessment = risk_assessment

            return ticker_state
