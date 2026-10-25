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
import logging
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
from trading_agents.contracts.trading import ExecutionCosts
from trading_agents.data.evidence.base import EvidenceProvider
from trading_agents.indicators.technical import bars_to_frame
from trading_agents.orchestration.state import DecisionCycleState, TickerPipelineState

logger = logging.getLogger(__name__)


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
        execution_costs: ExecutionCosts | None = None,
        evidence_provider: EvidenceProvider | None = None,
        screener_top_n: int = 10,
        max_concurrent_tickers: int = 5,
    ) -> None:
        """`execution_costs` should match the broker the decisions are
        submitted to, so cash-limited buys are sized to what the broker
        will actually charge. `evidence_provider`, when given, supplies
        point-in-time fundamentals/sentiment/news evidence for each
        shortlisted ticker; without one those analysts see empty evidence
        unless the caller passes it to `run_cycle`."""
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
        self.execution_costs = execution_costs or ExecutionCosts()
        self.evidence_provider = evidence_provider
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
        """Evidence dicts are optional, keyed by ticker. For each ticker and
        evidence type, a caller-supplied dict takes precedence; otherwise
        the orchestrator's `evidence_provider` (if any) fills it in, and
        failing that it stays empty."""
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
        # Repeats are dropped (keeping rank order) and the list is capped at
        # `screener_top_n`, so the LLM can't multiply per-ticker pipelines.
        valid_tickers = {candidate.ticker for candidate in candidates}
        shortlist = list(
            dict.fromkeys(
                result.ticker for result in state.screen_results if result.ticker in valid_tickers
            )
        )[: self.screener_top_n]

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
            if ts.error is None and ts.trade_proposal is not None and ts.risk_assessment is not None
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
                execution_costs=self.execution_costs,
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
            # One failed LLM call (API error after retries, an output that
            # fails validation, ...) costs this ticker for this cycle only —
            # it must not abort the other tickers or the whole backtest.
            try:
                await self._run_ticker_stages(
                    ticker_state,
                    as_of=as_of,
                    bars=bars,
                    portfolio_state=portfolio_state,
                    fundamentals_evidence=fundamentals_evidence,
                    sentiment_evidence=sentiment_evidence,
                    news_evidence=news_evidence,
                )
            except Exception as exc:
                logger.exception("Ticker pipeline failed for %s at %s", ticker, as_of)
                # A TaskGroup wraps failures; report the underlying error.
                while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
                    exc = exc.exceptions[0]
                ticker_state.error = f"{type(exc).__name__}: {exc}"
            return ticker_state

    async def _run_ticker_stages(
        self,
        ticker_state: TickerPipelineState,
        *,
        as_of: datetime,
        bars: list[MarketBar],
        portfolio_state: PortfolioState,
        fundamentals_evidence: dict[str, object],
        sentiment_evidence: dict[str, object],
        news_evidence: dict[str, object],
    ) -> None:
        ticker = ticker_state.ticker
        if (
            self.evidence_provider is not None
            and bars
            and not (fundamentals_evidence and sentiment_evidence and news_evidence)
        ):
            bundle = await self.evidence_provider.get_evidence(ticker, as_of, bars[-1].close)
            fundamentals_evidence = fundamentals_evidence or bundle.fundamentals
            sentiment_evidence = sentiment_evidence or bundle.sentiment
            news_evidence = news_evidence or bundle.news
        # A TaskGroup (not gather): if one analyst call fails, the others are
        # cancelled instead of running on unobserved, past the per-ticker
        # concurrency cap, still spending tokens.
        async with asyncio.TaskGroup() as analysts:
            fundamentals_task = analysts.create_task(
                self.fundamentals_analyst.run(
                    AnalystInput(ticker=ticker, as_of=as_of, evidence=fundamentals_evidence)
                )
            )
            sentiment_task = analysts.create_task(
                self.sentiment_analyst.run(
                    AnalystInput(ticker=ticker, as_of=as_of, evidence=sentiment_evidence)
                )
            )
            news_task = analysts.create_task(
                self.news_analyst.run(
                    AnalystInput(ticker=ticker, as_of=as_of, evidence=news_evidence)
                )
            )
            technical_task = analysts.create_task(
                self.technical_analyst.run(
                    TechnicalAnalystInput(ticker=ticker, as_of=as_of, bars=bars)
                )
            )
        fundamentals_report = fundamentals_task.result()
        sentiment_report = sentiment_task.result()
        news_report = news_task.result()
        technical_report = technical_task.result()
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
                execution_costs=self.execution_costs,
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
