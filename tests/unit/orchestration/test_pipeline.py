from datetime import datetime

import pytest

from trading_agents.agents.analysts.base import AnalystLLMOutput
from trading_agents.agents.analysts.fundamentals import FundamentalsAnalystAgent
from trading_agents.agents.analysts.news import NewsAnalystAgent
from trading_agents.agents.analysts.sentiment import SentimentAnalystAgent
from trading_agents.agents.analysts.technical import TechnicalAnalystAgent
from trading_agents.agents.portfolio_manager import (
    PortfolioManagerAgent,
    PortfolioManagerDecisionLLM,
    PortfolioManagerLLMOutput,
)
from trading_agents.agents.researcher import ResearcherAgent, ResearcherLLMOutput
from trading_agents.agents.risk_manager import RiskManagerAgent, RiskManagerLLMOutput
from trading_agents.agents.screener import ScreenerAgent, ScreenerLLMOutput, ScreenerRanking
from trading_agents.agents.trader import TraderAgent, TraderLLMOutput
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.llm.fakes import FakeLLMClient
from trading_agents.orchestration.pipeline import DecisionCycleOrchestrator


@pytest.fixture
def canned_responses():
    return {
        "screener": ScreenerLLMOutput(
            rankings=[
                ScreenerRanking(ticker="AAPL", score=0.9, rationale="momentum"),
                ScreenerRanking(ticker="MSFT", score=0.8, rationale="momentum"),
            ]
        ),
        "fundamentals_analyst": AnalystLLMOutput(summary="ok", signal="bullish", confidence=0.6),
        "sentiment_analyst": AnalystLLMOutput(summary="ok", signal="bullish", confidence=0.6),
        "news_analyst": AnalystLLMOutput(summary="ok", signal="neutral", confidence=0.5),
        "technical_analyst": AnalystLLMOutput(summary="ok", signal="bullish", confidence=0.7),
        "researcher": ResearcherLLMOutput(
            bull_thesis="b",
            bear_thesis="c",
            synthesis="d",
            recommended_action="buy",
            conviction=0.7,
        ),
        "trader": TraderLLMOutput(
            action="buy", quantity=5.0, order_type="market", rationale="go", time_horizon="1w"
        ),
        "risk_manager": RiskManagerLLMOutput(
            llm_approved=True, max_position_size_pct=0.5, adjusted_quantity=5.0, rationale="ok"
        ),
        "portfolio_manager": PortfolioManagerLLMOutput(
            decisions=[
                PortfolioManagerDecisionLLM(
                    ticker="AAPL", final_action="buy", final_quantity=5.0, rationale="go"
                ),
                PortfolioManagerDecisionLLM(
                    ticker="MSFT", final_action="buy", final_quantity=5.0, rationale="go"
                ),
            ]
        ),
    }


def _build_orchestrator(llm_client) -> DecisionCycleOrchestrator:
    return DecisionCycleOrchestrator(
        screener=ScreenerAgent(llm_client),
        fundamentals_analyst=FundamentalsAnalystAgent(llm_client),
        sentiment_analyst=SentimentAnalystAgent(llm_client),
        news_analyst=NewsAnalystAgent(llm_client),
        technical_analyst=TechnicalAnalystAgent(llm_client),
        researcher=ResearcherAgent(llm_client),
        trader=TraderAgent(llm_client),
        risk_manager=RiskManagerAgent(llm_client),
        portfolio_manager=PortfolioManagerAgent(llm_client),
    )


async def test_run_cycle_wires_the_full_pipeline_for_every_shortlisted_ticker(
    bars_factory, canned_responses
):
    orchestrator = _build_orchestrator(FakeLLMClient(canned_responses))

    market_data = {
        "AAPL": bars_factory("AAPL", 60),
        "MSFT": bars_factory("MSFT", 60, start_price=200.0),
    }
    portfolio_state = PortfolioState(
        cash=100_000.0, equity=100_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    state = await orchestrator.run_cycle(
        as_of=datetime(2024, 1, 1),
        universe=["AAPL", "MSFT"],
        market_data=market_data,
        portfolio_state=portfolio_state,
    )

    assert set(state.per_ticker) == {"AAPL", "MSFT"}
    for ticker_state in state.per_ticker.values():
        assert len(ticker_state.analyst_reports) == 4
        assert ticker_state.research is not None
        assert ticker_state.trade_proposal is not None
        assert ticker_state.risk_assessment is not None
        assert ticker_state.portfolio_decision is not None
        assert ticker_state.portfolio_decision.final_action == "buy"


async def test_run_cycle_excludes_tickers_the_prefilter_never_saw(bars_factory, canned_responses):
    # The screener's canned rankings mention "GOOG", which was never part of
    # the deterministic pre-filter candidates (too little history) — it
    # must not reach the per-ticker pipeline.
    canned_responses["screener"] = ScreenerLLMOutput(
        rankings=[ScreenerRanking(ticker="GOOG", score=0.9, rationale="hallucinated")]
    )
    orchestrator = _build_orchestrator(FakeLLMClient(canned_responses))

    market_data = {"AAPL": bars_factory("AAPL", 60), "GOOG": bars_factory("GOOG", 1)}
    portfolio_state = PortfolioState(
        cash=100_000.0, equity=100_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    state = await orchestrator.run_cycle(
        as_of=datetime(2024, 1, 1),
        universe=["AAPL", "GOOG"],
        market_data=market_data,
        portfolio_state=portfolio_state,
    )

    assert state.per_ticker == {}
