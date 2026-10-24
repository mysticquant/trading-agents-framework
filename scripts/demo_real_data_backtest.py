"""Exercises the real data path (yfinance fetch -> parquet cache -> technical
indicators -> paper broker) without needing an Anthropic API key, by wiring
`FakeLLMClient` (canned per-role responses) in place of `AnthropicLLMClient`.

This does NOT test agent *reasoning* quality — the LLM responses are
scripted — but it validates everything else: data ingestion/caching, the
point-in-time analyst evidence (fundamentals, analyst actions, news), the
orchestrator's wiring and concurrency, indicator computation, and the paper
broker's fills, over real historical prices. Needs internet access (to
Yahoo Finance via yfinance), but no Claude API key. Set SEC_USER_AGENT
(e.g. "Jane Doe jane@example.com") to also pull SEC EDGAR filings.

Run from the project root after `pip install -e ".[dev]"`:

    python scripts/demo_real_data_backtest.py
"""

import asyncio
import os
from datetime import date, timedelta
from pathlib import Path

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
from trading_agents.backtest.runner import BacktestRunner
from trading_agents.brokers.paper import PaperBrokerAdapter
from trading_agents.data.cache import CachedDataProvider, ParquetCache
from trading_agents.data.evidence.provider import DefaultEvidenceProvider
from trading_agents.data.evidence.sec_edgar import SecEdgarFilingsSource
from trading_agents.data.evidence.yfinance_source import YFinanceMarketDataSource
from trading_agents.data.providers.yfinance_provider import YFinanceProvider
from trading_agents.llm.fakes import FakeLLMClient
from trading_agents.orchestration.pipeline import DecisionCycleOrchestrator

UNIVERSE = ["AAPL", "MSFT"]


def build_canned_responses(universe: list[str]) -> dict:
    """One fixed, always-bullish-on-everything script, keyed by role — good
    enough to drive the pipeline end to end, not meant to resemble real
    trading judgment."""
    return {
        "screener": ScreenerLLMOutput(
            rankings=[
                ScreenerRanking(ticker=ticker, score=0.75, rationale="Recent price/volume move.")
                for ticker in universe
            ]
        ),
        "fundamentals_analyst": AnalystLLMOutput(
            summary="Scripted placeholder response.", signal="neutral", confidence=0.5
        ),
        "sentiment_analyst": AnalystLLMOutput(
            summary="Scripted placeholder response.", signal="neutral", confidence=0.5
        ),
        "news_analyst": AnalystLLMOutput(
            summary="Scripted placeholder response.", signal="neutral", confidence=0.5
        ),
        "technical_analyst": AnalystLLMOutput(
            summary="Scripted placeholder response.", signal="bullish", confidence=0.6
        ),
        "researcher": ResearcherLLMOutput(
            bull_thesis="Scripted bull case.",
            bear_thesis="Scripted bear case.",
            synthesis="Scripted synthesis favoring a small long position.",
            recommended_action="buy",
            conviction=0.6,
        ),
        "trader": TraderLLMOutput(
            action="buy",
            quantity=5.0,
            order_type="market",
            rationale="Scripted small buy for demo purposes.",
            time_horizon="1w",
        ),
        "risk_manager": RiskManagerLLMOutput(
            llm_approved=True,
            max_position_size_pct=0.1,
            adjusted_quantity=5.0,
            rationale="Scripted approval.",
        ),
        "portfolio_manager": PortfolioManagerLLMOutput(
            decisions=[
                PortfolioManagerDecisionLLM(
                    ticker=ticker,
                    final_action="buy",
                    final_quantity=5.0,
                    rationale="Scripted approval.",
                )
                for ticker in universe
            ]
        ),
    }


async def main() -> None:
    llm_client = FakeLLMClient(build_canned_responses(UNIVERSE))
    sec_user_agent = os.environ.get("SEC_USER_AGENT")
    evidence_provider = DefaultEvidenceProvider(
        YFinanceMarketDataSource(),
        SecEdgarFilingsSource(sec_user_agent) if sec_user_agent else None,
    )
    orchestrator = DecisionCycleOrchestrator(
        screener=ScreenerAgent(llm_client),
        fundamentals_analyst=FundamentalsAnalystAgent(llm_client),
        sentiment_analyst=SentimentAnalystAgent(llm_client),
        news_analyst=NewsAnalystAgent(llm_client),
        technical_analyst=TechnicalAnalystAgent(llm_client),
        researcher=ResearcherAgent(llm_client),
        trader=TraderAgent(llm_client),
        risk_manager=RiskManagerAgent(llm_client),
        portfolio_manager=PortfolioManagerAgent(llm_client),
        evidence_provider=evidence_provider,
    )

    data_provider = CachedDataProvider(YFinanceProvider(), ParquetCache(Path("data/cache")))
    broker = PaperBrokerAdapter(initial_cash=100_000.0, historical_provider=data_provider)
    await broker.connect()

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=90)

    runner = BacktestRunner(orchestrator=orchestrator, broker=broker, data_provider=data_provider)
    result = await runner.run(start_date=start_date, end_date=end_date, universe=UNIVERSE)

    print(f"Period: {result.start_date} -> {result.end_date}")
    print(f"Universe: {', '.join(result.ticker_universe)}")
    print(f"Equity curve points: {len(result.equity_curve)}")
    print(f"Trades: {len(result.trade_log)}")
    print(f"Total return: {result.total_return_pct:.2f}%")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Win rate: {result.win_rate:.0%}")
    print(f"Final equity: {result.final_portfolio_state.equity:.2f}")
    print(f"SEC filings: {'on' if sec_user_agent else 'off (set SEC_USER_AGENT to enable)'}")
    if result.cycle_errors:
        print(f"Skipped after errors: {len(result.cycle_errors)}")

    await broker.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
