"""CLI entrypoint: `trading-agents run-backtest [--universe ... --start ... --end ...]`.

Wires every agent, the LLM client, the paper broker, and a cached
yfinance data provider together, then runs a backtest and prints summary
stats. This is the first thing to run end-to-end once an Anthropic API key
is in `.env` — see the README for the manual opt-in live-LLM smoke test.
"""

import argparse
import asyncio

from trading_agents.agents.analysts.fundamentals import FundamentalsAnalystAgent
from trading_agents.agents.analysts.news import NewsAnalystAgent
from trading_agents.agents.analysts.sentiment import SentimentAnalystAgent
from trading_agents.agents.analysts.technical import TechnicalAnalystAgent
from trading_agents.agents.base import Agent
from trading_agents.agents.portfolio_manager import PortfolioManagerAgent
from trading_agents.agents.researcher import ResearcherAgent
from trading_agents.agents.risk_manager import RiskManagerAgent
from trading_agents.agents.screener import ScreenerAgent
from trading_agents.agents.trader import TraderAgent
from trading_agents.backtest.runner import BacktestRunner
from trading_agents.brokers.paper import PaperBrokerAdapter
from trading_agents.config.settings import Settings
from trading_agents.contracts.backtest import BacktestResult
from trading_agents.contracts.trading import ExecutionCosts
from trading_agents.data.cache import CachedDataProvider, ParquetCache
from trading_agents.data.evidence.base import EvidenceProvider
from trading_agents.data.evidence.provider import DefaultEvidenceProvider
from trading_agents.data.evidence.sec_edgar import SecEdgarFilingsSource
from trading_agents.data.evidence.yfinance_source import YFinanceMarketDataSource
from trading_agents.data.providers.yfinance_provider import YFinanceProvider
from trading_agents.llm.client import AnthropicLLMClient, LLMClient
from trading_agents.llm.config import LLMRoleRegistry, RoleModelConfig
from trading_agents.orchestration.pipeline import DecisionCycleOrchestrator


def build_orchestrator(
    llm_client: LLMClient,
    role_registry: LLMRoleRegistry | None = None,
    execution_costs: ExecutionCosts | None = None,
    evidence_provider: EvidenceProvider | None = None,
) -> DecisionCycleOrchestrator:
    registry = role_registry or LLMRoleRegistry()

    def config(agent_cls: type[Agent]) -> RoleModelConfig:
        return registry.for_role(agent_cls.role)

    return DecisionCycleOrchestrator(
        screener=ScreenerAgent(llm_client, config(ScreenerAgent)),
        fundamentals_analyst=FundamentalsAnalystAgent(llm_client, config(FundamentalsAnalystAgent)),
        sentiment_analyst=SentimentAnalystAgent(llm_client, config(SentimentAnalystAgent)),
        news_analyst=NewsAnalystAgent(llm_client, config(NewsAnalystAgent)),
        technical_analyst=TechnicalAnalystAgent(llm_client, config(TechnicalAnalystAgent)),
        researcher=ResearcherAgent(llm_client, config(ResearcherAgent)),
        trader=TraderAgent(llm_client, config(TraderAgent)),
        risk_manager=RiskManagerAgent(llm_client, config(RiskManagerAgent)),
        portfolio_manager=PortfolioManagerAgent(llm_client, config(PortfolioManagerAgent)),
        execution_costs=execution_costs,
        evidence_provider=evidence_provider,
    )


def build_evidence_provider(settings: Settings) -> EvidenceProvider:
    filings = SecEdgarFilingsSource(settings.sec_user_agent) if settings.sec_user_agent else None
    return DefaultEvidenceProvider(YFinanceMarketDataSource(), filings)


async def run_backtest(settings: Settings) -> BacktestResult:
    llm_client = AnthropicLLMClient(api_key=settings.anthropic_api_key.get_secret_value())
    # One cost model shared by the broker and the agents that size buys, so
    # cash-limited trades are sized to what the broker will actually charge.
    execution_costs = ExecutionCosts()
    orchestrator = build_orchestrator(
        llm_client,
        role_registry=LLMRoleRegistry(settings.llm_role_overrides),
        execution_costs=execution_costs,
        evidence_provider=build_evidence_provider(settings),
    )

    data_provider = CachedDataProvider(YFinanceProvider(), ParquetCache(settings.data_cache_dir))
    broker = PaperBrokerAdapter(
        initial_cash=settings.initial_cash,
        commission_bps=execution_costs.commission_bps,
        slippage_bps=execution_costs.slippage_bps,
        historical_provider=data_provider,
    )
    await broker.connect()
    try:
        runner = BacktestRunner(
            orchestrator=orchestrator, broker=broker, data_provider=data_provider
        )
        return await runner.run(
            start_date=settings.backtest_start_date,
            end_date=settings.backtest_end_date,
            universe=settings.ticker_universe,
        )
    finally:
        await broker.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading-agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("run-backtest")
    backtest_parser.add_argument("--universe", help="Comma-separated tickers, e.g. AAPL,MSFT")
    backtest_parser.add_argument("--start", help="YYYY-MM-DD")
    backtest_parser.add_argument("--end", help="YYYY-MM-DD")

    args = parser.parse_args()

    overrides = {}
    if args.universe:
        overrides["ticker_universe"] = args.universe
    if args.start:
        overrides["backtest_start_date"] = args.start
    if args.end:
        overrides["backtest_end_date"] = args.end
    settings = Settings(**overrides)

    result = asyncio.run(run_backtest(settings))
    print(f"Period: {result.start_date} -> {result.end_date}")
    print(f"Universe: {', '.join(result.ticker_universe)}")
    print(f"Total return: {result.total_return_pct:.2f}%")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Win rate: {result.win_rate:.0%}")
    print(f"Trades: {len(result.trade_log)}")
    print(f"Final equity: {result.final_portfolio_state.equity:.2f}")
    if result.cycle_errors:
        print(f"Skipped after errors: {len(result.cycle_errors)} (see log for details)")


if __name__ == "__main__":
    main()
