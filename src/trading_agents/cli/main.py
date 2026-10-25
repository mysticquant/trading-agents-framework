"""CLI entrypoint: `trading-agents run-backtest [--universe ... --start ... --end ...]`
and `trading-agents ib-check` (read-only Interactive Brokers connectivity check).

Wires every agent, the LLM client, the paper broker, and a cached
yfinance data provider together, then runs a backtest and prints summary
stats. The LLM is Claude by default (needs ANTHROPIC_API_KEY in `.env`) or a
local Ollama model with LLM_PROVIDER=ollama — see the README for both and
for the manual opt-in live-LLM smoke tests.
"""

import argparse
import asyncio
import sys

from pydantic import ValidationError

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
from trading_agents.brokers.base import BrokerError
from trading_agents.brokers.paper import PaperBrokerAdapter
from trading_agents.config.settings import ConfigurationError, Settings
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
from trading_agents.llm.ollama_client import OllamaError, OllamaLLMClient
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


def build_llm(settings: Settings) -> tuple[LLMClient, LLMRoleRegistry]:
    """The configured LLM client, plus a role registry whose default models
    belong to that provider (role overrides apply to either)."""
    if settings.llm_provider == "ollama":
        client: LLMClient = OllamaLLMClient(
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
            max_concurrency=settings.ollama_max_concurrency,
        )
        registry = LLMRoleRegistry(
            settings.llm_role_overrides,
            default_model=settings.ollama_model,
            high_stakes_model=settings.ollama_model,
        )
        return client, registry

    if settings.anthropic_api_key is None:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic' "
            "(set it in .env, or set LLM_PROVIDER=ollama to use a local model)."
        )
    client = AnthropicLLMClient(api_key=settings.anthropic_api_key.get_secret_value())
    return client, LLMRoleRegistry(settings.llm_role_overrides)


AGENT_ROLES = (
    ScreenerAgent.role,
    FundamentalsAnalystAgent.role,
    SentimentAnalystAgent.role,
    NewsAnalystAgent.role,
    TechnicalAnalystAgent.role,
    ResearcherAgent.role,
    TraderAgent.role,
    RiskManagerAgent.role,
    PortfolioManagerAgent.role,
)


async def run_backtest(settings: Settings) -> BacktestResult:
    llm_client, role_registry = build_llm(settings)
    if isinstance(llm_client, OllamaLLMClient):
        # Fail now, not after every simulated day has errored.
        await llm_client.check_models({role_registry.for_role(r).model for r in AGENT_ROLES})
    # One cost model shared by the broker and the agents that size buys, so
    # cash-limited trades are sized to what the broker will actually charge.
    execution_costs = ExecutionCosts()
    orchestrator = build_orchestrator(
        llm_client,
        role_registry=role_registry,
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


async def ib_check(settings: Settings) -> list[str]:
    """Connect to TWS/IB Gateway read-only and describe the account. Read-only
    mode cannot place orders, so live accounts are allowed here; the report
    says which kind it is."""
    # Imported lazily: ib_async is an optional extra (`pip install -e ".[ib]"`).
    from trading_agents.brokers.ib import InteractiveBrokersAdapter, is_paper_account

    adapter = InteractiveBrokersAdapter(
        host=settings.ib_host,
        port=settings.ib_port,
        client_id=settings.ib_client_id,
        account=settings.ib_account,
        readonly=True,
        allow_live=True,
    )
    await adapter.connect()
    try:
        balance = await adapter.get_account_balance()
        positions = await adapter.get_positions()
    finally:
        await adapter.disconnect()

    kind = "paper" if is_paper_account(adapter.account) else "LIVE"
    lines = [
        f"Connected to {settings.ib_host}:{settings.ib_port} - account {adapter.account} ({kind})",
        f"Cash: {balance.cash:,.2f}  Equity: {balance.equity:,.2f}  "
        f"Buying power: {balance.buying_power:,.2f}",
        f"Positions: {len(positions)}",
    ]
    lines += [
        f"  {p.ticker}: {p.quantity:g} @ {p.avg_cost:,.2f} (value {p.market_value:,.2f})"
        for p in positions.values()
    ]
    return lines


def _print_backtest(result: BacktestResult) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading-agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("run-backtest")
    backtest_parser.add_argument("--universe", help="Comma-separated tickers, e.g. AAPL,MSFT")
    backtest_parser.add_argument("--start", help="YYYY-MM-DD")
    backtest_parser.add_argument("--end", help="YYYY-MM-DD")

    subparsers.add_parser(
        "ib-check", help="Connect to TWS/IB Gateway read-only and show the account"
    )

    args = parser.parse_args()

    overrides = {}
    if args.command == "run-backtest":
        if args.universe:
            overrides["ticker_universe"] = args.universe
        if args.start:
            overrides["backtest_start_date"] = args.start
        if args.end:
            overrides["backtest_end_date"] = args.end
    try:
        settings = Settings(**overrides)
        if args.command == "ib-check":
            for line in asyncio.run(ib_check(settings)):
                print(line)
        else:
            _print_backtest(asyncio.run(run_backtest(settings)))
    except ValidationError as exc:
        problems = "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors())
        sys.exit(f"trading-agents: invalid configuration: {problems}")
    except ModuleNotFoundError as exc:
        if exc.name != "ib_async":
            raise
        sys.exit('trading-agents: ib-check needs the IB extra: pip install -e ".[ib]"')
    except (ConfigurationError, OllamaError, BrokerError) as exc:
        sys.exit(f"trading-agents: {exc}")


if __name__ == "__main__":
    main()
