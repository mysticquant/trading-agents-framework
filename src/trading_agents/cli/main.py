"""CLI entrypoint: `trading-agents run-backtest [--universe ... --start ... --end ...]`
plus read-only broker checks (`ib-check`, `kite-check`) and the daily Zerodha
login (`kite-login`).

Wires every agent, the LLM client, the paper broker, and a cached
yfinance data provider together, then runs a backtest and prints summary
stats. The LLM is Claude by default (needs ANTHROPIC_API_KEY in `.env`) or a
local Ollama model with LLM_PROVIDER=ollama — see the README for both and
for the manual opt-in live-LLM smoke tests.
"""

import argparse
import asyncio
import logging
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
from trading_agents.brokers.kite import (
    KiteBrokerAdapter,
    KiteRestClient,
    create_session,
    login_url,
)
from trading_agents.brokers.paper import PaperBrokerAdapter
from trading_agents.cli.trade import build_trading_loop, describe
from trading_agents.config.settings import ConfigurationError, Settings
from trading_agents.contracts.backtest import BacktestResult
from trading_agents.contracts.trading import ExecutionCosts
from trading_agents.data.cache import CachedDataProvider, ParquetCache
from trading_agents.data.evidence.base import EvidenceProvider
from trading_agents.data.evidence.provider import DefaultEvidenceProvider
from trading_agents.data.evidence.sec_edgar import SecEdgarFilingsSource
from trading_agents.data.evidence.yfinance_source import YFinanceMarketDataSource
from trading_agents.data.providers.yfinance_provider import YFinanceProvider
from trading_agents.live.loop import Mode
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


async def preflight_llm(llm_client: LLMClient, role_registry: LLMRoleRegistry) -> None:
    """Fail before a long run, not after every cycle has errored."""
    if isinstance(llm_client, OllamaLLMClient):
        await llm_client.check_models({role_registry.for_role(r).model for r in AGENT_ROLES})


async def run_backtest(settings: Settings) -> BacktestResult:
    llm_client, role_registry = build_llm(settings)
    await preflight_llm(llm_client, role_registry)
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


def _kite_client(settings: Settings, *, need_token: bool) -> KiteRestClient:
    if not settings.kite_api_key:
        raise ConfigurationError("KITE_API_KEY is not set in .env.")
    token = settings.kite_access_token.get_secret_value() if settings.kite_access_token else ""
    if need_token and not token:
        raise ConfigurationError(
            "KITE_ACCESS_TOKEN is not set; run `trading-agents kite-login` first."
        )
    return KiteRestClient(settings.kite_api_key, token)


async def kite_login(settings: Settings, request_token: str | None) -> list[str]:
    """Without a request token: how to log in. With one: exchange it for
    the day's access token."""
    client = _kite_client(settings, need_token=False)
    if not request_token:
        return [
            "1. Open this URL and log in to Kite:",
            f"   {login_url(client.api_key)}",
            "2. Kite redirects to your app's redirect URL with ?request_token=... "
            "(valid for a few minutes).",
            "3. Run: trading-agents kite-login --request-token <that token>",
        ]
    if settings.kite_api_secret is None:
        raise ConfigurationError("KITE_API_SECRET is not set in .env.")
    session = await create_session(
        client.api_key, settings.kite_api_secret.get_secret_value(), request_token
    )
    return [
        f"Logged in as {session.get('user_id', '?')}. Add this to .env "
        "(valid until 6 AM IST tomorrow):",
        f"KITE_ACCESS_TOKEN={session['access_token']}",
    ]


async def kite_check(settings: Settings) -> list[str]:
    """Read-only: the adapter is built without allow_live, so it cannot
    place orders."""
    adapter = KiteBrokerAdapter(_kite_client(settings, need_token=True))
    await adapter.connect()
    try:
        balance = await adapter.get_account_balance()
        positions = await adapter.get_positions()
    finally:
        await adapter.disconnect()
    lines = [
        f"Connected to Kite as {adapter.user_id} (live account; orders disabled here)",
        f"Cash: {balance.cash:,.2f}  Equity: {balance.equity:,.2f}  "
        f"Buying power: {balance.buying_power:,.2f}",
        f"Positions: {len(positions)}",
    ]
    lines += [
        f"  {p.ticker}: {p.quantity:g} @ {p.avg_cost:,.2f} (value {p.market_value:,.2f})"
        for p in positions.values()
    ]
    return lines


async def run_trade(settings: Settings, mode: Mode, *, once: bool) -> None:
    loop = await build_trading_loop(settings, mode)
    for line in describe(loop, settings):
        print(line)
    try:
        if once:
            report = await loop.run_once()
            print(f"Cycle done: {report.skipped_reason or f'{len(report.decisions)} decision(s)'}")
        else:
            await loop.run_forever()
    finally:
        if loop.broker.is_connected():
            await loop.broker.disconnect()


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

    login_parser = subparsers.add_parser(
        "kite-login", help="Zerodha Kite daily login: get the day's access token"
    )
    login_parser.add_argument("--request-token", help="request_token from Kite's redirect")
    subparsers.add_parser(
        "kite-check", help="Connect to Zerodha Kite read-only and show the account"
    )
    trade_parser = subparsers.add_parser(
        "trade", help="Run the live trading loop (dry-run unless told otherwise)"
    )
    trade_parser.add_argument(
        "--mode",
        choices=["dry-run", "paper", "live"],
        default="dry-run",
        help="dry-run (default): journal decisions, send nothing; paper: paper account; "
        "live: real orders (also needs ALLOW_LIVE_TRADING=true)",
    )
    trade_parser.add_argument(
        "--once", action="store_true", help="Run one decision cycle now and exit"
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
            lines = asyncio.run(ib_check(settings))
        elif args.command == "kite-login":
            lines = asyncio.run(kite_login(settings, args.request_token))
        elif args.command == "kite-check":
            lines = asyncio.run(kite_check(settings))
        elif args.command == "trade":
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
            asyncio.run(run_trade(settings, args.mode, once=args.once))
            lines = []
        else:
            lines = []
            _print_backtest(asyncio.run(run_backtest(settings)))
        for line in lines:
            print(line)
    except ValidationError as exc:
        problems = "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors())
        sys.exit(f"trading-agents: invalid configuration: {problems}")
    except ModuleNotFoundError as exc:
        if exc.name != "ib_async":
            raise
        sys.exit('trading-agents: this command needs the IB extra: pip install -e ".[ib]"')
    except (ConfigurationError, OllamaError, BrokerError) as exc:
        sys.exit(f"trading-agents: {exc}")


if __name__ == "__main__":
    main()
