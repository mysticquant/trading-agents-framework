"""Wiring for `trading-agents trade`: builds the broker, data, agents, and
guardrails for the live loop from settings, refusing any combination that
could send real orders without both explicit opt-ins.

How each mode reaches the broker:

| mode    | IB                         | Kite                       | execd (Rust service)         |
|---------|----------------------------|----------------------------|------------------------------|
| dry-run | read-only connection       | orders disabled            | orders blocked client-side   |
| paper   | paper accounts only        | refused: Kite has no paper | daemon must run `paper`      |
| live    | needs ALLOW_LIVE_TRADING   | needs ALLOW_LIVE_TRADING   | daemon must run `kite` with  |
|         |                            |                            | --allow-live, + ALLOW_LIVE…  |
"""

from datetime import date, time
from zoneinfo import ZoneInfo

from trading_agents.brokers.base import BrokerAdapter
from trading_agents.brokers.kite import (
    KiteBrokerAdapter,
    KiteHistoricalDataProvider,
    KiteRestClient,
)
from trading_agents.config.settings import ConfigurationError, Settings
from trading_agents.contracts.market import MarketBar, Timeframe
from trading_agents.contracts.trading import ExecutionCosts
from trading_agents.data.cache import CachedDataProvider, ParquetCache
from trading_agents.data.providers.base import HistoricalDataProvider
from trading_agents.data.providers.yfinance_provider import YFinanceProvider
from trading_agents.live.guardrails import Guardrails, StateStore
from trading_agents.live.loop import LiveTradingLoop, Mode, Schedule

_US = (ZoneInfo("America/New_York"), time(15, 30))
_NSE = (ZoneInfo("Asia/Kolkata"), time(15, 0))


def _market(settings: Settings) -> str:
    if settings.live_broker == "execd":
        return settings.execd_market
    return "us" if settings.live_broker == "ib" else "nse"


class _NseYFinanceProvider:
    """yfinance lists NSE stocks as "<SYMBOL>.NS"; Kite and execd use the bare
    symbol. Without the suffix yfinance can return a different listing
    (e.g. INFY is Infosys' US ADR), so translate both ways."""

    def __init__(self, inner: HistoricalDataProvider) -> None:
        self._inner = inner

    async def get_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]:
        bars = await self._inner.get_bars(f"{ticker}.NS", start, end, timeframe)
        return [bar.model_copy(update={"ticker": ticker}) for bar in bars]


def require_live_opt_in(settings: Settings, mode: Mode) -> None:
    if mode == "live" and not settings.allow_live_trading:
        raise ConfigurationError(
            "--mode live sends REAL orders. Also set ALLOW_LIVE_TRADING=true in .env "
            "to confirm; until then use --mode dry-run (default) or paper."
        )


def build_live_broker(
    settings: Settings, mode: Mode
) -> tuple[BrokerAdapter, HistoricalDataProvider]:
    """The broker for `mode`, plus the market-data source for that market."""
    require_live_opt_in(settings, mode)
    if settings.live_broker == "ib":
        # Imported lazily: ib_async is an optional extra.
        from trading_agents.brokers.ib import InteractiveBrokersAdapter

        broker = InteractiveBrokersAdapter(
            host=settings.ib_host,
            port=settings.ib_port,
            client_id=settings.ib_client_id,
            account=settings.ib_account,
            # Dry runs connect read-only (cannot place orders), which is safe
            # against any account; paper mode only accepts paper accounts.
            readonly=mode == "dry-run",
            allow_live=mode in ("dry-run", "live"),
        )
        data = CachedDataProvider(YFinanceProvider(), ParquetCache(settings.data_cache_dir))
        return broker, data

    if settings.live_broker == "execd":
        return _build_execd(settings, mode)

    if mode == "paper":
        raise ConfigurationError(
            "Kite has no paper-trading accounts. Use --mode dry-run to run the loop "
            "without orders, or --mode live (with ALLOW_LIVE_TRADING=true)."
        )
    if not settings.kite_api_key or settings.kite_access_token is None:
        raise ConfigurationError(
            "KITE_API_KEY and KITE_ACCESS_TOKEN must be set; run `trading-agents kite-login`."
        )
    client = KiteRestClient(settings.kite_api_key, settings.kite_access_token.get_secret_value())
    historical = KiteHistoricalDataProvider(client)
    return KiteBrokerAdapter(client, allow_live=mode == "live", historical=historical), historical


def _build_execd(settings: Settings, mode: Mode) -> tuple[BrokerAdapter, HistoricalDataProvider]:
    from trading_agents.brokers.execution_service import ExecutionServiceAdapter

    if settings.execd_token is None:
        raise ConfigurationError("EXECD_TOKEN must be set (the token execd was started with).")
    yfinance: HistoricalDataProvider = CachedDataProvider(
        YFinanceProvider(), ParquetCache(settings.data_cache_dir)
    )
    if settings.execd_market == "nse":
        yfinance = _NseYFinanceProvider(yfinance)
    adapter = ExecutionServiceAdapter(
        token=settings.execd_token.get_secret_value(),
        host=settings.execd_host,
        port=settings.execd_port,
        # Checked against the daemon on connect, not taken on trust.
        require_backend={"paper": "paper", "live": "kite"}.get(mode),
        require_orders_enabled=mode == "live",
        block_orders=mode == "dry-run",
        # The kite backend serves its own history; the paper backend has none.
        fallback_historical=yfinance,
    )
    return adapter, adapter


def build_schedule(settings: Settings) -> Schedule:
    timezone, default_time = _US if _market(settings) == "us" else _NSE
    if not settings.live_decision_time:
        return Schedule(timezone, default_time)
    try:
        decision_time = time.fromisoformat(settings.live_decision_time)
    except ValueError as exc:
        raise ConfigurationError(
            f"LIVE_DECISION_TIME must be HH:MM, got {settings.live_decision_time!r}."
        ) from exc
    return Schedule(timezone, decision_time)


def build_guardrails(settings: Settings) -> Guardrails:
    return Guardrails(
        max_order_pct_of_equity=settings.live_max_order_pct,
        max_orders_per_day=settings.live_max_orders_per_day,
        max_daily_drawdown_pct=settings.live_max_daily_drawdown_pct,
        kill_switch_path=settings.live_kill_switch_path,
    )


async def build_trading_loop(settings: Settings, mode: Mode) -> LiveTradingLoop:
    # Imported here to avoid a cycle: main imports this module.
    from trading_agents.cli.main import (
        build_evidence_provider,
        build_llm,
        build_orchestrator,
        preflight_llm,
    )

    broker, data = build_live_broker(settings, mode)
    llm_client, role_registry = build_llm(settings)
    await preflight_llm(llm_client, role_registry)
    orchestrator = build_orchestrator(
        llm_client,
        role_registry=role_registry,
        execution_costs=ExecutionCosts(),
        # The evidence sources (yfinance fundamentals, SEC EDGAR) cover US
        # listings; NSE symbols run on technicals alone for now.
        evidence_provider=build_evidence_provider(settings) if _market(settings) == "us" else None,
    )
    state_dir = settings.live_state_dir
    return LiveTradingLoop(
        orchestrator=orchestrator,
        broker=broker,
        data_provider=data,
        universe=settings.ticker_universe,
        guardrails=build_guardrails(settings),
        mode=mode,
        schedule=build_schedule(settings),
        # Per broker and mode: a dry or paper run must never set the drawdown
        # baseline or use up the order limit of a live run (and vice versa).
        state_store=StateStore(state_dir / f"{settings.live_broker}-{mode}-state.json"),
        journal_path=state_dir / f"{settings.live_broker}-{mode}-journal.jsonl",
    )


def describe(loop: LiveTradingLoop, settings: Settings) -> list[str]:
    g = loop.guardrails
    warning = {
        "dry-run": "DRY RUN - decisions are journaled, no orders are sent",
        "paper": "PAPER - orders go to a paper account",
        "live": "LIVE - REAL ORDERS WITH REAL MONEY",
    }[loop.mode]
    return [
        f"Mode: {warning}",
        f"Broker: {settings.live_broker}  Universe: {', '.join(loop.universe)}",
        f"Decision time: {loop.schedule.decision_time.strftime('%H:%M')} "
        f"{loop.schedule.timezone.key} on weekdays",
        f"Guardrails: buys <= {g.max_order_pct_of_equity:.0%} of equity, "
        f"<= {g.max_orders_per_day} orders/day, halt at {g.max_daily_drawdown_pct:.0%} "
        f"daily drawdown, kill switch: create {g.kill_switch_path}",
        f"Journal: {loop.journal_path}",
    ]
