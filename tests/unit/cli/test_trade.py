from datetime import time

import pytest

from trading_agents.brokers.kite import KiteBrokerAdapter
from trading_agents.cli.trade import build_live_broker, build_schedule, build_trading_loop
from trading_agents.config.settings import ConfigurationError, Settings
from trading_agents.contracts.market import MarketBar


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


_KITE = {"live_broker": "kite", "kite_api_key": "key", "kite_access_token": "tok"}


def test_live_mode_needs_the_second_opt_in():
    with pytest.raises(ConfigurationError, match="ALLOW_LIVE_TRADING=true"):
        build_live_broker(_settings(), "live")
    with pytest.raises(ConfigurationError, match="ALLOW_LIVE_TRADING=true"):
        build_live_broker(_settings(**_KITE), "live")


def test_kite_has_no_paper_mode():
    with pytest.raises(ConfigurationError, match="no paper-trading"):
        build_live_broker(_settings(**_KITE), "paper")


@pytest.mark.parametrize(
    ("mode", "allow_live_trading", "orders_enabled"),
    [("dry-run", False, False), ("live", True, True)],
)
def test_kite_orders_are_enabled_only_in_live_mode(mode, allow_live_trading, orders_enabled):
    broker, data = build_live_broker(
        _settings(**_KITE, allow_live_trading=allow_live_trading), mode
    )

    assert isinstance(broker, KiteBrokerAdapter)
    assert broker._allow_live is orders_enabled


def test_kite_needs_a_login():
    with pytest.raises(ConfigurationError, match="kite-login"):
        build_live_broker(_settings(live_broker="kite", kite_api_key="key"), "dry-run")


@pytest.mark.parametrize(
    ("mode", "allow_live_trading", "readonly", "allow_live"),
    [
        ("dry-run", False, True, True),  # read-only: cannot trade any account
        ("paper", False, False, False),  # paper accounts only
        ("live", True, False, True),
    ],
)
def test_ib_connection_per_mode(mode, allow_live_trading, readonly, allow_live):
    pytest.importorskip("ib_async")

    broker, _ = build_live_broker(_settings(allow_live_trading=allow_live_trading), mode)

    assert (broker._readonly, broker._allow_live) == (readonly, allow_live)


def test_schedule_defaults_to_30_minutes_before_each_close():
    us = build_schedule(_settings())
    india = build_schedule(_settings(live_broker="kite"))

    assert (us.timezone.key, us.decision_time) == ("America/New_York", time(15, 30))
    assert (india.timezone.key, india.decision_time) == ("Asia/Kolkata", time(15, 0))


def test_schedule_time_can_be_configured_and_is_validated():
    assert build_schedule(_settings(live_decision_time="15:45")).decision_time == time(15, 45)
    with pytest.raises(ConfigurationError, match="HH:MM"):
        build_schedule(_settings(live_decision_time="3:45pm"))


async def test_trading_loop_wiring(tmp_path):
    loop = await build_trading_loop(
        _settings(
            **_KITE,
            anthropic_api_key="k",
            ticker_universe=["INFY", "TCS"],
            live_max_orders_per_day=3,
            live_state_dir=tmp_path,
        ),
        "dry-run",
    )

    assert loop.mode == "dry-run"
    assert loop.universe == ["INFY", "TCS"]
    assert loop.guardrails.max_orders_per_day == 3
    # Separate per mode, so dry/paper runs never share live guardrail state.
    assert loop.journal_path == tmp_path / "kite-dry-run-journal.jsonl"
    assert loop.state_store._path == tmp_path / "kite-dry-run-state.json"
    # US-only evidence sources aren't used for NSE symbols.
    assert loop.orchestrator.evidence_provider is None


_EXECD = {"live_broker": "execd", "execd_token": "t" * 20}


@pytest.mark.parametrize(
    ("mode", "allow_live_trading", "backend", "orders_needed", "blocked"),
    [
        ("dry-run", False, None, False, True),
        ("paper", False, "paper", False, False),
        ("live", True, "kite", True, False),
    ],
)
def test_execd_adapter_checks_per_mode(mode, allow_live_trading, backend, orders_needed, blocked):
    broker, data = build_live_broker(
        _settings(**_EXECD, allow_live_trading=allow_live_trading), mode
    )

    assert data is broker  # the adapter is also the market-data source
    assert broker._require_backend == backend
    assert broker._require_orders_enabled is orders_needed
    assert broker._block_orders is blocked


def test_execd_needs_its_token_and_the_live_opt_in():
    with pytest.raises(ConfigurationError, match="EXECD_TOKEN"):
        build_live_broker(_settings(live_broker="execd"), "dry-run")
    with pytest.raises(ConfigurationError, match="ALLOW_LIVE_TRADING=true"):
        build_live_broker(_settings(**_EXECD), "live")


async def test_execd_nse_fallback_data_uses_the_ns_suffix(monkeypatch):
    from trading_agents.cli import trade as trade_module

    requested = []

    class FakeYFinance:
        async def get_bars(self, ticker, start, end, timeframe):
            requested.append(ticker)
            return [
                MarketBar(
                    ticker=ticker,
                    timestamp="2026-10-26T00:00:00",
                    open=1,
                    high=1,
                    low=1,
                    close=1,
                    volume=1,
                    timeframe="1d",
                )
            ]

    monkeypatch.setattr(trade_module, "CachedDataProvider", lambda inner, cache: FakeYFinance())
    broker, _ = build_live_broker(_settings(**_EXECD, execd_market="nse"), "dry-run")

    (bar,) = await broker._fallback_historical.get_bars("INFY", None, None, "1d")

    assert requested == ["INFY.NS"]
    assert bar.ticker == "INFY"


def test_execd_market_picks_the_schedule_and_evidence():
    assert build_schedule(_settings(**_EXECD)).timezone.key == "Asia/Kolkata"
    assert build_schedule(_settings(**_EXECD, execd_market="us")).timezone.key == "America/New_York"
