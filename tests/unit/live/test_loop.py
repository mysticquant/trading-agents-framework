import json
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading_agents.brokers.base import BrokerError, OrderRejectedError
from trading_agents.contracts.broker import AccountBalance, Order, Position
from trading_agents.contracts.market import MarketBar
from trading_agents.contracts.trading import PortfolioDecision
from trading_agents.live.guardrails import DailyState, Guardrails, StateStore
from trading_agents.live.loop import LiveTradingLoop, Schedule
from trading_agents.orchestration.state import DecisionCycleState, TickerPipelineState

NY = ZoneInfo("America/New_York")
# Monday 2026-10-26, 15:30 New York.
DECISION_AT = datetime(2026, 10, 26, 15, 30, tzinfo=NY)
TODAY = date(2026, 10, 26)


def _bars(ticker: str, last_day: date, price: float = 100.0) -> list[MarketBar]:
    return [
        MarketBar(
            ticker=ticker,
            timestamp=datetime.combine(last_day - timedelta(days=n), time.min),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1.0,
            timeframe="1d",
        )
        for n in range(5, -1, -1)
    ]


class FakeBroker:
    def __init__(self, *, equity: float = 100_000.0, connect_error: Exception | None = None):
        self.equity = equity
        self.connect_error = connect_error
        self.connected = False
        self.placed: list = []
        self.reject: set[str] = set()
        self.after_place = None

    async def connect(self):
        if self.connect_error:
            raise self.connect_error
        self.connected = True

    async def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    async def get_account_balance(self):
        return AccountBalance(cash=self.equity, equity=self.equity, buying_power=self.equity)

    async def get_positions(self):
        return {
            "MSFT": Position(
                ticker="MSFT", quantity=10, avg_cost=90, market_value=1000, unrealized_pnl=100
            )
        }

    async def place_order(self, request):
        if request.ticker in self.reject:
            raise OrderRejectedError("insufficient funds")
        self.placed.append(request)
        if self.after_place:
            self.after_place()
        return Order(
            order_id=str(len(self.placed)),
            ticker=request.ticker,
            side=request.side,
            quantity=request.quantity,
            order_type="market",
            status="filled",
            submitted_at=datetime.now(UTC),
            fill_price=100.0,
        )


class FakeData:
    def __init__(self, bars: dict[str, list[MarketBar]]):
        self.bars = bars

    async def get_bars(self, ticker, start, end, timeframe):
        if isinstance(self.bars[ticker], Exception):
            raise self.bars[ticker]
        return self.bars[ticker]


class ScriptedOrchestrator:
    """Stands in for DecisionCycleOrchestrator: returns fixed decisions."""

    def __init__(self, decisions: list[tuple[str, str, float]]):
        self.decisions = decisions
        self.calls: list[dict] = []

    async def run_cycle(self, *, as_of, universe, market_data, portfolio_state):
        self.calls.append({"as_of": as_of, "universe": universe, "portfolio": portfolio_state})
        state = DecisionCycleState(
            cycle_id="c",
            as_of=as_of,
            ticker_universe=universe,
            portfolio_state_snapshot=portfolio_state,
        )
        for ticker, action, quantity in self.decisions:
            if ticker in universe:
                state.per_ticker[ticker] = TickerPipelineState(
                    ticker=ticker,
                    portfolio_decision=PortfolioDecision(
                        ticker=ticker,
                        final_action=action,
                        final_quantity=quantity,
                        approved_by="portfolio_manager",
                        rationale="why",
                        decided_at=as_of,
                    ),
                )
        return state


@pytest.fixture
def env(tmp_path):
    return {
        "guardrails": Guardrails(kill_switch_path=tmp_path / "KILL_SWITCH"),
        "store": StateStore(tmp_path / "state.json"),
        "journal": tmp_path / "journal.jsonl",
    }


def _loop(env, *, mode="paper", broker=None, data=None, decisions=None, now=DECISION_AT):
    return LiveTradingLoop(
        orchestrator=ScriptedOrchestrator(
            decisions if decisions is not None else [("AAPL", "buy", 10), ("MSFT", "sell", 5)]
        ),
        broker=broker or FakeBroker(),
        data_provider=data
        or FakeData({"AAPL": _bars("AAPL", TODAY), "MSFT": _bars("MSFT", TODAY)}),
        universe=["AAPL", "MSFT"],
        guardrails=env["guardrails"],
        mode=mode,
        schedule=Schedule(NY, time(15, 30)),
        state_store=env["store"],
        journal_path=env["journal"],
        now=lambda: now,
    )


def _journal(env) -> list[dict]:
    return [json.loads(line) for line in env["journal"].read_text().splitlines()]


# --- schedule -------------------------------------------------------------------


def test_schedule_runs_at_the_decision_time_on_weekdays():
    schedule = Schedule(NY, time(15, 30))

    assert schedule.next_run(datetime(2026, 10, 26, 9, 0, tzinfo=NY)) == DECISION_AT
    # Exactly at / after the time -> next weekday.
    assert schedule.next_run(DECISION_AT) == DECISION_AT + timedelta(days=1)
    friday_evening = datetime(2026, 10, 30, 18, 0, tzinfo=NY)
    assert schedule.next_run(friday_evening) == datetime(2026, 11, 2, 15, 30, tzinfo=NY)


def test_schedule_converts_from_other_timezones():
    schedule = Schedule(ZoneInfo("Asia/Kolkata"), time(15, 0))

    # 08:00 UTC = 13:30 IST -> today 15:00 IST.
    assert schedule.next_run(datetime(2026, 10, 26, 8, 0, tzinfo=UTC)) == datetime(
        2026, 10, 26, 15, 0, tzinfo=ZoneInfo("Asia/Kolkata")
    )


# --- one cycle ------------------------------------------------------------------


async def test_dry_run_journals_orders_but_sends_none(env):
    broker = FakeBroker()
    loop = _loop(env, mode="dry-run", broker=broker)

    report = await loop.run_once()

    assert broker.placed == []
    assert [d["outcome"] for d in report.decisions] == ["dry-run (not sent)"] * 2
    # 10 AAPL at 100 = 1,000 <= 5% of 100,000, so untrimmed.
    assert report.decisions[0]["sent_quantity"] == 10
    assert env["store"].load(TODAY).orders_sent == 0  # dry runs don't use up the limit
    (entry,) = _journal(env)
    assert entry["mode"] == "dry-run" and len(entry["decisions"]) == 2


async def test_paper_mode_sends_guardrail_checked_orders_and_persists_the_count(env):
    broker = FakeBroker(equity=10_000.0)  # 5% = 500 -> 5 shares of AAPL at 100
    loop = _loop(env, broker=broker)

    report = await loop.run_once()

    assert [(o.ticker, o.side, o.quantity) for o in broker.placed] == [
        ("AAPL", "buy", 5),
        ("MSFT", "sell", 5),
    ]
    assert "trimmed from 10 to 5" in report.decisions[0]["guardrail"]
    assert report.decisions[0]["outcome"] == "filled"
    assert env["store"].load(TODAY).orders_sent == 2
    # The agents saw the broker's real account state.
    portfolio = loop.orchestrator.calls[0]["portfolio"]
    assert portfolio.equity == 10_000.0 and "MSFT" in portfolio.positions


async def test_daily_order_limit_survives_a_restart(env):
    env["store"].save(DailyState(trading_day=TODAY, starting_equity=100_000.0, orders_sent=10))
    broker = FakeBroker()

    report = await _loop(env, broker=broker).run_once()

    assert broker.placed == []
    assert all("daily order limit" in d["guardrail"] for d in report.decisions)


async def test_kill_switch_skips_the_whole_cycle(env):
    env["guardrails"].kill_switch_path.touch()
    loop = _loop(env)

    report = await loop.run_once()

    assert "kill switch" in report.skipped_reason
    assert loop.orchestrator.calls == []  # no LLM spend either


async def test_kill_switch_engaged_mid_cycle_stops_the_remaining_orders(env):
    broker = FakeBroker()
    broker.after_place = lambda: env["guardrails"].kill_switch_path.touch()

    report = await _loop(env, broker=broker).run_once()

    assert [o.ticker for o in broker.placed] == ["AAPL"]
    assert report.decisions[1]["outcome"] == "blocked"
    assert "kill switch" in report.decisions[1]["guardrail"]


async def test_drawdown_halts_trading_for_the_day(env):
    env["store"].save(DailyState(trading_day=TODAY, starting_equity=100_000.0))
    broker = FakeBroker(equity=97_000.0)
    loop = _loop(env, broker=broker)

    report = await loop.run_once()

    assert "daily drawdown limit" in report.skipped_reason
    assert broker.placed == [] and loop.orchestrator.calls == []
    assert env["store"].load(TODAY).halted_reason == report.skipped_reason


async def test_the_first_cycle_of_the_day_records_starting_equity(env):
    await _loop(env, broker=FakeBroker(equity=123_456.0), decisions=[]).run_once()

    assert env["store"].load(TODAY).starting_equity == 123_456.0


async def test_market_closed_day_is_skipped(env):
    yesterday = TODAY - timedelta(days=1)
    data = FakeData({"AAPL": _bars("AAPL", yesterday), "MSFT": _bars("MSFT", yesterday)})
    loop = _loop(env, data=data)

    report = await loop.run_once()

    assert "market closed" in report.skipped_reason
    assert loop.orchestrator.calls == []


async def test_tickers_without_a_fresh_bar_are_not_traded(env):
    data = FakeData({"AAPL": _bars("AAPL", TODAY), "MSFT": ConnectionError("yahoo down")})
    broker = FakeBroker()
    loop = _loop(env, broker=broker, data=data)

    report = await loop.run_once()

    assert loop.orchestrator.calls[0]["universe"] == ["AAPL"]
    assert [o.ticker for o in broker.placed] == ["AAPL"]
    assert any("MSFT: market data failed" in e for e in report.errors)


async def test_broker_failure_skips_the_cycle(env):
    loop = _loop(env, broker=FakeBroker(connect_error=BrokerError("TWS not running")))

    report = await loop.run_once()

    assert "broker unavailable: TWS not running" in report.skipped_reason
    assert len(_journal(env)) == 1


async def test_a_rejected_order_does_not_stop_the_others(env):
    broker = FakeBroker()
    broker.reject = {"AAPL"}

    report = await _loop(env, broker=broker).run_once()

    assert report.decisions[0]["outcome"] == "rejected: insufficient funds"
    assert [o.ticker for o in broker.placed] == ["MSFT"]


@pytest.mark.parametrize("is_async", [False, True])
async def test_paper_brokers_are_marked_at_todays_close_before_the_balance(env, is_async):
    events = []

    class MarkingBroker(FakeBroker):
        async def get_account_balance(self):
            events.append("balance")
            return await super().get_account_balance()

    broker = MarkingBroker()
    if is_async:

        async def mark_price(ticker, price):
            events.append(("mark", ticker, price))
    else:

        def mark_price(ticker, price):
            events.append(("mark", ticker, price))

    broker.mark_price = mark_price
    data = FakeData({"AAPL": _bars("AAPL", TODAY, price=123.0), "MSFT": _bars("MSFT", TODAY)})

    await _loop(env, broker=broker, data=data, decisions=[]).run_once()

    assert events == [("mark", "AAPL", 123.0), ("mark", "MSFT", 100.0), "balance"]


# --- scheduling loop ------------------------------------------------------------


async def test_run_forever_waits_for_the_decision_time_then_runs(env):
    clock = {"now": datetime(2026, 10, 26, 15, 0, tzinfo=NY)}
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += timedelta(seconds=seconds)
        if len(_journal_lines(env)) >= 1 and clock["now"] > DECISION_AT + timedelta(hours=1):
            raise KeyboardInterrupt  # stop after the first cycle

    loop = _loop(env, mode="dry-run")
    loop._now = lambda: clock["now"]
    loop._sleep = fake_sleep

    with pytest.raises(KeyboardInterrupt):
        await loop.run_forever()

    # 30 minutes waited in <= 60s chunks, then exactly one cycle at 15:30.
    assert max(sleeps) <= 60
    assert sum(sleeps[:30]) == pytest.approx(30 * 60)
    (entry,) = _journal(env)
    assert entry["started_at"].startswith("2026-10-26T15:30")


def _journal_lines(env) -> list[str]:
    return env["journal"].read_text().splitlines() if env["journal"].exists() else []
