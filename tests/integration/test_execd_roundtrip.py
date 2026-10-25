"""Python against the real `execd` binary (execution_service/, Rust), paper
backend. Skipped unless the binary is built: `cargo build` in
execution_service/, or point EXECD_BIN at one. CI builds it first.
"""

import os
import subprocess
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trading_agents.brokers.base import BrokerError, OrderRejectedError
from trading_agents.brokers.execution_service import ExecutionServiceAdapter
from trading_agents.contracts.broker import OrderRequest
from trading_agents.contracts.market import MarketBar
from trading_agents.contracts.trading import PortfolioDecision
from trading_agents.live.guardrails import Guardrails, StateStore
from trading_agents.live.loop import LiveTradingLoop, Schedule
from trading_agents.orchestration.state import DecisionCycleState, TickerPipelineState

TOKEN = "integration-token-0123456789"
_ROOT = Path(__file__).resolve().parents[2]
_EXE = "execd.exe" if sys.platform == "win32" else "execd"


def _find_binary() -> Path | None:
    if os.environ.get("EXECD_BIN"):
        return Path(os.environ["EXECD_BIN"])
    for profile in ("release", "debug"):
        candidate = _ROOT / "execution_service" / "target" / profile / _EXE
        if candidate.exists():
            return candidate
    return None


BINARY = _find_binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="execd binary not built")


def _launch(*args: str, token: str = TOKEN) -> subprocess.Popen:
    return subprocess.Popen(
        [str(BINARY), *args],
        env={**os.environ, "EXECD_TOKEN": token},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture
def execd_port():
    process = _launch(
        "--listen",
        "127.0.0.1:0",
        "--backend",
        "paper",
        "--initial-cash",
        "10000",
        "--commission-bps",
        "0",
        "--slippage-bps",
        "0",
    )
    try:
        ready = process.stdout.readline()
        assert ready.startswith("execd listening on 127.0.0.1:"), ready + process.stderr.read()
        assert "backend=paper orders_enabled=true" in ready
        yield int(ready.split()[3].rsplit(":", 1)[1])
    finally:
        process.terminate()
        process.wait(timeout=10)


async def test_adapter_round_trips_through_the_real_daemon(execd_port):
    adapter = ExecutionServiceAdapter(token=TOKEN, port=execd_port, require_backend="paper")
    await adapter.connect()
    try:
        await adapter.mark_price("AAPL", 100.0)
        order = await adapter.place_order(
            OrderRequest(ticker="AAPL", side="buy", quantity=12.5, order_type="market")
        )
        assert (order.status, order.quantity, order.fill_price) == ("filled", 12.0, 100.0)

        positions = await adapter.get_positions()
        balance = await adapter.get_account_balance()
        assert positions["AAPL"].quantity == 12.0
        assert balance.cash == 10_000.0 - 1_200.0
        assert (await adapter.get_order_status(order.order_id)).status == "filled"

        with pytest.raises(OrderRejectedError, match="only hold 12"):
            await adapter.place_order(
                OrderRequest(ticker="AAPL", side="sell", quantity=50, order_type="market")
            )
    finally:
        await adapter.disconnect()


async def test_wrong_token_is_refused(execd_port):
    adapter = ExecutionServiceAdapter(token="wrong-token-0123456789", port=execd_port)

    with pytest.raises(BrokerError, match="Unauthorized"):
        await adapter.connect()


async def test_live_mode_checks_refuse_the_paper_daemon(execd_port):
    adapter = ExecutionServiceAdapter(
        token=TOKEN, port=execd_port, require_backend="kite", require_orders_enabled=True
    )

    with pytest.raises(BrokerError, match="'paper' backend, but 'kite' is required"):
        await adapter.connect()


class _Orchestrator:
    async def run_cycle(self, *, as_of, universe, market_data, portfolio_state):
        state = DecisionCycleState(
            cycle_id="c",
            as_of=as_of,
            ticker_universe=universe,
            portfolio_state_snapshot=portfolio_state,
        )
        state.per_ticker["AAPL"] = TickerPipelineState(
            ticker="AAPL",
            portfolio_decision=PortfolioDecision(
                ticker="AAPL",
                final_action="buy",
                final_quantity=50,
                approved_by="pm",
                rationale="r",
                decided_at=as_of,
            ),
        )
        return state


async def test_live_loop_paper_cycle_through_the_real_daemon(execd_port, tmp_path):
    ny = ZoneInfo("America/New_York")
    decision_at = datetime(2026, 10, 26, 15, 30, tzinfo=ny)

    class TodaysBars:
        async def get_bars(self, ticker, start, end, timeframe):
            return [
                MarketBar(
                    ticker=ticker,
                    timestamp=datetime.combine(end - timedelta(days=n), time.min),
                    open=100,
                    high=100,
                    low=100,
                    close=100,
                    volume=1,
                    timeframe="1d",
                )
                for n in range(3, -1, -1)
            ]

    adapter = ExecutionServiceAdapter(
        token=TOKEN, port=execd_port, require_backend="paper", fallback_historical=TodaysBars()
    )
    loop = LiveTradingLoop(
        orchestrator=_Orchestrator(),
        broker=adapter,
        data_provider=adapter,  # paper backend has no history -> falls back
        universe=["AAPL"],
        guardrails=Guardrails(kill_switch_path=tmp_path / "KILL_SWITCH"),
        mode="paper",
        schedule=Schedule(ny, time(15, 30)),
        state_store=StateStore(tmp_path / "state.json"),
        journal_path=tmp_path / "journal.jsonl",
        now=lambda: decision_at.astimezone(UTC),
    )
    try:
        report = await loop.run_once()
    finally:
        await adapter.disconnect()

    (decision,) = report.decisions
    # 5% of 10,000 equity at 100/share -> 5 shares, filled by the Rust ledger
    # at the price the loop marked.
    assert decision["sent_quantity"] == 5
    assert decision["outcome"] == "filled"
    assert decision["fill_price"] == 100.0
    assert StateStore(tmp_path / "state.json").load(date(2026, 10, 26)).orders_sent == 1


@pytest.mark.parametrize(
    ("args", "token", "message"),
    [
        (["--listen", "0.0.0.0:0"], TOKEN, "non-loopback"),
        (["--listen", "127.0.0.1:0"], "short", "at least 16 characters"),
        (["--listen", "127.0.0.1:0", "--backend", "kite"], TOKEN, "KITE_API_KEY is required"),
    ],
)
def test_daemon_refuses_unsafe_or_incomplete_configuration(args, token, message):
    process = _launch(*args, token=token)
    _, stderr = process.communicate(timeout=30)

    assert process.returncode == 1
    assert message in stderr
