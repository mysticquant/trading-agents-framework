"""ExecutionServiceAdapter against an in-process fake `execd` speaking the
same newline-delimited JSON protocol."""

import asyncio
import json
from datetime import date

import pytest

from trading_agents.brokers.base import BrokerAuthExpiredError, BrokerError, OrderRejectedError
from trading_agents.brokers.execution_service import (
    ExecutionServiceAdapter,
    ExecutionServiceNotSupportedError,
)
from trading_agents.contracts.broker import OrderRequest
from trading_agents.contracts.market import MarketBar

TOKEN = "test-token-0123456789"
_ORDER = {
    "order_id": "1",
    "ticker": "AAPL",
    "side": "buy",
    "quantity": 5.0,
    "order_type": "market",
    "limit_price": None,
    "status": "filled",
    "submitted_at": "2026-10-26T15:30:00Z",
    "filled_at": "2026-10-26T15:30:00Z",
    "fill_price": 100.05,
    "commission": 0.25,
}


class FakeExecd:
    def __init__(self, *, backend: str = "paper", orders_enabled: bool = True) -> None:
        self.health = {"backend": backend, "orders_enabled": orders_enabled, "account": None}
        self.handlers: dict = {"health": lambda p: self.health}
        self.requests: list[dict] = []
        self.server: asyncio.base_events.Server | None = None
        self.port = 0
        self.writers: list[asyncio.StreamWriter] = []

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader, writer) -> None:
        self.writers.append(writer)
        while line := await reader.readline():
            request = json.loads(line)
            self.requests.append(request)
            if request["token"] != TOKEN:
                response = {
                    "id": request["id"],
                    "error": {"type": "Unauthorized", "message": "bad"},
                }
            else:
                handler = self.handlers[request["method"]]
                try:
                    result = handler(request["params"])
                    if asyncio.iscoroutine(result):
                        result = await result
                    response = {"id": request["id"], "result": result}
                except _Error as err:
                    response = {
                        "id": request["id"],
                        "error": {"type": err.kind, "message": err.message},
                    }
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        writer.close()

    async def stop(self) -> None:
        # Server.wait_closed() waits for open client connections (3.12+), so
        # close them from this side first, as a stopping daemon would.
        self.server.close()
        for writer in self.writers:
            writer.close()
        await self.server.wait_closed()


class _Error(Exception):
    def __init__(self, kind: str, message: str) -> None:
        self.kind, self.message = kind, message


def _raise(kind: str, message: str):
    def handler(params):
        raise _Error(kind, message)

    return handler


@pytest.fixture
async def execd():
    fake = FakeExecd()
    await fake.start()
    yield fake
    await fake.stop()


def _adapter(fake: FakeExecd, **kwargs) -> ExecutionServiceAdapter:
    return ExecutionServiceAdapter(token=TOKEN, port=fake.port, timeout_seconds=5, **kwargs)


def _buy() -> OrderRequest:
    return OrderRequest(ticker="AAPL", side="buy", quantity=5, order_type="market")


async def test_connect_reads_the_daemons_backend(execd):
    adapter = _adapter(execd)

    await adapter.connect()

    assert adapter.is_connected()
    assert (adapter.backend, adapter.orders_enabled) == ("paper", True)


async def test_connect_refuses_a_backend_other_than_required(execd):
    adapter = _adapter(execd, require_backend="kite")

    with pytest.raises(BrokerError, match="'paper' backend, but 'kite' is required"):
        await adapter.connect()
    assert not adapter.is_connected()


async def test_connect_refuses_a_daemon_without_orders_when_orders_are_needed():
    fake = FakeExecd(backend="kite", orders_enabled=False)
    await fake.start()
    try:
        with pytest.raises(BrokerError, match="--allow-live"):
            await _adapter(fake, require_backend="kite", require_orders_enabled=True).connect()
    finally:
        await fake.stop()


async def test_unreachable_daemon_says_so():
    adapter = ExecutionServiceAdapter(token=TOKEN, port=1, timeout_seconds=2)

    with pytest.raises(BrokerError, match="Is `execd` running"):
        await adapter.connect()


async def test_orders_round_trip_into_the_python_contract(execd):
    execd.handlers["place_order"] = lambda params: {**_ORDER, "quantity": params["quantity"]}
    adapter = _adapter(execd)
    await adapter.connect()

    order = await adapter.place_order(_buy())

    assert order.status == "filled" and order.fill_price == 100.05
    sent = execd.requests[-1]
    assert sent["token"] == TOKEN
    assert sent["params"] == {
        "ticker": "AAPL",
        "side": "buy",
        "quantity": 5.0,
        "order_type": "market",
        "limit_price": None,
    }


@pytest.mark.parametrize(
    ("kind", "error_class"),
    [
        ("OrderRejected", OrderRejectedError),
        ("AuthExpired", BrokerAuthExpiredError),
        ("Broker", BrokerError),
    ],
)
async def test_service_errors_map_onto_broker_exceptions(execd, kind, error_class):
    execd.handlers["place_order"] = _raise(kind, "nope")
    adapter = _adapter(execd)
    await adapter.connect()

    with pytest.raises(error_class, match=f"nope \\[execd {kind}\\]"):
        await adapter.place_order(_buy())
    assert adapter.is_connected()  # an error response doesn't drop the connection


async def test_dry_run_blocks_orders_before_anything_is_sent(execd):
    adapter = _adapter(execd, block_orders=True)
    await adapter.connect()

    with pytest.raises(OrderRejectedError, match="dry run"):
        await adapter.place_order(_buy())
    assert [r["method"] for r in execd.requests] == ["health"]


async def test_history_falls_back_when_the_backend_has_none(execd):
    execd.handlers["get_historical_bars"] = _raise("NotSupported", "no market data")

    class Fallback:
        async def get_bars(self, ticker, start, end, timeframe):
            return [
                MarketBar(
                    ticker=ticker,
                    timestamp="2026-10-26T00:00:00",
                    open=1,
                    high=1,
                    low=1,
                    close=1,
                    volume=1,
                    timeframe=timeframe,
                )
            ]

    with_fallback = _adapter(execd, fallback_historical=Fallback())
    without = _adapter(execd)
    await with_fallback.connect()
    await without.connect()

    bars = await with_fallback.get_bars("AAPL", date(2026, 10, 26), date(2026, 10, 26), "1d")
    assert [b.ticker for b in bars] == ["AAPL"]
    with pytest.raises(ExecutionServiceNotSupportedError):
        await without.get_bars("AAPL", date(2026, 10, 26), date(2026, 10, 26), "1d")


async def test_history_from_the_service_validates_into_market_bars(execd):
    execd.handlers["get_historical_bars"] = lambda p: [
        {
            "ticker": "INFY",
            "timestamp": "2026-10-26T00:00:00",
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 100,
            "timeframe": "1d",
        }
    ]
    adapter = _adapter(execd)
    await adapter.connect()

    (bar,) = await adapter.get_bars("INFY", date(2026, 10, 1), date(2026, 10, 26), "1d")

    assert bar.close == 1.5
    assert execd.requests[-1]["params"]["start"] == "2026-10-01"


async def test_mark_price_only_reaches_a_paper_backend():
    calls = []
    for backend in ("paper", "kite"):
        fake = FakeExecd(backend=backend)
        fake.handlers["mark_price"] = lambda p, b=backend: calls.append(b)
        await fake.start()
        adapter = _adapter(fake)
        await adapter.connect()
        await adapter.mark_price("AAPL", 100.0)
        await adapter.disconnect()
        await fake.stop()

    assert calls == ["paper"]


async def test_a_closed_connection_raises_and_marks_disconnected(execd):
    adapter = _adapter(execd)
    await adapter.connect()
    await execd.stop()  # the daemon goes away

    with pytest.raises(BrokerError):
        await adapter.get_account_balance()
    assert not adapter.is_connected()


# --- review fixes -------------------------------------------------------------


async def test_orders_get_a_longer_timeout_than_other_calls(execd):
    async def slow(params):
        await asyncio.sleep(0.5)  # like a slow Kite POST plus status polling
        return {**_ORDER, "quantity": params["quantity"]}

    execd.handlers["place_order"] = slow
    execd.handlers["get_account_balance"] = slow
    adapter = ExecutionServiceAdapter(
        token=TOKEN, port=execd.port, timeout_seconds=0.2, order_timeout_seconds=5
    )
    await adapter.connect()

    order = await adapter.place_order(_buy())  # waited out, not abandoned
    assert order.order_id == "1"

    with pytest.raises(BrokerError, match="failed"):
        await adapter.get_account_balance()


async def test_a_daemon_restart_is_noticed_so_the_loop_reconnects(execd):
    adapter = _adapter(execd)
    await adapter.connect()
    assert adapter.is_connected()

    await execd.stop()  # e.g. restarted with a new Kite token
    await asyncio.sleep(0.05)

    assert not adapter.is_connected()
