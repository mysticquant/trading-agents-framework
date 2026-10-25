"""InteractiveBrokersAdapter against a fake IB connection. Everything the
fake hands back (contracts, orders, trades, fills, account values) is a real
ib_async type, so the adapter's mapping is checked against IB's shapes."""

import asyncio
from datetime import UTC, date, datetime

import pytest

pytest.importorskip("ib_async")

from ib_async import (  # noqa: E402
    AccountValue,
    BarData,
    CommissionReport,
    Execution,
    Fill,
    OrderStatus,
    PortfolioItem,
    Stock,
    Trade,
    TradeLogEntry,
)

from trading_agents.brokers.base import BrokerError, OrderRejectedError  # noqa: E402
from trading_agents.brokers.ib import InteractiveBrokersAdapter, duration_for  # noqa: E402
from trading_agents.contracts.broker import OrderRequest  # noqa: E402

_NOW = datetime(2026, 10, 26, 15, 0, tzinfo=UTC)


def fill(trade: Trade, price: float, commission: float = 1.0) -> None:
    qty = trade.order.totalQuantity
    trade.orderStatus.status = OrderStatus.Filled
    trade.orderStatus.filled = qty
    trade.orderStatus.remaining = 0
    trade.orderStatus.avgFillPrice = price
    trade.fills.append(
        Fill(
            contract=trade.contract,
            execution=Execution(shares=qty, price=price, time=_NOW),
            commissionReport=CommissionReport(commission=commission),
            time=_NOW,
        )
    )


def reject(trade: Trade, message: str) -> None:
    trade.orderStatus.status = OrderStatus.Cancelled
    trade.log.append(TradeLogEntry(time=_NOW, status="Cancelled", message=message, errorCode=201))


def accept_only(trade: Trade) -> None:
    trade.orderStatus.status = OrderStatus.PreSubmitted  # e.g. queued before the open


class FakeIB:
    def __init__(self, *, accounts=("DU1234567",), connect_error=None, on_place=None):
        self.accounts = list(accounts)
        self.connect_error = connect_error
        self.on_place = on_place or (lambda trade: fill(trade, 100.0))
        self.connected = False
        self.connect_kwargs: dict = {}
        self.placed: list[Trade] = []
        self.cancelled: list = []
        self.summary: list[AccountValue] = []
        self.portfolio_items: list[PortfolioItem] = []
        self.bars: list[BarData] = []
        self.history_kwargs: dict = {}
        self.unknown_symbols: set[str] = set()
        self._next_id = 1

    async def connectAsync(self, host, port, **kwargs):  # noqa: N802 - ib_async API
        self.connect_kwargs = {"host": host, "port": port, **kwargs}
        if self.connect_error:
            raise self.connect_error
        self.connected = True

    def disconnect(self):
        self.connected = False

    def isConnected(self):  # noqa: N802
        return self.connected

    def managedAccounts(self):  # noqa: N802
        return self.accounts

    async def accountSummaryAsync(self, account=""):  # noqa: N802
        return self.summary

    def portfolio(self, account=""):
        return self.portfolio_items

    async def qualifyContractsAsync(self, *contracts):  # noqa: N802
        return [None if c.symbol in self.unknown_symbols else c for c in contracts]

    async def reqHistoricalDataAsync(self, contract, **kwargs):  # noqa: N802
        self.history_kwargs = kwargs
        return self.bars

    def placeOrder(self, contract, order):  # noqa: N802
        order.orderId = self._next_id
        self._next_id += 1
        trade = Trade(
            contract=contract,
            order=order,
            orderStatus=OrderStatus(orderId=order.orderId, status=OrderStatus.PendingSubmit),
            log=[TradeLogEntry(time=_NOW, status=OrderStatus.PendingSubmit)],
        )
        self.placed.append(trade)
        # IB acknowledges asynchronously, after placeOrder has returned.
        asyncio.get_running_loop().call_soon(self.on_place, trade)
        return trade

    def cancelOrder(self, order):  # noqa: N802
        self.cancelled.append(order)

    def trades(self):
        return self.placed


async def _connected(
    fake: FakeIB | None = None, **kwargs
) -> tuple[InteractiveBrokersAdapter, FakeIB]:
    fake = fake or FakeIB()
    adapter = InteractiveBrokersAdapter(ib=fake, order_ack_timeout_seconds=0.3, **kwargs)
    await adapter.connect()
    return adapter, fake


def _buy(quantity: float = 10.0, **kwargs) -> OrderRequest:
    return OrderRequest(ticker="AAPL", side="buy", quantity=quantity, order_type="market", **kwargs)


# --- connection and account safety ------------------------------------------


async def test_connects_to_a_paper_account():
    adapter, fake = await _connected(port=4002, client_id=7)

    assert adapter.is_connected()
    assert adapter.account == "DU1234567"
    assert fake.connect_kwargs["port"] == 4002
    assert fake.connect_kwargs["clientId"] == 7
    assert fake.connect_kwargs["readonly"] is False


async def test_refuses_a_live_account_unless_explicitly_allowed():
    fake = FakeIB(accounts=["U7654321"])
    adapter = InteractiveBrokersAdapter(ib=fake)

    with pytest.raises(BrokerError, match="not a paper account"):
        await adapter.connect()
    assert not fake.connected  # disconnected again before anything else

    allowed = InteractiveBrokersAdapter(ib=FakeIB(accounts=["U7654321"]), allow_live=True)
    await allowed.connect()
    assert allowed.is_connected()


async def test_connection_failure_explains_what_to_check():
    adapter = InteractiveBrokersAdapter(ib=FakeIB(connect_error=ConnectionRefusedError()))

    with pytest.raises(BrokerError, match="Is it running with the API enabled"):
        await adapter.connect()


async def test_multiple_accounts_need_an_explicit_choice():
    fake = FakeIB(accounts=["DU1", "DU2"])
    with pytest.raises(BrokerError, match="pass `account=`"):
        await InteractiveBrokersAdapter(ib=fake).connect()

    with pytest.raises(BrokerError, match="not managed by this login"):
        await InteractiveBrokersAdapter(ib=FakeIB(accounts=["DU1"]), account="DU9").connect()

    adapter, _ = await _connected(FakeIB(accounts=["DU1", "DU2"]), account="DU2")
    assert adapter.account == "DU2"


async def test_calls_before_connect_raise_a_broker_error():
    adapter = InteractiveBrokersAdapter(ib=FakeIB())

    with pytest.raises(BrokerError, match="call connect"):
        await adapter.place_order(_buy())


# --- account data -------------------------------------------------------------


async def test_account_balance_uses_the_account_currency():
    adapter, fake = await _connected()
    fake.summary = [
        AccountValue("DU1234567", "TotalCashValue", "5000.5", "EUR", ""),
        AccountValue("DU1234567", "TotalCashValue", "10000.25", "USD", ""),
        AccountValue("DU1234567", "NetLiquidation", "25000", "USD", ""),
        AccountValue("DU1234567", "BuyingPower", "40000", "USD", ""),
        AccountValue("DU1234567", "AccountType", "INDIVIDUAL", "", ""),
    ]

    balance = await adapter.get_account_balance()

    assert (balance.cash, balance.equity, balance.buying_power) == (10000.25, 25000.0, 40000.0)


async def test_account_balance_reports_missing_fields():
    adapter, fake = await _connected()
    fake.summary = [AccountValue("DU1234567", "TotalCashValue", "1", "USD", "")]

    with pytest.raises(BrokerError, match="NetLiquidation, BuyingPower"):
        await adapter.get_account_balance()


async def test_positions_include_only_open_stock_positions():
    adapter, fake = await _connected()
    option = Stock("AAPL", "SMART", "USD")
    option.secType = "OPT"
    fake.portfolio_items = [
        PortfolioItem(Stock("AAPL", "SMART", "USD"), 10, 200.0, 2000.0, 180.0, 200.0, 0.0, "DU1"),
        PortfolioItem(Stock("MSFT", "SMART", "USD"), 0, 400.0, 0.0, 390.0, 0.0, 50.0, "DU1"),
        PortfolioItem(option, 2, 5.0, 1000.0, 4.0, 200.0, 0.0, "DU1"),
    ]

    positions = await adapter.get_positions()

    assert list(positions) == ["AAPL"]
    aapl = positions["AAPL"]
    assert (aapl.quantity, aapl.avg_cost, aapl.market_value, aapl.unrealized_pnl) == (
        10.0,
        180.0,
        2000.0,
        200.0,
    )


# --- orders ---------------------------------------------------------------------


async def test_market_buy_fills_and_floors_to_whole_shares():
    adapter, fake = await _connected()

    order = await adapter.place_order(_buy(9.99))

    sent = fake.placed[0].order
    assert (sent.action, sent.totalQuantity, sent.orderType) == ("BUY", 9, "MKT")
    assert (sent.tif, sent.account) == ("DAY", "DU1234567")
    assert order.status == "filled"
    assert order.quantity == 9.0
    assert order.fill_price == 100.0
    assert order.commission == 1.0
    assert order.filled_at == _NOW


async def test_limit_sell_sends_the_limit_price():
    adapter, fake = await _connected()

    order = await adapter.place_order(
        OrderRequest(ticker="AAPL", side="sell", quantity=5, order_type="limit", limit_price=210.5)
    )

    sent = fake.placed[0].order
    assert (sent.action, sent.orderType, sent.lmtPrice) == ("SELL", "LMT", 210.5)
    assert order.order_type == "limit" and order.limit_price == 210.5


async def test_order_below_one_share_is_rejected_without_being_sent():
    adapter, fake = await _connected()

    with pytest.raises(OrderRejectedError, match="rounds down to 0"):
        await adapter.place_order(_buy(0.6))
    assert fake.placed == []


async def test_ib_rejection_raises_with_ibs_reason():
    fake = FakeIB(on_place=lambda t: reject(t, "Order rejected - insufficient buying power"))
    adapter, _ = await _connected(fake)

    with pytest.raises(OrderRejectedError, match="insufficient buying power"):
        await adapter.place_order(_buy())


async def test_order_queued_before_the_open_is_returned_as_submitted():
    adapter, fake = await _connected(FakeIB(on_place=accept_only))

    order = await adapter.place_order(_buy())

    assert order.status == "submitted"
    assert order.fill_price is None

    # The fill arrives later; status lookups see it.
    fill(fake.placed[0], 101.5)
    later = await adapter.get_order_status(order.order_id)
    assert later.status == "filled" and later.fill_price == 101.5


async def test_unacknowledged_order_is_returned_as_pending_after_the_timeout():
    adapter, _ = await _connected(FakeIB(on_place=lambda trade: None))

    order = await adapter.place_order(_buy())

    assert order.status == "pending"


async def test_read_only_connection_refuses_orders():
    adapter, fake = await _connected(readonly=True)

    with pytest.raises(OrderRejectedError, match="read-only"):
        await adapter.place_order(_buy())
    assert fake.connect_kwargs["readonly"] is True


async def test_unknown_ticker_is_rejected():
    fake = FakeIB()
    fake.unknown_symbols = {"NOPE"}
    adapter, _ = await _connected(fake)

    with pytest.raises(OrderRejectedError, match="does not recognize NOPE"):
        await adapter.place_order(
            OrderRequest(ticker="NOPE", side="buy", quantity=1, order_type="market")
        )


async def test_cancel_order_cancels_working_orders_only():
    adapter, fake = await _connected(FakeIB(on_place=accept_only))
    working = await adapter.place_order(_buy())

    await adapter.cancel_order(working.order_id)
    assert fake.cancelled == [fake.placed[0].order]

    fill(fake.placed[0], 100.0)
    await adapter.cancel_order(working.order_id)  # already done: no second cancel
    assert len(fake.cancelled) == 1

    with pytest.raises(BrokerError, match="Unknown order_id"):
        await adapter.cancel_order("999")


# --- historical data -----------------------------------------------------------


async def test_historical_bars_request_and_inclusive_date_filter():
    adapter, fake = await _connected()
    fake.bars = [
        BarData(date=date(2024, 5, 30), open=1, high=2, low=0.5, close=1.5, volume=100),
        BarData(date=date(2024, 5, 31), open=1, high=2, low=0.5, close=1.6, volume=100),
        BarData(date=date(2024, 6, 3), open=1, high=2, low=0.5, close=1.7, volume=100),
    ]

    bars = await adapter.get_historical_bars("AAPL", date(2024, 5, 31), date(2024, 5, 31), "1d")

    assert [b.close for b in bars] == [1.6]
    assert bars[0].timestamp == datetime(2024, 5, 31)
    assert fake.history_kwargs["endDateTime"] == datetime(2024, 6, 1)
    assert fake.history_kwargs["durationStr"] == "1 D"
    assert fake.history_kwargs["barSizeSetting"] == "1 day"
    assert fake.history_kwargs["useRTH"] is True


def test_duration_switches_to_years_past_365_days():
    assert duration_for(date(2024, 1, 1), date(2024, 1, 10)) == "10 D"
    assert duration_for(date(2024, 1, 1), date(2024, 12, 30)) == "365 D"
    assert duration_for(date(2023, 1, 1), date(2024, 6, 30)) == "2 Y"


# --- CLI ------------------------------------------------------------------------


async def test_ib_check_reports_the_account_read_only(monkeypatch):
    from trading_agents.brokers import ib as ib_module
    from trading_agents.cli.main import ib_check
    from trading_agents.config.settings import Settings

    fake = FakeIB()
    fake.summary = [
        AccountValue("DU1234567", "TotalCashValue", "1000", "USD", ""),
        AccountValue("DU1234567", "NetLiquidation", "3000", "USD", ""),
        AccountValue("DU1234567", "BuyingPower", "4000", "USD", ""),
    ]
    fake.portfolio_items = [
        PortfolioItem(Stock("AAPL", "SMART", "USD"), 10, 200.0, 2000.0, 180.0, 200.0, 0.0, "DU1")
    ]
    monkeypatch.setattr(ib_module, "IB", lambda: fake)

    lines = await ib_check(Settings(ib_port=4002, _env_file=None))

    assert fake.connect_kwargs["readonly"] is True
    assert not fake.connected  # disconnects when done
    assert lines[0] == "Connected to 127.0.0.1:4002 - account DU1234567 (paper)"
    assert "Equity: 3,000.00" in lines[1]
    assert lines[2:] == ["Positions: 1", "  AAPL: 10 @ 180.00 (value 2,000.00)"]
