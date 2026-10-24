import pytest

from trading_agents.brokers.base import BrokerError, OrderRejectedError
from trading_agents.brokers.paper import PaperBrokerAdapter
from trading_agents.contracts.broker import OrderRequest


async def test_connect_lifecycle():
    broker = PaperBrokerAdapter(initial_cash=10_000.0)
    assert broker.is_connected() is False
    await broker.connect()
    assert broker.is_connected() is True
    await broker.disconnect()
    assert broker.is_connected() is False


async def test_buy_order_reduces_cash_and_opens_a_position():
    broker = PaperBrokerAdapter(initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    broker.mark_price("AAPL", 100.0)

    order = await broker.place_order(
        OrderRequest(ticker="AAPL", side="buy", quantity=10.0, order_type="market")
    )

    assert order.status == "filled"
    assert order.fill_price == 100.0
    balance = await broker.get_account_balance()
    assert balance.cash == pytest.approx(9_000.0)
    positions = await broker.get_positions()
    assert positions["AAPL"].quantity == 10.0
    assert positions["AAPL"].avg_cost == pytest.approx(100.0)


async def test_sell_order_realizes_pnl_and_can_close_the_position():
    broker = PaperBrokerAdapter(initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    broker.mark_price("AAPL", 100.0)
    await broker.place_order(
        OrderRequest(ticker="AAPL", side="buy", quantity=10.0, order_type="market")
    )

    broker.mark_price("AAPL", 120.0)
    sell_order = await broker.place_order(
        OrderRequest(ticker="AAPL", side="sell", quantity=10.0, order_type="market")
    )

    assert sell_order.fill_price == 120.0
    assert broker.realized_pnl == pytest.approx(200.0)
    positions = await broker.get_positions()
    assert "AAPL" not in positions
    balance = await broker.get_account_balance()
    assert balance.cash == pytest.approx(10_200.0)


async def test_buy_order_commission_and_slippage_increase_cost():
    broker = PaperBrokerAdapter(initial_cash=10_000.0, commission_bps=10.0, slippage_bps=10.0)
    broker.mark_price("AAPL", 100.0)

    order = await broker.place_order(
        OrderRequest(ticker="AAPL", side="buy", quantity=10.0, order_type="market")
    )

    expected_fill_price = 100.0 * 1.001
    assert order.fill_price == pytest.approx(expected_fill_price)
    assert order.commission == pytest.approx(expected_fill_price * 10.0 * 10.0 / 10_000)


async def test_insufficient_cash_rejects_the_order():
    broker = PaperBrokerAdapter(initial_cash=100.0)
    broker.mark_price("AAPL", 100.0)

    with pytest.raises(OrderRejectedError):
        await broker.place_order(
            OrderRequest(ticker="AAPL", side="buy", quantity=10.0, order_type="market")
        )


async def test_selling_more_than_held_rejects_the_order():
    broker = PaperBrokerAdapter(initial_cash=10_000.0)
    broker.mark_price("AAPL", 100.0)
    await broker.place_order(
        OrderRequest(ticker="AAPL", side="buy", quantity=1.0, order_type="market")
    )

    with pytest.raises(OrderRejectedError):
        await broker.place_order(
            OrderRequest(ticker="AAPL", side="sell", quantity=5.0, order_type="market")
        )


async def test_placing_an_order_without_marking_price_first_is_rejected():
    broker = PaperBrokerAdapter(initial_cash=10_000.0)

    with pytest.raises(OrderRejectedError):
        await broker.place_order(
            OrderRequest(ticker="AAPL", side="buy", quantity=1.0, order_type="market")
        )


async def test_unknown_order_id_raises_broker_error():
    broker = PaperBrokerAdapter(initial_cash=10_000.0)

    with pytest.raises(BrokerError):
        await broker.get_order_status("does-not-exist")
    with pytest.raises(BrokerError):
        await broker.cancel_order("does-not-exist")


async def test_average_cost_updates_across_multiple_buys():
    broker = PaperBrokerAdapter(initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    broker.mark_price("AAPL", 100.0)
    await broker.place_order(
        OrderRequest(ticker="AAPL", side="buy", quantity=10.0, order_type="market")
    )

    broker.mark_price("AAPL", 200.0)
    await broker.place_order(
        OrderRequest(ticker="AAPL", side="buy", quantity=10.0, order_type="market")
    )

    positions = await broker.get_positions()
    assert positions["AAPL"].quantity == 20.0
    assert positions["AAPL"].avg_cost == pytest.approx(150.0)
