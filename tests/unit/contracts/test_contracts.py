from datetime import datetime

import pytest
from pydantic import ValidationError

from trading_agents.contracts.analysis import AnalystReport
from trading_agents.contracts.broker import Order, OrderRequest, Position
from trading_agents.contracts.market import MarketBar
from trading_agents.contracts.portfolio import PortfolioState


def test_market_bar_round_trip():
    bar = MarketBar(
        ticker="AAPL",
        timestamp=datetime(2024, 1, 1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000_000,
        timeframe="1d",
    )
    assert bar.ticker == "AAPL"
    assert bar.timeframe == "1d"


def test_analyst_report_rejects_confidence_outside_unit_interval():
    with pytest.raises(ValidationError):
        AnalystReport(
            ticker="AAPL",
            analyst_type="technical",
            as_of=datetime(2024, 1, 1),
            summary="x",
            signal="bullish",
            confidence=1.5,
        )


def test_order_request_rejects_non_positive_quantity():
    with pytest.raises(ValidationError):
        OrderRequest(ticker="AAPL", side="buy", quantity=0.0, order_type="market")


def test_order_defaults_to_zero_commission():
    order = Order(
        order_id="1",
        ticker="AAPL",
        side="buy",
        quantity=10.0,
        order_type="market",
        status="filled",
        submitted_at=datetime(2024, 1, 1),
    )
    assert order.commission == 0.0


def test_portfolio_state_defaults_to_empty_positions():
    state = PortfolioState(cash=1000.0, equity=1000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1))
    assert state.positions == {}


def test_position_fields_are_required():
    with pytest.raises(ValidationError):
        Position(
            ticker="AAPL", quantity=10.0, avg_cost=100.0
        )  # missing market_value/unrealized_pnl
