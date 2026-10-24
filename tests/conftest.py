"""Shared test fixtures."""

from datetime import datetime, timedelta

import pytest

from trading_agents.contracts.market import MarketBar


def make_bars(
    ticker: str, count: int, start_price: float = 100.0, daily_change: float = 1.0
) -> list[MarketBar]:
    """Synthetic ascending-price daily bars, for indicator/pipeline tests
    that just need "enough history", not realistic market behavior."""
    bars = []
    price = start_price
    start = datetime(2024, 1, 1)
    for i in range(count):
        price += daily_change
        bars.append(
            MarketBar(
                ticker=ticker,
                timestamp=start + timedelta(days=i),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1_000_000 + i * 1_000,
                timeframe="1d",
            )
        )
    return bars


@pytest.fixture
def bars_factory():
    return make_bars
