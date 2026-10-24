"""Historical market data provider interface.

`YFinanceProvider` is the first implementation (US tickers, free, no auth).
A future Zerodha Kite historical-data provider (Indian tickers) implements
the same Protocol and slots in without touching the backtest runner.
"""

from datetime import date
from typing import Protocol

from trading_agents.contracts.market import MarketBar, Timeframe


class HistoricalDataProvider(Protocol):
    async def get_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]: ...
