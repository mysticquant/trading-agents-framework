"""Historical OHLCV via `yfinance`. First/default `HistoricalDataProvider`
implementation — free, no auth, US tickers."""

import asyncio
from datetime import date

import yfinance as yf

from trading_agents.contracts.market import MarketBar, Timeframe

_YFINANCE_INTERVAL = {"1d": "1d", "1h": "1h", "5m": "5m"}


class YFinanceProvider:
    async def get_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]:
        # yfinance's `download` is synchronous network I/O; run it off the
        # event loop so it doesn't block other concurrent agent/data calls.
        frame = await asyncio.to_thread(
            yf.download,
            ticker,
            start=start,
            end=end,
            interval=_YFINANCE_INTERVAL[timeframe],
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )

        bars = []
        for timestamp, row in frame.iterrows():
            bars.append(
                MarketBar(
                    ticker=ticker,
                    timestamp=timestamp.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    timeframe=timeframe,
                )
            )
        return bars
