"""Local parquet cache for historical bars, keyed by (ticker, timeframe),
so a backtest re-run or a wider date range doesn't refetch data already on
disk. `CachedDataProvider` wraps any `HistoricalDataProvider` transparently
behind the same Protocol.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from trading_agents.contracts.market import MarketBar, Timeframe
from trading_agents.data.providers.base import HistoricalDataProvider

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class ParquetCache:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, timeframe: Timeframe) -> Path:
        return self._cache_dir / f"{ticker}_{timeframe}.parquet"

    def read(self, ticker: str, timeframe: Timeframe) -> list[MarketBar] | None:
        path = self._path(ticker, timeframe)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        return [
            MarketBar(ticker=ticker, timeframe=timeframe, **row)
            for row in frame.to_dict(orient="records")
        ]

    def write(self, ticker: str, timeframe: Timeframe, bars: list[MarketBar]) -> None:
        frame = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ],
            columns=_COLUMNS,
        )
        frame.to_parquet(self._path(ticker, timeframe), index=False)

    def covers(self, ticker: str, timeframe: Timeframe, start: date, end: date) -> bool:
        cached = self.read(ticker, timeframe)
        if not cached:
            return False
        cached_dates = [bar.timestamp.date() for bar in cached]
        return min(cached_dates) <= start and max(cached_dates) >= end


class CachedDataProvider:
    """Wraps any `HistoricalDataProvider`; serves from the parquet cache
    when the requested range is already covered, otherwise fetches from
    `inner` and writes the result back to the cache."""

    def __init__(self, inner: HistoricalDataProvider, cache: ParquetCache) -> None:
        self._inner = inner
        self._cache = cache

    async def get_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]:
        if self._cache.covers(ticker, timeframe, start, end):
            cached = self._cache.read(ticker, timeframe) or []
            return [bar for bar in cached if start <= bar.timestamp.date() <= end]

        fetched = await self._inner.get_bars(ticker, start, end, timeframe)
        if fetched:
            self._cache.write(ticker, timeframe, fetched)
        return fetched
