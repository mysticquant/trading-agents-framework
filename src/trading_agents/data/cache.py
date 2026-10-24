"""Local parquet cache for historical bars, keyed by (ticker, timeframe),
so a backtest re-run or a wider date range doesn't refetch data already on
disk. `CachedDataProvider` wraps any `HistoricalDataProvider` transparently
behind the same Protocol.

Coverage is tracked as the date range that was *requested* from the inner
provider (stored in a small JSON sidecar next to the parquet file), not
inferred from the first/last cached bar: requested ranges routinely start or
end on weekends/holidays, so bar dates alone almost never "cover" them.
"""

import json
from datetime import date, timedelta
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

    def _range_path(self, ticker: str, timeframe: Timeframe) -> Path:
        return self._cache_dir / f"{ticker}_{timeframe}.range.json"

    def read(self, ticker: str, timeframe: Timeframe) -> list[MarketBar] | None:
        path = self._path(ticker, timeframe)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        return [
            MarketBar(ticker=ticker, timeframe=timeframe, **row)
            for row in frame.to_dict(orient="records")
        ]

    def covered_range(self, ticker: str, timeframe: Timeframe) -> tuple[date, date] | None:
        """The inclusive date range the cached bars are complete for, or
        None if unknown (no cache, or a file written before ranges were
        tracked)."""
        range_path = self._range_path(ticker, timeframe)
        if not range_path.exists() or not self._path(ticker, timeframe).exists():
            return None
        meta = json.loads(range_path.read_text(encoding="utf-8"))
        return date.fromisoformat(meta["start"]), date.fromisoformat(meta["end"])

    def write(
        self,
        ticker: str,
        timeframe: Timeframe,
        bars: list[MarketBar],
        start: date,
        end: date,
    ) -> None:
        """`bars` must be the complete set the provider returned for
        `start`..`end` (inclusive); that range is recorded as covered."""
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
        self._range_path(ticker, timeframe).write_text(
            json.dumps({"start": start.isoformat(), "end": end.isoformat()}), encoding="utf-8"
        )

    def covers(self, ticker: str, timeframe: Timeframe, start: date, end: date) -> bool:
        covered = self.covered_range(ticker, timeframe)
        return covered is not None and covered[0] <= start and covered[1] >= end


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

        # Fetch the union of the request and what's already cached, so the
        # rewritten file stays one contiguous, complete range instead of
        # discarding previously cached history.
        fetch_start, fetch_end = start, end
        covered = self._cache.covered_range(ticker, timeframe)
        if covered is not None:
            fetch_start, fetch_end = min(start, covered[0]), max(end, covered[1])

        fetched = await self._inner.get_bars(ticker, fetch_start, fetch_end, timeframe)
        if fetched:
            # Today's (and any future) bars may not exist or be final yet;
            # only record coverage through yesterday so a later run refetches.
            covered_end = min(fetch_end, date.today() - timedelta(days=1))
            if covered_end >= fetch_start:
                self._cache.write(ticker, timeframe, fetched, fetch_start, covered_end)
        return [bar for bar in fetched if start <= bar.timestamp.date() <= end]
