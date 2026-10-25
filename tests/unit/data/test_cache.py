from datetime import date, datetime, timedelta

from trading_agents.contracts.market import MarketBar
from trading_agents.data.cache import CachedDataProvider, ParquetCache


class _WeekdayProvider:
    """Returns one bar per weekday in the inclusive range (like a real
    provider: no bars on weekends) and records every fetch."""

    def __init__(self) -> None:
        self.fetches: list[tuple[date, date]] = []

    async def get_bars(self, ticker, start, end, timeframe):
        self.fetches.append((start, end))
        bars = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                bars.append(
                    MarketBar(
                        ticker=ticker,
                        timestamp=datetime.combine(current, datetime.min.time()),
                        open=1.0,
                        high=1.0,
                        low=1.0,
                        close=float(current.toordinal()),
                        volume=1.0,
                        timeframe=timeframe,
                    )
                )
            current += timedelta(days=1)
        return bars


# Saturday 2024-01-06 .. Sunday 2024-02-04: both ends land on weekends, so
# no cached bar is dated exactly at either boundary.
_START, _END = date(2024, 1, 6), date(2024, 2, 4)


async def test_repeat_request_with_weekend_boundaries_is_served_from_cache(tmp_path):
    inner = _WeekdayProvider()
    provider = CachedDataProvider(inner, ParquetCache(tmp_path))

    first = await provider.get_bars("AAPL", _START, _END, "1d")
    second = await provider.get_bars("AAPL", _START, _END, "1d")

    assert len(inner.fetches) == 1
    assert [b.timestamp for b in second] == [b.timestamp for b in first]


async def test_sub_range_of_a_cached_range_is_served_from_cache(tmp_path):
    inner = _WeekdayProvider()
    provider = CachedDataProvider(inner, ParquetCache(tmp_path))
    await provider.get_bars("AAPL", _START, _END, "1d")

    bars = await provider.get_bars("AAPL", date(2024, 1, 10), date(2024, 1, 20), "1d")

    assert len(inner.fetches) == 1
    assert min(b.timestamp.date() for b in bars) >= date(2024, 1, 10)
    assert max(b.timestamp.date() for b in bars) <= date(2024, 1, 20)


async def test_extending_the_range_keeps_previously_cached_history(tmp_path):
    inner = _WeekdayProvider()
    cache = ParquetCache(tmp_path)
    provider = CachedDataProvider(inner, cache)
    await provider.get_bars("AAPL", _START, _END, "1d")

    later = await provider.get_bars("AAPL", date(2024, 2, 1), date(2024, 3, 1), "1d")

    assert min(b.timestamp.date() for b in later) >= date(2024, 2, 1)
    assert cache.covered_range("AAPL", "1d") == (_START, date(2024, 3, 1))
    # The original range is still a cache hit — nothing was overwritten away.
    await provider.get_bars("AAPL", _START, _END, "1d")
    assert len(inner.fetches) == 2


async def test_coverage_is_never_recorded_past_yesterday(tmp_path):
    inner = _WeekdayProvider()
    cache = ParquetCache(tmp_path)
    provider = CachedDataProvider(inner, cache)
    today = date.today()

    await provider.get_bars("AAPL", today - timedelta(days=30), today, "1d")
    await provider.get_bars("AAPL", today - timedelta(days=30), today, "1d")

    covered = cache.covered_range("AAPL", "1d")
    assert covered is not None and covered[1] == today - timedelta(days=1)
    # Today may not have a final bar yet, so a request through today refetches.
    assert len(inner.fetches) == 2


async def test_a_request_far_from_the_cached_range_is_not_widened_across_the_gap(tmp_path):
    inner = _WeekdayProvider()
    cache = ParquetCache(tmp_path)
    provider = CachedDataProvider(inner, cache)
    await provider.get_bars("AAPL", date(2020, 1, 1), date(2020, 2, 1), "1d")

    await provider.get_bars("AAPL", date(2024, 1, 1), date(2024, 2, 1), "1d")

    # Fetched only what was asked for, not the four years in between.
    assert inner.fetches[-1] == (date(2024, 1, 1), date(2024, 2, 1))
    assert cache.covered_range("AAPL", "1d") == (date(2024, 1, 1), date(2024, 2, 1))


async def test_intraday_requests_are_never_widened(tmp_path):
    inner = _WeekdayProvider()
    provider = CachedDataProvider(inner, ParquetCache(tmp_path))
    await provider.get_bars("AAPL", date(2024, 1, 1), date(2024, 2, 29), "5m")

    # Adjacent to the cached range, but a union would exceed yfinance's
    # ~60-day intraday history and could come back empty.
    await provider.get_bars("AAPL", date(2024, 3, 1), date(2024, 3, 5), "5m")

    assert inner.fetches[-1] == (date(2024, 3, 1), date(2024, 3, 5))
