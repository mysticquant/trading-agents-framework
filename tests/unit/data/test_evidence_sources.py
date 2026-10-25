from datetime import UTC, date, datetime

import pandas as pd
import pytest

from trading_agents.data.evidence.base import AnalystAction
from trading_agents.data.evidence.memo import AsyncTTLCache
from trading_agents.data.evidence.sec_edgar import (
    SecEdgarFilingsSource,
    parse_submissions,
    parse_ticker_map,
)
from trading_agents.data.evidence.yfinance_source import (
    parse_earnings_dates,
    parse_search_news,
    parse_splits,
    parse_statements,
    parse_upgrades_downgrades,
    split_adjust,
)


def test_parse_earnings_dates_handles_tz_aware_index_and_missing_values():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-11-02 15:00", tz="America/New_York"),
            pd.Timestamp("2026-07-30 16:00", tz="America/New_York"),
        ]
    )
    frame = pd.DataFrame(
        {
            "EPS Estimate": [1.98, 1.89],
            "Reported EPS": [float("nan"), 2.02],
            "Surprise(%)": [float("nan"), 6.74],
        },
        index=index,
    )

    released, scheduled = parse_earnings_dates(frame)  # oldest first

    assert released.reported_at == datetime(2026, 7, 30, 20, 0, tzinfo=UTC)
    assert released.surprise_pct == 6.74
    assert scheduled.eps_reported is None  # scheduled, not yet reported


def test_parse_earnings_dates_keeps_one_release_per_date():
    # yfinance sometimes lists a release twice; summing the last four
    # releases for TTM EPS would then double-count it.
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-08-01 16:00", tz="America/New_York"),
            pd.Timestamp("2024-08-01 16:00", tz="America/New_York"),
            pd.Timestamp("2024-05-02 16:00", tz="America/New_York"),
        ]
    )
    frame = pd.DataFrame(
        {
            "EPS Estimate": [1.3, 1.3, 1.5],
            "Reported EPS": [float("nan"), 1.4, 1.53],
            "Surprise(%)": [float("nan"), 7.7, 2.0],
        },
        index=index,
    )

    events = parse_earnings_dates(frame)

    assert [e.eps_reported for e in events] == [1.53, 1.4]


def test_parse_statements_merges_income_and_balance_sheet_by_quarter():
    periods = [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31")]
    income = pd.DataFrame(
        {periods[0]: [100.0, 40.0], periods[1]: [90.0, float("nan")]},
        index=["Total Revenue", "Gross Profit"],
    )
    balance = pd.DataFrame({periods[0]: [50.0], periods[1]: [45.0]}, index=["Total Debt"])

    statements = parse_statements(income, balance)

    assert [s.period_end for s in statements] == [date(2026, 3, 31), date(2026, 6, 30)]
    assert statements[1].values == {
        "Total Revenue": 100.0,
        "Gross Profit": 40.0,
        "Total Debt": 50.0,
    }
    assert "Gross Profit" not in statements[0].values


def test_parse_upgrades_downgrades_treats_naive_timestamps_as_utc():
    frame = pd.DataFrame(
        {
            "Firm": ["JP Morgan"],
            "ToGrade": ["Overweight"],
            "FromGrade": ["Neutral"],
            "Action": ["up"],
            "priceTargetAction": ["Raises"],
            "currentPriceTarget": [360.0],
            "priorPriceTarget": [0.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-10-12 18:47:46")], name="GradeDate"),
    )

    (action,) = parse_upgrades_downgrades(frame)

    assert action.at == datetime(2026, 10, 12, 18, 47, 46, tzinfo=UTC)
    assert action.price_target == 360.0
    assert action.prior_price_target is None  # 0 means "no prior target"


def test_parse_search_news_drops_items_about_other_tickers():
    items = [
        {
            "title": "Apple up",
            "publisher": "X",
            "providerPublishTime": 1792876140,
            "relatedTickers": ["AAPL"],
        },
        {
            "title": "Unrelated",
            "publisher": "X",
            "providerPublishTime": 1792876140,
            "relatedTickers": ["MSFT"],
        },
        {"title": "No tickers listed", "publisher": "X", "providerPublishTime": 1792876140},
    ]

    headlines = parse_search_news("AAPL", items)

    assert [h.title for h in headlines] == ["Apple up", "No tickers listed"]


def test_parse_submissions_reads_recent_filings():
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q"],
                "acceptanceDateTime": ["2026-07-30T20:30:28.000Z", "2026-08-01T12:00:00.000Z"],
                "items": ["2.02,9.01", ""],
            }
        }
    }

    first, second = parse_submissions(payload)

    assert first.accepted_at == datetime(2026, 7, 30, 20, 30, 28, tzinfo=UTC)
    assert first.items == ("2.02", "9.01")
    assert second.form == "10-Q" and second.items == ()


async def test_sec_source_maps_ticker_to_cik_and_caches_requests():
    requested: list[str] = []
    payloads = {
        "https://www.sec.gov/files/company_tickers.json": {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        },
        "https://data.sec.gov/submissions/CIK0000320193.json": {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "acceptanceDateTime": ["2026-07-30T20:30:28.000Z"],
                    "items": ["2.02"],
                }
            }
        },
    }

    async def fake_fetch(url: str):
        requested.append(url)
        return payloads[url]

    source = SecEdgarFilingsSource("Test test@example.com", fetch_json=fake_fetch)

    assert len(await source.filings("aapl")) == 1
    assert len(await source.filings("AAPL")) == 1
    assert await source.filings("NOTUS") == []
    assert len(requested) == 2  # ticker map once, submissions once


def test_parse_ticker_map_uppercases_tickers():
    assert parse_ticker_map({"0": {"cik_str": 1, "ticker": "brk-b"}}) == {"BRK-B": 1}


async def test_ttl_cache_refetches_only_after_expiry():
    clock = {"now": 0.0}
    cache: AsyncTTLCache[int] = AsyncTTLCache(10.0, clock=lambda: clock["now"])
    calls = 0

    async def fetch() -> int:
        nonlocal calls
        calls += 1
        return calls

    assert await cache.get("k", fetch) == 1
    clock["now"] = 9.0
    assert await cache.get("k", fetch) == 1
    clock["now"] = 10.5
    assert await cache.get("k", fetch) == 2


async def test_ttl_cache_remembers_failures_briefly_then_retries():
    clock = {"now": 0.0}
    cache: AsyncTTLCache[int] = AsyncTTLCache(
        600.0, clock=lambda: clock["now"], failure_ttl_seconds=60.0
    )
    attempts = 0

    async def boom() -> int:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("SEC says 403")

    async def ok() -> int:
        return 7

    for _ in range(3):  # e.g. three tickers on the same simulated day
        with pytest.raises(RuntimeError, match="403"):
            await cache.get("k", boom)
    assert attempts == 1  # the failing source was only hit once

    clock["now"] = 61.0
    assert await cache.get("k", ok) == 7  # retried after the failure window


def _action(at: datetime, target: float | None, prior: float | None) -> AnalystAction:
    return AnalystAction(
        at=at,
        firm="F",
        action="main",
        to_grade="Buy",
        from_grade="Buy",
        price_target=target,
        prior_price_target=prior,
    )


def test_split_adjust_rescales_targets_issued_before_a_split():
    # NVDA: targets stored as issued (~$1,200), prices split-adjusted (~$95).
    splits = parse_splits(
        pd.Series([4.0, 10.0], index=pd.DatetimeIndex(["2021-07-20", "2024-06-10"]))
    )
    actions = [
        _action(datetime(2024, 5, 23, 17, tzinfo=UTC), 1200.0, 1000.0),  # before 10:1
        _action(datetime(2021, 1, 5, 17, tzinfo=UTC), 640.0, None),  # before both
        _action(datetime(2024, 6, 20, 17, tzinfo=UTC), 140.0, 130.0),  # after both
    ]

    before_10, before_both, after = split_adjust(actions, splits)

    assert (before_10.price_target, before_10.prior_price_target) == (120.0, 100.0)
    assert before_both.price_target == 16.0  # / 4 / 10
    assert before_both.prior_price_target is None
    assert (after.price_target, after.prior_price_target) == (140.0, 130.0)


def test_parse_splits_handles_no_splits():
    assert parse_splits(pd.Series(dtype=float)) == []
    assert (
        split_adjust([_action(datetime(2024, 1, 1, tzinfo=UTC), 10.0, 9.0)], [])[0].price_target
        == 10.0
    )
