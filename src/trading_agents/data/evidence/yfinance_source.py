"""`MarketDataSource` backed by yfinance: earnings releases, quarterly
statements, analyst rating changes, recent headlines, and a current
snapshot. Network calls are synchronous, so each runs in a worker thread.

History available through yfinance (as of this writing):
- earnings releases with report timestamps: decades for large caps (~100
  releases requested; reported EPS present for roughly the last 20 years)
- quarterly income statement / balance sheet: only the last ~5 quarters,
  so margins/leverage are missing for backtest dates more than ~a year back
- analyst upgrades/downgrades: years. Their price targets are stored as
  issued (not split-adjusted) while yfinance prices are split-adjusted, so
  targets are rescaled with the ticker's split history (`split_adjust`).
- headlines (`yf.Search`): only the last few days
"""

import asyncio
import math
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from trading_agents.data.evidence.base import (
    AnalystAction,
    EarningsEvent,
    Headline,
    MarketSnapshot,
    QuarterlyStatement,
)
from trading_agents.data.evidence.memo import AsyncTTLCache

_SIX_HOURS = 6 * 60 * 60


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _aware_utc(timestamp: pd.Timestamp) -> datetime:
    # yfinance returns some indexes tz-aware (earnings) and some naive UTC
    # (rating changes, converted from epoch seconds).
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def parse_earnings_dates(frame: pd.DataFrame | None) -> list[EarningsEvent]:
    if frame is None or frame.empty:
        return []
    surprise_col = "Surprise(%)" if "Surprise(%)" in frame.columns else "Surprise (%)"
    # yfinance occasionally lists one release twice; summing "the last four"
    # would then double-count it. Keep one per release date, preferring the
    # row that has a reported EPS.
    by_date: dict[object, EarningsEvent] = {}
    for ts, row in frame.iterrows():
        event = EarningsEvent(
            reported_at=_aware_utc(ts),
            eps_estimate=_float_or_none(row.get("EPS Estimate")),
            eps_reported=_float_or_none(row.get("Reported EPS")),
            surprise_pct=_float_or_none(row.get(surprise_col)),
        )
        key = event.reported_at.date()
        existing = by_date.get(key)
        if existing is None or (existing.eps_reported is None and event.eps_reported is not None):
            by_date[key] = event
    return sorted(by_date.values(), key=lambda e: e.reported_at)


def parse_statements(*frames: pd.DataFrame | None) -> list[QuarterlyStatement]:
    """Merge statement frames (rows = line items, columns = period ends)
    into one record per quarter."""
    by_period: dict[Any, dict[str, float]] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for period, column in frame.items():
            values = by_period.setdefault(pd.Timestamp(period).date(), {})
            for label, value in column.items():
                number = _float_or_none(value)
                if number is not None:
                    values[str(label)] = number
    return [
        QuarterlyStatement(period_end=period, values=values)
        for period, values in sorted(by_period.items())
        if values
    ]


def parse_upgrades_downgrades(frame: pd.DataFrame | None) -> list[AnalystAction]:
    if frame is None or frame.empty:
        return []

    def target(value: Any) -> float | None:
        number = _float_or_none(value)
        return number if number is not None and number > 0 else None

    return [
        AnalystAction(
            at=_aware_utc(ts),
            firm=str(row.get("Firm", "")),
            action=str(row.get("Action", "")),
            to_grade=str(row.get("ToGrade", "")),
            from_grade=str(row.get("FromGrade", "")),
            price_target=target(row.get("currentPriceTarget")),
            prior_price_target=target(row.get("priorPriceTarget")),
        )
        for ts, row in frame.iterrows()
    ]


def parse_splits(series: pd.Series | None) -> list[tuple[date, float]]:
    """yfinance `Ticker.splits` (index = split date, value = ratio, e.g. 10.0
    for 10-for-1) as (date, ratio) pairs."""
    if series is None or series.empty:
        return []
    return [
        (pd.Timestamp(ts).date(), float(ratio))
        for ts, ratio in series.items()
        if _float_or_none(ratio) and float(ratio) > 0
    ]


def split_adjust(
    actions: list[AnalystAction], splits: list[tuple[date, float]]
) -> list[AnalystAction]:
    """Rescale price targets onto today's split-adjusted basis, matching the
    split-adjusted closes they are compared against. A target issued before a
    split is divided by every split ratio dated after it (e.g. a pre-split
    NVDA target of $1,200 becomes $120 after its 10-for-1 split)."""
    if not splits:
        return actions

    def factor(issued: date) -> float:
        total = 1.0
        for split_date, ratio in splits:
            if split_date > issued:
                total *= ratio
        return total

    adjusted = []
    for action in actions:
        f = factor(action.at.date())
        if f == 1.0:
            adjusted.append(action)
            continue
        adjusted.append(
            replace(
                action,
                price_target=action.price_target / f if action.price_target else None,
                prior_price_target=(
                    action.prior_price_target / f if action.prior_price_target else None
                ),
            )
        )
    return adjusted


def parse_search_news(ticker: str, items: list[dict[str, Any]]) -> list[Headline]:
    headlines = []
    for item in items:
        related = item.get("relatedTickers")
        if related and ticker.upper() not in {t.upper() for t in related}:
            continue
        published = item.get("providerPublishTime")
        if published is None or not item.get("title"):
            continue
        headlines.append(
            Headline(
                published_at=datetime.fromtimestamp(int(published), tz=UTC),
                title=str(item["title"]),
                publisher=str(item.get("publisher", "")),
            )
        )
    return headlines


class YFinanceMarketDataSource:
    def __init__(self, ttl_seconds: float = _SIX_HOURS) -> None:
        self._earnings: AsyncTTLCache[list[EarningsEvent]] = AsyncTTLCache(ttl_seconds)
        self._statements: AsyncTTLCache[list[QuarterlyStatement]] = AsyncTTLCache(ttl_seconds)
        self._actions: AsyncTTLCache[list[AnalystAction]] = AsyncTTLCache(ttl_seconds)
        self._headlines: AsyncTTLCache[list[Headline]] = AsyncTTLCache(ttl_seconds)
        self._snapshots: AsyncTTLCache[MarketSnapshot] = AsyncTTLCache(ttl_seconds)

    async def earnings_events(self, ticker: str) -> list[EarningsEvent]:
        async def fetch() -> list[EarningsEvent]:
            frame = await asyncio.to_thread(lambda: yf.Ticker(ticker).get_earnings_dates(limit=100))
            return parse_earnings_dates(frame)

        return await self._earnings.get(ticker, fetch)

    async def quarterly_statements(self, ticker: str) -> list[QuarterlyStatement]:
        async def fetch() -> list[QuarterlyStatement]:
            def load() -> list[QuarterlyStatement]:
                yf_ticker = yf.Ticker(ticker)
                return parse_statements(
                    yf_ticker.quarterly_income_stmt, yf_ticker.quarterly_balance_sheet
                )

            return await asyncio.to_thread(load)

        return await self._statements.get(ticker, fetch)

    async def analyst_actions(self, ticker: str) -> list[AnalystAction]:
        async def fetch() -> list[AnalystAction]:
            def load() -> list[AnalystAction]:
                yf_ticker = yf.Ticker(ticker)
                actions = parse_upgrades_downgrades(yf_ticker.upgrades_downgrades)
                return split_adjust(actions, parse_splits(yf_ticker.splits))

            return await asyncio.to_thread(load)

        return await self._actions.get(ticker, fetch)

    async def headlines(self, ticker: str) -> list[Headline]:
        # `Ticker.news` currently returns nothing; `Search` still works.
        async def fetch() -> list[Headline]:
            items = await asyncio.to_thread(lambda: yf.Search(ticker, news_count=20).news)
            return parse_search_news(ticker, items or [])

        return await self._headlines.get(ticker, fetch)

    async def snapshot(self, ticker: str) -> MarketSnapshot:
        async def fetch() -> MarketSnapshot:
            info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info) or {}
            return MarketSnapshot(
                short_percent_of_float=_float_or_none(info.get("shortPercentOfFloat")),
                recommendation_mean=_float_or_none(info.get("recommendationMean")),
            )

        return await self._snapshots.get(ticker, fetch)
