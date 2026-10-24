"""Evidence for the fundamentals, sentiment, and news analysts.

Raw data comes from *sources* (yfinance, SEC EDGAR) as normalized, timestamped
records; pure functions in `point_in_time.py` turn those records into the
evidence dicts the analysts read, keeping only what was public before the
decision cutoff. Every record therefore carries the moment it became public
(`reported_at`, `accepted_at`, ...), not just the period it describes —
that's what keeps backtests free of look-ahead bias.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True)
class EarningsEvent:
    """One quarterly earnings release. `reported_at` is timezone-aware."""

    reported_at: datetime
    eps_estimate: float | None
    eps_reported: float | None
    surprise_pct: float | None


@dataclass(frozen=True)
class QuarterlyStatement:
    """Income-statement and balance-sheet line items for one fiscal quarter,
    keyed by yfinance's row labels (e.g. "Total Revenue")."""

    period_end: date
    values: dict[str, float]


@dataclass(frozen=True)
class AnalystAction:
    """A sell-side rating or price-target change. `at` is timezone-aware.
    `action` is yfinance's code: up, down, init, main (maintain), reit."""

    at: datetime
    firm: str
    action: str
    to_grade: str
    from_grade: str
    price_target: float | None
    prior_price_target: float | None


@dataclass(frozen=True)
class Headline:
    published_at: datetime
    title: str
    publisher: str


@dataclass(frozen=True)
class Filing:
    """An SEC filing. `accepted_at` is EDGAR's acceptance time (UTC), the
    moment the filing became public; `items` holds 8-K item codes."""

    accepted_at: datetime
    form: str
    items: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketSnapshot:
    """Current-only values (no history available), used only when the
    decision date is today — see `point_in_time.is_live`."""

    short_percent_of_float: float | None = None
    recommendation_mean: float | None = None


class MarketDataSource(Protocol):
    async def earnings_events(self, ticker: str) -> list[EarningsEvent]: ...

    async def quarterly_statements(self, ticker: str) -> list[QuarterlyStatement]: ...

    async def analyst_actions(self, ticker: str) -> list[AnalystAction]: ...

    async def headlines(self, ticker: str) -> list[Headline]: ...

    async def snapshot(self, ticker: str) -> MarketSnapshot: ...


class FilingsSource(Protocol):
    async def filings(self, ticker: str) -> list[Filing]: ...


@dataclass
class EvidenceBundle:
    fundamentals: dict[str, object] = field(default_factory=dict)
    sentiment: dict[str, object] = field(default_factory=dict)
    news: dict[str, object] = field(default_factory=dict)


class EvidenceProvider(Protocol):
    async def get_evidence(self, ticker: str, as_of: datetime, price: float) -> EvidenceBundle:
        """`as_of` is the decision date; only data public before it may be
        returned. `price` is the latest close, used for valuation ratios."""
        ...
