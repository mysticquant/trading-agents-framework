"""Pure functions that turn timestamped source records into analyst
evidence, keeping only what was public strictly before the decision time.

The backtest runner's `as_of` is midnight of the decision day, so anything
released on that day itself (pre-market news, after-close earnings) is
excluded — conservative, but never leaks the future. A naive `as_of` is
read as exchange-local (US/Eastern) time.

Each evidence dict has the same shape: a few descriptive fields, a flat
`metrics` dict of floats (what analysts echo into their reports), and a
`note` whenever coverage is missing so the LLM never reads meaning into an
absence of data.
"""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from trading_agents.data.evidence.base import (
    AnalystAction,
    EarningsEvent,
    Filing,
    Headline,
    MarketSnapshot,
    QuarterlyStatement,
)

EXCHANGE_TZ = ZoneInfo("America/New_York")

# A quarter's statements are public once its earnings are released. When no
# matching release is known, assume the 10-Q filing deadline (40-45 days).
_RELEASE_SEARCH_WINDOW = timedelta(days=120)
_FALLBACK_REPORT_LAG = timedelta(days=45)

# Snapshot-only values (no history) are used only for decisions this recent.
_LIVE_WINDOW = timedelta(days=3)

UNAVAILABLE_NOTE = (
    "No data of this kind was available for this ticker as of this date. "
    "Do not infer anything from its absence."
)

# Form 8-K item codes -> plain-English event type.
EIGHT_K_ITEMS = {
    "1.01": "Entry into a material definitive agreement",
    "1.02": "Termination of a material definitive agreement",
    "1.03": "Bankruptcy or receivership",
    "1.05": "Material cybersecurity incident",
    "2.01": "Completion of acquisition or disposition of assets",
    "2.02": "Results of operations and financial condition",
    "2.03": "Creation of a direct financial obligation",
    "2.04": "Triggering events that accelerate a financial obligation",
    "2.05": "Costs associated with exit or disposal activities",
    "2.06": "Material impairments",
    "3.01": "Notice of delisting or failure to satisfy a listing rule",
    "3.02": "Unregistered sales of equity securities",
    "3.03": "Material modification to rights of security holders",
    "4.01": "Change in certifying accountant",
    "4.02": "Non-reliance on previously issued financial statements",
    "5.01": "Change in control of registrant",
    "5.02": "Departure or appointment of directors or officers",
    "5.03": "Amendments to articles or bylaws; change in fiscal year",
    "5.07": "Submission of matters to a vote of security holders",
    "7.01": "Regulation FD disclosure",
    "8.01": "Other events",
    "9.01": "Financial statements and exhibits",
}
# Exhibit-only filings carry no event of their own.
_NON_MATERIAL_ITEMS = {"9.01"}
_PERIODIC_FORMS = ("10-Q", "10-K")


def decision_cutoff(as_of: datetime) -> datetime:
    """`as_of` as an aware UTC datetime; naive values are exchange-local."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=EXCHANGE_TZ)
    return as_of.astimezone(UTC)


def is_live(as_of: datetime, now: datetime | None = None) -> bool:
    now = now or datetime.now(UTC)
    return decision_cutoff(as_of) >= now - _LIVE_WINDOW


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _put(metrics: dict[str, float], key: str, value: float | None) -> None:
    if value is not None:
        metrics[key] = round(value, 4)


def _sum_eps(releases: list[EarningsEvent]) -> float:
    return sum(e.eps_reported or 0.0 for e in releases)


def _period_end_utc(period_end: date) -> datetime:
    return datetime.combine(period_end, time.max, tzinfo=EXCHANGE_TZ).astimezone(UTC)


def statement_public_at(statement: QuarterlyStatement, earnings: list[EarningsEvent]) -> datetime:
    """When a quarter's statements became public: its earnings release (the
    first one after the period end), else period end + the filing lag."""
    period_end = _period_end_utc(statement.period_end)
    releases = [
        event.reported_at
        for event in earnings
        if period_end < event.reported_at <= period_end + _RELEASE_SEARCH_WINDOW
    ]
    return min(releases) if releases else period_end + _FALLBACK_REPORT_LAG


def fundamentals_evidence(
    *,
    earnings: list[EarningsEvent] | None,
    statements: list[QuarterlyStatement] | None,
    as_of: datetime,
    price: float,
) -> dict[str, object]:
    cutoff = decision_cutoff(as_of)
    releases = sorted(
        (e for e in earnings or [] if e.reported_at < cutoff and e.eps_reported is not None),
        key=lambda e: e.reported_at,
    )
    quarters = sorted(
        (s for s in statements or [] if statement_public_at(s, earnings or []) < cutoff),
        key=lambda s: s.period_end,
    )

    evidence: dict[str, object] = {"price": price}
    metrics: dict[str, float] = {}

    if releases:
        last = releases[-1]
        evidence["last_earnings_release"] = last.reported_at.date().isoformat()
        _put(metrics, "last_eps_reported", last.eps_reported)
        _put(metrics, "last_eps_surprise_pct", last.surprise_pct)
        surprises = [e.surprise_pct for e in releases[-4:] if e.surprise_pct is not None]
        if surprises:
            metrics["eps_beats_last_4"] = float(sum(1 for s in surprises if s > 0))
            _put(metrics, "avg_eps_surprise_pct_last_4", sum(surprises) / len(surprises))
    if len(releases) >= 4:
        eps_ttm = _sum_eps(releases[-4:])
        _put(metrics, "eps_ttm", eps_ttm)
        if eps_ttm > 0 and price > 0:
            _put(metrics, "pe_ttm", price / eps_ttm)
        if len(releases) >= 8:
            prior_ttm = _sum_eps(releases[-8:-4])
            if prior_ttm > 0:
                _put(metrics, "eps_ttm_growth_yoy", eps_ttm / prior_ttm - 1)

    if quarters:
        latest = quarters[-1]
        values = latest.values
        evidence["latest_quarter_end"] = latest.period_end.isoformat()
        revenue = values.get("Total Revenue")
        _put(metrics, "revenue_latest_quarter", revenue)
        _put(metrics, "gross_margin", _ratio(values.get("Gross Profit"), revenue))
        _put(metrics, "operating_margin", _ratio(values.get("Operating Income"), revenue))
        _put(metrics, "net_margin", _ratio(values.get("Net Income"), revenue))
        _put(
            metrics,
            "debt_to_equity",
            _ratio(values.get("Total Debt"), values.get("Stockholders Equity")),
        )
        _put(
            metrics,
            "current_ratio",
            _ratio(values.get("Current Assets"), values.get("Current Liabilities")),
        )

        year_ago = next(
            (q for q in quarters[:-1] if abs((latest.period_end - q.period_end).days - 365) <= 20),
            None,
        )
        if year_ago is not None:
            growth = _ratio(revenue, year_ago.values.get("Total Revenue"))
            _put(metrics, "revenue_growth_yoy", growth - 1 if growth is not None else None)

    evidence["metrics"] = metrics
    evidence["coverage"] = {
        "earnings_releases_known": len(releases),
        "quarterly_statements_known": len(quarters),
    }
    if not metrics:
        evidence["note"] = UNAVAILABLE_NOTE
    return evidence


def sentiment_evidence(
    *,
    actions: list[AnalystAction] | None,
    snapshot: MarketSnapshot | None,
    as_of: datetime,
    price: float,
    lookback_days: int = 90,
    now: datetime | None = None,
) -> dict[str, object]:
    cutoff = decision_cutoff(as_of)
    window_start = cutoff - timedelta(days=lookback_days)
    recent = sorted((a for a in actions or [] if window_start <= a.at < cutoff), key=lambda a: a.at)

    metrics: dict[str, float] = {}
    if recent:
        counts = {code: 0 for code in ("up", "down", "init", "main", "reit")}
        for action in recent:
            if action.action in counts:
                counts[action.action] += 1
        metrics["analyst_actions"] = float(len(recent))
        metrics["upgrades"] = float(counts["up"])
        metrics["downgrades"] = float(counts["down"])
        metrics["initiations"] = float(counts["init"])
        metrics["net_upgrades"] = float(counts["up"] - counts["down"])

        targeted = [(a, a.price_target) for a in recent if a.price_target is not None]
        changes = [
            target - a.prior_price_target
            for a, target in targeted
            if a.prior_price_target is not None
        ]
        metrics["price_target_raises"] = float(sum(1 for c in changes if c > 0))
        metrics["price_target_lowers"] = float(sum(1 for c in changes if c < 0))
        # Latest target per firm, so one prolific firm doesn't dominate.
        latest_by_firm = {a.firm: target for a, target in targeted}
        if latest_by_firm:
            mean_target = sum(latest_by_firm.values()) / len(latest_by_firm)
            _put(metrics, "mean_price_target", mean_target)
            if price > 0:
                _put(metrics, "implied_upside_pct", (mean_target / price - 1) * 100)

    live = is_live(as_of, now)
    if live and snapshot is not None:
        _put(metrics, "short_percent_of_float", snapshot.short_percent_of_float)
        _put(metrics, "recommendation_mean", snapshot.recommendation_mean)

    evidence: dict[str, object] = {
        "lookback_days": lookback_days,
        "metrics": metrics,
        "recent_analyst_actions": [
            {
                "date": a.at.date().isoformat(),
                "firm": a.firm,
                "action": a.action,
                "from_grade": a.from_grade,
                "to_grade": a.to_grade,
                "price_target": a.price_target,
            }
            for a in reversed(recent[-10:])
        ],
    }
    if not live:
        evidence["coverage_note"] = (
            "Historical date: short interest and consensus rating are only "
            "available as current snapshots, so they are omitted."
        )
    if not metrics:
        evidence["note"] = UNAVAILABLE_NOTE
    return evidence


def news_evidence(
    *,
    filings: list[Filing] | None,
    headlines: list[Headline] | None,
    as_of: datetime,
    filings_lookback_days: int = 30,
    headlines_lookback_days: int = 7,
    now: datetime | None = None,
) -> dict[str, object]:
    """`filings=None` means no filings source is configured (or it failed),
    as distinct from an empty list (configured, nothing filed)."""
    cutoff = decision_cutoff(as_of)

    filing_window = cutoff - timedelta(days=filings_lookback_days)
    recent_filings = sorted(
        (
            f
            for f in filings or []
            if filing_window <= f.accepted_at < cutoff
            and (f.form.startswith("8-K") or f.form.startswith(_PERIODIC_FORMS))
        ),
        key=lambda f: f.accepted_at,
        reverse=True,
    )
    headline_window = cutoff - timedelta(days=headlines_lookback_days)
    recent_headlines = sorted(
        (h for h in headlines or [] if headline_window <= h.published_at < cutoff),
        key=lambda h: h.published_at,
        reverse=True,
    )

    material = [
        f for f in recent_filings if f.form.startswith("8-K") and set(f.items) - _NON_MATERIAL_ITEMS
    ]
    evidence: dict[str, object] = {
        "metrics": {
            "material_8k_filings": float(len(material)),
            "headline_count": float(len(recent_headlines)),
        },
        "sec_filings": [
            {
                "date": f.accepted_at.date().isoformat(),
                "form": f.form,
                "events": [f"{code} {EIGHT_K_ITEMS.get(code, 'Other item')}" for code in f.items],
            }
            for f in recent_filings[:20]
        ],
        "headlines": [
            {
                "date": h.published_at.date().isoformat(),
                "title": h.title,
                "publisher": h.publisher,
            }
            for h in recent_headlines[:15]
        ],
    }

    notes = []
    if filings is None:
        notes.append("SEC filings were not available for this run.")
    if not recent_headlines and not is_live(as_of, now):
        notes.append(
            "Headlines exist only for recent dates, so none are available for "
            "this historical date — do not read anything into their absence."
        )
    if not recent_filings and not recent_headlines:
        notes.append(UNAVAILABLE_NOTE)
    if notes:
        evidence["note"] = " ".join(notes)
    return evidence
