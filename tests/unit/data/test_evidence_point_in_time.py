from datetime import UTC, date, datetime, timedelta

import pytest

from trading_agents.data.evidence.base import (
    AnalystAction,
    EarningsEvent,
    Filing,
    Headline,
    MarketSnapshot,
    QuarterlyStatement,
)
from trading_agents.data.evidence.point_in_time import (
    UNAVAILABLE_NOTE,
    decision_cutoff,
    fundamentals_evidence,
    news_evidence,
    sentiment_evidence,
    statement_public_at,
)


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _release(day: date, eps: float, surprise: float | None = 5.0) -> EarningsEvent:
    # 16:00 US/Eastern after-close release == 20:00 or 21:00 UTC.
    return EarningsEvent(
        reported_at=_utc(day.year, day.month, day.day, 20, 30),
        eps_estimate=eps * 0.95,
        eps_reported=eps,
        surprise_pct=surprise,
    )


# Quarterly releases about a month after each quarter end.
_RELEASES = [
    _release(date(2023, 2, 2), 1.0),
    _release(date(2023, 5, 4), 1.0),
    _release(date(2023, 8, 3), 1.0),
    _release(date(2023, 11, 2), 1.0),
    _release(date(2024, 2, 1), 1.5),
    _release(date(2024, 5, 2), 1.5),
    _release(date(2024, 8, 1), 1.5),
    _release(date(2024, 10, 31), 1.5, surprise=-2.0),
]


def _quarter(period_end: date, revenue: float) -> QuarterlyStatement:
    return QuarterlyStatement(
        period_end=period_end,
        values={
            "Total Revenue": revenue,
            "Gross Profit": revenue * 0.4,
            "Operating Income": revenue * 0.3,
            "Net Income": revenue * 0.25,
            "Total Debt": 50.0,
            "Stockholders Equity": 100.0,
            "Current Assets": 150.0,
            "Current Liabilities": 100.0,
        },
    )


_QUARTERS = [
    _quarter(date(2023, 6, 30), 100.0),
    _quarter(date(2023, 9, 30), 100.0),
    _quarter(date(2023, 12, 31), 100.0),
    _quarter(date(2024, 3, 31), 100.0),
    _quarter(date(2024, 6, 30), 120.0),
    _quarter(date(2024, 9, 30), 130.0),
]


def test_naive_as_of_is_read_as_exchange_local_time():
    # Midnight in New York (EST) is 05:00 UTC.
    assert decision_cutoff(datetime(2024, 1, 10)) == _utc(2024, 1, 10, 5)


def test_statements_become_public_at_their_earnings_release():
    q = _quarter(date(2024, 6, 30), 120.0)
    assert statement_public_at(q, _RELEASES) == _RELEASES[6].reported_at


def test_statements_without_a_release_use_the_filing_lag():
    q = _quarter(date(2024, 6, 30), 120.0)
    public_at = statement_public_at(q, [])
    # 90 days after the end of the quarter's last day (US/Eastern): late
    # enough even for fiscal-year results, which come with the 10-K.
    assert public_at.date() == date(2024, 7, 1) + timedelta(days=90)


def test_fundamentals_exclude_an_earnings_release_on_the_decision_day():
    # The 2024-08-01 after-close release is not usable for a decision made on
    # 2024-08-01 (as_of = midnight that day), only from the next day.
    on_release_day = fundamentals_evidence(
        earnings=_RELEASES, statements=_QUARTERS, as_of=datetime(2024, 8, 1), price=60.0
    )
    next_day = fundamentals_evidence(
        earnings=_RELEASES, statements=_QUARTERS, as_of=datetime(2024, 8, 2), price=60.0
    )

    assert on_release_day["last_earnings_release"] == "2024-05-02"
    assert on_release_day["latest_quarter_end"] == "2024-03-31"
    assert next_day["last_earnings_release"] == "2024-08-01"
    assert next_day["latest_quarter_end"] == "2024-06-30"


def test_fundamentals_compute_ttm_valuation_margins_and_growth():
    evidence = fundamentals_evidence(
        earnings=_RELEASES, statements=_QUARTERS, as_of=datetime(2024, 8, 2), price=60.0
    )
    metrics = evidence["metrics"]

    # Last four known releases: 1.0 (2023-11) + 1.5 * 3.
    assert metrics["eps_ttm"] == pytest.approx(5.5)
    assert metrics["pe_ttm"] == pytest.approx(60.0 / 5.5, rel=1e-3)
    # Only 7 releases are known, too few for a year-over-year TTM comparison.
    assert "eps_ttm_growth_yoy" not in metrics
    assert metrics["gross_margin"] == pytest.approx(0.4)
    assert metrics["debt_to_equity"] == pytest.approx(0.5)
    assert metrics["current_ratio"] == pytest.approx(1.5)
    # 2024-06-30 revenue 120 vs 2023-06-30 revenue 100.
    assert metrics["revenue_growth_yoy"] == pytest.approx(0.2)
    assert evidence["coverage"] == {"earnings_releases_known": 7, "quarterly_statements_known": 5}


def test_fundamentals_compute_ttm_eps_growth_once_eight_releases_are_known():
    evidence = fundamentals_evidence(
        earnings=_RELEASES, statements=_QUARTERS, as_of=datetime(2024, 11, 1), price=60.0
    )

    # 1.5 * 4 vs 1.0 * 4 a year earlier.
    assert evidence["metrics"]["eps_ttm_growth_yoy"] == pytest.approx(0.5)


def test_fundamentals_never_include_future_releases():
    evidence = fundamentals_evidence(
        earnings=_RELEASES, statements=_QUARTERS, as_of=datetime(2024, 9, 1), price=60.0
    )

    # The 2024-10-31 miss is in the future relative to 2024-09-01.
    assert evidence["metrics"]["last_eps_surprise_pct"] == 5.0
    assert evidence["metrics"]["eps_beats_last_4"] == 4.0


def test_fundamentals_without_data_carry_the_unavailable_note():
    evidence = fundamentals_evidence(
        earnings=None, statements=None, as_of=datetime(2024, 8, 2), price=60.0
    )

    assert evidence["metrics"] == {}
    assert evidence["note"] == UNAVAILABLE_NOTE


def _action(at: datetime, action: str, target: float | None, prior: float | None, firm: str):
    return AnalystAction(
        at=at,
        firm=firm,
        action=action,
        to_grade="Buy",
        from_grade="Hold",
        price_target=target,
        prior_price_target=prior,
    )


_ACTIONS = [
    _action(_utc(2024, 1, 2, 14), "up", 110.0, 100.0, "A"),  # outside 90-day window
    _action(_utc(2024, 4, 15, 14), "up", 120.0, 100.0, "A"),
    _action(_utc(2024, 5, 1, 14), "down", 90.0, 100.0, "B"),
    _action(_utc(2024, 5, 20, 14), "main", 130.0, 120.0, "A"),
    _action(_utc(2024, 6, 1, 14), "init", 100.0, None, "C"),
    _action(_utc(2024, 6, 3, 14), "up", 200.0, 100.0, "D"),  # on the decision day
]


def test_sentiment_counts_only_actions_inside_the_window_and_before_the_cutoff():
    evidence = sentiment_evidence(
        actions=_ACTIONS,
        snapshot=None,
        as_of=datetime(2024, 6, 3),
        price=100.0,
        now=_utc(2026, 1, 1),
    )
    metrics = evidence["metrics"]

    assert metrics["analyst_actions"] == 4.0
    assert metrics["upgrades"] == 1.0
    assert metrics["downgrades"] == 1.0
    assert metrics["net_upgrades"] == 0.0
    assert metrics["price_target_raises"] == 2.0
    assert metrics["price_target_lowers"] == 1.0
    # Latest target per firm: A=130, B=90, C=100 -> mean ~106.67.
    assert metrics["mean_price_target"] == pytest.approx(106.6667, rel=1e-4)
    assert metrics["implied_upside_pct"] == pytest.approx(6.6667, rel=1e-3)
    assert evidence["recent_analyst_actions"][0]["firm"] == "C"  # newest first


def test_snapshot_values_are_used_only_for_live_decisions():
    snapshot = MarketSnapshot(short_percent_of_float=0.02, recommendation_mean=1.8)
    now = _utc(2024, 6, 3, 18)

    historical = sentiment_evidence(
        actions=[], snapshot=snapshot, as_of=datetime(2024, 1, 3), price=100.0, now=now
    )
    live = sentiment_evidence(  # 13:30 New York = 17:30 UTC, 30 min before now
        actions=[], snapshot=snapshot, as_of=datetime(2024, 6, 3, 13, 30), price=100.0, now=now
    )
    two_days_back = sentiment_evidence(  # e.g. a recent backtest day
        actions=[], snapshot=snapshot, as_of=datetime(2024, 6, 1), price=100.0, now=now
    )

    assert "short_percent_of_float" not in historical["metrics"]
    assert historical["note"] == UNAVAILABLE_NOTE
    assert live["metrics"]["short_percent_of_float"] == 0.02
    assert live["metrics"]["recommendation_mean"] == 1.8
    # Today's snapshot must not leak into a recent backtest day.
    assert "short_percent_of_float" not in two_days_back["metrics"]


_FILINGS = [
    Filing(accepted_at=_utc(2024, 4, 1, 21), form="8-K", items=("5.02",)),  # outside window
    Filing(accepted_at=_utc(2024, 5, 2, 21), form="8-K", items=("2.02", "9.01")),
    Filing(accepted_at=_utc(2024, 5, 3, 21), form="10-Q"),
    Filing(accepted_at=_utc(2024, 5, 10, 21), form="8-K", items=("9.01",)),
    Filing(accepted_at=_utc(2024, 5, 15, 21), form="4"),  # insider form, ignored
    Filing(accepted_at=_utc(2024, 5, 31, 12), form="8-K", items=("1.01",)),  # decision day
]


def test_news_lists_recent_filings_before_the_cutoff():
    evidence = news_evidence(
        filings=_FILINGS, headlines=[], as_of=datetime(2024, 5, 31), now=_utc(2026, 1, 1)
    )

    filings = evidence["sec_filings"]
    assert [f["form"] for f in filings] == ["8-K", "10-Q", "8-K"]
    assert filings[2]["events"] == [
        "2.02 Results of operations and financial condition",
        "9.01 Financial statements and exhibits",
    ]
    # The exhibit-only 8-K isn't a material event.
    assert evidence["metrics"]["material_8k_filings"] == 1.0
    assert "Headlines exist only for recent dates" in evidence["coverage_note"]
    # Filings are present, so nothing says the evidence is unavailable
    # (the prompt turns that into a forced low-confidence neutral).
    assert "note" not in evidence


def test_news_marks_a_missing_filings_source_as_unavailable():
    evidence = news_evidence(
        filings=None, headlines=None, as_of=datetime(2024, 6, 3), now=_utc(2026, 1, 1)
    )

    assert "SEC filings were not available" in evidence["coverage_note"]
    assert evidence["note"] == UNAVAILABLE_NOTE


def test_news_includes_recent_headlines_for_live_decisions():
    now = _utc(2024, 6, 3, 18)
    headlines = [
        Headline(published_at=_utc(2024, 6, 3, 13), title="Fresh", publisher="X"),
        Headline(published_at=_utc(2024, 5, 1, 13), title="Stale", publisher="X"),
        Headline(published_at=_utc(2024, 6, 3, 19), title="Future", publisher="X"),
    ]

    evidence = news_evidence(
        filings=[], headlines=headlines, as_of=datetime(2024, 6, 3, 14), now=now
    )

    assert [h["title"] for h in evidence["headlines"]] == ["Fresh"]
    assert "note" not in evidence
