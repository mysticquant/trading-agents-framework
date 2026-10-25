from datetime import UTC, datetime

from trading_agents.data.evidence.base import (
    AnalystAction,
    EarningsEvent,
    Filing,
    MarketSnapshot,
)
from trading_agents.data.evidence.provider import DefaultEvidenceProvider


class _FakeMarket:
    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.fail = fail or set()

    def _check(self, name: str) -> None:
        if name in self.fail:
            raise ConnectionError(f"{name} down")

    async def earnings_events(self, ticker):
        self._check("earnings")
        return [
            EarningsEvent(
                reported_at=datetime(2024, 5, 2, 20, tzinfo=UTC),
                eps_estimate=1.0,
                eps_reported=1.1,
                surprise_pct=10.0,
            )
        ]

    async def quarterly_statements(self, ticker):
        self._check("statements")
        return []

    async def analyst_actions(self, ticker):
        self._check("analyst_actions")
        return [
            AnalystAction(
                at=datetime(2024, 5, 20, tzinfo=UTC),
                firm="A",
                action="up",
                to_grade="Buy",
                from_grade="Hold",
                price_target=120.0,
                prior_price_target=100.0,
            )
        ]

    async def headlines(self, ticker):
        self._check("headlines")
        return []

    async def snapshot(self, ticker):
        self._check("snapshot")
        return MarketSnapshot()


class _FakeFilings:
    async def filings(self, ticker):
        return [Filing(accepted_at=datetime(2024, 5, 25, tzinfo=UTC), form="8-K", items=("8.01",))]


def _provider(market, filings=None):
    return DefaultEvidenceProvider(market, filings, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))


async def test_provider_builds_all_three_evidence_types():
    bundle = await _provider(_FakeMarket(), _FakeFilings()).get_evidence(
        "AAPL", datetime(2024, 6, 3), 100.0
    )

    assert bundle.fundamentals["metrics"]["last_eps_surprise_pct"] == 10.0
    assert bundle.sentiment["metrics"]["upgrades"] == 1.0
    assert bundle.news["metrics"]["material_8k_filings"] == 1.0
    for evidence in (bundle.fundamentals, bundle.sentiment, bundle.news):
        assert "source_errors" not in evidence


async def test_a_failing_source_only_degrades_the_evidence_it_feeds():
    bundle = await _provider(_FakeMarket(fail={"analyst_actions"}), _FakeFilings()).get_evidence(
        "AAPL", datetime(2024, 6, 3), 100.0
    )

    assert bundle.sentiment["source_errors"] == {
        "analyst_actions": "ConnectionError: analyst_actions down"
    }
    assert bundle.fundamentals["metrics"]["last_eps_surprise_pct"] == 10.0
    assert "source_errors" not in bundle.news


async def test_without_a_filings_source_news_says_filings_are_unavailable():
    bundle = await _provider(_FakeMarket()).get_evidence("AAPL", datetime(2024, 6, 3), 100.0)

    assert "SEC filings were not available" in bundle.news["coverage_note"]
    assert "source_errors" not in bundle.news
