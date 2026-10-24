"""Default `EvidenceProvider`: fetches raw records from a `MarketDataSource`
(and optionally a `FilingsSource`) and builds point-in-time evidence for
the fundamentals, sentiment, and news analysts.

A failing source degrades only the evidence it feeds — the failure is
logged and recorded under `source_errors` — rather than costing the ticker
its whole decision cycle.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TypeVar

from trading_agents.data.evidence.base import (
    EvidenceBundle,
    Filing,
    FilingsSource,
    MarketDataSource,
)
from trading_agents.data.evidence.point_in_time import (
    fundamentals_evidence,
    news_evidence,
    sentiment_evidence,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DefaultEvidenceProvider:
    def __init__(
        self,
        market: MarketDataSource,
        filings: FilingsSource | None = None,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._market = market
        self._filings = filings
        self._now = now

    async def get_evidence(self, ticker: str, as_of: datetime, price: float) -> EvidenceBundle:
        errors: dict[str, str] = {}

        async def attempt(name: str, call: Callable[[], Awaitable[T]]) -> T | None:
            try:
                return await call()
            except Exception as exc:
                logger.warning("Evidence source %s failed for %s: %s", name, ticker, exc)
                errors[name] = f"{type(exc).__name__}: {exc}"
                return None

        async def fetch_filings() -> list[Filing] | None:
            # None (not []) tells news_evidence there's no filings source.
            if self._filings is None:
                return None
            return await self._filings.filings(ticker)

        earnings, statements, actions, headlines, snapshot, filings = await asyncio.gather(
            attempt("earnings", lambda: self._market.earnings_events(ticker)),
            attempt("statements", lambda: self._market.quarterly_statements(ticker)),
            attempt("analyst_actions", lambda: self._market.analyst_actions(ticker)),
            attempt("headlines", lambda: self._market.headlines(ticker)),
            attempt("snapshot", lambda: self._market.snapshot(ticker)),
            attempt("sec_filings", fetch_filings),
        )
        now = self._now()

        bundle = EvidenceBundle(
            fundamentals=fundamentals_evidence(
                earnings=earnings, statements=statements, as_of=as_of, price=price
            ),
            sentiment=sentiment_evidence(
                actions=actions, snapshot=snapshot, as_of=as_of, price=price, now=now
            ),
            news=news_evidence(filings=filings, headlines=headlines, as_of=as_of, now=now),
        )
        for evidence, names in (
            (bundle.fundamentals, ("earnings", "statements")),
            (bundle.sentiment, ("analyst_actions", "snapshot")),
            (bundle.news, ("sec_filings", "headlines")),
        ):
            failed = {name: errors[name] for name in names if name in errors}
            if failed:
                evidence["source_errors"] = failed
        return bundle
