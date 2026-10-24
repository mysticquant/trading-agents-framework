"""`FilingsSource` backed by SEC EDGAR's free submissions API. Gives a
point-in-time record of material events (8-K items) and periodic reports,
timestamped with EDGAR's acceptance time — usable for backtests, unlike
headline feeds that only cover recent days. US-listed companies only;
unknown tickers return no filings.

SEC's fair-access policy requires a User-Agent naming the requester with a
contact email (e.g. "Jane Doe jane@example.com") and at most 10 requests
per second; this source makes one request for the ticker map plus one per
ticker per cache period. Only the "recent" filings page is read (at least
the last year, typically far more), not older paginated history.
"""

import asyncio
import json
import urllib.request
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from trading_agents.data.evidence.base import Filing
from trading_agents.data.evidence.memo import AsyncTTLCache

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_SIX_HOURS = 6 * 60 * 60

JsonFetcher = Callable[[str], Awaitable[Any]]


def parse_ticker_map(payload: dict[str, Any]) -> dict[str, int]:
    return {str(row["ticker"]).upper(): int(row["cik_str"]) for row in payload.values()}


def parse_submissions(payload: dict[str, Any]) -> list[Filing]:
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accepted = recent.get("acceptanceDateTime", [])
    items = recent.get("items", [""] * len(forms))
    filings = []
    for form, accepted_at, item_codes in zip(forms, accepted, items, strict=False):
        if not accepted_at:
            continue
        filings.append(
            Filing(
                accepted_at=datetime.fromisoformat(accepted_at.replace("Z", "+00:00")),
                form=str(form),
                items=tuple(code.strip() for code in str(item_codes).split(",") if code.strip()),
            )
        )
    return filings


class SecEdgarFilingsSource:
    def __init__(
        self,
        user_agent: str,
        *,
        ttl_seconds: float = _SIX_HOURS,
        fetch_json: JsonFetcher | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._fetch_json = fetch_json or self._http_get_json
        self._ticker_map: AsyncTTLCache[dict[str, int]] = AsyncTTLCache(ttl_seconds * 4)
        self._filings: AsyncTTLCache[list[Filing]] = AsyncTTLCache(ttl_seconds)

    async def _http_get_json(self, url: str) -> Any:
        def load() -> Any:
            request = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)

        return await asyncio.to_thread(load)

    async def filings(self, ticker: str) -> list[Filing]:
        async def fetch_map() -> dict[str, int]:
            return parse_ticker_map(await self._fetch_json(_TICKER_MAP_URL))

        cik = (await self._ticker_map.get("all", fetch_map)).get(ticker.upper())
        if cik is None:
            return []

        async def fetch_filings() -> list[Filing]:
            return parse_submissions(await self._fetch_json(_SUBMISSIONS_URL.format(cik=cik)))

        return await self._filings.get(ticker.upper(), fetch_filings)
