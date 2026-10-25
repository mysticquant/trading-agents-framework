"""Zerodha Kite Connect adapter and historical-data provider, talking to the
Kite Connect v3 REST API directly (stdlib HTTP — the official `kiteconnect`
package pins a 2019 Twisted/autobahn stack that this code doesn't need).

Kite has **no paper-trading accounts**: every order it accepts is real
money. `KiteBrokerAdapter` therefore refuses to place orders unless built
with `allow_live=True`; reads (balances, positions, order status, history)
always work.

Operational requirements (Zerodha / SEBI retail algo rules):
- An access token from the daily login flow (`login_url` +
  `create_session`, or `trading-agents kite-login`). It expires at 6 AM IST
  the next day; an expired token raises `BrokerAuthExpiredError`.
- Placing orders through the API requires a static IP registered with
  Zerodha. Reads work from any IP.
- Market orders must carry market protection; this adapter sends -1
  (Kite's automatic protection), which converts them to protected limits.

Tickers are NSE trading symbols (e.g. "INFY", "RELIANCE"), not yfinance's
"INFY.NS" form. Quantities are floored to whole shares. The default product
is CNC (delivery).
"""

import asyncio
import csv
import gzip
import hashlib
import io
import json
import math
import re
import time as time_module
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from trading_agents.brokers.base import BrokerAuthExpiredError, BrokerError, OrderRejectedError
from trading_agents.contracts.broker import (
    AccountBalance,
    Order,
    OrderRequest,
    OrderStatus,
    Position,
)
from trading_agents.contracts.market import MarketBar, Timeframe

API_BASE = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login"
IST = ZoneInfo("Asia/Kolkata")

# (status, body) for one HTTP exchange; injectable for tests.
Transport = Callable[[str, str, dict[str, str], bytes | None], Awaitable[tuple[int, bytes]]]

_INTERVALS: dict[str, str] = {"1d": "day", "1h": "60minute", "5m": "5minute"}
# Longest range Kite serves per historical request, by interval.
_MAX_DAYS_PER_REQUEST: dict[str, int] = {"day": 2000, "60minute": 400, "5minute": 100}
# Kite's published rate limits: historical 3/s, everything else 10/s.
_MIN_INTERVAL_SECONDS = {"historical": 1 / 3, "default": 1 / 10}
# Errors that mean the order itself was refused, as opposed to the session
# or the service failing.
_ORDER_REFUSALS = {"InputException", "OrderException", "MarginException", "HoldingException"}
_MENTIONS_IP = re.compile(r"\bip\b", re.IGNORECASE)
_INTERIM_STATUSES = {
    "PUT ORDER REQ RECEIVED",
    "VALIDATION PENDING",
    "OPEN PENDING",
    "MODIFY VALIDATION PENDING",
    "MODIFY PENDING",
    "CANCEL PENDING",
}


def login_url(api_key: str) -> str:
    return f"{LOGIN_URL}?{urllib.parse.urlencode({'v': 3, 'api_key': api_key})}"


def session_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    return hashlib.sha256((api_key + request_token + api_secret).encode()).hexdigest()


class KiteApiError(BrokerError):
    def __init__(self, message: str, *, error_type: str, status: int) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status = status


async def _urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, bytes]:
    def send() -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except urllib.error.URLError as exc:
            raise BrokerError(f"Could not reach Kite at {url}: {exc.reason}") from exc

    return await asyncio.to_thread(send)


class KiteRestClient:
    def __init__(
        self,
        api_key: str,
        access_token: str = "",
        *,
        base_url: str = API_BASE,
        transport: Transport | None = None,
        clock: Callable[[], float] = time_module.monotonic,
        min_interval_seconds: dict[str, float] | None = None,
    ) -> None:
        """`min_interval_seconds` overrides the per-category spacing between
        requests (keys "historical", "default"); defaults to Kite's limits."""
        self.api_key = api_key
        self.access_token = access_token
        self._base_url = base_url.rstrip("/")
        self._transport = transport or _urllib_transport
        self._clock = clock
        self._min_interval = {**_MIN_INTERVAL_SECONDS, **(min_interval_seconds or {})}
        self._throttle_lock = asyncio.Lock()
        self._last_call: dict[str, float] = {}

    async def _throttle(self, category: str) -> None:
        async with self._throttle_lock:
            min_interval = self._min_interval[category]
            wait = self._last_call.get(category, -math.inf) + min_interval - self._clock()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call[category] = self._clock()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        form: dict[str, Any] | None = None,
        authenticated: bool = True,
        raw: bool = False,
    ) -> Any:
        await self._throttle(
            "historical" if path.startswith("/instruments/historical") else "default"
        )
        url = self._base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"X-Kite-Version": "3"}
        if authenticated:
            headers["Authorization"] = f"token {self.api_key}:{self.access_token}"
        body = None
        if form is not None:
            body = urllib.parse.urlencode(form).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        status, payload = await self._transport(method, url, headers, body)
        if raw and status == 200:
            return gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
        try:
            decoded = json.loads(payload or b"{}")
        except json.JSONDecodeError as exc:
            raise BrokerError(f"Kite returned HTTP {status} with a non-JSON body.") from exc
        if status == 200 and decoded.get("status") == "success":
            return decoded.get("data")

        error_type = str(decoded.get("error_type", "GeneralException"))
        message = str(decoded.get("message", f"HTTP {status}"))
        if error_type == "TokenException" or (status == 403 and "token" in message.lower()):
            raise BrokerAuthExpiredError(
                f"Kite session expired or invalid ({message}). Run `trading-agents kite-login` "
                "to get a new access token (tokens expire at 6 AM IST)."
            )
        if status == 429:
            message += " (rate limited)"
        raise KiteApiError(message, error_type=error_type, status=status)


async def create_session(
    api_key: str, api_secret: str, request_token: str, *, transport: Transport | None = None
) -> dict[str, Any]:
    """Exchange a login `request_token` for a session (`access_token`, ...)."""
    client = KiteRestClient(api_key, transport=transport)
    return await client.request(
        "POST",
        "/session/token",
        form={
            "api_key": api_key,
            "request_token": request_token,
            "checksum": session_checksum(api_key, request_token, api_secret),
        },
        authenticated=False,
    )


def _parse_kite_time(value: str | None) -> datetime | None:
    """Order timestamps are "YYYY-MM-DD HH:MM:SS" in IST."""
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST).astimezone(UTC)


def map_status(record: dict[str, Any]) -> OrderStatus:
    status = str(record.get("status", "")).upper()
    filled = float(record.get("filled_quantity") or 0)
    if status == "COMPLETE":
        return "filled"
    if status == "REJECTED":
        return "rejected"
    if status == "CANCELLED":
        return "partially_filled" if filled > 0 else "cancelled"
    if status in _INTERIM_STATUSES or not status:
        return "pending"
    return "partially_filled" if filled > 0 else "submitted"


def record_to_order(record: dict[str, Any]) -> Order:
    status = map_status(record)
    filled = float(record.get("filled_quantity") or 0)
    average_price = float(record.get("average_price") or 0)
    submitted = _parse_kite_time(record.get("order_timestamp")) or datetime.now(UTC)
    return Order(
        order_id=str(record["order_id"]),
        ticker=str(record.get("tradingsymbol", "")),
        side="buy" if str(record.get("transaction_type", "")).upper() == "BUY" else "sell",
        quantity=float(record.get("quantity") or 0),
        order_type="limit" if record.get("order_type") == "LIMIT" else "market",
        limit_price=float(record["price"]) if record.get("order_type") == "LIMIT" else None,
        status=status,
        submitted_at=submitted,
        filled_at=(
            (_parse_kite_time(record.get("exchange_update_timestamp")) or submitted)
            if status == "filled"
            else None
        ),
        fill_price=average_price if filled > 0 and average_price > 0 else None,
        # Kite's order APIs don't report charges (brokerage/STT/fees).
        commission=0.0,
    )


class KiteHistoricalDataProvider:
    """`HistoricalDataProvider` over Kite's candle API (may need the
    historical-data add-on on the Kite Connect app)."""

    def __init__(self, client: KiteRestClient, exchange: str = "NSE") -> None:
        self._client = client
        self._exchange = exchange
        self._tokens: dict[str, int] | None = None
        self._tokens_loaded_on: date | None = None

    async def instrument_token(self, ticker: str) -> int:
        # The instrument dump is regenerated daily; reload once a day.
        if self._tokens is None or self._tokens_loaded_on != date.today():
            dump = await self._client.request("GET", f"/instruments/{self._exchange}", raw=True)
            self._tokens = parse_instruments(dump, self._exchange)
            self._tokens_loaded_on = date.today()
        token = self._tokens.get(ticker.upper())
        if token is None:
            raise BrokerError(f"{ticker} is not a {self._exchange} equity instrument on Kite.")
        return token

    async def get_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]:
        token = await self.instrument_token(ticker)
        interval = _INTERVALS[timeframe]
        span = timedelta(days=_MAX_DAYS_PER_REQUEST[interval] - 1)
        bars: list[MarketBar] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(end, chunk_start + span)
            data = await self._client.request(
                "GET",
                f"/instruments/historical/{token}/{interval}",
                params={
                    "from": f"{chunk_start.isoformat()} 00:00:00",
                    "to": f"{chunk_end.isoformat()} 23:59:59",
                },
            )
            bars.extend(_candles_to_bars(ticker, timeframe, data.get("candles", [])))
            chunk_start = chunk_end + timedelta(days=1)
        return bars


def parse_instruments(dump: bytes, exchange: str) -> dict[str, int]:
    """Map trading symbol -> instrument token for plain equities ("EQ")."""
    rows = csv.DictReader(io.StringIO(dump.decode("utf-8")))
    return {
        row["tradingsymbol"].upper(): int(row["instrument_token"])
        for row in rows
        if row.get("exchange") == exchange and row.get("instrument_type") == "EQ"
    }


def _candles_to_bars(
    ticker: str, timeframe: Timeframe, candles: list[list[Any]]
) -> list[MarketBar]:
    bars = []
    for stamp, open_, high, low, close, volume, *_ in candles:
        moment = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
        # Daily bars are keyed by trading date (naive midnight), matching the
        # rest of the system; intraday bars keep their exchange timestamp.
        timestamp = datetime.combine(moment.date(), time.min) if timeframe == "1d" else moment
        bars.append(
            MarketBar(
                ticker=ticker,
                timestamp=timestamp,
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(volume),
                timeframe=timeframe,
            )
        )
    return bars


class KiteBrokerAdapter:
    def __init__(
        self,
        client: KiteRestClient,
        *,
        exchange: str = "NSE",
        product: str = "CNC",
        allow_live: bool = False,
        order_ack_timeout_seconds: float = 10.0,
        order_poll_seconds: float = 0.5,
        historical: KiteHistoricalDataProvider | None = None,
    ) -> None:
        self._client = client
        self._exchange = exchange
        self._product = product
        self._allow_live = allow_live
        self._ack_timeout = order_ack_timeout_seconds
        self._poll = order_poll_seconds
        self._historical = historical or KiteHistoricalDataProvider(client, exchange)
        self._connected = False
        self.user_id: str | None = None

    async def connect(self) -> None:
        if not self._client.access_token:
            raise BrokerAuthExpiredError(
                "No Kite access token; run `trading-agents kite-login` first."
            )
        profile = await self._request("GET", "/user/profile")
        self.user_id = profile.get("user_id")
        self._connected = True

    async def disconnect(self) -> None:
        # The access token stays valid for the rest of the day; logging out
        # (DELETE /session/token) would also end the user's other sessions.
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            return await self._client.request(method, path, **kwargs)
        except BrokerAuthExpiredError:
            self._connected = False
            raise

    def _require_connection(self) -> None:
        if not self._connected:
            raise BrokerError("Not connected to Kite; call connect() first.")

    async def get_account_balance(self) -> AccountBalance:
        self._require_connection()
        margins = await self._request("GET", "/user/margins/equity")
        cash = float(margins["available"]["live_balance"])
        positions = await self.get_positions()
        return AccountBalance(
            cash=cash,
            equity=cash + sum(p.market_value for p in positions.values()),
            buying_power=float(margins["net"]),
        )

    async def get_positions(self) -> dict[str, Position]:
        """Delivery holdings (settled + T1) plus today's net equity
        positions on the configured exchange; same-day buys appear only in
        positions and same-day sells as negative position quantities."""
        self._require_connection()
        holdings = await self._request("GET", "/portfolio/holdings")
        positions = await self._request("GET", "/portfolio/positions")

        quantity: dict[str, float] = {}
        cost: dict[str, float] = {}
        last_price: dict[str, float] = {}
        for row in holdings or []:
            if row.get("exchange") != self._exchange:
                continue
            symbol = row["tradingsymbol"]
            qty = float(row.get("quantity") or 0) + float(row.get("t1_quantity") or 0)
            quantity[symbol] = quantity.get(symbol, 0.0) + qty
            cost[symbol] = cost.get(symbol, 0.0) + qty * float(row.get("average_price") or 0)
            last_price[symbol] = float(row.get("last_price") or 0)
        for row in (positions or {}).get("net", []):
            if row.get("exchange") != self._exchange or row.get("product") not in ("CNC", "MIS"):
                continue
            symbol = row["tradingsymbol"]
            qty = float(row.get("quantity") or 0)
            quantity[symbol] = quantity.get(symbol, 0.0) + qty
            if qty > 0:
                cost[symbol] = cost.get(symbol, 0.0) + qty * float(row.get("average_price") or 0)
            last_price[symbol] = float(row.get("last_price") or last_price.get(symbol, 0))

        result = {}
        for symbol, qty in quantity.items():
            if qty <= 0:
                continue
            avg_cost = cost.get(symbol, 0.0) / qty
            market_value = qty * last_price[symbol]
            result[symbol] = Position(
                ticker=symbol,
                quantity=qty,
                avg_cost=avg_cost,
                market_value=market_value,
                unrealized_pnl=market_value - qty * avg_cost,
            )
        return result

    async def get_historical_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]:
        return await self._historical.get_bars(ticker, start, end, timeframe)

    async def place_order(self, request: OrderRequest) -> Order:
        self._require_connection()
        if not self._allow_live:
            raise OrderRejectedError(
                "Kite has no paper trading, so every order is real money; this adapter "
                "was built without allow_live=True, so orders are disabled."
            )
        quantity = math.floor(request.quantity)
        if quantity < 1:
            raise OrderRejectedError(
                f"{request.quantity:g} shares of {request.ticker} rounds down to 0 whole shares."
            )
        form: dict[str, Any] = {
            "tradingsymbol": request.ticker.upper(),
            "exchange": self._exchange,
            "transaction_type": "BUY" if request.side == "buy" else "SELL",
            "quantity": quantity,
            "product": self._product,
            "validity": "DAY",
            "tag": "tradingagents",
        }
        if request.order_type == "limit":
            if request.limit_price is None:
                raise OrderRejectedError("Limit orders need a limit_price.")
            form.update(order_type="LIMIT", price=request.limit_price)
        else:
            # Required for API market orders; -1 = Kite's automatic protection.
            form.update(order_type="MARKET", market_protection=-1)

        try:
            placed = await self._request("POST", "/orders/regular", form=form)
        except KiteApiError as exc:
            if exc.error_type in _ORDER_REFUSALS or exc.status in (400, 403):
                hint = (
                    " Kite only accepts API orders from your registered static IP."
                    if _MENTIONS_IP.search(str(exc))
                    else ""
                )
                raise OrderRejectedError(f"Kite refused the order: {exc}.{hint}") from exc
            raise

        record = await self._wait_for_ack(str(placed["order_id"]))
        order = record_to_order(record)
        if order.status in ("rejected", "cancelled"):
            raise OrderRejectedError(
                f"Kite {order.status} the order for {quantity} {request.ticker}: "
                f"{record.get('status_message') or 'no reason given'}"
            )
        return order

    async def _wait_for_ack(self, order_id: str) -> dict[str, Any]:
        """Poll the order until it leaves Kite's interim states or the ack
        timeout passes; returns the latest order record either way."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ack_timeout
        while True:
            record = await self._latest_record(order_id)
            if map_status(record) != "pending" or loop.time() >= deadline:
                return record
            await asyncio.sleep(self._poll)

    async def _latest_record(self, order_id: str) -> dict[str, Any]:
        history = await self._request("GET", f"/orders/{order_id}")
        if not history:
            raise BrokerError(f"Unknown order_id: {order_id}")
        return history[-1]

    async def cancel_order(self, order_id: str) -> None:
        self._require_connection()
        record = await self._latest_record(order_id)
        if map_status(record) in ("filled", "cancelled", "rejected"):
            return
        await self._request("DELETE", f"/orders/regular/{order_id}")

    async def get_order_status(self, order_id: str) -> Order:
        self._require_connection()
        return record_to_order(await self._latest_record(order_id))
