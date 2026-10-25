import gzip
import hashlib
import json
import time
import urllib.parse
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from trading_agents.brokers import kite as kite_module
from trading_agents.brokers.base import BrokerAuthExpiredError, BrokerError, OrderRejectedError
from trading_agents.brokers.kite import (
    KiteApiError,
    KiteBrokerAdapter,
    KiteHistoricalDataProvider,
    KiteRestClient,
    create_session,
    login_url,
    parse_instruments,
    session_checksum,
)
from trading_agents.contracts.broker import OrderRequest


def ok(data) -> tuple[int, bytes]:
    return 200, json.dumps({"status": "success", "data": data}).encode()


def error(status: int, error_type: str, message: str) -> tuple[int, bytes]:
    return status, json.dumps(
        {"status": "error", "error_type": error_type, "message": message}
    ).encode()


class FakeKite:
    """Routes (method, path) to scripted responses and records requests.
    A route's value may be a response, or a list consumed in order."""

    def __init__(self, routes: dict[tuple[str, str], object] | None = None) -> None:
        self.routes = routes or {}
        self.requests: list[dict] = []

    async def __call__(self, method, url, headers, body):
        parsed = urllib.parse.urlsplit(url)
        self.requests.append(
            {
                "method": method,
                "path": parsed.path,
                "query": dict(urllib.parse.parse_qsl(parsed.query)),
                "headers": headers,
                "form": dict(urllib.parse.parse_qsl(body.decode())) if body else None,
            }
        )
        response = self.routes[(method, parsed.path)]
        if isinstance(response, list):
            return response.pop(0) if len(response) > 1 else response[0]
        return response

    def sent(self, method: str, path: str) -> list[dict]:
        return [r for r in self.requests if (r["method"], r["path"]) == (method, path)]


_PROFILE = ok({"user_id": "AB1234", "user_name": "Test"})


_NO_THROTTLE = {"historical": 0.0, "default": 0.0}


def _client(fake: FakeKite, token: str = "tok", **kwargs) -> KiteRestClient:
    kwargs.setdefault("min_interval_seconds", _NO_THROTTLE)
    return KiteRestClient("key", token, transport=fake, **kwargs)


async def _adapter(fake: FakeKite, **kwargs) -> KiteBrokerAdapter:
    fake.routes.setdefault(("GET", "/user/profile"), _PROFILE)
    adapter = KiteBrokerAdapter(
        _client(fake), order_ack_timeout_seconds=0.2, order_poll_seconds=0.01, **kwargs
    )
    await adapter.connect()
    return adapter


def _order(status: str, **fields) -> dict:
    return {
        "order_id": "250101000000001",
        "status": status,
        "tradingsymbol": "INFY",
        "transaction_type": "BUY",
        "order_type": "MARKET",
        "quantity": 9,
        "filled_quantity": 0,
        "average_price": 0,
        "price": 0,
        "order_timestamp": "2026-10-26 10:15:00",
        "status_message": None,
        **fields,
    }


def _buy(quantity: float = 9.7, **kwargs) -> OrderRequest:
    fields = {"ticker": "INFY", "side": "buy", "quantity": quantity, "order_type": "market"}
    return OrderRequest(**{**fields, **kwargs})


# --- login and transport ------------------------------------------------------


def test_login_url_and_checksum():
    assert login_url("key") == "https://kite.zerodha.com/connect/login?v=3&api_key=key"
    expected = hashlib.sha256(b"key" + b"reqtok" + b"secret").hexdigest()
    assert session_checksum("key", "reqtok", "secret") == expected


async def test_create_session_posts_the_checksum_without_auth():
    fake = FakeKite({("POST", "/session/token"): ok({"access_token": "new", "user_id": "AB"})})

    session = await create_session("key", "secret", "reqtok", transport=fake)

    assert session["access_token"] == "new"
    (sent,) = fake.requests
    assert sent["form"] == {
        "api_key": "key",
        "request_token": "reqtok",
        "checksum": session_checksum("key", "reqtok", "secret"),
    }
    assert "Authorization" not in sent["headers"]


async def test_requests_carry_kite_auth_headers():
    fake = FakeKite({("GET", "/user/profile"): _PROFILE})

    await _client(fake).request("GET", "/user/profile")

    headers = fake.requests[0]["headers"]
    assert headers["Authorization"] == "token key:tok"
    assert headers["X-Kite-Version"] == "3"


async def test_expired_token_raises_auth_expired_and_disconnects():
    fake = FakeKite({("GET", "/user/margins/equity"): error(403, "TokenException", "Expired")})
    adapter = await _adapter(fake)

    with pytest.raises(BrokerAuthExpiredError, match="kite-login"):
        await adapter.get_account_balance()
    assert not adapter.is_connected()


async def test_connect_needs_a_token():
    adapter = KiteBrokerAdapter(_client(FakeKite(), token=""))

    with pytest.raises(BrokerAuthExpiredError, match="kite-login"):
        await adapter.connect()


async def test_rate_limit_errors_are_labelled():
    fake = FakeKite({("GET", "/orders"): error(429, "NetworkException", "Too many requests")})

    with pytest.raises(KiteApiError, match="rate limited"):
        await _client(fake).request("GET", "/orders")


async def test_historical_requests_are_throttled_to_three_per_second():
    fake = FakeKite({("GET", "/instruments/historical/1/day"): ok({"candles": []})})
    client = _client(fake, min_interval_seconds=None)  # Kite's real limits

    started = time.monotonic()
    for _ in range(3):
        await client.request("GET", "/instruments/historical/1/day")

    assert time.monotonic() - started >= 2 / 3 - 0.05


# --- account data ---------------------------------------------------------------


_HOLDINGS = [
    {
        "tradingsymbol": "INFY",
        "exchange": "NSE",
        "quantity": 10,
        "t1_quantity": 2,
        "average_price": 1500.0,
        "last_price": 1600.0,
    },
    {
        "tradingsymbol": "TCS",
        "exchange": "NSE",
        "quantity": 5,
        "t1_quantity": 0,
        "average_price": 3000.0,
        "last_price": 3100.0,
    },
    {
        "tradingsymbol": "SBIN",
        "exchange": "BSE",
        "quantity": 7,
        "t1_quantity": 0,
        "average_price": 500.0,
        "last_price": 550.0,
    },
]
_POSITIONS = {
    "net": [
        # Bought 3 more INFY today (delivery), sold all TCS today.
        {
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "product": "CNC",
            "quantity": 3,
            "average_price": 1580.0,
            "last_price": 1600.0,
        },
        {
            "tradingsymbol": "TCS",
            "exchange": "NSE",
            "product": "CNC",
            "quantity": -5,
            "average_price": 0,
            "last_price": 3100.0,
        },
    ],
    "day": [],
}


def _portfolio_routes() -> dict:
    return {
        ("GET", "/portfolio/holdings"): ok(_HOLDINGS),
        ("GET", "/portfolio/positions"): ok(_POSITIONS),
    }


async def test_positions_combine_holdings_and_todays_positions():
    adapter = await _adapter(FakeKite(_portfolio_routes()))

    positions = await adapter.get_positions()

    assert list(positions) == ["INFY"]  # TCS sold out today; SBIN is on BSE
    infy = positions["INFY"]
    assert infy.quantity == 15.0  # 10 settled + 2 T1 + 3 bought today
    assert infy.avg_cost == pytest.approx((12 * 1500 + 3 * 1580) / 15)
    assert infy.market_value == pytest.approx(15 * 1600.0)


async def test_balance_uses_live_cash_and_net_buying_power():
    routes = _portfolio_routes()
    routes[("GET", "/user/margins/equity")] = ok(
        {"net": 50000.0, "available": {"cash": 60000.0, "live_balance": 45000.0}}
    )
    adapter = await _adapter(FakeKite(routes))

    balance = await adapter.get_account_balance()

    assert balance.cash == 45000.0
    assert balance.buying_power == 50000.0
    assert balance.equity == pytest.approx(45000.0 + 15 * 1600.0)


# --- orders ---------------------------------------------------------------------


async def test_orders_are_disabled_unless_live_trading_is_allowed():
    fake = FakeKite()
    adapter = await _adapter(fake)

    with pytest.raises(OrderRejectedError, match="no paper trading"):
        await adapter.place_order(_buy())
    assert fake.sent("POST", "/orders/regular") == []


async def test_market_order_sends_market_protection_and_reports_the_fill():
    fake = FakeKite(
        {
            ("POST", "/orders/regular"): ok({"order_id": "250101000000001"}),
            ("GET", "/orders/250101000000001"): ok(
                [
                    _order("PUT ORDER REQ RECEIVED"),
                    _order(
                        "COMPLETE",
                        filled_quantity=9,
                        average_price=1601.5,
                        exchange_update_timestamp="2026-10-26 10:15:02",
                    ),
                ]
            ),
        }
    )
    adapter = await _adapter(fake, allow_live=True)

    order = await adapter.place_order(_buy(9.7))

    (sent,) = fake.sent("POST", "/orders/regular")
    assert sent["form"] == {
        "tradingsymbol": "INFY",
        "exchange": "NSE",
        "transaction_type": "BUY",
        "quantity": "9",
        "product": "CNC",
        "validity": "DAY",
        "tag": "tradingagents",
        "order_type": "MARKET",
        "market_protection": "-1",
    }
    assert order.status == "filled"
    assert order.fill_price == 1601.5
    assert order.filled_at == datetime(2026, 10, 26, 4, 45, 2, tzinfo=UTC)  # 10:15:02 IST


async def test_limit_order_sends_the_price_and_no_market_protection():
    fake = FakeKite(
        {
            ("POST", "/orders/regular"): ok({"order_id": "250101000000001"}),
            ("GET", "/orders/250101000000001"): ok(
                [_order("OPEN", order_type="LIMIT", price=1550.0)]
            ),
        }
    )
    adapter = await _adapter(fake, allow_live=True)

    order = await adapter.place_order(_buy(5, order_type="limit", limit_price=1550.0))

    form = fake.sent("POST", "/orders/regular")[0]["form"]
    assert (form["order_type"], form["price"]) == ("LIMIT", "1550.0")
    assert "market_protection" not in form
    assert order.status == "submitted" and order.limit_price == 1550.0


async def test_order_still_interim_after_the_timeout_is_returned_as_pending():
    fake = FakeKite(
        {
            ("POST", "/orders/regular"): ok({"order_id": "250101000000001"}),
            ("GET", "/orders/250101000000001"): ok([_order("VALIDATION PENDING")]),
        }
    )
    adapter = await _adapter(fake, allow_live=True)

    assert (await adapter.place_order(_buy())).status == "pending"
    assert len(fake.sent("GET", "/orders/250101000000001")) > 1  # it polled


async def test_rejected_order_raises_with_kites_reason():
    fake = FakeKite(
        {
            ("POST", "/orders/regular"): ok({"order_id": "250101000000001"}),
            ("GET", "/orders/250101000000001"): ok(
                [_order("REJECTED", status_message="Insufficient funds. Required margin is 95000")]
            ),
        }
    )
    adapter = await _adapter(fake, allow_live=True)

    with pytest.raises(OrderRejectedError, match="Insufficient funds"):
        await adapter.place_order(_buy())


async def test_order_from_an_unregistered_ip_explains_the_static_ip_rule():
    fake = FakeKite(
        {("POST", "/orders/regular"): error(403, "PermissionException", "IP not allowed")}
    )
    adapter = await _adapter(fake, allow_live=True)

    with pytest.raises(OrderRejectedError, match="registered static IP"):
        await adapter.place_order(_buy())


async def test_margin_refusal_is_an_order_rejection():
    fake = FakeKite(
        {("POST", "/orders/regular"): error(400, "MarginException", "Insufficient margin")}
    )
    adapter = await _adapter(fake, allow_live=True)

    with pytest.raises(OrderRejectedError, match="Insufficient margin") as exc_info:
        await adapter.place_order(_buy())
    assert "static IP" not in str(exc_info.value)


async def test_fractional_quantity_below_one_share_is_never_sent():
    fake = FakeKite()
    adapter = await _adapter(fake, allow_live=True)

    with pytest.raises(OrderRejectedError, match="rounds down to 0"):
        await adapter.place_order(_buy(0.9))
    assert fake.sent("POST", "/orders/regular") == []


async def test_cancel_only_touches_open_orders():
    fake = FakeKite(
        {
            ("GET", "/orders/1"): ok([_order("OPEN", order_id="1")]),
            ("GET", "/orders/2"): ok([_order("COMPLETE", order_id="2", filled_quantity=9)]),
            ("DELETE", "/orders/regular/1"): ok({"order_id": "1"}),
        }
    )
    adapter = await _adapter(fake)

    await adapter.cancel_order("1")
    await adapter.cancel_order("2")

    assert [r["path"] for r in fake.requests if r["method"] == "DELETE"] == ["/orders/regular/1"]


async def test_unknown_order_id_raises():
    adapter = await _adapter(FakeKite({("GET", "/orders/9"): ok([])}))

    with pytest.raises(BrokerError, match="Unknown order_id"):
        await adapter.get_order_status("9")


# --- historical data -----------------------------------------------------------


_INSTRUMENTS_CSV = (
    "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,"
    "tick_size,lot_size,instrument_type,segment,exchange\n"
    "408065,1594,INFY,INFOSYS,0,,0,0.05,1,EQ,NSE,NSE\n"
    "12345,48,INFY25NOVFUT,INFOSYS,0,2025-11-27,0,0.05,400,FUT,NFO-FUT,NFO\n"
    "2953217,11536,TCS,TATA CONSULTANCY,0,,0,0.05,1,EQ,NSE,NSE\n"
)


def test_parse_instruments_keeps_exchange_equities_only():
    assert parse_instruments(_INSTRUMENTS_CSV.encode(), "NSE") == {"INFY": 408065, "TCS": 2953217}


async def test_daily_bars_are_chunked_and_keyed_by_trading_date():
    ist = timezone(timedelta(hours=5, minutes=30))
    candle = [datetime(2020, 1, 2, tzinfo=ist).strftime("%Y-%m-%dT%H:%M:%S%z"), 1, 2, 0.5, 1.5, 100]
    fake = FakeKite(
        {
            ("GET", "/instruments/NSE"): (200, gzip.compress(_INSTRUMENTS_CSV.encode())),
            ("GET", "/instruments/historical/408065/day"): ok({"candles": [candle]}),
        }
    )
    provider = KiteHistoricalDataProvider(_client(fake))

    # 2500 days > Kite's 2000-day limit for daily candles -> two requests.
    bars = await provider.get_bars("INFY", date(2018, 1, 1), date(2024, 11, 3), "1d")
    await provider.get_bars("INFY", date(2020, 1, 1), date(2020, 1, 5), "1d")

    history = fake.sent("GET", "/instruments/historical/408065/day")
    assert [r["query"] for r in history[:2]] == [
        {"from": "2018-01-01 00:00:00", "to": "2023-06-23 23:59:59"},
        {"from": "2023-06-24 00:00:00", "to": "2024-11-03 23:59:59"},
    ]
    assert len(fake.sent("GET", "/instruments/NSE")) == 1  # dump cached
    assert bars[0].timestamp == datetime(2020, 1, 2)
    assert bars[0].close == 1.5


async def test_intraday_bars_keep_their_exchange_timestamp():
    fake = FakeKite(
        {
            ("GET", "/instruments/NSE"): (200, _INSTRUMENTS_CSV.encode()),
            ("GET", "/instruments/historical/408065/5minute"): ok(
                {"candles": [["2026-10-26T09:15:00+0530", 1, 2, 0.5, 1.5, 100]]}
            ),
        }
    )

    (bar,) = await KiteHistoricalDataProvider(_client(fake)).get_bars(
        "INFY", date(2026, 10, 26), date(2026, 10, 26), "5m"
    )

    assert bar.timestamp == datetime(2026, 10, 26, 3, 45, tzinfo=UTC)


async def test_unknown_symbol_raises():
    fake = FakeKite({("GET", "/instruments/NSE"): (200, _INSTRUMENTS_CSV.encode())})

    with pytest.raises(BrokerError, match="not a NSE equity"):
        await KiteHistoricalDataProvider(_client(fake)).instrument_token("NOPE")


# --- CLI ------------------------------------------------------------------------


async def test_kite_login_without_a_token_prints_the_login_url():
    from trading_agents.cli.main import kite_login
    from trading_agents.config.settings import Settings

    lines = await kite_login(Settings(kite_api_key="key", _env_file=None), None)

    assert "https://kite.zerodha.com/connect/login?v=3&api_key=key" in lines[1]


async def test_kite_login_exchanges_the_request_token(monkeypatch):
    from trading_agents.cli.main import kite_login
    from trading_agents.config.settings import Settings

    fake = FakeKite(
        {("POST", "/session/token"): ok({"access_token": "day-token", "user_id": "AB"})}
    )
    monkeypatch.setattr(kite_module, "_urllib_transport", fake)

    lines = await kite_login(
        Settings(kite_api_key="key", kite_api_secret="secret", _env_file=None), "reqtok"
    )

    assert lines[-1] == "KITE_ACCESS_TOKEN=day-token"


async def test_kite_check_needs_an_access_token():
    from trading_agents.cli.main import kite_check
    from trading_agents.config.settings import ConfigurationError, Settings

    with pytest.raises(ConfigurationError, match="kite-login"):
        await kite_check(Settings(kite_api_key="key", _env_file=None))
