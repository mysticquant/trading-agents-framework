"""Interactive Brokers adapter over `ib_async`, talking to a running TWS or
IB Gateway process through its socket API. Requires the optional extra:
`pip install -e ".[ib]"`.

Safety:
- Only **paper** accounts (IDs starting with "D") are accepted unless the
  adapter is built with `allow_live=True`. This is checked on `connect()`,
  so a mis-pointed port (live TWS is 7496, paper 7497; Gateway 4001/4002)
  fails before any order can be sent.
- `readonly=True` connects in IB's read-only API mode; orders are refused.

Behavior worth knowing:
- Quantities are floored to whole shares (IB's API rejects fractional
  orders unless the account is enabled for them); an order that floors to
  zero is rejected rather than silently changed.
- `place_order` waits briefly for IB to accept or reject the order. A
  market order placed outside regular hours legitimately stays
  "PreSubmitted" and is returned as `submitted`, not as an error; poll
  `get_order_status` for the fill.
- Only US stocks (`Stock(symbol, "SMART", "USD")` by default) are mapped;
  other position types are left out of `get_positions`.
"""

import asyncio
import math
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from ib_async import IB, LimitOrder, MarketOrder, Stock, Trade
from ib_async import Order as IBOrder
from ib_async import OrderStatus as IBOrderStatus

from trading_agents.brokers.base import BrokerError, OrderRejectedError
from trading_agents.contracts.broker import (
    AccountBalance,
    Order,
    OrderRequest,
    OrderStatus,
    Position,
)
from trading_agents.contracts.market import MarketBar, Timeframe

_BAR_SIZES: dict[str, str] = {"1d": "1 day", "1h": "1 hour", "5m": "5 mins"}
_REJECTED_STATES = {IBOrderStatus.Inactive}
_CANCELLED_STATES = {IBOrderStatus.Cancelled, IBOrderStatus.ApiCancelled}
# Not yet acknowledged by IB. (ib_async's WaitingStates also includes
# PreSubmitted, but that means IB accepted the order and is holding it, e.g.
# until the open — so it counts as submitted here.)
_UNACKNOWLEDGED_STATES = {"", IBOrderStatus.PendingSubmit, IBOrderStatus.ApiPending}


def is_paper_account(account_id: str) -> bool:
    return account_id.upper().startswith("D")


def _finite(value: float | Decimal | None) -> float | None:
    # IB uses float max ("UNSET_DOUBLE") for fields it hasn't filled in.
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) and abs(number) < 1e300 else None


def map_status(trade: Trade) -> OrderStatus:
    status = trade.orderStatus.status
    if status == IBOrderStatus.Filled:
        return "filled"
    if status in _CANCELLED_STATES:
        return "partially_filled" if trade.orderStatus.filled > 0 else "cancelled"
    if status in _REJECTED_STATES:
        return "rejected"
    if status in _UNACKNOWLEDGED_STATES:
        return "pending"
    return "partially_filled" if trade.orderStatus.filled > 0 else "submitted"


def trade_to_order(trade: Trade) -> Order:
    status = map_status(trade)
    filled = trade.orderStatus.filled
    commissions = [
        c
        for fill in trade.fills
        if fill.commissionReport is not None
        and (c := _finite(fill.commissionReport.commission)) is not None
    ]
    fill_times = [fill.time for fill in trade.fills if fill.time is not None]
    order = trade.order
    return Order(
        order_id=str(order.orderId),
        ticker=trade.contract.symbol,
        side="buy" if order.action.upper() == "BUY" else "sell",
        quantity=float(order.totalQuantity),
        order_type="limit" if order.orderType == "LMT" else "market",
        limit_price=_finite(order.lmtPrice) if order.orderType == "LMT" else None,
        status=status,
        submitted_at=trade.log[0].time if trade.log else datetime.now(UTC),
        filled_at=max(fill_times) if status == "filled" and fill_times else None,
        fill_price=_finite(trade.orderStatus.avgFillPrice) if filled > 0 else None,
        commission=sum(commissions),
    )


def duration_for(start: date, end: date) -> str:
    """IB `durationStr` covering `start`..`end` (inclusive); IB takes days
    up to 365, then whole years."""
    days = (end - start).days + 1
    return f"{days} D" if days <= 365 else f"{math.ceil(days / 365)} Y"


class InteractiveBrokersAdapter:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account: str = "",
        readonly: bool = False,
        allow_live: bool = False,
        exchange: str = "SMART",
        currency: str = "USD",
        connect_timeout_seconds: float = 10.0,
        order_ack_timeout_seconds: float = 10.0,
        ib: Any = None,
    ) -> None:
        """`ib` is injectable for tests; by default a fresh `ib_async.IB`."""
        self._ib = ib if ib is not None else IB()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._account = account
        self._readonly = readonly
        self._allow_live = allow_live
        self._exchange = exchange
        self._currency = currency
        self._connect_timeout = connect_timeout_seconds
        self._ack_timeout = order_ack_timeout_seconds
        self._contracts: dict[str, Any] = {}
        self._trades: dict[str, Trade] = {}

    @property
    def account(self) -> str:
        return self._account

    async def connect(self) -> None:
        try:
            await self._ib.connectAsync(
                self._host,
                self._port,
                clientId=self._client_id,
                timeout=self._connect_timeout,
                readonly=self._readonly,
                account=self._account,
            )
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise BrokerError(
                f"Could not connect to TWS/IB Gateway at {self._host}:{self._port} "
                f"({type(exc).__name__}: {exc}). Is it running with the API enabled "
                "(Configure > API > Settings) on this port? Paper TWS uses 7497, "
                "paper Gateway 4002."
            ) from exc

        accounts = list(self._ib.managedAccounts())
        try:
            self._account = self._select_account(accounts)
            if not (self._allow_live or is_paper_account(self._account)):
                raise BrokerError(
                    f"Account {self._account} is not a paper account (paper IDs start "
                    "with 'D'). Refusing to trade a live account; pass allow_live=True "
                    "only if that is really intended."
                )
        except BrokerError:
            self._ib.disconnect()
            raise

    def _select_account(self, accounts: list[str]) -> str:
        if self._account:
            if self._account not in accounts:
                raise BrokerError(
                    f"Account {self._account} is not managed by this login "
                    f"(available: {', '.join(accounts) or 'none'})."
                )
            return self._account
        if len(accounts) != 1:
            raise BrokerError(
                f"This login manages {len(accounts)} accounts ({', '.join(accounts)}); "
                "pass `account=` to choose one."
            )
        return accounts[0]

    async def disconnect(self) -> None:
        self._ib.disconnect()

    def is_connected(self) -> bool:
        return bool(self._ib.isConnected())

    def _require_connection(self) -> None:
        if not self.is_connected():
            raise BrokerError("Not connected to TWS/IB Gateway; call connect() first.")

    async def _contract(self, ticker: str) -> Any:
        if ticker not in self._contracts:
            qualified = await self._ib.qualifyContractsAsync(
                Stock(ticker, self._exchange, self._currency)
            )
            if not qualified or qualified[0] is None:
                raise OrderRejectedError(f"IB does not recognize {ticker} as a stock.")
            self._contracts[ticker] = qualified[0]
        return self._contracts[ticker]

    async def get_account_balance(self) -> AccountBalance:
        self._require_connection()
        values = await self._ib.accountSummaryAsync(self._account)
        by_tag: dict[str, float] = {}
        for value in values:
            if value.currency in (self._currency, "BASE") and value.tag not in by_tag:
                try:
                    by_tag[value.tag] = float(value.value)
                except ValueError:
                    continue
        missing = [
            t for t in ("TotalCashValue", "NetLiquidation", "BuyingPower") if t not in by_tag
        ]
        if missing:
            raise BrokerError(f"IB account summary is missing {', '.join(missing)}.")
        return AccountBalance(
            cash=by_tag["TotalCashValue"],
            equity=by_tag["NetLiquidation"],
            buying_power=by_tag["BuyingPower"],
        )

    async def get_positions(self) -> dict[str, Position]:
        self._require_connection()
        positions: dict[str, Position] = {}
        for item in self._ib.portfolio(self._account):
            if item.contract.secType != "STK" or item.position == 0:
                continue
            positions[item.contract.symbol] = Position(
                ticker=item.contract.symbol,
                quantity=float(item.position),
                avg_cost=float(item.averageCost),
                market_value=float(item.marketValue),
                unrealized_pnl=float(item.unrealizedPNL),
            )
        return positions

    async def get_historical_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]:
        self._require_connection()
        contract = await self._contract(ticker)
        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime=datetime.combine(end + timedelta(days=1), time.min),
            durationStr=duration_for(start, end),
            barSizeSetting=_BAR_SIZES[timeframe],
            whatToShow="TRADES",
            useRTH=True,
        )
        result = []
        for bar in bars:
            timestamp = (
                bar.date if isinstance(bar.date, datetime) else datetime.combine(bar.date, time.min)
            )
            if not start <= timestamp.date() <= end:
                continue
            result.append(
                MarketBar(
                    ticker=ticker,
                    timestamp=timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                    timeframe=timeframe,
                )
            )
        return result

    async def place_order(self, request: OrderRequest) -> Order:
        self._require_connection()
        if self._readonly:
            raise OrderRejectedError("Adapter is connected read-only; orders are disabled.")
        quantity = math.floor(request.quantity)
        if quantity < 1:
            raise OrderRejectedError(
                f"{request.quantity:g} shares of {request.ticker} rounds down to 0 whole shares."
            )
        action = "BUY" if request.side == "buy" else "SELL"
        common = {"tif": "DAY", "account": self._account}
        ib_order: IBOrder
        if request.order_type == "limit":
            if request.limit_price is None:
                raise OrderRejectedError("Limit orders need a limit_price.")
            ib_order = LimitOrder(action, quantity, request.limit_price, **common)
        else:
            ib_order = MarketOrder(action, quantity, **common)

        contract = await self._contract(request.ticker)
        trade = self._ib.placeOrder(contract, ib_order)
        self._trades[str(trade.order.orderId)] = trade

        await self._wait_for_ack(trade)
        order = trade_to_order(trade)
        if order.status in ("rejected", "cancelled"):
            reasons = "; ".join(entry.message for entry in trade.log if entry.message)
            raise OrderRejectedError(
                f"IB {trade.orderStatus.status.lower()} the {action} order for "
                f"{quantity} {request.ticker}: {reasons or 'no reason given'}"
            )
        return order

    async def _wait_for_ack(self, trade: Trade) -> None:
        """Wait until IB has accepted (working/filled) or rejected the order,
        or the ack timeout passes (e.g. a market order queued pre-open)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ack_timeout
        while loop.time() < deadline:
            if trade.orderStatus.status not in _UNACKNOWLEDGED_STATES:
                return
            await asyncio.sleep(0.05)

    def _find_trade(self, order_id: str) -> Trade:
        trade = self._trades.get(order_id)
        if trade is None:
            trade = next((t for t in self._ib.trades() if str(t.order.orderId) == order_id), None)
        if trade is None:
            raise BrokerError(f"Unknown order_id: {order_id}")
        return trade

    async def cancel_order(self, order_id: str) -> None:
        self._require_connection()
        trade = self._find_trade(order_id)
        if trade.isDone():
            return
        self._ib.cancelOrder(trade.order)

    async def get_order_status(self, order_id: str) -> Order:
        return trade_to_order(self._find_trade(order_id))
