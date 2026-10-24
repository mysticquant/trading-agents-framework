"""Simulated broker: an in-memory cash/positions ledger with immediate fills
at the last marked price plus fixed commission/slippage. Deliberately simple
— no tick-level order-book matching — since this phase is backtesting and
paper trading, not live execution.
"""

import itertools
from datetime import UTC, datetime

from trading_agents.brokers.base import BrokerAuthExpiredError, BrokerError, OrderRejectedError  # noqa: F401
from trading_agents.contracts.broker import AccountBalance, Order, OrderRequest, Position
from trading_agents.contracts.market import MarketBar, Timeframe
from trading_agents.data.providers.base import HistoricalDataProvider


class PaperBrokerAdapter:
    """Implements the `BrokerAdapter` Protocol with a simulated ledger.

    Callers must call `mark_price(ticker, price)` for a ticker before
    placing a market/limit order against it — this stands in for the "last
    trade price" a real broker would already know. In the backtest runner,
    this is called once per bar per ticker before orders for that bar are
    submitted.
    """

    def __init__(
        self,
        initial_cash: float,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
        historical_provider: HistoricalDataProvider | None = None,
    ) -> None:
        self._cash = initial_cash
        self._commission_bps = commission_bps
        self._slippage_bps = slippage_bps
        self._historical_provider = historical_provider

        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._last_price: dict[str, float] = {}
        self._realized_pnl = 0.0
        self._connected = False
        self._order_ids = itertools.count(1)

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    def mark_price(self, ticker: str, price: float) -> None:
        """Record the current/last price for `ticker`, used both to fill
        future orders and to mark open positions to market."""
        self._last_price[ticker] = price
        position = self._positions.get(ticker)
        if position is not None:
            market_value = position.quantity * price
            self._positions[ticker] = position.model_copy(
                update={
                    "market_value": market_value,
                    "unrealized_pnl": market_value - position.quantity * position.avg_cost,
                }
            )

    async def get_account_balance(self) -> AccountBalance:
        equity = self._cash + sum(p.market_value for p in self._positions.values())
        return AccountBalance(cash=self._cash, equity=equity, buying_power=self._cash)

    async def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    async def get_historical_bars(
        self, ticker: str, start, end, timeframe: Timeframe
    ) -> list[MarketBar]:
        if self._historical_provider is None:
            raise NotImplementedError(
                "PaperBrokerAdapter was built without a historical_provider; "
                "use a HistoricalDataProvider directly for backtesting instead."
            )
        return await self._historical_provider.get_bars(ticker, start, end, timeframe)

    async def place_order(self, request: OrderRequest) -> Order:
        if request.ticker not in self._last_price:
            raise OrderRejectedError(
                f"No current price known for {request.ticker}; call mark_price() first."
            )

        reference_price = (
            request.limit_price
            if request.order_type == "limit" and request.limit_price is not None
            else self._last_price[request.ticker]
        )
        slippage_mult = (
            1 + self._slippage_bps / 10_000
            if request.side == "buy"
            else 1 - self._slippage_bps / 10_000
        )
        fill_price = reference_price * slippage_mult
        commission = fill_price * request.quantity * self._commission_bps / 10_000

        if request.side == "buy":
            cost = fill_price * request.quantity + commission
            if cost > self._cash:
                raise OrderRejectedError(
                    f"Insufficient cash for {request.ticker}: need {cost:.2f}, "
                    f"have {self._cash:.2f}"
                )
        else:
            existing_qty = self._positions.get(request.ticker)
            held = existing_qty.quantity if existing_qty is not None else 0.0
            if request.quantity > held:
                raise OrderRejectedError(
                    f"Cannot sell {request.quantity} of {request.ticker}; only hold {held}"
                )

        order = Order(
            order_id=str(next(self._order_ids)),
            ticker=request.ticker,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.limit_price,
            status="filled",
            submitted_at=datetime.now(UTC),
            filled_at=datetime.now(UTC),
            fill_price=fill_price,
            commission=commission,
        )
        self._orders[order.order_id] = order
        self._apply_fill(order, fill_price)
        return order

    def _apply_fill(self, order: Order, fill_price: float) -> None:
        """`fill_price` is passed in (rather than read from `order.fill_price`,
        which is `float | None` on the shared `Order` contract) because every
        order this adapter creates is filled immediately with a known price."""
        position = self._positions.get(order.ticker)
        mark = self._last_price[order.ticker]

        if order.side == "buy":
            self._cash -= fill_price * order.quantity + order.commission
            if position is None:
                new_qty = order.quantity
                new_avg_cost = fill_price
            else:
                new_qty = position.quantity + order.quantity
                new_avg_cost = (
                    position.quantity * position.avg_cost + order.quantity * fill_price
                ) / new_qty
            market_value = new_qty * mark
            self._positions[order.ticker] = Position(
                ticker=order.ticker,
                quantity=new_qty,
                avg_cost=new_avg_cost,
                market_value=market_value,
                unrealized_pnl=market_value - new_qty * new_avg_cost,
            )
        else:
            assert position is not None  # guaranteed by the sell-size check above
            self._cash += fill_price * order.quantity - order.commission
            self._realized_pnl += (fill_price - position.avg_cost) * order.quantity
            new_qty = position.quantity - order.quantity
            if new_qty == 0:
                del self._positions[order.ticker]
            else:
                market_value = new_qty * mark
                self._positions[order.ticker] = Position(
                    ticker=order.ticker,
                    quantity=new_qty,
                    avg_cost=position.avg_cost,
                    market_value=market_value,
                    unrealized_pnl=market_value - new_qty * position.avg_cost,
                )

    async def cancel_order(self, order_id: str) -> None:
        if order_id not in self._orders:
            raise BrokerError(f"Unknown order_id: {order_id}")
        # Orders fill immediately in this simple model, so there is never
        # anything pending left to cancel.

    async def get_order_status(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise BrokerError(f"Unknown order_id: {order_id}") from exc
