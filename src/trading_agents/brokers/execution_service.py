"""`BrokerAdapter` that proxies to the `execd` execution service (the Rust
daemon in `execution_service/`) over newline-delimited JSON on a loopback
TCP port, authenticated with a shared token.

The adapter checks the daemon against what the caller expects instead of
trusting its configuration: on `connect()` it asks the daemon which
backend it runs and whether orders are enabled, and refuses to continue if
that doesn't match (`require_backend`, `require_orders_enabled`).
`block_orders=True` refuses orders locally, before anything is sent — the
live loop uses it for dry runs.
"""

import asyncio
import contextlib
import json
from datetime import date
from typing import Any

from trading_agents.brokers.base import BrokerAuthExpiredError, BrokerError, OrderRejectedError
from trading_agents.contracts.broker import AccountBalance, Order, OrderRequest, Position
from trading_agents.contracts.market import MarketBar, Timeframe
from trading_agents.data.providers.base import HistoricalDataProvider


class ExecutionServiceNotSupportedError(BrokerError):
    """The daemon's backend doesn't implement the requested method."""


_ERRORS: dict[str, type[BrokerError]] = {
    "OrderRejected": OrderRejectedError,
    "AuthExpired": BrokerAuthExpiredError,
    "NotSupported": ExecutionServiceNotSupportedError,
}


class ExecutionServiceAdapter:
    def __init__(
        self,
        *,
        token: str,
        host: str = "127.0.0.1",
        port: int = 7600,
        timeout_seconds: float = 30.0,
        order_timeout_seconds: float = 120.0,
        require_backend: str | None = None,
        require_orders_enabled: bool = False,
        block_orders: bool = False,
        fallback_historical: HistoricalDataProvider | None = None,
    ) -> None:
        """`fallback_historical` serves bars when the daemon's backend has
        no market data of its own (the paper backend).

        `order_timeout_seconds` must exceed the daemon's worst case for one
        order (Kite: a 30s POST plus ~10s of status polling, each poll up to
        30s): giving up early would leave a real order placed but untracked."""
        self._token = token
        self._host = host
        self._port = port
        self._timeout = timeout_seconds
        self._order_timeout = order_timeout_seconds
        self._require_backend = require_backend
        self._require_orders_enabled = require_orders_enabled
        self._block_orders = block_orders
        self._fallback_historical = fallback_historical
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0
        self.backend: str | None = None
        self.orders_enabled = False
        self.account: str | None = None

    async def connect(self) -> None:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port), timeout=self._timeout
            )
        except (OSError, TimeoutError) as exc:
            raise BrokerError(
                f"Could not reach the execution service at {self._host}:{self._port} "
                f"({type(exc).__name__}: {exc}). Is `execd` running?"
            ) from exc

        try:
            health = await self._call("health")
            self.backend = str(health.get("backend"))
            self.orders_enabled = bool(health.get("orders_enabled"))
            self.account = health.get("account")
            if self._require_backend and self.backend != self._require_backend:
                raise BrokerError(
                    f"The execution service runs the '{self.backend}' backend, but "
                    f"'{self._require_backend}' is required for this mode."
                )
            if self._require_orders_enabled and not self.orders_enabled:
                raise BrokerError(
                    "The execution service has orders disabled (started without "
                    "--allow-live), but this mode needs to place orders."
                )
        except BrokerError:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(OSError):
                await self._writer.wait_closed()
        self._reader = self._writer = None

    def is_connected(self) -> bool:
        # A daemon that exits or restarts closes its end: the socket stays
        # half-open on ours, so also check for the end-of-stream it left.
        return (
            self._writer is not None
            and self._reader is not None
            and not self._writer.is_closing()
            and not self._reader.at_eof()
        )

    async def _call(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> Any:
        if self._reader is None or self._writer is None:
            raise BrokerError("Not connected to the execution service; call connect() first.")
        async with self._lock:
            self._next_id += 1
            request_id = self._next_id
            line = json.dumps(
                {"id": request_id, "token": self._token, "method": method, "params": params}
            )
            try:
                self._writer.write(line.encode() + b"\n")
                await self._writer.drain()
                raw = await asyncio.wait_for(
                    self._reader.readline(), timeout=timeout or self._timeout
                )
            except (OSError, TimeoutError) as exc:
                await self.disconnect()
                raise BrokerError(f"Execution service call '{method}' failed: {exc}") from exc
            if not raw:
                await self.disconnect()
                raise BrokerError("The execution service closed the connection.")

        response = json.loads(raw)
        if response.get("id") != request_id:
            await self.disconnect()
            raise BrokerError("Execution service response did not match the request.")
        error = response.get("error")
        if error:
            error_class = _ERRORS.get(error.get("type", ""), BrokerError)
            raise error_class(f"{error.get('message')} [execd {error.get('type')}]")
        return response.get("result")

    async def get_account_balance(self) -> AccountBalance:
        return AccountBalance.model_validate(await self._call("get_account_balance"))

    async def get_positions(self) -> dict[str, Position]:
        result = await self._call("get_positions")
        return {ticker: Position.model_validate(p) for ticker, p in (result or {}).items()}

    async def get_historical_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]:
        try:
            result = await self._call(
                "get_historical_bars",
                {
                    "ticker": ticker,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "timeframe": timeframe,
                },
            )
        except ExecutionServiceNotSupportedError:
            if self._fallback_historical is None:
                raise
            return await self._fallback_historical.get_bars(ticker, start, end, timeframe)
        return [MarketBar.model_validate(bar) for bar in result]

    async def get_bars(
        self, ticker: str, start: date, end: date, timeframe: Timeframe
    ) -> list[MarketBar]:
        """`HistoricalDataProvider` view of `get_historical_bars`, so the
        adapter can also be the live loop's market-data source."""
        return await self.get_historical_bars(ticker, start, end, timeframe)

    async def mark_price(self, ticker: str, price: float) -> None:
        """Sets the price the paper backend's simulated fills use; a no-op
        for real backends, which know their own prices."""
        if self.backend == "paper":
            await self._call("mark_price", {"ticker": ticker, "price": price})

    async def place_order(self, request: OrderRequest) -> Order:
        if self._block_orders:
            raise OrderRejectedError("Orders are blocked on this connection (dry run).")
        return Order.model_validate(
            await self._call(
                "place_order", request.model_dump(mode="json"), timeout=self._order_timeout
            )
        )

    async def cancel_order(self, order_id: str) -> None:
        await self._call("cancel_order", {"order_id": order_id})

    async def get_order_status(self, order_id: str) -> Order:
        return Order.model_validate(await self._call("get_order_status", {"order_id": order_id}))
