"""Broker adapter interface.

Every broker — paper, and later Interactive Brokers (`ib_async`) and
Zerodha Kite (`kiteconnect`) — implements this same `BrokerAdapter`
Protocol, so orchestration code never needs to know which one it's talking
to.

Lifecycle methods (`connect`/`disconnect`/`is_connected`) are part of the
interface now, even though `PaperBrokerAdapter` no-ops them, because both
real brokers need this exact seam: IB requires a persistent socket
connection to a running TWS/Gateway process, and Zerodha Kite requires a
daily OAuth-style login that produces a short-lived access token which can
expire mid-session. Designing the seam in from the start means adding those
adapters later touches only `brokers/`, never the agents or orchestrator.

A future low-latency execution service (Rust/C++, out of scope for this
phase) would also implement this Protocol — proxying `place_order` etc.
over gRPC or a local socket/queue to the external process — using the same
`OrderRequest`/`Order` contracts as the boundary.
"""

from typing import Protocol

from trading_agents.contracts.broker import AccountBalance, Order, OrderRequest, Position
from trading_agents.contracts.market import MarketBar, Timeframe


class BrokerError(Exception):
    """Base class for all broker adapter errors."""


class BrokerAuthExpiredError(BrokerError):
    """Raised when a broker's auth/session has expired (e.g. Kite's daily
    access token) and the adapter needs `connect()` called again."""


class OrderRejectedError(BrokerError):
    """Raised when a broker rejects an order outright (e.g. insufficient
    funds, invalid ticker)."""


class BrokerAdapter(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    async def get_account_balance(self) -> AccountBalance: ...

    async def get_positions(self) -> dict[str, Position]: ...

    async def get_historical_bars(
        self, ticker: str, start, end, timeframe: Timeframe
    ) -> list[MarketBar]: ...

    async def place_order(self, request: OrderRequest) -> Order: ...

    async def cancel_order(self, order_id: str) -> None: ...

    async def get_order_status(self, order_id: str) -> Order: ...
