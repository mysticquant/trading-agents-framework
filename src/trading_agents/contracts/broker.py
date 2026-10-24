"""Broker-facing order and account contracts.

These are the boundary types every `BrokerAdapter` implementation speaks —
paper, and later Interactive Brokers / Zerodha Kite / a low-latency
execution service all accept `OrderRequest` and return `Order`.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
OrderStatus = Literal["pending", "submitted", "filled", "partially_filled", "cancelled", "rejected"]


class OrderRequest(BaseModel):
    """What callers submit to `BrokerAdapter.place_order`."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    side: OrderSide
    quantity: float = Field(gt=0.0)
    order_type: OrderType
    limit_price: float | None = None


class Order(BaseModel):
    """What a broker returns/tracks for a submitted order."""

    order_id: str
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: float | None = None
    status: OrderStatus
    submitted_at: datetime
    filled_at: datetime | None = None
    fill_price: float | None = None
    commission: float = 0.0


class Position(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float


class AccountBalance(BaseModel):
    cash: float
    equity: float
    buying_power: float
