"""Market data contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Timeframe = Literal["1d", "1h", "5m"]


class MarketBar(BaseModel):
    """A single OHLCV bar for one ticker at one timeframe."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Timeframe
