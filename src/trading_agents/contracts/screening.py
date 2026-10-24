"""Screener output contracts."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class StockScreenResult(BaseModel):
    """A ticker that passed the deterministic pre-filter, with an LLM-assigned
    rank/rationale for why it's worth deeper per-ticker analysis."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    score: float = Field(description="Higher means more worth analyzing this cycle.")
    rationale: str
    tags: list[str] = Field(default_factory=list)
    screened_at: datetime
