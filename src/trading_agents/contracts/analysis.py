"""Analyst and research contracts.

A single `AnalystReport` shape is shared by all four analyst roles
(fundamentals, sentiment, news, technical) — the Researcher consumes all
four uniformly, and `metrics` is the extension point for type-specific
numeric evidence (e.g. precomputed technical-indicator values).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnalystType = Literal["fundamentals", "sentiment", "news", "technical"]
Signal = Literal["bullish", "bearish", "neutral"]
RecommendedAction = Literal["buy", "sell", "hold"]


class AnalystReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    analyst_type: AnalystType
    as_of: datetime
    summary: str
    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class ResearchDebateResult(BaseModel):
    """Bull/bear synthesis produced by the Researcher from all analyst reports."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    bull_thesis: str
    bear_thesis: str
    synthesis: str
    recommended_action: RecommendedAction
    conviction: float = Field(ge=0.0, le=1.0)
    supporting_reports: list[AnalystReport] = Field(default_factory=list)
