"""Screener: a deterministic quantitative pre-filter followed by an LLM pass
that ranks and explains the shortlist. The pre-filter runs first and keeps
every LLM call scoped to a small shortlist, not the whole universe — this
is the framework's main lever against per-cycle LLM cost.
"""

import json
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from trading_agents.agents.base import Agent
from trading_agents.contracts.screening import StockScreenResult


class PrefilterCandidate(BaseModel):
    ticker: str
    return_pct: float
    volume_zscore: float


def compute_prefilter_candidates(
    bars_by_ticker: dict[str, pd.DataFrame], top_n: int = 10
) -> list[PrefilterCandidate]:
    """Rank tickers by |return| and volume z-score over the trailing window
    in `bars_by_ticker` (each a DataFrame with a `close`/`volume` column,
    most recent row last) and keep the top `top_n` by combined magnitude.
    Pure Python/pandas — no LLM call."""
    candidates = []
    for ticker, bars in bars_by_ticker.items():
        if len(bars) < 2:
            continue
        close = bars["close"]
        return_pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        volume = bars["volume"]
        volume_std = volume.std()
        volume_zscore = (volume.iloc[-1] - volume.mean()) / volume_std if volume_std else 0.0
        candidates.append(
            PrefilterCandidate(
                ticker=ticker,
                return_pct=round(float(return_pct), 4),
                volume_zscore=round(float(volume_zscore), 4),
            )
        )
    candidates.sort(key=lambda c: abs(c.return_pct) + abs(c.volume_zscore), reverse=True)
    return candidates[:top_n]


class ScreenerRanking(BaseModel):
    ticker: str
    score: float = Field(description="Higher means more worth analyzing this cycle.")
    rationale: str
    tags: list[str] = Field(default_factory=list)


class ScreenerLLMOutput(BaseModel):
    rankings: list[ScreenerRanking]


class ScreenerInput(BaseModel):
    candidates: list[PrefilterCandidate]
    as_of: datetime


class ScreenerAgent(Agent[ScreenerInput, list[StockScreenResult]]):
    """`as_of` is always the orchestrator/backtest runner's simulated clock
    for the current cycle, never wall-clock time — this keeps backtests
    free of lookahead and reproducible."""

    role = "screener"

    async def run(self, agent_input: ScreenerInput) -> list[StockScreenResult]:
        if not agent_input.candidates:
            return []
        user_content = json.dumps(
            {"candidates": [c.model_dump() for c in agent_input.candidates]}, default=str
        )
        llm_output = await self._generate(user_content=user_content, output_model=ScreenerLLMOutput)
        return [
            StockScreenResult(
                ticker=ranking.ticker,
                score=ranking.score,
                rationale=ranking.rationale,
                tags=ranking.tags,
                screened_at=agent_input.as_of,
            )
            for ranking in llm_output.rankings
        ]
