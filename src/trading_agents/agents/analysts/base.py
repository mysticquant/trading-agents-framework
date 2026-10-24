"""Shared base for the three evidence-interpreting analysts (fundamentals,
sentiment, news). All three take the same shape of input — a ticker plus an
arbitrary evidence payload — and differ only in `role`/`analyst_type` and
their prompt file. The TechnicalAnalyst is NOT built on this base: it
computes its own evidence deterministically from price bars rather than
receiving a pre-built evidence dict (see `agents/analysts/technical.py`).

Evidence is assembled outside the agent — normally by an
`EvidenceProvider` (see `data/evidence/`) that keeps it point-in-time for
backtests — so the agent logic stays independently testable via
`FakeLLMClient`.
"""

import json
from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, Field

from trading_agents.agents.base import Agent
from trading_agents.contracts.analysis import AnalystReport, AnalystType, Signal


class AnalystInput(BaseModel):
    ticker: str
    as_of: datetime
    evidence: dict[str, object] = Field(default_factory=dict)


class AnalystLLMOutput(BaseModel):
    summary: str
    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class AnalystAgent(Agent[AnalystInput, AnalystReport]):
    analyst_type: ClassVar[AnalystType]

    async def run(self, agent_input: AnalystInput) -> AnalystReport:
        user_content = json.dumps(
            {"ticker": agent_input.ticker, "evidence": agent_input.evidence}, default=str
        )
        llm_output = await self._generate(user_content=user_content, output_model=AnalystLLMOutput)
        return AnalystReport(
            ticker=agent_input.ticker,
            analyst_type=self.analyst_type,
            as_of=agent_input.as_of,
            summary=llm_output.summary,
            signal=llm_output.signal,
            confidence=llm_output.confidence,
            key_points=llm_output.key_points,
            metrics=llm_output.metrics,
        )
