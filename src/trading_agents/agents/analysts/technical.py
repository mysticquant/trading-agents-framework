"""Technical analyst: computes indicators deterministically from price bars
(never via the LLM), then asks the LLM only to interpret the already-computed
values."""

import json
from datetime import datetime

from pydantic import BaseModel

from trading_agents.agents.analysts.base import AnalystLLMOutput
from trading_agents.agents.base import Agent
from trading_agents.contracts.analysis import AnalystReport
from trading_agents.contracts.market import MarketBar
from trading_agents.indicators.technical import bars_to_frame, latest_snapshot


class TechnicalAnalystInput(BaseModel):
    ticker: str
    as_of: datetime
    bars: list[MarketBar]


class TechnicalAnalystAgent(Agent[TechnicalAnalystInput, AnalystReport]):
    role = "technical_analyst"

    async def run(self, agent_input: TechnicalAnalystInput) -> AnalystReport:
        indicator_values = latest_snapshot(bars_to_frame(agent_input.bars))
        user_content = json.dumps({"ticker": agent_input.ticker, "indicators": indicator_values})
        llm_output = await self._generate(user_content=user_content, output_model=AnalystLLMOutput)
        return AnalystReport(
            ticker=agent_input.ticker,
            analyst_type="technical",
            as_of=agent_input.as_of,
            summary=llm_output.summary,
            signal=llm_output.signal,
            confidence=llm_output.confidence,
            key_points=llm_output.key_points,
            metrics={**indicator_values, **llm_output.metrics},
        )
