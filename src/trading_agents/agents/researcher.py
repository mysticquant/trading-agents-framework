"""Researcher: synthesizes the four analyst reports for one ticker into a
bull/bear debate and a recommendation. LLM-only — there is no deterministic
computation to do here; `supporting_reports` is echoed back from the input
in Python rather than re-typed by the LLM, to keep the call cheap and avoid
transcription drift.
"""

import json

from pydantic import BaseModel, Field

from trading_agents.agents.base import Agent
from trading_agents.contracts.analysis import AnalystReport, RecommendedAction, ResearchDebateResult


class ResearcherInput(BaseModel):
    ticker: str
    analyst_reports: list[AnalystReport]


class ResearcherLLMOutput(BaseModel):
    bull_thesis: str
    bear_thesis: str
    synthesis: str
    recommended_action: RecommendedAction
    conviction: float = Field(ge=0.0, le=1.0)


class ResearcherAgent(Agent[ResearcherInput, ResearchDebateResult]):
    role = "researcher"

    async def run(self, agent_input: ResearcherInput) -> ResearchDebateResult:
        user_content = json.dumps(
            {
                "ticker": agent_input.ticker,
                "analyst_reports": [r.model_dump(mode="json") for r in agent_input.analyst_reports],
            }
        )
        llm_output = await self._generate(
            user_content=user_content, output_model=ResearcherLLMOutput
        )
        return ResearchDebateResult(
            ticker=agent_input.ticker,
            bull_thesis=llm_output.bull_thesis,
            bear_thesis=llm_output.bear_thesis,
            synthesis=llm_output.synthesis,
            recommended_action=llm_output.recommended_action,
            conviction=llm_output.conviction,
            supporting_reports=agent_input.analyst_reports,
        )
