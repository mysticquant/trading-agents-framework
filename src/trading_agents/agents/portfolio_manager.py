"""Portfolio Manager: the final, cross-ticker allocation authority for one
decision cycle. The LLM reasons about prioritization across all of this
cycle's proposals under the shared cash constraint; a deterministic Python
pass afterward asserts final cash feasibility (sequential allocation in
the order proposals were given, which callers should order by conviction)
so the realized decisions can never collectively overspend available cash.
"""

import json
from datetime import datetime

from pydantic import BaseModel, Field

from trading_agents.agents.base import Agent
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.contracts.trading import (
    FinalAction,
    PortfolioDecision,
    RiskAssessment,
    TradeProposal,
)


class ProposalAssessmentPair(BaseModel):
    proposal: TradeProposal
    assessment: RiskAssessment


class PortfolioManagerInput(BaseModel):
    pairs: list[ProposalAssessmentPair]
    portfolio_state: PortfolioState
    current_prices: dict[str, float]
    as_of: datetime


class PortfolioManagerDecisionLLM(BaseModel):
    ticker: str
    final_action: FinalAction
    final_quantity: float = Field(ge=0.0)
    rationale: str


class PortfolioManagerLLMOutput(BaseModel):
    decisions: list[PortfolioManagerDecisionLLM]


class PortfolioManagerAgent(Agent[PortfolioManagerInput, list[PortfolioDecision]]):
    role = "portfolio_manager"

    async def run(self, agent_input: PortfolioManagerInput) -> list[PortfolioDecision]:
        if not agent_input.pairs:
            return []

        user_content = json.dumps(
            {
                "pairs": [
                    {
                        "proposal": pair.proposal.model_dump(mode="json"),
                        "assessment": pair.assessment.model_dump(mode="json"),
                    }
                    for pair in agent_input.pairs
                ],
                "portfolio_state": agent_input.portfolio_state.model_dump(mode="json"),
                "current_prices": agent_input.current_prices,
            }
        )
        llm_output = await self._generate(
            user_content=user_content, output_model=PortfolioManagerLLMOutput
        )
        decisions_by_ticker = {d.ticker: d for d in llm_output.decisions}

        remaining_cash = agent_input.portfolio_state.cash
        final_decisions: list[PortfolioDecision] = []
        for pair in agent_input.pairs:
            proposal, assessment = pair.proposal, pair.assessment
            llm_decision = decisions_by_ticker.get(proposal.ticker)

            if not assessment.approved or llm_decision is None:
                final_decisions.append(
                    PortfolioDecision(
                        ticker=proposal.ticker,
                        final_action="reject",
                        final_quantity=0.0,
                        approved_by="portfolio_manager",
                        rationale=(
                            "Risk manager did not approve this trade."
                            if not assessment.approved
                            else "Portfolio manager returned no decision for this ticker."
                        ),
                        decided_at=agent_input.as_of,
                    )
                )
                continue

            final_action = llm_decision.final_action
            final_quantity = min(llm_decision.final_quantity, assessment.adjusted_quantity)

            if final_action == "buy":
                price = agent_input.current_prices[proposal.ticker]
                # `remaining_cash` can drift a hair below zero from prior
                # iterations' floating-point subtraction, which would
                # otherwise make `affordable_qty` (and so `final_quantity`)
                # a tiny negative number and fail PortfolioDecision's >= 0
                # constraint.
                affordable_qty = max(0.0, remaining_cash) / price if price > 0 else 0.0
                final_quantity = max(0.0, min(final_quantity, affordable_qty))
                if final_quantity <= 0:
                    final_action = "reject"
                    final_quantity = 0.0
                else:
                    remaining_cash -= final_quantity * price

            final_decisions.append(
                PortfolioDecision(
                    ticker=proposal.ticker,
                    final_action=final_action,
                    final_quantity=final_quantity,
                    approved_by="portfolio_manager",
                    rationale=llm_decision.rationale,
                    decided_at=agent_input.as_of,
                )
            )

        return final_decisions
