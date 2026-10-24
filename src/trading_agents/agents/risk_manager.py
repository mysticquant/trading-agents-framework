"""Risk Manager: an independent check on one TradeProposal. The LLM supplies
qualitative judgment and risk flags; a hard, deterministic position-size
limit (percent of portfolio equity) is enforced in Python regardless of
what the LLM says, and can only ever tighten the outcome, never loosen it.
The hard limit only constrains *buys* — trimming or rejecting a sell could
itself be a risk (forced over-concentration), so sells/holds are governed
by the LLM's judgment alone.
"""

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from trading_agents.agents.base import Agent
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.contracts.trading import RiskAssessment, TradeProposal


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct_per_ticker: float = 0.10


class RiskManagerInput(BaseModel):
    proposal: TradeProposal
    portfolio_state: PortfolioState
    current_price: float
    risk_config: RiskConfig


class RiskManagerLLMOutput(BaseModel):
    llm_approved: bool
    max_position_size_pct: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)
    adjusted_quantity: float = Field(ge=0.0)
    rationale: str


class RiskManagerAgent(Agent[RiskManagerInput, RiskAssessment]):
    role = "risk_manager"

    async def run(self, agent_input: RiskManagerInput) -> RiskAssessment:
        proposal = agent_input.proposal
        hard_cap_pct = agent_input.risk_config.max_position_pct_per_ticker
        user_content = json.dumps(
            {
                "proposal": proposal.model_dump(mode="json"),
                "portfolio_state": agent_input.portfolio_state.model_dump(mode="json"),
                "current_price": agent_input.current_price,
                "risk_config": {"max_position_pct_per_ticker": hard_cap_pct},
            }
        )
        llm_output = await self._generate(
            user_content=user_content, output_model=RiskManagerLLMOutput
        )

        risk_flags = list(llm_output.risk_flags)
        if proposal.action == "buy" and agent_input.current_price > 0:
            max_allowed_pct = min(llm_output.max_position_size_pct, hard_cap_pct)
            # The limit bounds the *total* position, so shares already held
            # use up part of it — otherwise a buy every cycle would let the
            # position grow by a full cap's worth each day.
            existing = agent_input.portfolio_state.positions.get(proposal.ticker)
            existing_value = existing.quantity * agent_input.current_price if existing else 0.0
            headroom_value = max(
                0.0, agent_input.portfolio_state.equity * max_allowed_pct - existing_value
            )
            max_allowed_qty = headroom_value / agent_input.current_price
            final_quantity = min(llm_output.adjusted_quantity, proposal.quantity, max_allowed_qty)
            if final_quantity < min(llm_output.adjusted_quantity, proposal.quantity):
                risk_flags.append(
                    f"Quantity trimmed to {final_quantity:g} by the "
                    f"{max_allowed_pct:.0%} max-position-per-ticker limit "
                    f"(existing position: {existing_value:,.2f})."
                )
        else:
            max_allowed_pct = llm_output.max_position_size_pct
            final_quantity = min(llm_output.adjusted_quantity, proposal.quantity)

        approved = llm_output.llm_approved and (proposal.action != "buy" or final_quantity > 0)

        return RiskAssessment(
            ticker=proposal.ticker,
            approved=approved,
            max_position_size_pct=max_allowed_pct,
            risk_flags=risk_flags,
            adjusted_quantity=final_quantity,
            rationale=llm_output.rationale,
        )
