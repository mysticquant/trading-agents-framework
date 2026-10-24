"""Trader: turns the Researcher's recommendation into a concrete, sized
trade proposal. The LLM proposes direction/sizing/rationale; Python
hard-guards the quantity against available cash (buys) or held quantity
(sells) so a miscalibrated LLM proposal can never exceed what's actually
possible — RiskManager and PortfolioManager apply further limits downstream.
"""

import json

from pydantic import BaseModel, Field

from trading_agents.agents.base import Agent
from trading_agents.contracts.analysis import ResearchDebateResult
from trading_agents.contracts.broker import Position
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.contracts.trading import OrderType, TradeAction, TradeProposal


class TraderInput(BaseModel):
    ticker: str
    research: ResearchDebateResult
    portfolio_state: PortfolioState
    current_position: Position | None = None
    current_price: float


class TraderLLMOutput(BaseModel):
    action: TradeAction
    quantity: float = Field(ge=0.0)
    order_type: OrderType
    limit_price: float | None = None
    rationale: str
    time_horizon: str


class TraderAgent(Agent[TraderInput, TradeProposal]):
    role = "trader"

    async def run(self, agent_input: TraderInput) -> TradeProposal:
        user_content = json.dumps(
            {
                "ticker": agent_input.ticker,
                "research": agent_input.research.model_dump(mode="json"),
                "portfolio_state": agent_input.portfolio_state.model_dump(mode="json"),
                "current_position": (
                    agent_input.current_position.model_dump(mode="json")
                    if agent_input.current_position
                    else None
                ),
                "current_price": agent_input.current_price,
            }
        )
        llm_output = await self._generate(user_content=user_content, output_model=TraderLLMOutput)

        quantity = llm_output.quantity
        if llm_output.action == "buy" and agent_input.current_price > 0:
            max_affordable = agent_input.portfolio_state.cash / agent_input.current_price
            quantity = min(quantity, max_affordable)
        elif llm_output.action == "sell":
            held = agent_input.current_position.quantity if agent_input.current_position else 0.0
            quantity = min(quantity, held)
        else:
            quantity = 0.0

        return TradeProposal(
            ticker=agent_input.ticker,
            action=llm_output.action,
            quantity=quantity,
            order_type=llm_output.order_type,
            limit_price=llm_output.limit_price,
            rationale=llm_output.rationale,
            time_horizon=llm_output.time_horizon,
        )
