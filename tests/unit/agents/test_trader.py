from datetime import datetime

from trading_agents.agents.trader import TraderAgent, TraderInput, TraderLLMOutput
from trading_agents.contracts.analysis import ResearchDebateResult
from trading_agents.contracts.broker import Position
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.llm.fakes import FakeLLMClient


def _research() -> ResearchDebateResult:
    return ResearchDebateResult(
        ticker="AAPL",
        bull_thesis="b",
        bear_thesis="c",
        synthesis="d",
        recommended_action="buy",
        conviction=0.8,
    )


async def test_trader_caps_buy_quantity_to_available_cash():
    canned = TraderLLMOutput(
        action="buy", quantity=1000.0, order_type="market", rationale="go", time_horizon="1w"
    )
    agent = TraderAgent(FakeLLMClient({"trader": canned}))
    portfolio_state = PortfolioState(
        cash=1_000.0, equity=1_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    proposal = await agent.run(
        TraderInput(
            ticker="AAPL",
            research=_research(),
            portfolio_state=portfolio_state,
            current_price=100.0,
        )
    )

    assert proposal.quantity == 10.0


async def test_trader_caps_sell_quantity_to_held_position():
    canned = TraderLLMOutput(
        action="sell", quantity=1000.0, order_type="market", rationale="go", time_horizon="1w"
    )
    agent = TraderAgent(FakeLLMClient({"trader": canned}))
    portfolio_state = PortfolioState(
        cash=0.0, equity=500.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )
    position = Position(
        ticker="AAPL", quantity=5.0, avg_cost=90.0, market_value=500.0, unrealized_pnl=50.0
    )

    proposal = await agent.run(
        TraderInput(
            ticker="AAPL",
            research=_research(),
            portfolio_state=portfolio_state,
            current_position=position,
            current_price=100.0,
        )
    )

    assert proposal.quantity == 5.0


async def test_trader_hold_action_always_has_zero_quantity():
    canned = TraderLLMOutput(
        action="hold", quantity=10.0, order_type="market", rationale="wait", time_horizon="1w"
    )
    agent = TraderAgent(FakeLLMClient({"trader": canned}))
    portfolio_state = PortfolioState(
        cash=1_000.0, equity=1_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    proposal = await agent.run(
        TraderInput(
            ticker="AAPL",
            research=_research(),
            portfolio_state=portfolio_state,
            current_price=100.0,
        )
    )

    assert proposal.action == "hold"
    assert proposal.quantity == 0.0
