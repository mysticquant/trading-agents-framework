from datetime import datetime

import pytest

from trading_agents.agents.risk_manager import (
    RiskConfig,
    RiskManagerAgent,
    RiskManagerInput,
    RiskManagerLLMOutput,
)
from trading_agents.contracts.broker import Position
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.contracts.trading import TradeProposal
from trading_agents.llm.fakes import FakeLLMClient


async def test_hard_position_limit_trims_a_buy_even_when_the_llm_approves_more():
    proposal = TradeProposal(
        ticker="AAPL",
        action="buy",
        quantity=100.0,
        order_type="market",
        rationale="r",
        time_horizon="1w",
    )
    canned = RiskManagerLLMOutput(
        llm_approved=True, max_position_size_pct=0.5, adjusted_quantity=100.0, rationale="ok"
    )
    agent = RiskManagerAgent(FakeLLMClient({"risk_manager": canned}))
    portfolio_state = PortfolioState(
        cash=10_000.0, equity=10_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    assessment = await agent.run(
        RiskManagerInput(
            proposal=proposal,
            portfolio_state=portfolio_state,
            current_price=100.0,
            risk_config=RiskConfig(max_position_pct_per_ticker=0.1),
        )
    )

    # 10% of 10,000 equity = 1,000 max value -> 10 shares at price 100.
    assert assessment.adjusted_quantity == 10.0
    assert assessment.approved is True
    assert any("trimmed" in flag for flag in assessment.risk_flags)


async def test_llm_veto_rejects_the_trade_regardless_of_sizing():
    proposal = TradeProposal(
        ticker="AAPL",
        action="buy",
        quantity=5.0,
        order_type="market",
        rationale="r",
        time_horizon="1w",
    )
    canned = RiskManagerLLMOutput(
        llm_approved=False,
        max_position_size_pct=0.1,
        risk_flags=["too risky"],
        adjusted_quantity=5.0,
        rationale="no",
    )
    agent = RiskManagerAgent(FakeLLMClient({"risk_manager": canned}))
    portfolio_state = PortfolioState(
        cash=10_000.0, equity=10_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    assessment = await agent.run(
        RiskManagerInput(
            proposal=proposal,
            portfolio_state=portfolio_state,
            current_price=100.0,
            risk_config=RiskConfig(),
        )
    )

    assert assessment.approved is False


async def test_sell_proposals_are_not_constrained_by_the_position_size_limit():
    proposal = TradeProposal(
        ticker="AAPL",
        action="sell",
        quantity=50.0,
        order_type="market",
        rationale="r",
        time_horizon="1w",
    )
    canned = RiskManagerLLMOutput(
        llm_approved=True, max_position_size_pct=0.1, adjusted_quantity=50.0, rationale="ok"
    )
    agent = RiskManagerAgent(FakeLLMClient({"risk_manager": canned}))
    portfolio_state = PortfolioState(
        cash=0.0, equity=10_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    assessment = await agent.run(
        RiskManagerInput(
            proposal=proposal,
            portfolio_state=portfolio_state,
            current_price=100.0,
            risk_config=RiskConfig(max_position_pct_per_ticker=0.01),
        )
    )

    assert assessment.adjusted_quantity == 50.0


def _buy(quantity: float) -> TradeProposal:
    return TradeProposal(
        ticker="AAPL",
        action="buy",
        quantity=quantity,
        order_type="market",
        rationale="r",
        time_horizon="1w",
    )


def _state_holding_aapl(shares: float, price: float = 100.0) -> PortfolioState:
    position = Position(
        ticker="AAPL",
        quantity=shares,
        avg_cost=price,
        market_value=shares * price,
        unrealized_pnl=0.0,
    )
    return PortfolioState(
        cash=10_000.0 - shares * price,
        positions={"AAPL": position},
        equity=10_000.0,
        realized_pnl=0.0,
        as_of=datetime(2024, 1, 1),
    )


async def test_hard_position_limit_counts_shares_already_held():
    canned = RiskManagerLLMOutput(
        llm_approved=True, max_position_size_pct=0.5, adjusted_quantity=100.0, rationale="ok"
    )
    agent = RiskManagerAgent(FakeLLMClient({"risk_manager": canned}))

    assessment = await agent.run(
        RiskManagerInput(
            proposal=_buy(100.0),
            portfolio_state=_state_holding_aapl(6.0),
            current_price=100.0,
            risk_config=RiskConfig(max_position_pct_per_ticker=0.1),
        )
    )

    # Cap is 10% of 10,000 = 1,000 total; 600 is already held -> 4 more shares.
    assert assessment.adjusted_quantity == pytest.approx(4.0)
    assert assessment.approved is True


async def test_buy_is_rejected_when_the_position_is_already_at_the_cap():
    canned = RiskManagerLLMOutput(
        llm_approved=True, max_position_size_pct=0.5, adjusted_quantity=10.0, rationale="ok"
    )
    agent = RiskManagerAgent(FakeLLMClient({"risk_manager": canned}))

    assessment = await agent.run(
        RiskManagerInput(
            proposal=_buy(10.0),
            portfolio_state=_state_holding_aapl(12.0),
            current_price=100.0,
            risk_config=RiskConfig(max_position_pct_per_ticker=0.1),
        )
    )

    assert assessment.adjusted_quantity == 0.0
    assert assessment.approved is False
