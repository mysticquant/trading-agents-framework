from datetime import datetime

import pytest

from trading_agents.agents.portfolio_manager import (
    PortfolioManagerAgent,
    PortfolioManagerDecisionLLM,
    PortfolioManagerInput,
    PortfolioManagerLLMOutput,
    ProposalAssessmentPair,
)
from trading_agents.brokers.paper import PaperBrokerAdapter
from trading_agents.contracts.broker import OrderRequest
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.contracts.trading import ExecutionCosts, RiskAssessment, TradeProposal
from trading_agents.llm.fakes import FakeLLMClient


def _proposal(ticker: str) -> TradeProposal:
    return TradeProposal(
        ticker=ticker,
        action="buy",
        quantity=10.0,
        order_type="market",
        rationale="r",
        time_horizon="1w",
    )


def _assessment(ticker: str, approved: bool = True) -> RiskAssessment:
    return RiskAssessment(
        ticker=ticker,
        approved=approved,
        max_position_size_pct=0.5,
        adjusted_quantity=10.0,
        rationale="ok",
    )


async def test_cash_feasibility_trims_the_later_ticker_when_cash_runs_out():
    canned = PortfolioManagerLLMOutput(
        decisions=[
            PortfolioManagerDecisionLLM(
                ticker="AAPL", final_action="buy", final_quantity=10.0, rationale="go"
            ),
            PortfolioManagerDecisionLLM(
                ticker="MSFT", final_action="buy", final_quantity=10.0, rationale="go"
            ),
        ]
    )
    agent = PortfolioManagerAgent(FakeLLMClient({"portfolio_manager": canned}))
    portfolio_state = PortfolioState(
        cash=1_200.0, equity=1_200.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    decisions = await agent.run(
        PortfolioManagerInput(
            pairs=[
                ProposalAssessmentPair(proposal=_proposal("AAPL"), assessment=_assessment("AAPL")),
                ProposalAssessmentPair(proposal=_proposal("MSFT"), assessment=_assessment("MSFT")),
            ],
            portfolio_state=portfolio_state,
            current_prices={"AAPL": 100.0, "MSFT": 100.0},
            as_of=datetime(2024, 1, 1),
            execution_costs=ExecutionCosts(commission_bps=0.0, slippage_bps=0.0),
        )
    )

    aapl = next(d for d in decisions if d.ticker == "AAPL")
    msft = next(d for d in decisions if d.ticker == "MSFT")
    assert aapl.final_quantity == 10.0
    # only $200 of the $1,200 cash remains after AAPL
    assert msft.final_quantity == pytest.approx(2.0)


async def test_unapproved_risk_assessment_is_always_rejected():
    agent = PortfolioManagerAgent(
        FakeLLMClient({"portfolio_manager": PortfolioManagerLLMOutput(decisions=[])})
    )
    portfolio_state = PortfolioState(
        cash=1_000.0, equity=1_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    decisions = await agent.run(
        PortfolioManagerInput(
            pairs=[
                ProposalAssessmentPair(
                    proposal=_proposal("AAPL"), assessment=_assessment("AAPL", approved=False)
                )
            ],
            portfolio_state=portfolio_state,
            current_prices={"AAPL": 100.0},
            as_of=datetime(2024, 1, 1),
        )
    )

    assert decisions[0].final_action == "reject"


async def test_floating_point_cash_drift_rejects_cleanly_instead_of_raising():
    # 100.0 / 3.0 * 3.0 != 100.0 exactly in floating point — it leaves
    # `remaining_cash` a hair below zero after the first allocation. The
    # next ticker's affordable quantity must clamp to 0, not go negative
    # (which would fail PortfolioDecision's final_quantity >= 0 constraint).
    canned = PortfolioManagerLLMOutput(
        decisions=[
            PortfolioManagerDecisionLLM(
                ticker="AAPL", final_action="buy", final_quantity=1000.0, rationale="go"
            ),
            PortfolioManagerDecisionLLM(
                ticker="MSFT", final_action="buy", final_quantity=10.0, rationale="go"
            ),
        ]
    )
    agent = PortfolioManagerAgent(FakeLLMClient({"portfolio_manager": canned}))
    portfolio_state = PortfolioState(
        cash=100.0, equity=100.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    decisions = await agent.run(
        PortfolioManagerInput(
            pairs=[
                ProposalAssessmentPair(
                    proposal=_proposal("AAPL").model_copy(update={"quantity": 1000.0}),
                    assessment=_assessment("AAPL").model_copy(update={"adjusted_quantity": 1000.0}),
                ),
                ProposalAssessmentPair(proposal=_proposal("MSFT"), assessment=_assessment("MSFT")),
            ],
            portfolio_state=portfolio_state,
            current_prices={"AAPL": 3.0, "MSFT": 5.0},
            as_of=datetime(2024, 1, 1),
        )
    )

    msft = next(d for d in decisions if d.ticker == "MSFT")
    assert msft.final_action == "reject"
    assert msft.final_quantity == 0.0


async def test_no_pairs_returns_no_decisions_without_calling_the_llm():
    agent = PortfolioManagerAgent(FakeLLMClient({}))
    portfolio_state = PortfolioState(
        cash=1_000.0, equity=1_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    decisions = await agent.run(
        PortfolioManagerInput(
            pairs=[], portfolio_state=portfolio_state, current_prices={}, as_of=datetime(2024, 1, 1)
        )
    )

    assert decisions == []


def _pair(ticker: str, action: str, quantity: float = 10.0) -> ProposalAssessmentPair:
    return ProposalAssessmentPair(
        proposal=_proposal(ticker).model_copy(update={"action": action, "quantity": quantity}),
        assessment=_assessment(ticker).model_copy(update={"adjusted_quantity": quantity}),
    )


@pytest.mark.parametrize(
    ("proposal_action", "llm_action"), [("sell", "buy"), ("buy", "sell"), ("hold", "buy")]
)
async def test_llm_cannot_flip_the_direction_of_a_proposal(proposal_action, llm_action):
    # A sell proposal is never size-capped by the risk manager; letting the PM
    # turn it into a buy would bypass the per-ticker position limit.
    canned = PortfolioManagerLLMOutput(
        decisions=[
            PortfolioManagerDecisionLLM(
                ticker="AAPL", final_action=llm_action, final_quantity=500.0, rationale="flip"
            )
        ]
    )
    agent = PortfolioManagerAgent(FakeLLMClient({"portfolio_manager": canned}))
    portfolio_state = PortfolioState(
        cash=1_000_000.0, equity=1_000_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    decisions = await agent.run(
        PortfolioManagerInput(
            pairs=[_pair("AAPL", proposal_action, quantity=500.0)],
            portfolio_state=portfolio_state,
            current_prices={"AAPL": 100.0},
            as_of=datetime(2024, 1, 1),
        )
    )

    assert decisions[0].final_action == "reject"
    assert decisions[0].final_quantity == 0.0


async def test_llm_hold_on_a_buy_proposal_carries_zero_quantity():
    canned = PortfolioManagerLLMOutput(
        decisions=[
            PortfolioManagerDecisionLLM(
                ticker="AAPL", final_action="hold", final_quantity=10.0, rationale="wait"
            )
        ]
    )
    agent = PortfolioManagerAgent(FakeLLMClient({"portfolio_manager": canned}))
    portfolio_state = PortfolioState(
        cash=10_000.0, equity=10_000.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    decisions = await agent.run(
        PortfolioManagerInput(
            pairs=[_pair("AAPL", "buy")],
            portfolio_state=portfolio_state,
            current_prices={"AAPL": 100.0},
            as_of=datetime(2024, 1, 1),
        )
    )

    assert decisions[0].final_action == "hold"
    assert decisions[0].final_quantity == 0.0


async def test_cash_limited_buys_are_affordable_at_the_broker_after_fees():
    # Two buys that together want more than the available cash: the second
    # is trimmed to "the rest of the cash". With slippage and commission
    # the broker must still accept both orders.
    canned = PortfolioManagerLLMOutput(
        decisions=[
            PortfolioManagerDecisionLLM(
                ticker=ticker, final_action="buy", final_quantity=10.0, rationale="go"
            )
            for ticker in ("AAPL", "MSFT")
        ]
    )
    agent = PortfolioManagerAgent(FakeLLMClient({"portfolio_manager": canned}))
    costs = ExecutionCosts(commission_bps=10.0, slippage_bps=20.0)
    portfolio_state = PortfolioState(
        cash=1_500.0, equity=1_500.0, realized_pnl=0.0, as_of=datetime(2024, 1, 1)
    )

    decisions = await agent.run(
        PortfolioManagerInput(
            pairs=[_pair("AAPL", "buy"), _pair("MSFT", "buy")],
            portfolio_state=portfolio_state,
            current_prices={"AAPL": 100.0, "MSFT": 100.0},
            as_of=datetime(2024, 1, 1),
            execution_costs=costs,
        )
    )

    broker = PaperBrokerAdapter(
        initial_cash=1_500.0,
        commission_bps=costs.commission_bps,
        slippage_bps=costs.slippage_bps,
    )
    for decision in decisions:
        assert decision.final_action == "buy"
        broker.mark_price(decision.ticker, 100.0)
        await broker.place_order(
            OrderRequest(
                ticker=decision.ticker,
                side="buy",
                quantity=decision.final_quantity,
                order_type="market",
            )
        )
    msft = next(d for d in decisions if d.ticker == "MSFT")
    assert 0 < msft.final_quantity < 5.0
    assert (await broker.get_account_balance()).cash == pytest.approx(0.0, abs=1e-3)
