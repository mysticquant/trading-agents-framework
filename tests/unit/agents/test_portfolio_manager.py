from datetime import datetime

from trading_agents.agents.portfolio_manager import (
    PortfolioManagerAgent,
    PortfolioManagerDecisionLLM,
    PortfolioManagerInput,
    PortfolioManagerLLMOutput,
    ProposalAssessmentPair,
)
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.contracts.trading import RiskAssessment, TradeProposal
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
        )
    )

    aapl = next(d for d in decisions if d.ticker == "AAPL")
    msft = next(d for d in decisions if d.ticker == "MSFT")
    assert aapl.final_quantity == 10.0
    assert msft.final_quantity == 2.0  # only $200 of the $1,200 cash remains after AAPL


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
