from datetime import date, datetime

import pytest

from trading_agents.contracts.trading import PortfolioDecision
from trading_agents.live.guardrails import (
    DailyState,
    Guardrails,
    StateStore,
    check_order,
    halt_reason,
)


def _decision(action: str, quantity: float) -> PortfolioDecision:
    return PortfolioDecision(
        ticker="AAPL",
        final_action=action,
        final_quantity=quantity,
        approved_by="portfolio_manager",
        rationale="r",
        decided_at=datetime(2026, 10, 26),
    )


@pytest.fixture
def guardrails(tmp_path) -> Guardrails:
    return Guardrails(kill_switch_path=tmp_path / "KILL_SWITCH")


def _state(**kwargs) -> DailyState:
    return DailyState(trading_day=date(2026, 10, 26), starting_equity=100_000.0, **kwargs)


def test_buys_are_capped_at_the_equity_fraction(guardrails):
    # 5% of 100,000 = 5,000 -> 25 shares at 200 (not 25.x).
    check = check_order(guardrails, _state(), _decision("buy", 40), price=199.0, equity=100_000)

    assert check.quantity == 25
    assert "5%-of-equity" in check.reason


def test_a_buy_that_trims_to_zero_is_blocked(guardrails):
    check = check_order(guardrails, _state(), _decision("buy", 1), price=6_000.0, equity=100_000)

    assert not check.allowed


def test_sells_are_not_size_capped(guardrails):
    check = check_order(guardrails, _state(), _decision("sell", 500), price=200.0, equity=100_000)

    assert check.quantity == 500 and check.reason is None


def test_daily_order_limit(guardrails):
    check = check_order(
        guardrails, _state(orders_sent=10), _decision("buy", 1), price=10.0, equity=100_000
    )

    assert not check.allowed
    assert "daily order limit" in check.reason


def test_drawdown_halts_for_the_rest_of_the_day(guardrails):
    state = _state()

    assert halt_reason(guardrails, state, equity=98_500.0) is None
    reason = halt_reason(guardrails, state, equity=97_900.0)

    assert "daily drawdown limit" in reason
    # Stays halted even if equity recovers later the same day.
    assert halt_reason(guardrails, state, equity=101_000.0) == reason


def test_kill_switch_blocks_every_order(guardrails):
    guardrails.kill_switch_path.touch()

    check = check_order(guardrails, _state(), _decision("sell", 1), price=10.0, equity=100_000)

    assert not check.allowed
    assert "kill switch" in check.reason


def test_state_store_round_trips_and_resets_each_day(tmp_path):
    store = StateStore(tmp_path / "live" / "state.json")
    store.save(_state(orders_sent=3, halted_reason="x"))

    restored = store.load(date(2026, 10, 26))

    assert restored == _state(orders_sent=3, halted_reason="x")
    assert store.load(date(2026, 10, 27)) is None
    assert not (tmp_path / "live" / "state.tmp").exists()
