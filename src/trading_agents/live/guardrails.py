"""Hard limits on what the live loop may send to a broker, enforced in
Python after the agents decide — they can only shrink or block an order,
never enlarge one, whatever the LLM said.

- Kill switch: while the kill-switch file exists, no order is sent.
- Daily drawdown: once equity falls the configured fraction below the
  day's starting equity, trading stops for the rest of that day.
- Orders per day: a cap on orders sent per trading day.
- Order size: a buy may not exceed the configured fraction of equity.
  Sells are not size-capped (they reduce exposure, and capping them could
  trap a position); the broker still refuses selling more than is held.

The per-day counters live in `DailyState`, persisted by `StateStore` so a
restart mid-day doesn't reset them.
"""

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from trading_agents.contracts.trading import PortfolioDecision


@dataclass(frozen=True)
class Guardrails:
    max_order_pct_of_equity: float = 0.05
    max_orders_per_day: int = 10
    max_daily_drawdown_pct: float = 0.02
    kill_switch_path: Path = Path("KILL_SWITCH")

    def kill_switch_engaged(self) -> bool:
        return self.kill_switch_path.exists()


@dataclass
class DailyState:
    trading_day: date
    starting_equity: float
    orders_sent: int = 0
    halted_reason: str | None = None


class StateStore:
    """Persists `DailyState` as JSON. A state from an earlier day is
    discarded, so each trading day starts fresh."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self, trading_day: date) -> DailyState | None:
        if not self._path.exists():
            return None
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if raw.get("trading_day") != trading_day.isoformat():
            return None
        return DailyState(
            trading_day=trading_day,
            starting_equity=float(raw["starting_equity"]),
            orders_sent=int(raw["orders_sent"]),
            halted_reason=raw.get("halted_reason"),
        )

    def save(self, state: DailyState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**asdict(state), "trading_day": state.trading_day.isoformat()}
        # Write-then-rename so a crash mid-write can't leave a corrupt file
        # that would reset the day's counters.
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._path)


@dataclass(frozen=True)
class OrderCheck:
    quantity: float
    reason: str | None = None  # why the order was trimmed or blocked

    @property
    def allowed(self) -> bool:
        return self.quantity > 0


def halt_reason(guardrails: Guardrails, state: DailyState, equity: float) -> str | None:
    """Why no orders may be sent right now, or None."""
    if guardrails.kill_switch_engaged():
        return f"kill switch engaged ({guardrails.kill_switch_path} exists)"
    if state.halted_reason:
        return state.halted_reason
    floor = state.starting_equity * (1 - guardrails.max_daily_drawdown_pct)
    if equity < floor:
        state.halted_reason = (
            f"daily drawdown limit hit: equity {equity:,.2f} is below "
            f"{floor:,.2f} ({guardrails.max_daily_drawdown_pct:.0%} under the day's "
            f"starting {state.starting_equity:,.2f})"
        )
        return state.halted_reason
    return None


def check_order(
    guardrails: Guardrails,
    state: DailyState,
    decision: PortfolioDecision,
    *,
    price: float,
    equity: float,
) -> OrderCheck:
    halted = halt_reason(guardrails, state, equity)
    if halted:
        return OrderCheck(0.0, halted)
    if state.orders_sent >= guardrails.max_orders_per_day:
        return OrderCheck(0.0, f"daily order limit reached ({guardrails.max_orders_per_day})")

    quantity = decision.final_quantity
    if decision.final_action == "buy" and price > 0:
        max_quantity = math.floor(equity * guardrails.max_order_pct_of_equity / price)
        if quantity > max_quantity:
            return OrderCheck(
                float(max_quantity),
                f"buy trimmed from {quantity:g} to {max_quantity} shares by the "
                f"{guardrails.max_order_pct_of_equity:.0%}-of-equity order limit",
            )
    return OrderCheck(quantity)
