"""Live trading loop: one decision cycle per trading day at a scheduled
time near the close, against a real broker.

Modes:
- "dry-run" (default): the full cycle runs and every order it *would*
  send is journaled, but nothing is sent.
- "paper": orders go to the broker; the caller must hand in a broker that
  can only reach a paper account (e.g. IB built without allow_live).
- "live": real orders. The CLI requires both `--mode live` and
  ALLOW_LIVE_TRADING=true before building this.

Every cycle appends one JSON line to the journal (decisions, guardrail
actions, orders, errors), so what happened — or would have — is auditable.
Market holidays are detected by the absence of a bar dated today.
"""

import asyncio
import dataclasses
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from trading_agents.brokers.base import (
    BrokerAdapter,
    BrokerAuthExpiredError,
    BrokerError,
    OrderRejectedError,
)
from trading_agents.contracts.broker import OrderRequest
from trading_agents.contracts.market import MarketBar
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.data.providers.base import HistoricalDataProvider
from trading_agents.live.guardrails import (
    DailyState,
    Guardrails,
    StateStore,
    check_order,
    halt_reason,
)
from trading_agents.orchestration.pipeline import DecisionCycleOrchestrator

logger = logging.getLogger(__name__)

Mode = Literal["dry-run", "paper", "live"]

# Long waits are split into short sleeps so a laptop that slept, or a clock
# change, doesn't make the loop oversleep its decision time.
_MAX_SLEEP_SECONDS = 60.0


@dataclass(frozen=True)
class Schedule:
    timezone: ZoneInfo
    decision_time: time

    def next_run(self, after: datetime) -> datetime:
        """The first weekday decision time strictly after `after`."""
        local = after.astimezone(self.timezone)
        candidate = datetime.combine(local.date(), self.decision_time, tzinfo=self.timezone)
        if candidate <= local:
            candidate += timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate


@dataclass
class CycleReport:
    started_at: datetime
    mode: Mode
    trading_day: date
    skipped_reason: str | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                **dataclasses.asdict(self),
                "started_at": self.started_at.isoformat(),
                "trading_day": self.trading_day.isoformat(),
            },
            default=str,
        )


class LiveTradingLoop:
    def __init__(
        self,
        *,
        orchestrator: DecisionCycleOrchestrator,
        broker: BrokerAdapter,
        data_provider: HistoricalDataProvider,
        universe: list[str],
        guardrails: Guardrails,
        mode: Mode,
        schedule: Schedule,
        state_store: StateStore,
        journal_path: Path,
        lookback_days: int = 150,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.orchestrator = orchestrator
        self.broker = broker
        self.data_provider = data_provider
        self.universe = universe
        self.guardrails = guardrails
        self.mode = mode
        self.schedule = schedule
        self.state_store = state_store
        self.journal_path = journal_path
        self.lookback_days = lookback_days
        self._now = now
        self._sleep = sleep

    async def run_forever(self) -> None:
        while True:
            next_at = self.schedule.next_run(self._now())
            logger.info("Next decision cycle at %s", next_at.isoformat())
            await self._wait_until(next_at)
            try:
                await self.run_once()
            except Exception:
                # Never let one bad day stop the loop; the next run retries.
                logger.exception("Decision cycle crashed")

    async def _wait_until(self, moment: datetime) -> None:
        while (remaining := (moment - self._now()).total_seconds()) > 0:
            await self._sleep(min(remaining, _MAX_SLEEP_SECONDS))

    async def run_once(self) -> CycleReport:
        now = self._now()
        trading_day = now.astimezone(self.schedule.timezone).date()
        report = CycleReport(started_at=now, mode=self.mode, trading_day=trading_day)
        try:
            await self._run_cycle(report, now)
        finally:
            self._journal(report)
        return report

    async def _run_cycle(self, report: CycleReport, now: datetime) -> None:
        if self.guardrails.kill_switch_engaged():
            report.skipped_reason = f"kill switch engaged ({self.guardrails.kill_switch_path})"
            return
        try:
            if not self.broker.is_connected():
                await self.broker.connect()
        except BrokerError as exc:
            report.skipped_reason = f"broker unavailable: {exc}"
            return

        market_data = await self._todays_market_data(report)
        if not market_data:
            report.skipped_reason = (
                f"no bar dated {report.trading_day} for any ticker (market closed?)"
            )
            return

        try:
            # Simulated (paper) brokers fill at the price they were last
            # marked at; mark before reading the balance so equity is current.
            await self._mark_prices(market_data)
            balance = await self.broker.get_account_balance()
            positions = await self.broker.get_positions()
        except BrokerError as exc:
            report.skipped_reason = f"broker unavailable: {exc}"
            return

        state = self.state_store.load(report.trading_day) or DailyState(
            trading_day=report.trading_day, starting_equity=balance.equity
        )
        self.state_store.save(state)
        if halted := halt_reason(self.guardrails, state, balance.equity):
            self.state_store.save(state)
            report.skipped_reason = halted
            return

        cycle = await self.orchestrator.run_cycle(
            as_of=now,
            universe=list(market_data),
            market_data=market_data,
            portfolio_state=PortfolioState(
                cash=balance.cash,
                positions=positions,
                equity=balance.equity,
                realized_pnl=0.0,
                as_of=now,
            ),
        )

        # Dry runs check limits against a scratch copy, so the journal shows
        # what a real run would have been allowed to send without touching
        # the persisted counters.
        limits_state = dataclasses.replace(state) if self.mode == "dry-run" else state
        for ticker_state in cycle.per_ticker.values():
            if ticker_state.error:
                report.errors.append(f"{ticker_state.ticker}: {ticker_state.error}")
            decision = ticker_state.portfolio_decision
            if decision is None or decision.final_action not in ("buy", "sell"):
                continue
            if decision.final_quantity <= 0:
                continue
            entry: dict[str, Any] = {
                "ticker": decision.ticker,
                "action": decision.final_action,
                "decided_quantity": decision.final_quantity,
                "rationale": decision.rationale,
            }
            report.decisions.append(entry)

            # Checked per order, after the agents have decided, so a kill
            # switch engaged mid-cycle stops every order not yet sent.
            check = check_order(
                self.guardrails,
                limits_state,
                decision,
                price=market_data[decision.ticker][-1].close,
                equity=balance.equity,
            )
            entry["sent_quantity"] = check.quantity
            if check.reason:
                entry["guardrail"] = check.reason
            if not check.allowed:
                entry["outcome"] = "blocked"
                continue

            # Counted before sending, so a send that fails after reaching
            # the broker still counts against the daily limit.
            limits_state.orders_sent += 1
            if self.mode == "dry-run":
                entry["outcome"] = "dry-run (not sent)"
                continue
            self.state_store.save(state)
            try:
                order = await self.broker.place_order(
                    OrderRequest(
                        ticker=decision.ticker,
                        side=decision.final_action,
                        quantity=check.quantity,
                        order_type="market",
                    )
                )
            except OrderRejectedError as exc:
                entry["outcome"] = f"rejected: {exc}"
                continue
            except BrokerAuthExpiredError as exc:
                entry["outcome"] = f"not sent: {exc}"
                report.errors.append(str(exc))
                break
            entry["outcome"] = order.status
            entry["order_id"] = order.order_id
            if order.fill_price is not None:
                entry["fill_price"] = order.fill_price

    async def _mark_prices(self, market_data: dict[str, list[MarketBar]]) -> None:
        mark = getattr(self.broker, "mark_price", None)
        if mark is None:
            return
        for ticker, bars in market_data.items():
            result = mark(ticker, bars[-1].close)
            if inspect.isawaitable(result):
                await result

    async def _todays_market_data(self, report: CycleReport) -> dict[str, list[MarketBar]]:
        """Bars through today for each ticker that has a bar dated today;
        tickers with stale data (or a failed fetch) are left out, so nothing
        trades on yesterday's price."""
        start = report.trading_day - timedelta(days=self.lookback_days)
        results = await asyncio.gather(
            *(
                self.data_provider.get_bars(ticker, start, report.trading_day, "1d")
                for ticker in self.universe
            ),
            return_exceptions=True,
        )
        market_data = {}
        for ticker, result in zip(self.universe, results, strict=True):
            if isinstance(result, BaseException):
                report.errors.append(f"{ticker}: market data failed: {result}")
            elif result and result[-1].timestamp.date() == report.trading_day:
                market_data[ticker] = result
        return market_data

    def _journal(self, report: CycleReport) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as journal:
            journal.write(report.to_json() + "\n")
        if report.skipped_reason:
            logger.warning("Cycle skipped: %s", report.skipped_reason)
        for entry in report.decisions:
            logger.info(
                "%s %s %s -> %s",
                entry["action"],
                entry["sent_quantity"],
                entry["ticker"],
                entry.get("outcome"),
            )
