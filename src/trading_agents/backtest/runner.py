"""Drives one `DecisionCycleOrchestrator` day-by-day over a date range and
ticker universe, submitting its decisions to a `PaperBrokerAdapter` and
recording the resulting equity curve and trade log.

Callers are responsible for the broker's connect/disconnect lifecycle (this
runner only calls `mark_price`/`place_order`/`get_account_balance`/
`get_positions`) — that mirrors how a live/paper loop would own the same
broker instance across many backtest or trading sessions.
"""

import asyncio
from datetime import date, datetime, time, timedelta

from trading_agents.backtest import metrics
from trading_agents.brokers.base import OrderRejectedError
from trading_agents.brokers.paper import PaperBrokerAdapter
from trading_agents.contracts.backtest import BacktestResult, EquityCurvePoint
from trading_agents.contracts.broker import Order, OrderRequest
from trading_agents.contracts.portfolio import PortfolioState
from trading_agents.data.providers.base import HistoricalDataProvider
from trading_agents.orchestration.pipeline import DecisionCycleOrchestrator


class BacktestRunner:
    def __init__(
        self,
        orchestrator: DecisionCycleOrchestrator,
        broker: PaperBrokerAdapter,
        data_provider: HistoricalDataProvider,
    ) -> None:
        self.orchestrator = orchestrator
        self.broker = broker
        self.data_provider = data_provider

    async def run(
        self,
        *,
        start_date: date,
        end_date: date,
        universe: list[str],
        lookback_days: int = 150,
    ) -> BacktestResult:
        """`lookback_days` of extra history is fetched before `start_date` so
        indicators (e.g. SMA-50) have enough data to be non-NaN from day
        one; it's a calendar-day buffer, not an exact trading-day count, so
        it slightly over-fetches — a deliberate, documented simplification."""
        lookback_start = start_date - timedelta(days=lookback_days)
        fetched = await asyncio.gather(
            *(
                self.data_provider.get_bars(ticker, lookback_start, end_date, "1d")
                for ticker in universe
            )
        )
        bars_by_ticker = dict(zip(universe, fetched, strict=True))

        trading_days = sorted(
            {
                bar.timestamp.date()
                for bars in bars_by_ticker.values()
                for bar in bars
                if start_date <= bar.timestamp.date() <= end_date
            }
        )

        equity_curve: list[EquityCurvePoint] = []
        trade_log: list[Order] = []

        for day in trading_days:
            as_of = datetime.combine(day, time.min)

            market_data_window = {
                ticker: [bar for bar in bars if bar.timestamp.date() <= day]
                for ticker, bars in bars_by_ticker.items()
            }
            for ticker, bars in market_data_window.items():
                if bars:
                    self.broker.mark_price(ticker, bars[-1].close)

            portfolio_state = await self._portfolio_state(as_of)
            cycle_state = await self.orchestrator.run_cycle(
                as_of=as_of,
                universe=universe,
                market_data=market_data_window,
                portfolio_state=portfolio_state,
            )

            for ticker_state in cycle_state.per_ticker.values():
                decision = ticker_state.portfolio_decision
                if (
                    decision is None
                    or decision.final_action not in ("buy", "sell")
                    or decision.final_quantity <= 0
                ):
                    continue
                try:
                    order = await self.broker.place_order(
                        OrderRequest(
                            ticker=decision.ticker,
                            side=decision.final_action,
                            quantity=decision.final_quantity,
                            order_type="market",
                        )
                    )
                    trade_log.append(order)
                except OrderRejectedError:
                    # Safety net only: the Portfolio Manager already enforces
                    # cash feasibility, but slippage/commission on fill can
                    # still push a trade just over the edge.
                    continue

            balance = await self.broker.get_account_balance()
            equity_curve.append(
                EquityCurvePoint(as_of=as_of, equity=balance.equity, cash=balance.cash)
            )

        final_portfolio_state = await self._portfolio_state(datetime.combine(end_date, time.min))

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            ticker_universe=universe,
            equity_curve=equity_curve,
            trade_log=trade_log,
            total_return_pct=metrics.total_return_pct(equity_curve),
            sharpe_ratio=metrics.sharpe_ratio(equity_curve),
            max_drawdown_pct=metrics.max_drawdown_pct(equity_curve),
            win_rate=metrics.win_rate(trade_log),
            final_portfolio_state=final_portfolio_state,
        )

    async def _portfolio_state(self, as_of: datetime) -> PortfolioState:
        balance = await self.broker.get_account_balance()
        positions = await self.broker.get_positions()
        return PortfolioState(
            cash=balance.cash,
            positions=positions,
            equity=balance.equity,
            realized_pnl=self.broker.realized_pnl,
            as_of=as_of,
        )
