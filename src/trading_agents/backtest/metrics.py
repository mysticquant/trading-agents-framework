"""Basic backtest performance stats computed from an equity curve and trade
log. Intentionally simple: daily Sharpe with a 0% risk-free rate, peak-to-
trough max drawdown, and an average-cost-basis win rate — no tick-level
attribution or risk-free-rate configuration.
"""

import math

from trading_agents.contracts.backtest import EquityCurvePoint
from trading_agents.contracts.broker import Order

TRADING_DAYS_PER_YEAR = 252


def total_return_pct(equity_curve: list[EquityCurvePoint]) -> float:
    if len(equity_curve) < 2 or equity_curve[0].equity == 0:
        return 0.0
    return (equity_curve[-1].equity / equity_curve[0].equity - 1) * 100


def sharpe_ratio(equity_curve: list[EquityCurvePoint]) -> float:
    """Annualized Sharpe over daily equity returns, risk-free rate = 0
    (a documented simplification for this phase)."""
    if len(equity_curve) < 3:
        return 0.0
    returns = [
        equity_curve[i].equity / equity_curve[i - 1].equity - 1
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1].equity != 0
    ]
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_return = math.sqrt(variance)
    if std_return == 0:
        return 0.0
    return (mean_return / std_return) * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown_pct(equity_curve: list[EquityCurvePoint]) -> float:
    """Returns a non-positive percentage (e.g. -12.5 for a 12.5% drawdown)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0].equity
    worst_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point.equity)
        if peak > 0:
            drawdown = (point.equity - peak) / peak
            worst_drawdown = min(worst_drawdown, drawdown)
    return worst_drawdown * 100


def win_rate(trade_log: list[Order]) -> float:
    """Fraction of sell fills whose fill price exceeded the running
    average cost basis for that ticker (average-cost, not FIFO lots)."""
    avg_cost: dict[str, float] = {}
    held_qty: dict[str, float] = {}
    wins = 0
    sells = 0

    for order in trade_log:
        if order.status != "filled" or order.fill_price is None:
            continue
        if order.side == "buy":
            prev_qty = held_qty.get(order.ticker, 0.0)
            prev_cost = avg_cost.get(order.ticker, 0.0)
            new_qty = prev_qty + order.quantity
            if new_qty > 0:
                avg_cost[order.ticker] = (
                    prev_qty * prev_cost + order.quantity * order.fill_price
                ) / new_qty
            held_qty[order.ticker] = new_qty
        else:
            sells += 1
            if order.fill_price > avg_cost.get(order.ticker, 0.0):
                wins += 1
            held_qty[order.ticker] = held_qty.get(order.ticker, 0.0) - order.quantity

    return wins / sells if sells else 0.0
