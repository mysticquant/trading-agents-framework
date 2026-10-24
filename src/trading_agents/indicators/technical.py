"""Deterministic technical indicators, computed in pandas.

These are the only source of truth for indicator values in this framework
— the TechnicalAnalyst agent interprets numbers computed here, it never
computes them itself.
"""

import pandas as pd

from trading_agents.contracts.market import MarketBar


def bars_to_frame(bars: list[MarketBar]) -> pd.DataFrame:
    """Bars must already be sorted ascending by timestamp."""
    return pd.DataFrame(
        [
            {"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
            for b in bars
        ]
    )


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(close, window)
    std = close.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(window=window).mean()


def latest_snapshot(bars: pd.DataFrame) -> dict[str, float]:
    """Compute every indicator above over `bars` (columns: open, high, low,
    close, volume) and return only the most recent value of each, rounded
    for prompt-friendliness. Returns an empty dict if there isn't enough
    history yet for a given indicator (NaN values are dropped)."""
    close, high, low = bars["close"], bars["high"], bars["low"]
    macd_line, signal_line, histogram = macd(close)
    upper, middle, lower = bollinger_bands(close)

    values = {
        "sma_20": sma(close, 20).iloc[-1],
        "sma_50": sma(close, 50).iloc[-1],
        "ema_12": ema(close, 12).iloc[-1],
        "ema_26": ema(close, 26).iloc[-1],
        "rsi_14": rsi(close, 14).iloc[-1],
        "macd": macd_line.iloc[-1],
        "macd_signal": signal_line.iloc[-1],
        "macd_histogram": histogram.iloc[-1],
        "bollinger_upper": upper.iloc[-1],
        "bollinger_middle": middle.iloc[-1],
        "bollinger_lower": lower.iloc[-1],
        "atr_14": atr(high, low, close, 14).iloc[-1],
    }
    return {name: round(float(value), 4) for name, value in values.items() if pd.notna(value)}
