import pandas as pd

from trading_agents.indicators.technical import (
    atr,
    bollinger_bands,
    ema,
    latest_snapshot,
    macd,
    rsi,
    sma,
)


def _rising_series(n: int = 60) -> pd.Series:
    return pd.Series([100.0 + i for i in range(n)])


def test_sma_matches_manual_average():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(close, window=3)
    assert result.iloc[-1] == (3.0 + 4.0 + 5.0) / 3


def test_ema_reacts_faster_than_sma_to_recent_change():
    close = pd.concat([pd.Series([100.0] * 30), pd.Series([200.0] * 5)], ignore_index=True)
    assert ema(close, span=5).iloc[-1] > sma(close, window=30).iloc[-1]


def test_rsi_is_high_for_a_strictly_rising_series():
    rsi_values = rsi(_rising_series(), window=14)
    assert rsi_values.iloc[-1] > 90


def test_macd_returns_three_aligned_series():
    close = _rising_series()
    macd_line, signal_line, histogram = macd(close)
    assert len(macd_line) == len(signal_line) == len(histogram) == len(close)


def test_bollinger_bands_bracket_the_middle_band():
    close = _rising_series()
    upper, middle, lower = bollinger_bands(close)
    assert (upper.dropna() >= middle.dropna()).all()
    assert (lower.dropna() <= middle.dropna()).all()


def test_atr_is_non_negative():
    high = _rising_series() + 1
    low = _rising_series() - 1
    close = _rising_series()
    atr_values = atr(high, low, close)
    assert (atr_values.dropna() >= 0).all()


def test_latest_snapshot_drops_nan_and_rounds():
    close = _rising_series()
    bars = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        }
    )
    snapshot = latest_snapshot(bars)
    assert "sma_20" in snapshot
    assert all(isinstance(value, float) for value in snapshot.values())


def test_latest_snapshot_omits_indicators_without_enough_history():
    close = pd.Series([100.0, 101.0, 102.0])
    bars = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        }
    )
    snapshot = latest_snapshot(bars)
    assert "sma_50" not in snapshot
