from datetime import date

import pandas as pd

from trading_agents.data.providers import yfinance_provider
from trading_agents.data.providers.yfinance_provider import YFinanceProvider


async def test_requested_end_date_is_inclusive(monkeypatch):
    # yfinance treats `end` as exclusive; the provider Protocol (and the
    # backtest runner) treat it as inclusive, so the provider must ask for
    # one extra day or the last day's bar is silently dropped.
    captured = {}

    def fake_download(ticker, **kwargs):
        captured.update(kwargs)
        frame = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-05-31")]),
        )
        return frame

    monkeypatch.setattr(yfinance_provider.yf, "download", fake_download)

    bars = await YFinanceProvider().get_bars("AAPL", date(2024, 5, 1), date(2024, 5, 31), "1d")

    assert captured["start"] == date(2024, 5, 1)
    assert captured["end"] == date(2024, 6, 1)
    assert [bar.timestamp.date() for bar in bars] == [date(2024, 5, 31)]
