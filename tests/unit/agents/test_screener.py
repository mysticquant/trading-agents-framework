from datetime import datetime

import pandas as pd

from trading_agents.agents.screener import (
    PrefilterCandidate,
    ScreenerAgent,
    ScreenerInput,
    ScreenerLLMOutput,
    ScreenerRanking,
    compute_prefilter_candidates,
)
from trading_agents.llm.fakes import FakeLLMClient


def test_compute_prefilter_candidates_ranks_by_combined_magnitude():
    bars_by_ticker = {
        "AAPL": pd.DataFrame({"close": [100.0, 101.0, 110.0], "volume": [1000.0, 1000.0, 5000.0]}),
        "MSFT": pd.DataFrame({"close": [200.0, 200.0, 200.5], "volume": [1000.0, 1000.0, 1000.0]}),
    }
    candidates = compute_prefilter_candidates(bars_by_ticker, top_n=5)
    assert candidates[0].ticker == "AAPL"


def test_compute_prefilter_candidates_skips_tickers_with_too_little_history():
    bars_by_ticker = {"AAPL": pd.DataFrame({"close": [100.0], "volume": [1000.0]})}
    assert compute_prefilter_candidates(bars_by_ticker) == []


async def test_screener_agent_wraps_llm_rankings_with_the_cycles_as_of():
    canned = ScreenerLLMOutput(
        rankings=[
            ScreenerRanking(ticker="AAPL", score=0.9, rationale="unusual volume", tags=["momentum"])
        ]
    )
    agent = ScreenerAgent(FakeLLMClient({"screener": canned}))
    as_of = datetime(2024, 1, 1)

    results = await agent.run(
        ScreenerInput(
            candidates=[PrefilterCandidate(ticker="AAPL", return_pct=5.0, volume_zscore=2.0)],
            as_of=as_of,
        )
    )

    assert results[0].ticker == "AAPL"
    assert results[0].screened_at == as_of


async def test_screener_agent_returns_empty_without_calling_the_llm():
    agent = ScreenerAgent(FakeLLMClient({}))
    results = await agent.run(ScreenerInput(candidates=[], as_of=datetime.now()))
    assert results == []
