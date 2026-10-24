from datetime import datetime

import pytest

from trading_agents.agents.analysts.base import AnalystInput, AnalystLLMOutput
from trading_agents.agents.analysts.fundamentals import FundamentalsAnalystAgent
from trading_agents.agents.analysts.news import NewsAnalystAgent
from trading_agents.agents.analysts.sentiment import SentimentAnalystAgent
from trading_agents.agents.analysts.technical import TechnicalAnalystAgent, TechnicalAnalystInput
from trading_agents.indicators.technical import bars_to_frame, latest_snapshot
from trading_agents.llm.fakes import FakeLLMClient


@pytest.mark.parametrize(
    ("agent_cls", "role", "expected_analyst_type"),
    [
        (FundamentalsAnalystAgent, "fundamentals_analyst", "fundamentals"),
        (SentimentAnalystAgent, "sentiment_analyst", "sentiment"),
        (NewsAnalystAgent, "news_analyst", "news"),
    ],
)
async def test_evidence_analyst_wraps_llm_output_into_a_report(
    agent_cls, role, expected_analyst_type
):
    canned = AnalystLLMOutput(
        summary="Strong signal",
        signal="bullish",
        confidence=0.8,
        key_points=["x"],
        metrics={"m": 1.0},
    )
    agent = agent_cls(FakeLLMClient({role: canned}))

    report = await agent.run(
        AnalystInput(ticker="AAPL", as_of=datetime(2024, 1, 1), evidence={"m": 1.0})
    )

    assert report.analyst_type == expected_analyst_type
    assert report.signal == "bullish"
    assert report.metrics == {"m": 1.0}


async def test_technical_analyst_computes_indicators_and_merges_them_into_metrics(bars_factory):
    canned = AnalystLLMOutput(
        summary="Overbought", signal="bearish", confidence=0.6, key_points=["RSI high"]
    )
    agent = TechnicalAnalystAgent(FakeLLMClient({"technical_analyst": canned}))
    bars = bars_factory("AAPL", 60)

    report = await agent.run(
        TechnicalAnalystInput(ticker="AAPL", as_of=datetime(2024, 1, 1), bars=bars)
    )

    assert report.analyst_type == "technical"
    assert report.signal == "bearish"
    assert "sma_20" in report.metrics
    assert "rsi_14" in report.metrics


async def test_technical_analyst_llm_echo_never_overrides_computed_indicators(bars_factory):
    bars = bars_factory("AAPL", 60)
    computed = latest_snapshot(bars_to_frame(bars))
    canned = AnalystLLMOutput(
        summary="Overbought",
        signal="bearish",
        confidence=0.6,
        metrics={"rsi_14": 999.0, "sma_20": -1.0, "llm_only_note": 1.0},
    )
    agent = TechnicalAnalystAgent(FakeLLMClient({"technical_analyst": canned}))

    report = await agent.run(
        TechnicalAnalystInput(ticker="AAPL", as_of=datetime(2024, 1, 1), bars=bars)
    )

    assert report.metrics["rsi_14"] == computed["rsi_14"]
    assert report.metrics["sma_20"] == computed["sma_20"]
    # Keys only the LLM supplied are still kept.
    assert report.metrics["llm_only_note"] == 1.0
