"""Opt-in integration test against a real local Ollama server. Skipped by
default (see `addopts` in pyproject.toml); run explicitly with:

    pytest -m live_ollama

Requires Ollama running locally with the configured model pulled
(`ollama pull llama3.1:8b`, or whatever OLLAMA_MODEL is set to). Free, but
slow on CPU-only machines.
"""

from datetime import datetime

import pytest

from trading_agents.agents.screener import PrefilterCandidate, ScreenerAgent, ScreenerInput
from trading_agents.cli.main import build_llm
from trading_agents.config.settings import Settings

pytestmark = pytest.mark.live_ollama


async def test_screener_agent_against_a_local_ollama_model():
    llm_client, registry = build_llm(Settings(llm_provider="ollama"))
    agent = ScreenerAgent(llm_client, registry.for_role(ScreenerAgent.role))

    results = await agent.run(
        ScreenerInput(
            candidates=[
                PrefilterCandidate(ticker="AAPL", return_pct=4.2, volume_zscore=2.1),
                PrefilterCandidate(ticker="MSFT", return_pct=0.1, volume_zscore=0.0),
            ],
            as_of=datetime.now(),
        )
    )

    assert results
    assert all(result.ticker in {"AAPL", "MSFT"} for result in results)
