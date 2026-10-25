"""Opt-in integration test against the real Anthropic API. Skipped by
default (see `addopts` in pyproject.toml); run explicitly with:

    pytest -m live_llm

Requires a valid ANTHROPIC_API_KEY in the environment/.env. This makes a
real, billed API call every time it runs — never add it to default CI.
"""

from datetime import datetime

import pytest

from trading_agents.agents.screener import PrefilterCandidate, ScreenerAgent, ScreenerInput
from trading_agents.config.settings import Settings
from trading_agents.llm.client import AnthropicLLMClient

pytestmark = pytest.mark.live_llm


async def test_screener_agent_against_the_real_anthropic_api():
    settings = Settings(llm_provider="anthropic")
    assert settings.anthropic_api_key is not None
    llm_client = AnthropicLLMClient(api_key=settings.anthropic_api_key.get_secret_value())
    agent = ScreenerAgent(llm_client)

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
