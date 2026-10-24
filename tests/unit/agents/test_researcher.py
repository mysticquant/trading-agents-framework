from datetime import datetime

from trading_agents.agents.researcher import ResearcherAgent, ResearcherInput, ResearcherLLMOutput
from trading_agents.contracts.analysis import AnalystReport
from trading_agents.llm.fakes import FakeLLMClient


async def test_researcher_echoes_supporting_reports_and_applies_llm_verdict():
    report = AnalystReport(
        ticker="AAPL",
        analyst_type="technical",
        as_of=datetime(2024, 1, 1),
        summary="overbought",
        signal="bearish",
        confidence=0.7,
    )
    canned = ResearcherLLMOutput(
        bull_thesis="b", bear_thesis="c", synthesis="d", recommended_action="buy", conviction=0.8
    )
    agent = ResearcherAgent(FakeLLMClient({"researcher": canned}))

    result = await agent.run(ResearcherInput(ticker="AAPL", analyst_reports=[report]))

    assert result.recommended_action == "buy"
    assert result.supporting_reports == [report]
