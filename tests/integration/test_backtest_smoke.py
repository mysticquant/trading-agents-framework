"""Short, network-free backtest smoke test: a fake LLM client and a
synthetic (non-yfinance) data provider drive the full
screener -> analysts -> researcher -> trader -> risk manager ->
portfolio manager -> paper broker pipeline end to end."""

from datetime import date, datetime, timedelta

from trading_agents.agents.analysts.base import AnalystLLMOutput
from trading_agents.agents.analysts.fundamentals import FundamentalsAnalystAgent
from trading_agents.agents.analysts.news import NewsAnalystAgent
from trading_agents.agents.analysts.sentiment import SentimentAnalystAgent
from trading_agents.agents.analysts.technical import TechnicalAnalystAgent
from trading_agents.agents.portfolio_manager import (
    PortfolioManagerAgent,
    PortfolioManagerDecisionLLM,
    PortfolioManagerLLMOutput,
)
from trading_agents.agents.researcher import ResearcherAgent, ResearcherLLMOutput
from trading_agents.agents.risk_manager import RiskManagerAgent, RiskManagerLLMOutput
from trading_agents.agents.screener import ScreenerAgent, ScreenerLLMOutput, ScreenerRanking
from trading_agents.agents.trader import TraderAgent, TraderLLMOutput
from trading_agents.backtest.runner import BacktestRunner
from trading_agents.brokers.paper import PaperBrokerAdapter
from trading_agents.contracts.market import MarketBar
from trading_agents.llm.fakes import FakeLLMClient
from trading_agents.orchestration.pipeline import DecisionCycleOrchestrator


class _SyntheticDataProvider:
    """Deterministic, network-free stand-in for `YFinanceProvider`: a daily
    upward-drifting price series for every calendar day in the requested
    range."""

    async def get_bars(
        self, ticker: str, start: date, end: date, timeframe: str
    ) -> list[MarketBar]:
        bars = []
        price = 100.0
        current = start
        while current <= end:
            price += 0.5
            bars.append(
                MarketBar(
                    ticker=ticker,
                    timestamp=datetime.combine(current, datetime.min.time()),
                    open=price - 0.2,
                    high=price + 0.5,
                    low=price - 0.5,
                    close=price,
                    volume=1_000_000,
                    timeframe=timeframe,
                )
            )
            current += timedelta(days=1)
        return bars


def _canned_responses() -> dict:
    return {
        "screener": ScreenerLLMOutput(
            rankings=[ScreenerRanking(ticker="AAPL", score=0.9, rationale="momentum")]
        ),
        "fundamentals_analyst": AnalystLLMOutput(summary="ok", signal="bullish", confidence=0.6),
        "sentiment_analyst": AnalystLLMOutput(summary="ok", signal="bullish", confidence=0.6),
        "news_analyst": AnalystLLMOutput(summary="ok", signal="neutral", confidence=0.5),
        "technical_analyst": AnalystLLMOutput(summary="ok", signal="bullish", confidence=0.7),
        "researcher": ResearcherLLMOutput(
            bull_thesis="b",
            bear_thesis="c",
            synthesis="d",
            recommended_action="buy",
            conviction=0.7,
        ),
        "trader": TraderLLMOutput(
            action="buy", quantity=1.0, order_type="market", rationale="go", time_horizon="1w"
        ),
        "risk_manager": RiskManagerLLMOutput(
            llm_approved=True, max_position_size_pct=0.5, adjusted_quantity=1.0, rationale="ok"
        ),
        "portfolio_manager": PortfolioManagerLLMOutput(
            decisions=[
                PortfolioManagerDecisionLLM(
                    ticker="AAPL", final_action="buy", final_quantity=1.0, rationale="go"
                )
            ]
        ),
    }


async def test_backtest_runner_produces_a_sane_result():
    llm_client = FakeLLMClient(_canned_responses())
    orchestrator = DecisionCycleOrchestrator(
        screener=ScreenerAgent(llm_client),
        fundamentals_analyst=FundamentalsAnalystAgent(llm_client),
        sentiment_analyst=SentimentAnalystAgent(llm_client),
        news_analyst=NewsAnalystAgent(llm_client),
        technical_analyst=TechnicalAnalystAgent(llm_client),
        researcher=ResearcherAgent(llm_client),
        trader=TraderAgent(llm_client),
        risk_manager=RiskManagerAgent(llm_client),
        portfolio_manager=PortfolioManagerAgent(llm_client),
    )
    data_provider = _SyntheticDataProvider()
    broker = PaperBrokerAdapter(initial_cash=100_000.0, historical_provider=data_provider)
    await broker.connect()

    runner = BacktestRunner(orchestrator=orchestrator, broker=broker, data_provider=data_provider)
    result = await runner.run(
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 10), universe=["AAPL"]
    )

    assert len(result.equity_curve) > 0
    assert len(result.trade_log) > 0
    assert result.final_portfolio_state.equity > 0
    assert result.ticker_universe == ["AAPL"]
    assert result.cycle_errors == []


class _ScreenerFailsOnceLLMClient(FakeLLMClient):
    """The screener call on the Nth cycle raises, as a rate-limit error
    surviving the SDK's retries would."""

    def __init__(self, responses, *, fail_on_call: int) -> None:
        super().__init__(responses)
        self._fail_on_call = fail_on_call
        self._screener_calls = 0

    async def generate_structured(self, **kwargs):
        if kwargs["role"] == "screener":
            self._screener_calls += 1
            if self._screener_calls == self._fail_on_call:
                raise RuntimeError("simulated rate limit")
        return await super().generate_structured(**kwargs)


async def test_backtest_survives_a_failed_decision_cycle():
    llm_client = _ScreenerFailsOnceLLMClient(_canned_responses(), fail_on_call=3)
    orchestrator = DecisionCycleOrchestrator(
        screener=ScreenerAgent(llm_client),
        fundamentals_analyst=FundamentalsAnalystAgent(llm_client),
        sentiment_analyst=SentimentAnalystAgent(llm_client),
        news_analyst=NewsAnalystAgent(llm_client),
        technical_analyst=TechnicalAnalystAgent(llm_client),
        researcher=ResearcherAgent(llm_client),
        trader=TraderAgent(llm_client),
        risk_manager=RiskManagerAgent(llm_client),
        portfolio_manager=PortfolioManagerAgent(llm_client),
    )
    data_provider = _SyntheticDataProvider()
    broker = PaperBrokerAdapter(initial_cash=100_000.0, historical_provider=data_provider)
    await broker.connect()

    runner = BacktestRunner(orchestrator=orchestrator, broker=broker, data_provider=data_provider)
    result = await runner.run(
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 10), universe=["AAPL"]
    )

    # Every day still gets an equity point; only the failed day is skipped.
    assert len(result.equity_curve) == 10
    assert len(result.cycle_errors) == 1
    assert result.cycle_errors[0].startswith("2024-01-03")
    assert "simulated rate limit" in result.cycle_errors[0]
    assert len(result.trade_log) == 9
