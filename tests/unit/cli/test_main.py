from trading_agents.cli.main import build_evidence_provider, build_orchestrator
from trading_agents.config.settings import Settings
from trading_agents.contracts.trading import ExecutionCosts
from trading_agents.data.evidence.sec_edgar import SecEdgarFilingsSource
from trading_agents.llm.config import DEFAULT_MODEL, HIGH_STAKES_MODEL, LLMRoleRegistry
from trading_agents.llm.fakes import FakeLLMClient


def _models_by_role(orchestrator) -> dict[str, str]:
    agents = [
        orchestrator.screener,
        orchestrator.fundamentals_analyst,
        orchestrator.sentiment_analyst,
        orchestrator.news_analyst,
        orchestrator.technical_analyst,
        orchestrator.researcher,
        orchestrator.trader,
        orchestrator.risk_manager,
        orchestrator.portfolio_manager,
    ]
    return {agent.role: agent._role_config.model for agent in agents}


def test_build_orchestrator_applies_role_overrides_to_every_agent():
    orchestrator = build_orchestrator(
        FakeLLMClient({}),
        role_registry=LLMRoleRegistry({"researcher": "cheap-model", "trader": "other-model"}),
    )

    models = _models_by_role(orchestrator)
    assert models["researcher"] == "cheap-model"
    assert models["trader"] == "other-model"
    # Roles without an override keep their tier defaults.
    assert models["portfolio_manager"] == HIGH_STAKES_MODEL
    assert models["screener"] == DEFAULT_MODEL


def test_sec_filings_are_enabled_only_when_a_user_agent_is_configured():
    without = build_evidence_provider(Settings(anthropic_api_key="k", _env_file=None))
    with_ua = build_evidence_provider(
        Settings(anthropic_api_key="k", sec_user_agent="Jane jane@example.com", _env_file=None)
    )

    assert without._filings is None
    assert isinstance(with_ua._filings, SecEdgarFilingsSource)


def test_build_orchestrator_passes_execution_costs_through():
    costs = ExecutionCosts(commission_bps=1.0, slippage_bps=2.0)

    orchestrator = build_orchestrator(FakeLLMClient({}), execution_costs=costs)

    assert orchestrator.execution_costs == costs
