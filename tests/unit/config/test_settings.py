import pytest

from trading_agents.config.settings import Settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    # Keep a developer's real .env / environment out of these tests.
    monkeypatch.chdir(tmp_path)
    for var in ("TICKER_UNIVERSE", "LLM_ROLE_OVERRIDES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_ticker_universe_parses_the_documented_csv_form_from_the_environment(monkeypatch):
    monkeypatch.setenv("TICKER_UNIVERSE", "AAPL, MSFT,GOOGL")

    assert Settings().ticker_universe == ["AAPL", "MSFT", "GOOGL"]


def test_ticker_universe_parses_the_csv_form_from_a_dotenv_file(tmp_path):
    # Mirrors `copy .env.example .env` from the README.
    (tmp_path / ".env").write_text("TICKER_UNIVERSE=AAPL,MSFT,GOOGL\n", encoding="utf-8")

    assert Settings().ticker_universe == ["AAPL", "MSFT", "GOOGL"]


def test_ticker_universe_also_accepts_a_json_list(monkeypatch):
    monkeypatch.setenv("TICKER_UNIVERSE", '["AAPL", "MSFT"]')

    assert Settings().ticker_universe == ["AAPL", "MSFT"]


def test_ticker_universe_csv_override_from_the_cli():
    assert Settings(ticker_universe="NVDA,AMD").ticker_universe == ["NVDA", "AMD"]


def test_llm_role_overrides_parse_from_json(monkeypatch):
    monkeypatch.setenv("LLM_ROLE_OVERRIDES", '{"researcher": "claude-sonnet-5"}')

    assert Settings().llm_role_overrides == {"researcher": "claude-sonnet-5"}
