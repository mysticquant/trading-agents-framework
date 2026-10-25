"""Runtime configuration, loaded once at the CLI entrypoint and passed down
via plain constructor arguments (no global singleton)."""

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ConfigurationError(ValueError):
    """A setting a command needs is missing or inconsistent."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Which LLMClient the CLI builds. "ollama" runs every agent on a local
    # Ollama server and needs no API key.
    llm_provider: Literal["anthropic", "ollama"] = "anthropic"
    # Required for LLM commands when llm_provider is "anthropic".
    anthropic_api_key: SecretStr | None = None

    ollama_base_url: str = "http://localhost:11434"
    # Model for every role unless LLM_ROLE_OVERRIDES says otherwise; must be
    # pulled first (`ollama pull <name>`).
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = 300.0
    ollama_max_concurrency: int = 2

    # `NoDecode` stops pydantic-settings from JSON-decoding the raw env value
    # before `_split_csv` sees it, so the documented `AAPL,MSFT,GOOGL` form works.
    ticker_universe: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL"]
    )
    backtest_start_date: date = date(2024, 1, 1)
    backtest_end_date: date = date(2024, 6, 1)
    initial_cash: float = 100_000.0

    data_cache_dir: Path = Path("data/cache")

    # Interactive Brokers (TWS / IB Gateway socket API). Paper TWS listens on
    # 7497 and paper Gateway on 4002; ib_account may be left empty when the
    # login manages a single account.
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    ib_account: str = ""
    # Price bars for the IB live loop: from IB itself (needs market-data
    # permissions) or from yfinance.
    ib_market_data: Literal["ib", "yfinance"] = "ib"

    # Zerodha Kite Connect. The access token comes from the daily login
    # (`trading-agents kite-login`) and expires at 6 AM IST the next day.
    kite_api_key: str | None = None
    kite_api_secret: SecretStr | None = None
    kite_access_token: SecretStr | None = None

    # Live trading loop (`trading-agents trade`). Real-money orders need BOTH
    # `--mode live` and allow_live_trading=true. The decision time is
    # exchange-local "HH:MM"; empty means 15:30 New York for IB and 15:00
    # Mumbai for Kite (both 30 minutes before the close).
    live_broker: Literal["ib", "kite", "execd"] = "ib"
    allow_live_trading: bool = False
    live_decision_time: str | None = None
    live_max_order_pct: float = Field(default=0.05, gt=0.0, le=1.0)
    live_max_orders_per_day: int = Field(default=10, ge=0)
    live_max_daily_drawdown_pct: float = Field(default=0.02, gt=0.0, le=1.0)
    live_kill_switch_path: Path = Path("KILL_SWITCH")
    live_state_dir: Path = Path("data/live")

    # The `execd` execution service (execution_service/, Rust) when
    # live_broker is "execd". execd_market picks the schedule, evidence, and
    # fallback data: "nse" (Kite symbols) or "us".
    execd_host: str = "127.0.0.1"
    execd_port: int = 7600
    execd_token: SecretStr | None = None
    execd_market: Literal["nse", "us"] = "nse"

    # SEC EDGAR filings (point-in-time news evidence) are fetched only when
    # this is set; SEC requires it to name you with a contact email.
    sec_user_agent: str | None = None

    llm_role_overrides: dict[str, str] | None = None

    @field_validator("ticker_universe", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            if value.strip().startswith("["):
                return json.loads(value)
            return [ticker.strip() for ticker in value.split(",") if ticker.strip()]
        return value

    @field_validator("llm_role_overrides", mode="before")
    @classmethod
    def _parse_json(cls, value: object) -> object:
        if isinstance(value, str) and value.strip():
            return json.loads(value)
        return value or None
