"""Runtime configuration, loaded once at the CLI entrypoint and passed down
via plain constructor arguments (no global singleton)."""

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Which LLMClient the CLI builds. "ollama" runs every agent on a local
    # Ollama server and needs no API key.
    llm_provider: Literal["anthropic", "ollama"] = "anthropic"
    # Required when llm_provider is "anthropic".
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

    @model_validator(mode="after")
    def _require_key_for_anthropic(self) -> "Settings":
        if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic' "
                "(set it in .env, or set LLM_PROVIDER=ollama to use a local model)."
            )
        return self

    @field_validator("llm_role_overrides", mode="before")
    @classmethod
    def _parse_json(cls, value: object) -> object:
        if isinstance(value, str) and value.strip():
            return json.loads(value)
        return value or None
