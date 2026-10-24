"""Runtime configuration, loaded once at the CLI entrypoint and passed down
via plain constructor arguments (no global singleton)."""

import json
from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: SecretStr

    # `NoDecode` stops pydantic-settings from JSON-decoding the raw env value
    # before `_split_csv` sees it, so the documented `AAPL,MSFT,GOOGL` form works.
    ticker_universe: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL"]
    )
    backtest_start_date: date = date(2024, 1, 1)
    backtest_end_date: date = date(2024, 6, 1)
    initial_cash: float = 100_000.0

    data_cache_dir: Path = Path("data/cache")

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
