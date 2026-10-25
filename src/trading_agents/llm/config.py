"""Per-agent-role LLM configuration."""

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_MODEL = "claude-sonnet-5"
HIGH_STAKES_MODEL = "claude-opus-5"

# Researcher and PortfolioManager are the lowest-frequency, highest-stakes
# calls in a decision cycle, so they default to the premium tier; every
# other (high-volume, per-ticker) role defaults to the cheaper/faster one.
HIGH_STAKES_ROLES = {"researcher", "portfolio_manager"}


@dataclass(frozen=True)
class RoleModelConfig:
    role: str
    model: str
    max_tokens: int = 2048
    system_prompt_path: Path | None = None

    @property
    def system_prompt(self) -> str:
        if self.system_prompt_path is None:
            raise ValueError(f"No system_prompt_path configured for role '{self.role}'.")
        return self.system_prompt_path.read_text(encoding="utf-8")


class LLMRoleRegistry:
    """Resolves a `RoleModelConfig` per agent role, honoring optional overrides.
    `default_model`/`high_stakes_model` set the two tiers (Claude models
    unless given, e.g. Ollama model names for a local provider)."""

    def __init__(
        self,
        overrides: dict[str, str] | None = None,
        *,
        default_model: str = DEFAULT_MODEL,
        high_stakes_model: str = HIGH_STAKES_MODEL,
    ) -> None:
        self._overrides = overrides or {}
        self._default_model = default_model
        self._high_stakes_model = high_stakes_model

    def for_role(self, role: str) -> RoleModelConfig:
        model = self._overrides.get(
            role, self._high_stakes_model if role in HIGH_STAKES_ROLES else self._default_model
        )
        prompt_path = PROMPTS_DIR / f"{role}.md"
        return RoleModelConfig(role=role, model=model, system_prompt_path=prompt_path)
