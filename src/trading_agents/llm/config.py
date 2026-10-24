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
    """Resolves a `RoleModelConfig` per agent role, honoring optional overrides."""

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self._overrides = overrides or {}

    def for_role(self, role: str) -> RoleModelConfig:
        model = self._overrides.get(
            role, HIGH_STAKES_MODEL if role in HIGH_STAKES_ROLES else DEFAULT_MODEL
        )
        prompt_path = PROMPTS_DIR / f"{role}.md"
        return RoleModelConfig(role=role, model=model, system_prompt_path=prompt_path)
