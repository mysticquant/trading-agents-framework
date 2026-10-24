"""Base class shared by every agent role.

Subclasses implement `run()`, translating their typed input into a prompt
and back into a typed output. The constructor only takes an `LLMClient` and
an optional `RoleModelConfig` — plain constructor-argument injection, so
tests can swap in `FakeLLMClient` without any framework magic.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from trading_agents.llm.client import LLMClient
from trading_agents.llm.config import LLMRoleRegistry, RoleModelConfig

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")
TLLMOutput = TypeVar("TLLMOutput", bound=BaseModel)

_DEFAULT_REGISTRY = LLMRoleRegistry()


class Agent(ABC, Generic[TInput, TOutput]):
    role: ClassVar[str]

    def __init__(self, llm_client: LLMClient, role_config: RoleModelConfig | None = None) -> None:
        self._llm_client = llm_client
        self._role_config = role_config or _DEFAULT_REGISTRY.for_role(self.role)

    @abstractmethod
    async def run(self, agent_input: TInput) -> TOutput: ...

    async def _generate(self, *, user_content: str, output_model: type[TLLMOutput]) -> TLLMOutput:
        return await self._llm_client.generate_structured(
            role=self.role,
            system_prompt=self._role_config.system_prompt,
            user_content=user_content,
            output_model=output_model,
            model=self._role_config.model,
            max_tokens=self._role_config.max_tokens,
        )
