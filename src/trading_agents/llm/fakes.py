"""Test double for `LLMClient` — returns pre-canned pydantic instances keyed
by role, with no network access. Used throughout agent/orchestrator tests."""

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient:
    def __init__(self, responses: dict[str, BaseModel]) -> None:
        """`responses` maps role -> the single canned instance to return for
        every call made with that role. Mutate `self.responses` between
        calls in a test to vary what's returned across a sequence."""
        self.responses = responses
        self.calls: list[dict] = []

    async def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_content: str,
        output_model: type[T],
        model: str,
        max_tokens: int = 2048,
        model_override: str | None = None,
    ) -> T:
        self.calls.append({"role": role, "user_content": user_content, "model": model})
        try:
            canned = self.responses[role]
        except KeyError as exc:
            raise KeyError(f"FakeLLMClient has no canned response for role '{role}'.") from exc
        if not isinstance(canned, output_model):
            raise TypeError(
                f"Canned response for role '{role}' is {type(canned).__name__}, "
                f"expected {output_model.__name__}."
            )
        return canned
