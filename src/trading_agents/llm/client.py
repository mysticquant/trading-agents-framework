"""LLM client abstraction.

Every agent talks to this `LLMClient` Protocol, never to the Anthropic SDK
directly — that's what lets `FakeLLMClient` (see fakes.py) stand in for
tests without any network access.
"""

import logging
from typing import Protocol, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from trading_agents.llm.structured_output import TOOL_NAME, model_to_tool_schema, parse_tool_input

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
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
    ) -> T: ...


class AnthropicLLMClient:
    """Wraps `anthropic.AsyncAnthropic`, forcing structured (tool-use) output
    so every agent gets back a validated pydantic instance, never raw text."""

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

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
        tool = model_to_tool_schema(output_model)
        # The SDK's TypedDict param types don't structurally match plain
        # dict literals under mypy even though they're valid at runtime;
        # this wrapper is covered by FakeLLMClient everywhere except the
        # opt-in live-LLM smoke test, so a narrow ignore here is cheaper
        # than retyping every request body against the SDK's TypedDicts.
        response = await self._client.messages.create(  # type: ignore[call-overload]
            model=model_override or model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )

        usage = response.usage
        logger.info(
            "llm_call role=%s model=%s input_tokens=%s cache_read_tokens=%s output_tokens=%s",
            role,
            model_override or model,
            usage.input_tokens,
            getattr(usage, "cache_read_input_tokens", 0),
            usage.output_tokens,
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == TOOL_NAME:
                return parse_tool_input(block.input, output_model)

        raise ValueError(f"No '{TOOL_NAME}' tool_use block in response for role '{role}'.")
