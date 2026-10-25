"""`LLMClient` backed by a local Ollama server (https://ollama.com).

Uses Ollama's `/api/chat` endpoint with `format` set to the output model's
JSON schema, so the server constrains generation to schema-shaped JSON.
Local models are less reliable than hosted ones, so this client also:

- states the schema in the system prompt (Ollama recommends grounding the
  model this way in addition to `format`),
- retries once, feeding the validation error back, when the JSON doesn't
  parse or fails the pydantic model's constraints,
- caps in-flight requests (a decision cycle can fire ~20 calls at once,
  which a local server would otherwise queue until they time out),
- raises a clear error when output was cut off by the token limit.
"""

import asyncio
import contextlib
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from trading_agents.llm.structured_output import TOOL_NAME, model_to_tool_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

JsonTransport = Callable[[str, dict[str, Any] | None], Awaitable[dict[str, Any]]]

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


def _schema_instructions(schema: dict[str, Any]) -> str:
    # The role prompts end with "Emit your result using the `emit_result`
    # tool"; Ollama has no tool here, so say what to do instead.
    return (
        f"\n\nOutput format: there is no `{TOOL_NAME}` tool in this session. "
        "Instead, reply with a single JSON object and nothing else (no prose, "
        "no code fences) that matches this JSON schema exactly:\n"
        f"{json.dumps(schema)}"
    )


class OllamaLLMClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 300.0,
        max_concurrency: int = 2,
        max_repair_attempts: int = 1,
        transport: JsonTransport | None = None,
    ) -> None:
        """`transport(url, body)` POSTs `body` as JSON (GET when `body` is
        None) and returns the decoded response; injectable for tests."""
        base = base_url.rstrip("/")
        self._chat_url = base + "/api/chat"
        self._tags_url = base + "/api/tags"
        self._timeout = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_repair_attempts = max_repair_attempts
        self._transport = transport or self._http_json

    async def check_models(self, models: set[str]) -> None:
        """Fail fast, before a long run, if the server is unreachable or any
        of `models` hasn't been pulled."""
        response = await self._transport(self._tags_url, None)
        available = {str(m.get("name", "")) for m in response.get("models", [])}

        def pulled(name: str) -> bool:
            # `ollama pull llama3.1` is stored as "llama3.1:latest".
            return name in available or (":" not in name and f"{name}:latest" in available)

        missing = sorted(name for name in models if not pulled(name))
        if missing:
            commands = "; ".join(f"ollama pull {name}" for name in missing)
            raise OllamaError(f"Ollama model(s) not pulled: {', '.join(missing)}. Run: {commands}")

    async def _http_json(self, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        def send() -> dict[str, Any]:
            request = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8") if body is not None else None,
                headers={"Content-Type": "application/json"},
                method="POST" if body is not None else "GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                with contextlib.suppress(json.JSONDecodeError, AttributeError):
                    detail = json.loads(detail).get("error", detail)
                raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise OllamaError(
                    f"Could not reach Ollama at {url} ({exc.reason}). "
                    "Is it installed and running (`ollama serve`)?"
                ) from exc

        return await asyncio.to_thread(send)

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
        model_name = model_override or model
        schema = model_to_tool_schema(output_model)["input_schema"]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt + _schema_instructions(schema)},
            {"role": "user", "content": user_content},
        ]

        for attempt in range(self._max_repair_attempts + 1):
            async with self._semaphore:
                response = await self._transport(
                    self._chat_url,
                    {
                        "model": model_name,
                        "messages": messages,
                        "stream": False,
                        "format": schema,
                        "options": {"num_predict": max_tokens, "temperature": 0},
                    },
                )
            if "error" in response:
                raise OllamaError(f"Ollama error for role '{role}': {response['error']}")
            logger.info(
                "llm_call role=%s model=%s provider=ollama input_tokens=%s output_tokens=%s",
                role,
                model_name,
                response.get("prompt_eval_count"),
                response.get("eval_count"),
            )
            if response.get("done_reason") == "length":
                raise OllamaError(
                    f"Ollama output for role '{role}' hit the {max_tokens}-token limit "
                    "before the JSON was complete."
                )

            content = str(response.get("message", {}).get("content", ""))
            try:
                return output_model.model_validate_json(content)
            except ValidationError as exc:
                problem = str(exc)
            if attempt < self._max_repair_attempts:
                logger.warning(
                    "Ollama output for role '%s' failed validation; retrying: %s", role, problem
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That JSON was invalid for the required schema:\n"
                            f"{problem}\n"
                            "Reply again with only the corrected JSON object."
                        ),
                    },
                ]

        raise OllamaError(
            f"Ollama output for role '{role}' was still invalid after "
            f"{self._max_repair_attempts} repair attempt(s): {problem}"
        )
