import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel, Field

from trading_agents.llm.ollama_client import OllamaError, OllamaLLMClient
from trading_agents.llm.structured_output import model_to_tool_schema


class _Verdict(BaseModel):
    signal: str
    confidence: float = Field(ge=0.0, le=1.0)


def _reply(content: str, **extra) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
        **extra,
    }


class _ScriptedServer:
    """Fake transport: returns queued replies and records request bodies."""

    def __init__(self, *replies: dict) -> None:
        self.replies = list(replies)
        self.requests: list[tuple[str, dict]] = []

    async def __call__(self, url: str, body: dict) -> dict:
        self.requests.append((url, body))
        return self.replies.pop(0)


async def _generate(client: OllamaLLMClient, **overrides):
    kwargs = {
        "role": "news_analyst",
        "system_prompt": "You are an analyst.",
        "user_content": '{"ticker": "AAPL"}',
        "output_model": _Verdict,
        "model": "llama3.1:8b",
        "max_tokens": 512,
    }
    return await client.generate_structured(**{**kwargs, **overrides})


async def test_sends_a_schema_constrained_chat_request():
    server = _ScriptedServer(_reply('{"signal": "bullish", "confidence": 0.7}'))
    client = OllamaLLMClient(base_url="http://ollama:11434/", transport=server)

    result = await _generate(client)

    assert result == _Verdict(signal="bullish", confidence=0.7)
    url, body = server.requests[0]
    assert url == "http://ollama:11434/api/chat"
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is False
    assert body["format"] == model_to_tool_schema(_Verdict)["input_schema"]
    assert body["options"] == {"num_predict": 512, "temperature": 0}
    system, user = body["messages"]
    assert system["role"] == "system" and system["content"].startswith("You are an analyst.")
    assert "single JSON object" in system["content"]
    assert user == {"role": "user", "content": '{"ticker": "AAPL"}'}


async def test_model_override_takes_precedence():
    server = _ScriptedServer(_reply('{"signal": "neutral", "confidence": 0.1}'))
    client = OllamaLLMClient(transport=server)

    await _generate(client, model_override="qwen3:8b")

    assert server.requests[0][1]["model"] == "qwen3:8b"


async def test_invalid_output_is_repaired_with_the_validation_error_fed_back():
    server = _ScriptedServer(
        _reply('{"signal": "bullish", "confidence": 75}'),  # out of range
        _reply('{"signal": "bullish", "confidence": 0.75}'),
    )
    client = OllamaLLMClient(transport=server)

    result = await _generate(client)

    assert result.confidence == 0.75
    retry_messages = server.requests[1][1]["messages"]
    assert retry_messages[2] == {
        "role": "assistant",
        "content": '{"signal": "bullish", "confidence": 75}',
    }
    assert "less than or equal to 1" in retry_messages[3]["content"]


async def test_gives_up_after_the_repair_attempts():
    server = _ScriptedServer(_reply("not json"), _reply("still not json"))
    client = OllamaLLMClient(transport=server, max_repair_attempts=1)

    with pytest.raises(OllamaError, match="still invalid after 1 repair"):
        await _generate(client)
    assert len(server.requests) == 2


async def test_truncated_output_raises_a_clear_error():
    server = _ScriptedServer(_reply('{"signal": "bull', done_reason="length"))
    client = OllamaLLMClient(transport=server)

    with pytest.raises(OllamaError, match="512-token limit"):
        await _generate(client)


async def test_server_error_payload_raises():
    server = _ScriptedServer({"error": "model 'llama3.1:8b' not found"})
    client = OllamaLLMClient(transport=server)

    with pytest.raises(OllamaError, match="not found"):
        await _generate(client)


async def test_in_flight_requests_are_capped():
    in_flight = 0
    peak = 0

    async def slow_server(url: str, body: dict) -> dict:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _reply('{"signal": "neutral", "confidence": 0.1}')

    client = OllamaLLMClient(transport=slow_server, max_concurrency=2)

    await asyncio.gather(*(_generate(client) for _ in range(6)))

    assert peak == 2


class _NotFoundHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server's naming
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"error": "model 'nope' not found, try pulling it first"}).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


async def test_http_errors_surface_ollamas_error_message():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NotFoundHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OllamaLLMClient(base_url=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(OllamaError, match=r"HTTP 404: model 'nope' not found"):
            await _generate(client, model="nope")
    finally:
        server.shutdown()


async def test_unreachable_server_says_how_to_start_ollama():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # closed again before the request

    client = OllamaLLMClient(base_url=f"http://127.0.0.1:{port}", timeout_seconds=5)

    with pytest.raises(OllamaError, match="ollama serve"):
        await _generate(client)


async def test_check_models_accepts_pulled_models_including_implicit_latest():
    server = _ScriptedServer({"models": [{"name": "llama3.1:8b"}, {"name": "qwen3:latest"}]})
    client = OllamaLLMClient(base_url="http://ollama:11434", transport=server)

    await client.check_models({"llama3.1:8b", "qwen3"})

    assert server.requests == [("http://ollama:11434/api/tags", None)]


async def test_check_models_names_the_pull_command_for_missing_models():
    server = _ScriptedServer({"models": [{"name": "llama3.1:8b"}]})
    client = OllamaLLMClient(transport=server)

    with pytest.raises(OllamaError, match=r"Run: ollama pull qwen3:32b"):
        await client.check_models({"llama3.1:8b", "qwen3:32b"})
