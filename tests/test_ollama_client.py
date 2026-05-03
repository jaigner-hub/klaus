import pytest

from local_agent.config import AgentConfig
from local_agent.ollama_client import OllamaClient

# ── unit: construction ─────────────────────────────────────────────────────────

def test_client_construction():
    cfg = AgentConfig()
    client = OllamaClient(cfg)
    assert client is not None


def test_client_exposes_config():
    cfg = AgentConfig(model="qwen2.5-coder:1.5b")
    client = OllamaClient(cfg)
    assert client.config.model == "qwen2.5-coder:1.5b"


# ── integration: live Ollama ───────────────────────────────────────────────────
# These tests require `ollama serve` to be running with qwen2.5-coder:7b pulled.
# They exercise the real streaming path end-to-end.

@pytest.mark.asyncio
async def test_chat_stream_yields_strings():
    cfg = AgentConfig()
    client = OllamaClient(cfg)
    messages = [{"role": "user", "content": "Reply with just the word 'pong'."}]

    tokens = []
    async for token in client.chat_stream(messages):
        assert isinstance(token, str)
        tokens.append(token)

    assert len(tokens) > 0


@pytest.mark.asyncio
async def test_chat_stream_response_nonempty():
    cfg = AgentConfig()
    client = OllamaClient(cfg)
    messages = [
        {"role": "user", "content": "What is 2 + 2? Answer with a single number."}
    ]

    full_response = ""
    async for token in client.chat_stream(messages):
        full_response += token

    assert full_response.strip() != ""


@pytest.mark.asyncio
async def test_chat_stream_multi_turn():
    cfg = AgentConfig()
    client = OllamaClient(cfg)
    messages = [
        {"role": "user", "content": "My favourite number is 42."},
        {"role": "assistant", "content": "Got it, your favourite number is 42."},
        {"role": "user", "content": "What is my favourite number?"},
    ]

    full_response = ""
    async for token in client.chat_stream(messages):
        full_response += token

    assert "42" in full_response
