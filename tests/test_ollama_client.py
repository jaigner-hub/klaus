import pytest

from local_agent.config import AgentConfig
from local_agent.ollama_client import OllamaClient, _parse_content_tool_call

# ── _parse_content_tool_call ───────────────────────────────────────────────────

def test_parse_plain_json_tool_call():
    text = '{"name": "read_file", "arguments": {"path": "/tmp/x"}}'
    result = _parse_content_tool_call(text)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "read_file"
    assert result[0].function.arguments == {"path": "/tmp/x"}


def test_parse_fenced_json_tool_call():
    text = '```json\n{"name": "list_directory", "arguments": {"path": "/tmp"}}\n```'
    result = _parse_content_tool_call(text)
    assert result is not None
    assert len(result) == 1
    assert result[0].function.name == "list_directory"


def test_parse_fenced_json_without_language_tag():
    text = '```\n{"name": "read_file", "arguments": {"path": "/a"}}\n```'
    result = _parse_content_tool_call(text)
    assert result is not None
    assert result[0].function.name == "read_file"


def test_parse_multiple_compact_json_lines():
    text = (
        '{"name": "read_file", "arguments": {"path": "/a"}}\n'
        '{"name": "read_file", "arguments": {"path": "/b"}}'
    )
    result = _parse_content_tool_call(text)
    assert result is not None
    assert len(result) == 2
    assert result[0].function.arguments["path"] == "/a"
    assert result[1].function.arguments["path"] == "/b"


def test_parse_fenced_json_embedded_after_text():
    text = (
        "Sure, let me list the directory for you:\n\n"
        "```json\n"
        '{"name": "list_directory", "arguments": {"path": "./src"}}\n'
        "```\n\n"
        "I'll summarise the results once I have them."
    )
    result = _parse_content_tool_call(text)
    assert result is not None
    assert result[0].function.name == "list_directory"


def test_parse_returns_none_for_plain_text():
    assert _parse_content_tool_call("Hello, how can I help?") is None


def test_parse_returns_none_for_json_without_name():
    assert _parse_content_tool_call('{"foo": "bar"}') is None

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
