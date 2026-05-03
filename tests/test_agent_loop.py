import pytest

from local_agent.agent import run_agent_turn
from local_agent.config import AgentConfig
from local_agent.ollama_client import OllamaClient
from local_agent.tools.registry import ToolRegistry


@pytest.fixture
def client():
    return OllamaClient(AgentConfig())


@pytest.fixture
def registry():
    return ToolRegistry.with_builtins()


# ── no-tool path ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_simple_question_returns_response(client, registry):
    messages: list[dict] = []
    response = await run_agent_turn(client, registry, messages, "What is 1 + 1?")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_agent_appends_user_and_assistant_messages(client, registry):
    messages: list[dict] = []
    await run_agent_turn(client, registry, messages, "Say hello.")
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles


# ── tool-call path ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_calls_read_file_tool(client, registry, tmp_path):
    f = tmp_path / "secret.txt"
    f.write_text("the secret value is XYZZY42")
    messages: list[dict] = []
    tool_calls_seen: list[tuple[str, dict]] = []

    response = await run_agent_turn(
        client, registry, messages,
        f"Use the read_file tool to read {f} and tell me the secret value.",
        on_tool_call=lambda name, args: tool_calls_seen.append((name, args)),
    )

    assert any(name == "read_file" for name, _ in tool_calls_seen)
    assert "XYZZY42" in response


@pytest.mark.asyncio
async def test_agent_calls_list_directory_tool(client, registry, tmp_path):
    (tmp_path / "alpha.py").write_text("pass")
    (tmp_path / "beta.py").write_text("pass")
    messages: list[dict] = []
    tool_calls_seen: list[str] = []

    response = await run_agent_turn(
        client, registry, messages,
        f"Use the list_directory tool to list the files in {tmp_path}.",
        on_tool_call=lambda name, args: tool_calls_seen.append(name),
    )

    assert "list_directory" in tool_calls_seen
    assert "alpha.py" in response or "beta.py" in response


# ── callbacks ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_on_tool_result_callback_fires(client, registry, tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("callback test content")
    messages: list[dict] = []
    results_seen: list[tuple[str, str]] = []

    await run_agent_turn(
        client, registry, messages,
        f"Use the read_file tool to read {f}.",
        on_tool_result=lambda name, result: results_seen.append((name, result)),
    )

    assert any(name == "read_file" for name, _ in results_seen)


@pytest.mark.asyncio
async def test_on_token_callback_fires_for_final_response(client, registry):
    messages: list[dict] = []
    tokens: list[str] = []

    await run_agent_turn(
        client, registry, messages, "Say the word pong.",
        on_token=tokens.append,
    )

    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)
