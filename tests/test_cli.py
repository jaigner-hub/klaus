import pytest

from local_agent.cli import run_turn
from local_agent.config import AgentConfig
from local_agent.ollama_client import OllamaClient


@pytest.fixture
def client():
    return OllamaClient(AgentConfig())


# ── message history management ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_turn_appends_user_message(client):
    messages = []
    await run_turn(client, messages, "hello")
    assert messages[0] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_run_turn_appends_assistant_message(client):
    messages = []
    await run_turn(client, messages, "say the word yes")
    assert messages[-1]["role"] == "assistant"
    assert len(messages[-1]["content"]) > 0


@pytest.mark.asyncio
async def test_run_turn_returns_response_text(client):
    messages = []
    response = await run_turn(client, messages, "say the word yes")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_run_turn_response_matches_appended_message(client):
    messages = []
    response = await run_turn(client, messages, "say the word yes")
    assert messages[-1]["content"] == response


@pytest.mark.asyncio
async def test_run_turn_preserves_existing_history(client):
    prior = [
        {"role": "user", "content": "prior message"},
        {"role": "assistant", "content": "prior response"},
    ]
    messages = list(prior)
    await run_turn(client, messages, "new message")
    assert messages[0] == prior[0]
    assert messages[1] == prior[1]


@pytest.mark.asyncio
async def test_multiple_turns_accumulate_history(client):
    messages = []
    await run_turn(client, messages, "my secret number is 7")
    await run_turn(client, messages, "what is my secret number?")
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[3]["role"] == "assistant"
    assert "7" in messages[3]["content"]


# ── on_token callback ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_on_token_callback_is_called(client):
    tokens_received = []
    messages = []
    await run_turn(
        client, messages, "say the word yes", on_token=tokens_received.append
    )
    assert len(tokens_received) > 0


@pytest.mark.asyncio
async def test_on_token_tokens_join_to_full_response(client):
    tokens_received = []
    messages = []
    response = await run_turn(
        client, messages, "say the word yes", on_token=tokens_received.append
    )
    assert "".join(tokens_received) == response


@pytest.mark.asyncio
async def test_no_on_token_still_works(client):
    messages = []
    response = await run_turn(client, messages, "say the word yes")
    assert len(response) > 0
