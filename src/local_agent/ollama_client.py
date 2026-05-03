import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import AsyncIterator

import ollama

from local_agent.config import AgentConfig


@dataclass
class _FakeFunction:
    name: str
    arguments: dict


@dataclass
class _FakeToolCall:
    function: _FakeFunction


def _try_parse_json_calls(text: str) -> list | None:
    if not text.strip().startswith("{"):
        return None
    try:
        data = json.loads(text)
        if "name" in data and "arguments" in data:
            return [_FakeToolCall(_FakeFunction(data["name"], data["arguments"]))]
        return None
    except json.JSONDecodeError:
        pass
    calls = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
            if "name" in data and "arguments" in data:
                calls.append(_FakeToolCall(_FakeFunction(data["name"], data["arguments"])))
        except json.JSONDecodeError:
            pass
    return calls if calls else None


def _parse_content_tool_call(text: str) -> list | None:
    """Fallback for models (e.g. qwen2.5-coder) that embed tool calls as JSON
    in content instead of using the native tool_calls field."""
    for fence in re.finditer(r"```(?:json)?\s*\n([\s\S]*?)\n```", text):
        calls = _try_parse_json_calls(fence.group(1).strip())
        if calls:
            return calls
    return _try_parse_json_calls(text.strip())


class OllamaClient:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client = ollama.AsyncClient(host=config.ollama_base_url)

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self._client.chat(
            model=self.config.model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            if chunk.message.content:
                yield chunk.message.content

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, list | None]:
        """Stream with tool support. Buffers response; replays tokens only on
        the final non-tool response so tool-call iterations stay clean.

        Tool calls can appear in any streaming chunk (llama3.2 puts them in
        the first chunk). If the model puts tool call JSON in content instead
        (qwen2.5-coder), we parse it as a fallback.
        """
        stream = await self._client.chat(
            model=self.config.model,
            messages=messages,
            tools=tools,
            stream=True,
        )
        parts: list[str] = []
        tool_calls = None
        async for chunk in stream:
            if chunk.message.content:
                parts.append(chunk.message.content)
            if chunk.message.tool_calls:
                tool_calls = chunk.message.tool_calls

        text = "".join(parts)

        # Fallback: some models embed tool calls as JSON in content
        if not tool_calls:
            tool_calls = _parse_content_tool_call(text)

        if not tool_calls and on_token:
            for token in parts:
                on_token(token)
        return text, tool_calls
