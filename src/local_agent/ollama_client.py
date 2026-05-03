from typing import AsyncIterator

import ollama

from local_agent.config import AgentConfig


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
