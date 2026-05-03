import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

from local_agent.config import McpServerConfig
from local_agent.tools.registry import ToolRegistry

_log = logging.getLogger(__name__)


def _tool_schema(server_name: str, tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": f"{server_name}__{tool.name}",
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def _result_to_str(result: CallToolResult) -> str:
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


class McpConnection:
    def __init__(self, config: McpServerConfig) -> None:
        self.server_name = config.name
        self._config = config
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env=self._config.env or None,
        )
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

    async def list_tools(self) -> list[dict]:
        response = await self._session.list_tools()
        return [_tool_schema(self.server_name, t) for t in response.tools]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = await self._session.call_tool(tool_name, arguments)
        return _result_to_str(result)

    async def close(self) -> None:
        await self._exit_stack.aclose()


async def register_mcp_servers(
    configs: list[McpServerConfig],
    registry: ToolRegistry,
) -> list[McpConnection]:
    connections: list[McpConnection] = []
    for cfg in configs:
        conn = McpConnection(cfg)
        try:
            await conn.connect()
            schemas = await conn.list_tools()
            for schema in schemas:
                qualified = schema["function"]["name"]
                local_name = qualified.split("__", 1)[-1]

                async def _handler(_n=local_name, _c=conn, **kwargs):
                    return await _c.call_tool(_n, kwargs)

                registry.register(qualified, _handler, schema)
            connections.append(conn)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            _log.warning("MCP server %r failed to connect: %s", cfg.name, exc)
            try:
                await conn.close()
            except BaseException:
                pass
    return connections
