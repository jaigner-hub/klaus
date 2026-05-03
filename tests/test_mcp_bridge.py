import pytest
from mcp.types import CallToolResult, TextContent, Tool

from local_agent.config import McpServerConfig
from local_agent.tools.mcp_bridge import McpConnection, _result_to_str, _tool_schema
from local_agent.tools.registry import ToolRegistry

# ── unit: schema translation ───────────────────────────────────────────────────

def _make_tool(name: str, description: str = "", schema: dict | None = None) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def test_tool_schema_name_is_namespaced():
    schema = _tool_schema("filesystem", _make_tool("read_file"))
    assert schema["function"]["name"] == "filesystem__read_file"


def test_tool_schema_description_preserved():
    schema = _tool_schema("filesystem", _make_tool("read_file", description="Read a file."))
    assert schema["function"]["description"] == "Read a file."


def test_tool_schema_parameters_is_input_schema():
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    schema = _tool_schema("filesystem", _make_tool("read_file", schema=input_schema))
    assert schema["function"]["parameters"] == input_schema


def test_tool_schema_type_is_function():
    schema = _tool_schema("myserver", _make_tool("list_directory"))
    assert schema["type"] == "function"


# ── unit: result extraction ────────────────────────────────────────────────────

def test_result_to_str_single_text_block():
    result = CallToolResult(content=[TextContent(type="text", text="hello world")])
    assert _result_to_str(result) == "hello world"


def test_result_to_str_multiple_blocks_joined():
    result = CallToolResult(content=[
        TextContent(type="text", text="line one"),
        TextContent(type="text", text="line two"),
    ])
    assert _result_to_str(result) == "line one\nline two"


def test_result_to_str_empty_content():
    result = CallToolResult(content=[])
    assert _result_to_str(result) == ""


# ── integration: real filesystem MCP server ────────────────────────────────────
# Requires npx; downloads @modelcontextprotocol/server-filesystem on first run.

@pytest.fixture
def fs_config(tmp_path):
    return McpServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
    )


@pytest.mark.asyncio
async def test_mcp_connection_discovers_tools(fs_config):
    conn = McpConnection(fs_config)
    await conn.connect()
    try:
        schemas = await conn.list_tools()
        names = [s["function"]["name"] for s in schemas]
        assert "filesystem__read_file" in names
        assert "filesystem__list_directory" in names
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mcp_connection_call_read_file(fs_config, tmp_path):
    (tmp_path / "hello.txt").write_text("mcp works")
    conn = McpConnection(fs_config)
    await conn.connect()
    try:
        result = await conn.call_tool("read_file", {"path": str(tmp_path / "hello.txt")})
        assert "mcp works" in result
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mcp_connection_call_list_directory(fs_config, tmp_path):
    (tmp_path / "alpha.py").write_text("pass")
    conn = McpConnection(fs_config)
    await conn.connect()
    try:
        result = await conn.call_tool("list_directory", {"path": str(tmp_path)})
        assert "alpha.py" in result
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_register_mcp_servers_adds_tools_to_registry(fs_config):
    from local_agent.tools.mcp_bridge import register_mcp_servers
    registry = ToolRegistry()
    connections = await register_mcp_servers([fs_config], registry)
    try:
        names = [s["function"]["name"] for s in registry.to_ollama_schemas()]
        assert "filesystem__read_file" in names
        assert "filesystem__list_directory" in names
    finally:
        for conn in connections:
            await conn.close()


@pytest.mark.asyncio
async def test_register_mcp_servers_dispatches_read_file(fs_config, tmp_path):
    from local_agent.tools.mcp_bridge import register_mcp_servers
    (tmp_path / "data.txt").write_text("dispatch test")
    registry = ToolRegistry()
    connections = await register_mcp_servers([fs_config], registry)
    try:
        result = await registry.dispatch(
            "filesystem__read_file", {"path": str(tmp_path / "data.txt")}
        )
        assert "dispatch test" in result
    finally:
        for conn in connections:
            await conn.close()


@pytest.mark.asyncio
async def test_register_mcp_servers_tolerates_bad_server():
    from local_agent.tools.mcp_bridge import register_mcp_servers
    bad = McpServerConfig(name="broken", command="false", args=[])
    registry = ToolRegistry()
    connections = await register_mcp_servers([bad], registry)
    assert connections == []
    assert registry.to_ollama_schemas() == []
