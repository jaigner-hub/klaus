import json

import pytest

from local_agent.tools.registry import ToolRegistry

# ── registration ───────────────────────────────────────────────────────────────

def test_registered_tool_appears_in_schemas():
    reg = ToolRegistry()
    schema = {
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "does a thing",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        },
    }
    reg.register("my_tool", lambda x: "ok", schema)
    names = [s["function"]["name"] for s in reg.to_ollama_schemas()]
    assert "my_tool" in names


def test_to_ollama_schemas_format():
    reg = ToolRegistry()
    schema = {
        "type": "function",
        "function": {
            "name": "greet",
            "description": "says hi",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
    reg.register("greet", lambda name: f"hi {name}", schema)
    schemas = reg.to_ollama_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert "function" in schemas[0]
    assert schemas[0]["function"]["name"] == "greet"


def test_empty_registry_returns_empty_schemas():
    reg = ToolRegistry()
    assert reg.to_ollama_schemas() == []


# ── dispatch ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_calls_registered_function():
    reg = ToolRegistry()
    calls = []
    reg.register(
        "echo",
        lambda msg: calls.append(msg) or f"echoed: {msg}",
        {"type": "function", "function": {"name": "echo", "description": "",
         "parameters": {"type": "object", "properties": {}}}},
    )
    result = await reg.dispatch("echo", {"msg": "hello"})
    assert calls == ["hello"]
    assert result == "echoed: hello"


@pytest.mark.asyncio
async def test_dispatch_returns_string():
    reg = ToolRegistry()
    reg.register(
        "num",
        lambda: 42,
        {"type": "function", "function": {"name": "num", "description": "",
         "parameters": {"type": "object", "properties": {}}}},
    )
    result = await reg.dispatch("num", {})
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error():
    reg = ToolRegistry()
    result = await reg.dispatch("no_such_tool", {})
    assert "error" in result.lower() or "unknown" in result.lower()


# ── default registry with builtins ─────────────────────────────────────────────

def test_default_registry_has_read_file():
    reg = ToolRegistry.with_builtins()
    names = [s["function"]["name"] for s in reg.to_ollama_schemas()]
    assert "read_file" in names


def test_default_registry_has_list_directory():
    reg = ToolRegistry.with_builtins()
    names = [s["function"]["name"] for s in reg.to_ollama_schemas()]
    assert "list_directory" in names


@pytest.mark.asyncio
async def test_default_registry_read_file_works(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello")
    reg = ToolRegistry.with_builtins()
    result = json.loads(await reg.dispatch("read_file", {"path": str(f)}))
    assert result["content"] == "hello"


@pytest.mark.asyncio
async def test_default_registry_list_directory_works(tmp_path):
    (tmp_path / "a.py").write_text("x")
    reg = ToolRegistry.with_builtins()
    result = json.loads(
        await reg.dispatch("list_directory", {"path": str(tmp_path)})
    )
    names = [e["name"] for e in result["entries"]]
    assert "a.py" in names
