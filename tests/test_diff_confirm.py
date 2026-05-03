import json
from pathlib import Path

import pytest

from local_agent.safety.confirm import compute_diff, guarded_dispatch, is_write_tool
from local_agent.tools.registry import ToolRegistry

# ── is_write_tool ──────────────────────────────────────────────────────────────

def test_is_write_tool_write_file():
    assert is_write_tool("write_file") is True

def test_is_write_tool_edit_file():
    assert is_write_tool("edit_file") is True

def test_is_write_tool_namespaced_write():
    assert is_write_tool("filesystem__write_file") is True

def test_is_write_tool_namespaced_edit():
    assert is_write_tool("filesystem__edit_file") is True

def test_is_write_tool_read_file():
    assert is_write_tool("read_file") is False

def test_is_write_tool_list_directory():
    assert is_write_tool("list_directory") is False

def test_is_write_tool_run_command():
    assert is_write_tool("shell__run_command") is False

# ── compute_diff: write_file ───────────────────────────────────────────────────

def test_compute_diff_new_file(tmp_path):
    path = str(tmp_path / "new.py")
    diff = compute_diff("write_file", {"path": path, "content": "print('hello')\n"})
    assert diff is not None
    assert "+" in diff
    assert "hello" in diff

def test_compute_diff_existing_file_with_change(tmp_path):
    f = tmp_path / "existing.py"
    f.write_text("x = 1\n")
    diff = compute_diff("write_file", {"path": str(f), "content": "x = 2\n"})
    assert diff is not None
    assert "-x = 1" in diff
    assert "+x = 2" in diff

def test_compute_diff_unchanged_returns_empty(tmp_path):
    f = tmp_path / "same.py"
    content = "x = 1\n"
    f.write_text(content)
    diff = compute_diff("write_file", {"path": str(f), "content": content})
    assert diff == ""

def test_compute_diff_namespaced_write_file(tmp_path):
    path = str(tmp_path / "f.py")
    diff = compute_diff("filesystem__write_file", {"path": path, "content": "y = 3\n"})
    assert diff is not None
    assert "y = 3" in diff

def test_compute_diff_edit_file_shows_change(tmp_path):
    f = tmp_path / "edit_me.py"
    f.write_text("x = 1\ny = 2\n")
    diff = compute_diff(
        "edit_file",
        {
            "path": str(f),
            "edits": [{"oldText": "x = 1", "newText": "x = 99"}],
        },
    )
    assert diff is not None
    assert "-x = 1" in diff
    assert "+x = 99" in diff

def test_compute_diff_unknown_tool_returns_none():
    assert compute_diff("read_file", {"path": "/tmp/x"}) is None

# ── guarded_dispatch ───────────────────────────────────────────────────────────

def _write_registry(writes: list[str]) -> ToolRegistry:
    """Registry with a fake write_file that records calls."""
    reg = ToolRegistry()

    async def fake_write(path: str, content: str) -> str:
        writes.append(path)
        return json.dumps({"written": True})

    reg.register(
        "write_file",
        fake_write,
        {"type": "function", "function": {
            "name": "write_file",
            "description": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        }},
    )
    return reg


@pytest.mark.asyncio
async def test_guarded_dispatch_confirmed_dispatches(tmp_path):
    writes = []
    reg = _write_registry(writes)
    path = str(tmp_path / "out.py")

    result = await guarded_dispatch(
        "write_file",
        {"path": path, "content": "x = 1\n"},
        reg,
        on_write_confirm=lambda name, args, diff: True,
    )

    assert writes == [path]
    assert json.loads(result)["written"] is True


@pytest.mark.asyncio
async def test_guarded_dispatch_rejected_skips(tmp_path):
    writes = []
    reg = _write_registry(writes)
    path = str(tmp_path / "out.py")

    result = await guarded_dispatch(
        "write_file",
        {"path": path, "content": "x = 1\n"},
        reg,
        on_write_confirm=lambda name, args, diff: False,
    )

    assert writes == []
    assert json.loads(result).get("skipped") is True


@pytest.mark.asyncio
async def test_guarded_dispatch_confirm_receives_diff(tmp_path):
    f = tmp_path / "existing.py"
    f.write_text("old = 1\n")
    received: list[str] = []
    reg = _write_registry([])

    await guarded_dispatch(
        "write_file",
        {"path": str(f), "content": "new = 2\n"},
        reg,
        on_write_confirm=lambda name, args, diff: received.append(diff) or True,
    )

    assert received
    assert "-old = 1" in received[0]


@pytest.mark.asyncio
async def test_guarded_dispatch_read_tool_skips_confirm(tmp_path):
    calls = []
    reg = ToolRegistry()

    async def fake_read(path: str) -> str:
        return json.dumps({"content": "hi"})

    reg.register(
        "read_file",
        fake_read,
        {"type": "function", "function": {
            "name": "read_file",
            "description": "read",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }},
    )

    await guarded_dispatch(
        "read_file",
        {"path": "/tmp/x"},
        reg,
        on_write_confirm=lambda name, args, diff: calls.append(1) or False,
    )

    assert calls == []  # confirm was never called


@pytest.mark.asyncio
async def test_guarded_dispatch_no_callback_dispatches_write(tmp_path):
    writes = []
    reg = _write_registry(writes)
    path = str(tmp_path / "out.py")

    await guarded_dispatch(
        "write_file",
        {"path": path, "content": "x = 1\n"},
        reg,
        on_write_confirm=None,
    )

    assert writes == [path]
