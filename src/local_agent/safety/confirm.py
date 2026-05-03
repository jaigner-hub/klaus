"""Diff-before-write and shell command confirmation UI."""

import difflib
import json
from collections.abc import Callable
from pathlib import Path

from local_agent.tools.registry import ToolRegistry

# Tool local-names (after stripping server namespace) that modify file content.
_WRITE_TOOL_NAMES = frozenset({"write_file", "edit_file"})


def is_write_tool(tool_name: str) -> bool:
    """True if the tool modifies file content (namespace-agnostic)."""
    local_name = tool_name.split("__", 1)[-1]
    return local_name in _WRITE_TOOL_NAMES


def compute_diff(tool_name: str, arguments: dict) -> str | None:
    """Return a unified diff string for a write tool call, or None if not applicable.

    Returns an empty string when the content is identical (no change).
    """
    local_name = tool_name.split("__", 1)[-1]

    if local_name == "write_file":
        return _write_file_diff(arguments.get("path", ""), arguments.get("content", ""))

    if local_name == "edit_file":
        return _edit_file_diff(arguments.get("path", ""), arguments.get("edits", []))

    return None


def _write_file_diff(path: str, new_content: str) -> str:
    p = Path(path)
    if p.exists():
        old_lines = p.read_text(errors="replace").splitlines(keepends=True)
        from_label = f"a/{path}"
    else:
        old_lines = []
        from_label = "/dev/null"

    new_lines = new_content.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(old_lines, new_lines, fromfile=from_label, tofile=f"b/{path}")
    )


def _edit_file_diff(path: str, edits: list[dict]) -> str:
    p = Path(path)
    if not p.exists():
        return f"[file not found: {path}]"

    old_content = p.read_text(errors="replace")
    new_content = old_content
    for edit in edits:
        old_text = edit.get("oldText", "")
        new_text = edit.get("newText", "")
        if old_text in new_content:
            new_content = new_content.replace(old_text, new_text, 1)

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
    )


async def guarded_dispatch(
    name: str,
    arguments: dict,
    registry: ToolRegistry,
    on_write_confirm: Callable[[str, dict, str], bool] | None = None,
) -> str:
    """Dispatch a tool call, intercepting write tools for confirmation.

    If `on_write_confirm` is provided and the tool is a write tool, calls
    on_write_confirm(name, arguments, diff). If it returns False the tool is
    skipped and a skipped-JSON result is returned instead.
    """
    if on_write_confirm is not None and is_write_tool(name):
        diff = compute_diff(name, arguments) or ""
        if not on_write_confirm(name, arguments, diff):
            return json.dumps({"skipped": True, "tool": name})

    return await registry.dispatch(name, arguments)
