import json

import pytest

from local_agent_shell_mcp.server import run_command

# ── denied commands ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_denied_command_returns_error():
    result = json.loads(await run_command("rm -rf /"))
    assert result.get("error") == "denied"


@pytest.mark.asyncio
async def test_denied_dd_returns_error():
    result = json.loads(await run_command("dd if=/dev/zero of=/dev/sda"))
    assert result.get("error") == "denied"


@pytest.mark.asyncio
async def test_denied_result_includes_command():
    result = json.loads(await run_command("mkfs.ext4 /dev/sda1"))
    assert result.get("command") == "mkfs.ext4 /dev/sda1"


# ── confirm-mode commands ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_confirm_command_returns_needs_confirmation():
    result = json.loads(await run_command("pip install requests"))
    assert result.get("needs_confirmation") is True


@pytest.mark.asyncio
async def test_confirm_result_echoes_command():
    result = json.loads(await run_command("git push origin main"))
    assert result.get("command") == "git push origin main"


@pytest.mark.asyncio
async def test_confirm_result_includes_cwd():
    result = json.loads(await run_command("python script.py", cwd="/tmp"))
    assert result.get("cwd") == "/tmp"


# ── allowed commands ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allowed_command_returns_stdout(tmp_path):
    (tmp_path / "hello.txt").write_text("hello from mcp")
    result = json.loads(await run_command(f"cat {tmp_path}/hello.txt"))
    assert "hello from mcp" in result["stdout"]
    assert result["returncode"] == 0


@pytest.mark.asyncio
async def test_allowed_command_captures_stderr(tmp_path):
    result = json.loads(await run_command(f"ls {tmp_path}/nonexistent"))
    assert result["returncode"] != 0
    assert result["stderr"] != "" or result["stdout"] != ""


@pytest.mark.asyncio
async def test_allowed_command_uses_cwd(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    result = json.loads(await run_command("ls", cwd=str(tmp_path)))
    assert "marker.txt" in result["stdout"]


@pytest.mark.asyncio
async def test_allowed_echo_command():
    result = json.loads(await run_command("echo hello"))
    assert "hello" in result["stdout"]
    assert result["returncode"] == 0
