import asyncio
import json

from mcp.server.fastmcp import FastMCP

from local_agent_shell_mcp.safety.patterns import classify

mcp = FastMCP("shell-mcp")

_TIMEOUT_SECONDS = 30


async def run_command(command: str, cwd: str | None = None) -> str:
    """Execute a shell command with safety enforcement.

    Returns JSON with one of three shapes:
      {"error": "denied", "command": ...}            — denylist hit
      {"needs_confirmation": true, "command": ...}   — confirm-mode
      {"stdout": ..., "stderr": ..., "returncode": ...}  — executed
    """
    classification = classify(command)

    if classification == "deny":
        return json.dumps({"error": "denied", "command": command})

    if classification == "confirm":
        return json.dumps({"needs_confirmation": True, "command": command, "cwd": cwd})

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_TIMEOUT_SECONDS
        )
        return json.dumps({
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "returncode": proc.returncode,
        })
    except TimeoutError:
        return json.dumps({"error": "timeout", "command": command})
    except Exception as exc:
        return json.dumps({"error": str(exc), "command": command})


# Register run_command as an MCP tool
mcp.tool()(run_command)
