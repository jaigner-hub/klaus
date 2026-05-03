# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Klaus is a local-first CLI coding agent built on Ollama + MCP. The default model is `qwen2.5-coder:7b` (Q4_K_M, fits a 12 GB card). The agent is an **MCP client**, not a server. It uses Ollama's native `tools` parameter — no custom JSON-extraction parsing.

## Commands

```bash
# Run the agent
uv run python -m local_agent

# Tests
uv run pytest
uv run pytest tests/test_agent_loop.py          # single file
uv run pytest -k "test_tool_call"               # single test

# Lint / format
uv run ruff check .
uv run ruff format .

# Install deps
uv sync

# Run the shell-mcp server standalone (Phase 4+)
uv run --package local-agent-shell-mcp python -m local_agent_shell_mcp
```

## Architecture

The agent loop lives in `agent.py`. It works like this:

1. Append user message to `messages`.
2. Call `ollama_client.chat_stream(messages, tools=registry.to_ollama_schemas())`.
3. If the response contains `tool_calls`, execute each via `registry.dispatch(call)`, append `tool` role results, loop back to step 2.
4. When the model returns a final assistant message with no tool calls, render it and return to the REPL.
5. Cap at 25 iterations; prompt user before continuing past the cap.

### Tool registry

`tools/registry.py` holds the unified tool catalog. It merges:
- **Built-in tools** (`tools/builtin.py`) — Phase 2 scaffolding only; retired once MCP filesystem server is wired.
- **MCP tools** (`tools/mcp_bridge.py`) — namespace as `<server>__<tool>` to avoid collisions.

`mcp_bridge.py` handles bidirectional translation: MCP JSON Schema → Ollama tool schema on startup; Ollama `tool_calls` → MCP `call_tool` requests at runtime; MCP results → Ollama `tool` role messages.

### MCP lifecycle

On startup, `agent.py` (or a dedicated `mcp_manager`) spawns each configured MCP server as a subprocess (stdio transport), performs the MCP handshake, calls `list_tools`, and registers results. If a server crashes at any point: log the error, mark its tools unavailable, continue — do not crash the agent.

### Safety (Phase 4+)

`packages/shell-mcp/` is a standalone package with its own `pyproject.toml`. Its safety model:
- **Allowlist mode** (default): read-only patterns (`ls`, `cat`, `grep`, `rg`, `find`, `git status`, `git log`, `git diff`, etc.) run without prompting.
- **Confirm mode**: anything off the allowlist surfaces a `y/n/a` prompt in the REPL showing exact command + cwd.
- **Denylist**: patterns like `rm -rf /`, `dd of=/dev/`, writes to `/etc /boot /sys`, fork bombs — rejected unconditionally, even if the user confirms.

The safety module (`safety/patterns.py`) is the thing to test adversarially. `tests/test_shell_safety.py` must include explicit attempts to escape the net.

### Diff-before-write (Phase 5+)

Tool calls that modify files are intercepted in the agent loop (not passed through to MCP transparently). `safety/confirm.py` computes a unified diff and prompts before applying. This is intentional: it's the demonstration that the agent understands what it's about to do.

### Session persistence

Every turn is appended to `~/.local-agent/sessions/<uuid>.jsonl` (one JSON object per line: role, content, tool calls, timestamps). A parallel `<uuid>.log` captures full request/response payloads at DEBUG level — this is the primary debugging artifact for "why did the model do that?"

`/load <id>` replays the JSONL back into `messages` and resumes from there.

### Context management (Phase 5+)

When `len(messages)` × avg_tokens approaches the model's context window, summarize the oldest N turns into a synthetic system message using the same Ollama model. Never silently drop turns.

## Key constraints

- **Phase gates are hard stops.** Do not begin Phase N+1 until Phase N's `Done when:` criterion is met and verified manually.
- **No sync wrappers** around async Ollama/MCP calls. The whole stack is async; keep it that way.
- **Pydantic for every external boundary**: config parsing, tool schema definitions, MCP message types, session record types. This is where agent bugs hide.
- **Cloud fallback (Phase 6+) is opt-in only.** The default codepath must not make any non-local network calls except to the configured Ollama base URL.

## Repo layout note

`packages/shell-mcp/` is an independent uv package. It has its own `pyproject.toml` and is not imported by the main `local_agent` package — it is invoked as a subprocess MCP server, same as any external MCP server.
