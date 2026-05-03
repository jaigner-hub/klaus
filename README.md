# Klaus

A CLI coding agent that runs entirely on local hardware via Ollama. Uses the Model Context Protocol (MCP) to connect to tool servers and executes a real agentic loop — plan, call tools, observe results, iterate. Think "Claude Code, but on your laptop with Qwen."

The engineering focus: end-to-end agent design — protocol implementation, native tool use, safety guardrails, and the parts of agent architecture that actually matter in production.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) running locally

## Quick start

```bash
# Pull the default model
ollama pull qwen2.5-coder:7b

# Install and run
uv run python -m local_agent
```

On first run, a default config is written to `~/.local-agent/config.toml`.

## Configuration

`~/.local-agent/config.toml`:

```toml
model = "qwen2.5-coder:7b"          # or 14b, 1.5b
ollama_base_url = "http://localhost:11434"
system_prompt_path = ""              # optional override

[[mcp_servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]

[[mcp_servers]]
name = "git"
command = "uvx"
args = ["mcp-server-git", "--repository", "/home/user/projects/myrepo"]
```

See `examples/config.example.toml` for the full reference.

## Hardware

| VRAM | Recommended model |
|------|-------------------|
| 12 GB (e.g. RTX 4070) | `qwen2.5-coder:7b` Q4_K_M — default |
| 24 GB | `qwen2.5-coder:14b` Q4_K_M |
| CPU only | `qwen2.5-coder:1.5b` — works, tool-call accuracy lower |
| Apple Silicon 16 GB+ | `qwen2.5-coder:7b` via Metal backend |

## REPL commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/tools` | List active tools and their MCP server origin |
| `/plan` | Ask the model to produce a numbered plan before executing |
| `/model <name>` | Switch model mid-session |
| `/save` | Persist the current session |
| `/load <id>` | Resume a previous session |
| `/clear` | Clear conversation history |
| `/cost` | Show token counts for this session |
| Ctrl-C / Ctrl-D | Exit cleanly |

## Architecture

```
local_agent/
├── cli.py           # REPL, slash commands, rich rendering
├── config.py        # config.toml parsing via Pydantic
├── ollama_client.py # async Ollama HTTP wrapper (streaming)
├── agent.py         # the loop: chat → tool calls → exec → repeat
├── session.py       # JSONL persistence, load/save
├── tools/
│   ├── registry.py  # unified tool catalog (built-ins + MCP)
│   ├── builtin.py   # Phase 2 scaffolding tools (retired in Phase 3)
│   └── mcp_bridge.py # MCP ↔ Ollama schema translation
└── safety/
    ├── confirm.py   # diff-before-write, command confirmation UI
    └── patterns.py  # shell allowlist / denylist patterns
```

The shell-exec MCP server lives in `packages/shell-mcp/` as a standalone package with its own `pyproject.toml`, so its safety logic is independently auditable.

Sessions are written to `~/.local-agent/sessions/<uuid>.jsonl` (one JSON line per turn) and `~/.local-agent/sessions/<uuid>.log` (DEBUG-level full payloads).

## Development phases

| Phase | Focus | Gate |
|-------|-------|------|
| 1 | Plain Ollama REPL with streaming | Multi-turn history works |
| 2 | Native tool calling with built-in tools | `list_directory` → `read_file` → summary |
| 3 | MCP client integration | `mcp-server-filesystem` + `mcp-server-git` live |
| 4 | Custom shell-exec MCP server (with safety) | Destructive commands blocked, confirmed by adversarial tests |
| 5 | Agentic polish: plan/act, diff-before-write, context mgmt | End-to-end task with diffs and session replay |
| 6+ | Eval harness, multi-model routing, vector memory | Optional |

## Design principles

- **Async everywhere** — Ollama client and MCP transport are both async; no sync wrappers.
- **Pydantic at every boundary** — config, tool schemas, MCP messages, session records.
- **Tool calls are first-class UI** — the user sees every call and result in the REPL in real time.
- **Safety is a property, not a feature** — the shell-exec server is tested adversarially; escaping the safety net should be impossible by construction.
- **Local-first means local-first** — no telemetry, no cloud calls in the default codepath.

## Resources

- [Ollama Python client](https://github.com/ollama/ollama-python)
- [Ollama tool calling](https://ollama.com/blog/tool-support)
- [MCP specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Reference MCP servers](https://github.com/modelcontextprotocol/servers)
