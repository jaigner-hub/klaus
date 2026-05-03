import asyncio
import json
from collections.abc import Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from local_agent.agent import run_agent_turn
from local_agent.config import AgentConfig, load_config
from local_agent.ollama_client import OllamaClient
from local_agent.session import Session
from local_agent.tools.mcp_bridge import register_mcp_servers
from local_agent.tools.registry import ToolRegistry


async def run_turn(
    client: OllamaClient,
    messages: list[dict],
    user_input: str,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Process one turn without tools. Mutates messages in-place."""
    messages.append({"role": "user", "content": user_input})
    parts: list[str] = []
    async for token in client.chat_stream(messages):
        parts.append(token)
        if on_token is not None:
            on_token(token)
    response = "".join(parts)
    messages.append({"role": "assistant", "content": response})
    return response


def _fmt_args(args: dict) -> str:
    return ", ".join(f'{k}="{v}"' for k, v in args.items())


def _fmt_result(tool_name: str, result: str) -> str:
    try:
        data = json.loads(result)
        if "error" in data:
            return f"[red]error:[/red] {data['error']}"
        if tool_name == "read_file":
            size = data.get("size", "?")
            suffix = " [dim](truncated)[/dim]" if data.get("truncated") else ""
            return f"{size} bytes{suffix}"
        if tool_name == "list_directory":
            count = len(data.get("entries", []))
            return f"{count} entr{'y' if count == 1 else 'ies'}"
    except (json.JSONDecodeError, KeyError):
        pass
    return (result[:72] + "…") if len(result) > 72 else result


async def run_repl(config: AgentConfig) -> None:
    console = Console()
    client = OllamaClient(config)
    registry = ToolRegistry.with_builtins()
    mcp_connections = await register_mcp_servers(config.mcp_servers, registry)
    messages: list[dict] = [{"role": "system", "content": config.system_prompt}]
    session = Session()

    console.print(
        "[bold cyan]Klaus[/bold cyan] — local coding agent  "
        f"[dim]{config.model}[/dim]"
    )
    console.print(
        f"[dim]Session {session.session_id[:8]}…  "
        "Ctrl-C or Ctrl-D to exit.[/dim]\n"
    )

    try:
        while True:
            try:
                user_input = Prompt.ask("[bold green]>[/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/dim]")
                break

            stripped = user_input.strip()
            if not stripped:
                continue

            # ── built-in REPL commands ─────────────────────────────────────
            if stripped.startswith("/load "):
                sid = stripped[6:].strip()
                try:
                    loaded = Session(session_id=sid)
                    loaded_messages = loaded.load()
                    messages.clear()
                    messages.append({"role": "system", "content": config.system_prompt})
                    messages.extend(loaded_messages)
                    session = loaded
                    console.print(
                        f"[dim]Loaded session {sid[:8]}… "
                        f"({len(loaded_messages)} messages)[/dim]\n"
                    )
                except FileNotFoundError:
                    console.print(f"[red]Session not found:[/red] {sid}\n")
                continue

            if stripped == "/sessions":
                ids = Session.list_sessions()
                if ids:
                    for sid in ids[:10]:
                        marker = " [bold cyan]←[/bold cyan]" if sid == session.session_id else ""
                        console.print(f"  {sid[:8]}…{marker}")
                else:
                    console.print("[dim]No sessions saved yet.[/dim]")
                console.print()
                continue

            # ── normal agent turn ──────────────────────────────────────────
            console.print()
            accumulated = ""
            prev_len = len(messages)

            with Live(
                console=console, refresh_per_second=15, vertical_overflow="visible"
            ) as live:

                def on_tool_call(name: str, args: dict) -> None:
                    live.console.print(
                        f"  [bold blue]→[/bold blue] {name}({_fmt_args(args)})"
                    )

                def on_tool_result(name: str, result: str) -> None:
                    live.console.print(
                        f"  [bold green]←[/bold green] {_fmt_result(name, result)}"
                    )

                def on_token(token: str) -> None:
                    nonlocal accumulated
                    accumulated += token
                    live.update(Markdown(accumulated))

                def on_write_confirm(name: str, args: dict, diff: str) -> bool:
                    live.console.print()
                    live.console.print(
                        f"  [bold yellow]✎[/bold yellow] {name}({_fmt_args(args)})"
                    )
                    if diff:
                        live.console.print(
                            Syntax(diff, "diff", theme="ansi_dark", word_wrap=True)
                        )
                    else:
                        live.console.print("[dim]  (no changes detected)[/dim]")
                    return Confirm.ask("  Apply?", default=True, console=live.console)

                await run_agent_turn(
                    client, registry, messages, user_input,
                    on_token=on_token,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    on_write_confirm=on_write_confirm,
                )

            for msg in messages[prev_len:]:
                session.append(msg)

            console.print()
    finally:
        for conn in mcp_connections:
            try:
                await conn.close()
            except BaseException:
                pass


def main() -> None:
    config = load_config()
    try:
        asyncio.run(run_repl(config))
    except KeyboardInterrupt:
        pass
