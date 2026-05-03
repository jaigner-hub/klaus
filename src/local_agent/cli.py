import asyncio
import json
from collections.abc import Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt

from local_agent.agent import run_agent_turn
from local_agent.config import AgentConfig, load_config
from local_agent.ollama_client import OllamaClient
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
    messages: list[dict] = [{"role": "system", "content": config.system_prompt}]

    console.print(
        "[bold cyan]Klaus[/bold cyan] — local coding agent  "
        f"[dim]{config.model}[/dim]"
    )
    console.print("[dim]Ctrl-C or Ctrl-D to exit.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]>[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input.strip():
            continue

        console.print()
        accumulated = ""

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

            await run_agent_turn(
                client, registry, messages, user_input,
                on_token=on_token,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
            )

        console.print()


def main() -> None:
    config = load_config()
    try:
        asyncio.run(run_repl(config))
    except KeyboardInterrupt:
        pass
