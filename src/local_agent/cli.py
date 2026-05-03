import asyncio
from collections.abc import Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt

from local_agent.config import AgentConfig, load_config
from local_agent.ollama_client import OllamaClient


async def run_turn(
    client: OllamaClient,
    messages: list[dict],
    user_input: str,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Process one turn. Mutates messages in-place; returns full response."""
    messages.append({"role": "user", "content": user_input})
    parts: list[str] = []
    async for token in client.chat_stream(messages):
        parts.append(token)
        if on_token is not None:
            on_token(token)
    response = "".join(parts)
    messages.append({"role": "assistant", "content": response})
    return response


async def run_repl(config: AgentConfig) -> None:
    console = Console()
    client = OllamaClient(config)
    messages: list[dict] = [{"role": "system", "content": config.system_prompt}]

    console.print("[bold cyan]Klaus[/bold cyan] — local coding agent  "
                  f"[dim]{config.model}[/dim]")
    console.print("[dim]Ctrl-C or Ctrl-D to exit.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]>[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input.strip():
            continue

        accumulated = ""

        def update_live(token: str, _live: Live) -> None:
            nonlocal accumulated
            accumulated += token
            _live.update(Markdown(accumulated))

        console.print()
        with Live(
            console=console, refresh_per_second=15, vertical_overflow="visible"
        ) as live:
            await run_turn(
                client, messages, user_input,
                on_token=lambda t: update_live(t, live),
            )
        console.print()


def main() -> None:
    config = load_config()
    try:
        asyncio.run(run_repl(config))
    except KeyboardInterrupt:
        pass
