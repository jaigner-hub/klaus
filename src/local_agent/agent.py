from collections.abc import Callable

from local_agent.ollama_client import OllamaClient
from local_agent.safety.confirm import guarded_dispatch
from local_agent.tools.registry import ToolRegistry

MAX_ITERATIONS = 25


async def run_agent_turn(
    client: OllamaClient,
    registry: ToolRegistry,
    messages: list[dict],
    user_input: str,
    on_token: Callable[[str], None] | None = None,
    on_tool_call: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str], None] | None = None,
    on_write_confirm: Callable[[str, dict, str], bool] | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Run the tool-use loop for one user turn.

    Appends user + assistant (+ tool) messages to `messages` in-place.
    Returns the final assistant response text.
    """
    messages.append({"role": "user", "content": user_input})
    tools = registry.to_ollama_schemas()

    for _ in range(max_iterations):
        text, tool_calls = await client.chat_with_tools(
            messages, tools, on_token=on_token
        )

        if not tool_calls:
            messages.append({"role": "assistant", "content": text})
            return text

        messages.append({"role": "assistant", "content": text})

        for call in tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)

            if on_tool_call:
                on_tool_call(name, args)

            result = await guarded_dispatch(name, args, registry, on_write_confirm)

            if on_tool_result:
                on_tool_result(name, result)

            messages.append({"role": "tool", "content": result})

    warning = "[Warning: agent reached the iteration limit without a final response]"
    messages.append({"role": "assistant", "content": warning})
    return warning
