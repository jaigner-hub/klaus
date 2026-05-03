import inspect
import json
from collections.abc import Callable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable, dict]] = {}

    def register(self, name: str, fn: Callable, schema: dict) -> None:
        self._tools[name] = (fn, schema)

    def to_ollama_schemas(self) -> list[dict]:
        return [schema for _, schema in self._tools.values()]

    async def dispatch(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        fn, _ = self._tools[name]
        result = fn(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) else json.dumps(result)

    @classmethod
    def with_builtins(cls) -> "ToolRegistry":
        from local_agent.tools.builtin import (
            SCHEMAS,
            list_directory,
            read_file,
        )

        reg = cls()
        reg.register("read_file", read_file, SCHEMAS[0])
        reg.register("list_directory", list_directory, SCHEMAS[1])
        return reg
