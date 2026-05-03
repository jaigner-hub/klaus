import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

_DEFAULT_CONFIG_PATH = Path.home() / ".local-agent" / "config.toml"

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local coding assistant with access to file system tools. "
    "When you use tools, interpret their results and respond in plain English — "
    "never output raw JSON or repeat the tool result verbatim. "
    "Call tools when the user asks you to read or list files; "
    "then summarise what you found in a clear, human-readable way. "
    "Think step by step and prefer reading code before modifying it."
)


class McpServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)


def load_config(path: Path | None = None) -> AgentConfig:
    config_path = path or _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        _write_default(config_path)
        return AgentConfig()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    servers = [McpServerConfig(**s) for s in data.pop("mcp_servers", [])]
    return AgentConfig(**data, mcp_servers=servers)


def _write_default(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'model = "qwen2.5-coder:7b"\n'
        'ollama_base_url = "http://localhost:11434"\n'
    )
