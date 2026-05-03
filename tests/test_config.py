import tomllib
from pathlib import Path

import pytest

from local_agent.config import AgentConfig, McpServerConfig, load_config


# ── defaults ──────────────────────────────────────────────────────────────────

def test_default_model():
    cfg = AgentConfig()
    assert cfg.model == "qwen2.5-coder:7b"


def test_default_ollama_base_url():
    cfg = AgentConfig()
    assert cfg.ollama_base_url == "http://localhost:11434"


def test_default_mcp_servers_empty():
    cfg = AgentConfig()
    assert cfg.mcp_servers == []


def test_default_system_prompt_is_nonempty():
    cfg = AgentConfig()
    assert len(cfg.system_prompt) > 0


# ── first-run file creation ────────────────────────────────────────────────────

def test_config_file_created_on_first_run(tmp_path):
    config_path = tmp_path / "config.toml"
    assert not config_path.exists()
    load_config(config_path)
    assert config_path.exists()


def test_created_config_is_valid_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    load_config(config_path)
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    assert "model" in data


# ── loading from file ──────────────────────────────────────────────────────────

def test_loads_model_from_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "qwen2.5-coder:14b"\n')
    cfg = load_config(config_path)
    assert cfg.model == "qwen2.5-coder:14b"


def test_loads_ollama_base_url_from_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('ollama_base_url = "http://192.168.1.10:11434"\n')
    cfg = load_config(config_path)
    assert cfg.ollama_base_url == "http://192.168.1.10:11434"


def test_missing_keys_fall_back_to_defaults(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "qwen2.5-coder:1.5b"\n')
    cfg = load_config(config_path)
    assert cfg.ollama_base_url == "http://localhost:11434"


def test_loads_mcp_servers(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[[mcp_servers]]\nname = "filesystem"\ncommand = "npx"\nargs = ["-y", "server-fs"]\n'
    )
    cfg = load_config(config_path)
    assert len(cfg.mcp_servers) == 1
    assert cfg.mcp_servers[0].name == "filesystem"
    assert cfg.mcp_servers[0].command == "npx"
    assert cfg.mcp_servers[0].args == ["-y", "server-fs"]


def test_mcp_server_model():
    srv = McpServerConfig(name="git", command="uvx", args=["mcp-server-git"])
    assert srv.name == "git"
    assert srv.args == ["mcp-server-git"]
