import json
from pathlib import Path

import pytest

from local_agent.session import Session

# ── construction ───────────────────────────────────────────────────────────────

def test_session_id_is_auto_generated(tmp_path):
    s = Session(base_dir=tmp_path)
    assert s.session_id
    assert len(s.session_id) > 0


def test_session_id_is_unique(tmp_path):
    a = Session(base_dir=tmp_path)
    b = Session(base_dir=tmp_path)
    assert a.session_id != b.session_id


def test_session_id_can_be_provided(tmp_path):
    s = Session(session_id="abc123", base_dir=tmp_path)
    assert s.session_id == "abc123"


def test_session_path_is_in_base_dir(tmp_path):
    s = Session(session_id="xyz", base_dir=tmp_path)
    assert s.path.parent == tmp_path
    assert s.path.name == "xyz.jsonl"


# ── append ─────────────────────────────────────────────────────────────────────

def test_append_creates_file(tmp_path):
    s = Session(base_dir=tmp_path)
    s.append({"role": "user", "content": "hello"})
    assert s.path.exists()


def test_append_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b"
    s = Session(base_dir=nested)
    s.append({"role": "user", "content": "hi"})
    assert s.path.exists()


def test_append_writes_valid_json(tmp_path):
    s = Session(base_dir=tmp_path)
    s.append({"role": "user", "content": "hello"})
    line = s.path.read_text().strip()
    record = json.loads(line)
    assert record["role"] == "user"
    assert record["content"] == "hello"


def test_append_includes_timestamp(tmp_path):
    s = Session(base_dir=tmp_path)
    s.append({"role": "user", "content": "hello"})
    record = json.loads(s.path.read_text().strip())
    assert "timestamp" in record
    assert record["timestamp"]


def test_append_multiple_messages_one_per_line(tmp_path):
    s = Session(base_dir=tmp_path)
    s.append({"role": "user", "content": "hi"})
    s.append({"role": "assistant", "content": "hello"})
    lines = [l for l in s.path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_append_tool_message(tmp_path):
    s = Session(base_dir=tmp_path)
    s.append({"role": "tool", "content": '{"result": "ok"}'})
    record = json.loads(s.path.read_text().strip())
    assert record["role"] == "tool"


def test_append_skips_system_message(tmp_path):
    s = Session(base_dir=tmp_path)
    s.append({"role": "system", "content": "you are helpful"})
    assert not s.path.exists()


# ── load ───────────────────────────────────────────────────────────────────────

def test_load_restores_messages(tmp_path):
    s = Session(session_id="test", base_dir=tmp_path)
    s.append({"role": "user", "content": "question"})
    s.append({"role": "assistant", "content": "answer"})

    loaded = s.load()
    assert len(loaded) == 2
    assert loaded[0]["role"] == "user"
    assert loaded[0]["content"] == "question"
    assert loaded[1]["role"] == "assistant"
    assert loaded[1]["content"] == "answer"


def test_load_preserves_order(tmp_path):
    s = Session(base_dir=tmp_path)
    for i in range(5):
        s.append({"role": "user", "content": str(i)})
    loaded = s.load()
    assert [m["content"] for m in loaded] == ["0", "1", "2", "3", "4"]


def test_load_missing_session_raises(tmp_path):
    s = Session(session_id="ghost", base_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        s.load()


def test_load_strips_timestamps(tmp_path):
    s = Session(base_dir=tmp_path)
    s.append({"role": "user", "content": "hi"})
    loaded = s.load()
    assert "timestamp" not in loaded[0]


# ── list_sessions ──────────────────────────────────────────────────────────────

def test_list_sessions_returns_ids(tmp_path):
    Session(session_id="aaa", base_dir=tmp_path).append({"role": "user", "content": "x"})
    Session(session_id="bbb", base_dir=tmp_path).append({"role": "user", "content": "y"})
    ids = Session.list_sessions(base_dir=tmp_path)
    assert "aaa" in ids
    assert "bbb" in ids


def test_list_sessions_empty_when_no_dir(tmp_path):
    ids = Session.list_sessions(base_dir=tmp_path / "nonexistent")
    assert ids == []


def test_list_sessions_ignores_non_jsonl(tmp_path):
    (tmp_path / "notes.txt").write_text("ignore me")
    Session(session_id="real", base_dir=tmp_path).append({"role": "user", "content": "x"})
    ids = Session.list_sessions(base_dir=tmp_path)
    assert ids == ["real"]
