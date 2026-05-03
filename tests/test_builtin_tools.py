import json

from local_agent.tools.builtin import list_directory, read_file

# ── read_file ──────────────────────────────────────────────────────────────────

def test_read_file_returns_content(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result = json.loads(read_file(str(f)))
    assert result["content"] == "hello world"


def test_read_file_returns_path_and_size(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result = json.loads(read_file(str(f)))
    assert result["path"] == str(f)
    assert result["size"] == len("hello world")


def test_read_file_not_truncated_for_small_file(tmp_path):
    f = tmp_path / "small.txt"
    f.write_text("small")
    result = json.loads(read_file(str(f)))
    assert result["truncated"] is False


def test_read_file_truncates_at_100kb(tmp_path):
    f = tmp_path / "big.txt"
    content = "x" * (110 * 1024)
    f.write_bytes(content.encode())
    result = json.loads(read_file(str(f)))
    assert result["truncated"] is True
    assert len(result["content"]) == 100 * 1024


def test_read_file_missing_returns_error():
    result = json.loads(read_file("/nonexistent/path/file.txt"))
    assert "error" in result


def test_read_file_on_directory_returns_error(tmp_path):
    result = json.loads(read_file(str(tmp_path)))
    assert "error" in result


# ── list_directory ─────────────────────────────────────────────────────────────

def test_list_directory_returns_entries(tmp_path):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    result = json.loads(list_directory(str(tmp_path)))
    names = {e["name"] for e in result["entries"]}
    assert "a.py" in names
    assert "b.txt" in names


def test_list_directory_entry_has_type_and_size(tmp_path):
    (tmp_path / "file.py").write_text("content")
    (tmp_path / "subdir").mkdir()
    result = json.loads(list_directory(str(tmp_path)))
    by_name = {e["name"]: e for e in result["entries"]}
    assert by_name["file.py"]["type"] == "file"
    assert by_name["file.py"]["size"] == len("content")
    assert by_name["subdir"]["type"] == "directory"


def test_list_directory_returns_path(tmp_path):
    result = json.loads(list_directory(str(tmp_path)))
    assert result["path"] == str(tmp_path)


def test_list_directory_missing_returns_error():
    result = json.loads(list_directory("/nonexistent/path"))
    assert "error" in result


def test_list_directory_on_file_returns_error(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    result = json.loads(list_directory(str(f)))
    assert "error" in result
