import json
from pathlib import Path

_MAX_READ_BYTES = 100 * 1024  # 100 KB


def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if not p.is_file():
        return json.dumps({"error": f"Not a file: {path}"})
    raw = p.read_bytes()
    truncated = len(raw) > _MAX_READ_BYTES
    content = raw[:_MAX_READ_BYTES].decode("utf-8", errors="replace")
    return json.dumps({
        "path": str(p),
        "size": len(raw),
        "truncated": truncated,
        "content": content,
    })


def list_directory(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"Directory not found: {path}"})
    if not p.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})
    entries = []
    for child in sorted(p.iterdir()):
        stat = child.stat()
        entries.append({
            "name": child.name,
            "type": "directory" if child.is_dir() else "file",
            "size": stat.st_size,
        })
    return json.dumps({"path": str(p), "entries": entries})


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. "
                "Returns the content, size in bytes, and whether it was truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List the contents of a directory. "
                "Returns each entry's name, type (file or directory), and size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the directory.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]
