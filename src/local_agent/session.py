"""JSONL session persistence."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT_SESSIONS_DIR = Path.home() / ".local-agent" / "sessions"


class Session:
    def __init__(
        self,
        session_id: str | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._base_dir = base_dir or _DEFAULT_SESSIONS_DIR
        self.path = self._base_dir / f"{self.session_id}.jsonl"

    def append(self, message: dict) -> None:
        """Append a message to the session file. System messages are skipped."""
        if message.get("role") == "system":
            return
        self._base_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "role": message["role"],
            "content": message.get("content", ""),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def load(self) -> list[dict]:
        """Load messages from the session file as plain role/content dicts."""
        if not self.path.exists():
            raise FileNotFoundError(f"Session not found: {self.session_id}")
        messages = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages.append({"role": record["role"], "content": record["content"]})
        return messages

    @classmethod
    def list_sessions(cls, base_dir: Path | None = None) -> list[str]:
        """Return session IDs sorted by modification time (newest first)."""
        d = base_dir or _DEFAULT_SESSIONS_DIR
        if not d.exists():
            return []
        files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files]
