from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from xueliu_ai.paths import resolve_path


class GameLogger:
    def __init__(
        self,
        path: str | Path = "data/games/session.jsonl",
        *,
        max_bytes: int = 100 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self.path = resolve_path(path)
        self.max_bytes = max(0, max_bytes)
        self.backup_count = max(0, backup_count)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._rotate_if_needed(len(line.encode("utf-8")))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if not self.max_bytes or not self.path.exists():
            return
        if self.path.stat().st_size + incoming_bytes <= self.max_bytes:
            return
        if self.backup_count == 0:
            self.path.unlink()
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))
