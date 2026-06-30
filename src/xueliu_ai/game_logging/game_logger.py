from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from xueliu_ai.paths import resolve_path


class GameLogger:
    def __init__(self, path: str | Path = "data/games/session.jsonl") -> None:
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
