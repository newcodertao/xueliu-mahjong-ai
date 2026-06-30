from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
