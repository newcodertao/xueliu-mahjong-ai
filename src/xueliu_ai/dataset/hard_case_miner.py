from __future__ import annotations

from pathlib import Path

from xueliu_ai.paths import resolve_path


def hard_case_dir(kind: str = "mis_detected") -> Path:
    root = resolve_path("data/hard_cases") / kind
    root.mkdir(parents=True, exist_ok=True)
    return root
