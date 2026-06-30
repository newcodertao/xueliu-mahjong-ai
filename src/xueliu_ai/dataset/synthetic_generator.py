from __future__ import annotations

from pathlib import Path

from xueliu_ai.paths import resolve_path


def generate_synthetic_dataset(output_dir: str | Path = "data/labeled/synthetic") -> Path:
    root = resolve_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Synthetic tile rendering needs tile artwork assets before it can run.")
