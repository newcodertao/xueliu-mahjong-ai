from __future__ import annotations

from pathlib import Path

from xueliu_ai.dataset.dataset_builder import build_yolo_dataset


def ensure_yolo_export(dataset_dir: str | Path = "datasets/xueliu_tiles_v1") -> Path:
    return build_yolo_dataset(dataset_dir)
