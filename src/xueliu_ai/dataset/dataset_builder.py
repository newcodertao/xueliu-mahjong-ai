from __future__ import annotations

from pathlib import Path

import yaml

from xueliu_ai.mahjong.tiles import TILE_NAMES
from xueliu_ai.paths import resolve_path


SUBDIRS = (
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test",
)


def build_yolo_dataset(dataset_dir: str | Path = "datasets/xueliu_tiles_v1") -> Path:
    root = resolve_path(dataset_dir)
    for subdir in SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    data = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(TILE_NAMES)},
    }
    with (root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return root
