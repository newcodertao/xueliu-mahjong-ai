from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from xueliu_ai.mahjong.tiles import TILE_TO_INDEX
from xueliu_ai.paths import resolve_path


SUFFIX_TO_TARGET_SUIT = {
    "C": "W",  # characters / wan
    "D": "T",  # dots / tong
    "B": "B",  # bamboo / tiao
}


@dataclass(frozen=True)
class ConversionResult:
    source: str
    output: str
    images_copied: int
    labels_written: int
    boxes_kept: int
    boxes_skipped: int


def convert_roboflow_mahjong_to_xueliu(
    source_dir: str | Path = "external_datasets/roboflow_mahjong",
    output_dir: str | Path = "datasets/xueliu_tiles_roboflow_v1",
) -> ConversionResult:
    source = resolve_path(source_dir)
    output = resolve_path(output_dir)
    source_yaml = source / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(source_yaml)

    source_data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_names = list(source_data["names"])
    class_map = _build_class_map(source_names)

    images_copied = 0
    labels_written = 0
    boxes_kept = 0
    boxes_skipped = 0

    for split, output_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        src_images = source / split / "images"
        src_labels = source / split / "labels"
        dst_images = output / "images" / output_split
        dst_labels = output / "labels" / output_split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for label_path in sorted(src_labels.glob("*.txt")):
            converted_lines, kept, skipped = _convert_label_file(label_path, class_map)
            boxes_kept += kept
            boxes_skipped += skipped
            if not converted_lines:
                continue

            image_path = _find_image_for_label(src_images, label_path.stem)
            if image_path is None:
                continue
            shutil.copy2(image_path, dst_images / image_path.name)
            (dst_labels / label_path.name).write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
            images_copied += 1
            labels_written += 1

    _write_data_yaml(output)
    return ConversionResult(
        source=str(source),
        output=str(output),
        images_copied=images_copied,
        labels_written=labels_written,
        boxes_kept=boxes_kept,
        boxes_skipped=boxes_skipped,
    )


def _build_class_map(source_names: list[str]) -> dict[int, int]:
    class_map: dict[int, int] = {}
    for source_index, source_name in enumerate(source_names):
        if len(source_name) < 2 or not source_name[0].isdigit():
            continue
        rank = source_name[0]
        suffix = source_name[1:]
        target_suit = SUFFIX_TO_TARGET_SUIT.get(suffix)
        if not target_suit:
            continue
        target_name = f"{rank}{target_suit}"
        if target_name in TILE_TO_INDEX:
            class_map[source_index] = TILE_TO_INDEX[target_name]
    return class_map


def _convert_label_file(label_path: Path, class_map: dict[int, int]) -> tuple[list[str], int, int]:
    converted: list[str] = []
    kept = 0
    skipped = 0
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            skipped += 1
            continue
        source_class = int(float(parts[0]))
        target_class = class_map.get(source_class)
        if target_class is None:
            skipped += 1
            continue
        converted.append(" ".join([str(target_class), *parts[1:]]))
        kept += 1
    return converted, kept, skipped


def _find_image_for_label(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _write_data_yaml(output: Path) -> None:
    names = {index: tile for tile, index in TILE_TO_INDEX.items()}
    data = {
        "path": str(output).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
