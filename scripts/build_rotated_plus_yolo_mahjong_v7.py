from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import yaml


BASE_DATASET = Path(r"F:\xueliu-mahjong-ai\datasets\xueliu_tiles_rotated_plus_human_round1")
EXTRA_DATASET = Path(r"F:\xueliu-mahjong-ai\datasets\xueliu_tiles_roboflow_yolo_mahjong_v7")
OUT = Path(r"F:\xueliu-mahjong-ai\datasets\xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7")

ROTATIONS = {
    "r0": None,
    "r90": cv2.ROTATE_90_CLOCKWISE,
    "r180": cv2.ROTATE_180,
    "r270": cv2.ROTATE_90_COUNTERCLOCKWISE,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    base_images, base_boxes = copy_base_dataset()
    extra_images, extra_boxes = add_rotated_extra_dataset()
    write_data_yaml()

    summary = {
        "base_dataset": str(BASE_DATASET),
        "extra_dataset": str(EXTRA_DATASET),
        "output": str(OUT),
        "base_images": base_images,
        "base_boxes": base_boxes,
        "extra_rotated_images": extra_images,
        "extra_rotated_boxes": extra_boxes,
        "rotations": list(ROTATIONS),
    }
    (OUT / "build_summary.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(summary)
    for split in ("train", "val", "test"):
        images = len(list((OUT / "images" / split).glob("*")))
        labels = len(list((OUT / "labels" / split).glob("*.txt")))
        print(split, images, labels)


def copy_base_dataset() -> tuple[int, int]:
    images = 0
    boxes = 0
    for split in ("train", "val", "test"):
        src_images = BASE_DATASET / "images" / split
        src_labels = BASE_DATASET / "labels" / split
        if not src_images.exists():
            continue
        for image_path in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label_path = src_labels / f"{image_path.stem}.txt"
            dst_image = OUT / "images" / split / f"base_{image_path.name}"
            dst_label = OUT / "labels" / split / f"base_{image_path.stem}.txt"
            link_or_copy(image_path, dst_image)
            if label_path.exists():
                text = label_path.read_text(encoding="utf-8")
                dst_label.write_text(text, encoding="utf-8")
                boxes += sum(1 for line in text.splitlines() if line.strip())
            else:
                dst_label.write_text("", encoding="utf-8")
            images += 1
    return images, boxes


def add_rotated_extra_dataset() -> tuple[int, int]:
    images = 0
    boxes = 0
    for split in ("train", "val", "test"):
        src_images = EXTRA_DATASET / "images" / split
        src_labels = EXTRA_DATASET / "labels" / split
        if not src_images.exists():
            continue
        for image_path in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            label_path = src_labels / f"{image_path.stem}.txt"
            raw_lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            for rotation_name, rotation_code in ROTATIONS.items():
                rotated = image if rotation_code is None else cv2.rotate(image, rotation_code)
                out_stem = f"ymv7_{image_path.stem}_{rotation_name}"
                out_image = OUT / "images" / split / f"{out_stem}.jpg"
                out_label = OUT / "labels" / split / f"{out_stem}.txt"
                cv2.imwrite(str(out_image), rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                converted = [
                    converted_line
                    for converted_line in (rotate_yolo_line(line, rotation_name) for line in raw_lines)
                    if converted_line
                ]
                out_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
                images += 1
                boxes += len(converted)
    return images, boxes


def rotate_yolo_line(line: str, rotation: str) -> str | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = parts[0]
    x, y, width, height = map(float, parts[1:5])
    if rotation == "r0":
        nx, ny, nw, nh = x, y, width, height
    elif rotation == "r90":
        nx, ny, nw, nh = 1.0 - y, x, height, width
    elif rotation == "r180":
        nx, ny, nw, nh = 1.0 - x, 1.0 - y, width, height
    elif rotation == "r270":
        nx, ny, nw, nh = y, 1.0 - x, height, width
    else:
        raise ValueError(rotation)
    return f"{cls} {clamp(nx):.6f} {clamp(ny):.6f} {clamp(nw):.6f} {clamp(nh):.6f}"


def clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_data_yaml() -> None:
    names = {index: f"{rank}{suit}" for index, (suit, rank) in enumerate((suit, rank) for suit in ("W", "T", "B") for rank in range(1, 10))}
    data = {
        "path": str(OUT).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (OUT / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
