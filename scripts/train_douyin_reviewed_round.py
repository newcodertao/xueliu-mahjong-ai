from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


PROJECT = Path(r"F:\xueliu-mahjong-ai")
TARGET_NAMES = [f"{rank}{suit}" for suit in ("W", "T", "B") for rank in range(1, 10)]
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on manually reviewed Douyin prelabels.")
    parser.add_argument(
        "--review-dir",
        default=str(PROJECT / "data" / "labeling" / "douyin_20260630_prelabeled_chinese_v1" / "images_to_review"),
    )
    parser.add_argument("--reviewed-after", default="2026-06-30 12:00:00")
    parser.add_argument(
        "--base-model",
        default=str(
            PROJECT
            / "models"
            / "yolo"
            / "xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7_merged_dense_chinese_v1.pt"
        ),
    )
    parser.add_argument(
        "--output-model",
        default=str(
            PROJECT
            / "models"
            / "yolo"
            / "xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7_merged_dense_chinese_douyin_reviewed_v1.pt"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT / "datasets" / "xueliu_tiles_douyin_reviewed_v1"),
    )
    parser.add_argument("--run-name", default="douyin_reviewed_after_chinese_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_dir)
    dataset = Path(args.dataset)
    base_model = Path(args.base_model)
    output_model = Path(args.output_model)
    threshold = datetime.strptime(args.reviewed_after, "%Y-%m-%d %H:%M:%S")

    reviewed = collect_reviewed_json(review_dir, threshold)
    if not reviewed:
        raise SystemExit("No reviewed JSON files found.")
    build_dataset(review_dir, reviewed, dataset)
    train(dataset, base_model, args)
    best = PROJECT / "runs" / "detect" / args.run_name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output_model)
    write_manifest(args, dataset, base_model, output_model, reviewed)
    print(f"reviewed_images={len(reviewed)}")
    print(f"dataset={dataset}")
    print(f"model={output_model}")


def collect_reviewed_json(review_dir: Path, threshold: datetime) -> list[Path]:
    return sorted(
        path for path in review_dir.glob("*.json") if datetime.fromtimestamp(path.stat().st_mtime) > threshold
    )


def build_dataset(review_dir: Path, reviewed_json: list[Path], dataset: Path) -> None:
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(20260630)
    shuffled = reviewed_json[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * 0.2)) if len(shuffled) > 4 else 0
    val_set = set(shuffled[:val_count])

    stats = {"train_images": 0, "val_images": 0, "boxes": 0, "skipped_shapes": 0}
    for json_path in shuffled:
        split = "val" if json_path in val_set else "train"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = review_dir / data.get("imagePath", f"{json_path.stem}.jpg")
        if not image_path.exists():
            image_path = find_image(review_dir, json_path.stem)
        if image_path is None:
            continue
        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
        lines, skipped = shapes_to_yolo(data.get("shapes", []), width, height)
        stats["skipped_shapes"] += skipped
        stats["boxes"] += len(lines)
        shutil.copy2(image_path, dataset / "images" / split / image_path.name)
        (dataset / "labels" / split / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        stats[f"{split}_images"] += 1

    write_data_yaml(dataset)
    (dataset / "build_summary.yaml").write_text(
        yaml.safe_dump(stats, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def find_image(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def shapes_to_yolo(shapes: list[dict], width: int, height: int) -> tuple[list[str], int]:
    lines: list[str] = []
    skipped = 0
    for shape in shapes:
        label = str(shape.get("label", "")).strip().upper()
        target = TARGET_INDEX.get(label)
        points = shape.get("points") or []
        if target is None or len(points) < 2:
            skipped += 1
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(f"{target} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    return lines, skipped


def write_data_yaml(dataset: Path) -> None:
    names = {index: name for index, name in enumerate(TARGET_NAMES)}
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def train(dataset: Path, base_model: Path, args: argparse.Namespace) -> None:
    if not base_model.exists():
        raise FileNotFoundError(base_model)
    model = YOLO(str(base_model))
    model.train(
        data=str(dataset / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        patience=6,
        project=str(PROJECT / "runs" / "detect"),
        name=args.run_name,
        exist_ok=True,
        pretrained=True,
        cache=False,
        verbose=True,
    )


def write_manifest(
    args: argparse.Namespace,
    dataset: Path,
    base_model: Path,
    output_model: Path,
    reviewed: list[Path],
) -> None:
    manifest = PROJECT / "models" / "yolo" / "training_rounds.yaml"
    existing = yaml.safe_load(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    rounds = existing.get("rounds", []) if isinstance(existing, dict) else []
    rounds.append(
        {
            "name": "douyin_reviewed_v1",
            "model": str(output_model),
            "previous_model": str(base_model),
            "run_dir": str(PROJECT / "runs" / "detect" / args.run_name),
            "dataset": str(dataset),
            "reviewed_images": len(reviewed),
            "rollback_note": "Use previous_model to roll back if this small reviewed Douyin round overfits.",
        }
    )
    manifest.write_text(
        yaml.safe_dump({"rounds": rounds}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
