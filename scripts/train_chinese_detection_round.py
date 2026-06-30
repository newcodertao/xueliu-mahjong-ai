from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


PROJECT = Path(r"F:\xueliu-mahjong-ai")

PREVIOUS_MODEL = (
    PROJECT
    / "models"
    / "yolo"
    / "xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7_merged_dense_v1.pt"
)
FINAL_MODEL = (
    PROJECT
    / "models"
    / "yolo"
    / "xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7_merged_dense_chinese_v1.pt"
)
SOURCE = PROJECT / "external_datasets" / "roboflow_chinese_mahjong_detection_v4"
DATASET = PROJECT / "datasets" / "xueliu_tiles_chinese_detection_v4_rotated"
RUN_NAME = "chinese_detection_after_merged_dense_v1"

BACKTESTS = [
    (
        Path(r"G:\video\20260625222825.mp4"),
        PROJECT / "data" / "backtests" / "video_20260625222825_chinese_detection_v1",
    ),
    (
        PROJECT / "data" / "input_videos" / "phone_landscape_20260629_154234.mp4",
        PROJECT / "data" / "backtests" / "phone_landscape_chinese_detection_v1",
    ),
]

ROTATIONS = {
    "r0": None,
    "r90": cv2.ROTATE_90_CLOCKWISE,
    "r180": cv2.ROTATE_180,
    "r270": cv2.ROTATE_90_COUNTERCLOCKWISE,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUIT_MAP = {"m": "W", "p": "T", "s": "B"}
TARGET_NAMES = [f"{rank}{suit}" for suit in ("W", "T", "B") for rank in range(1, 10)]
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_NAMES)}


def main() -> None:
    build_rotated_dataset()
    train_model()
    copy_best_model()
    run_backtests()
    write_round_manifest()


def build_rotated_dataset() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val", "test"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    source_yaml = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8"))
    class_map = build_class_map(list(source_yaml["names"]))

    images = 0
    boxes_kept = 0
    boxes_skipped = 0
    for source_split, target_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        src_images = SOURCE / source_split / "images"
        src_labels = SOURCE / source_split / "labels"
        for image_path in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            label_path = src_labels / f"{image_path.stem}.txt"
            raw_lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            converted_base, kept, skipped = convert_lines(raw_lines, class_map)
            boxes_skipped += skipped
            if not converted_base:
                continue
            for rotation_name, rotation_code in ROTATIONS.items():
                rotated = image if rotation_code is None else cv2.rotate(image, rotation_code)
                out_stem = f"chinese_{image_path.stem}_{rotation_name}"
                out_image = DATASET / "images" / target_split / f"{out_stem}.jpg"
                out_label = DATASET / "labels" / target_split / f"{out_stem}.txt"
                cv2.imwrite(str(out_image), rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                rotated_lines = [
                    line for line in (rotate_yolo_line(line, rotation_name) for line in converted_base) if line
                ]
                out_label.write_text(
                    "\n".join(rotated_lines) + ("\n" if rotated_lines else ""),
                    encoding="utf-8",
                )
                images += 1
                boxes_kept += len(rotated_lines)

    write_data_yaml(DATASET)
    summary = {
        "source": str(SOURCE),
        "output": str(DATASET),
        "images": images,
        "boxes_kept_after_rotation": boxes_kept,
        "source_boxes_skipped": boxes_skipped,
        "rotations": list(ROTATIONS),
        "class_mapping": "m->W, p->T, s->B; f/z skipped",
    }
    (DATASET / "build_summary.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(summary, flush=True)


def build_class_map(source_names: list[str]) -> dict[int, int]:
    class_map: dict[int, int] = {}
    for index, source_name in enumerate(source_names):
        source_name = str(source_name).strip().lower()
        if len(source_name) != 2 or not source_name[0].isdigit():
            continue
        rank = int(source_name[0])
        target_suit = SUIT_MAP.get(source_name[1])
        if not target_suit:
            continue
        target_name = f"{rank}{target_suit}"
        class_map[index] = TARGET_INDEX[target_name]
    return class_map


def convert_lines(raw_lines: list[str], class_map: dict[int, int]) -> tuple[list[str], int, int]:
    converted: list[str] = []
    kept = 0
    skipped = 0
    for raw_line in raw_lines:
        parts = raw_line.strip().split()
        if len(parts) != 5:
            skipped += 1
            continue
        source_cls = int(float(parts[0]))
        target_cls = class_map.get(source_cls)
        if target_cls is None:
            skipped += 1
            continue
        converted.append(" ".join([str(target_cls), *parts[1:]]))
        kept += 1
    return converted, kept, skipped


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


def write_data_yaml(dataset: Path) -> None:
    names = {index: name for index, name in enumerate(TARGET_NAMES)}
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def train_model() -> None:
    if not PREVIOUS_MODEL.exists():
        raise FileNotFoundError(PREVIOUS_MODEL)
    model = YOLO(str(PREVIOUS_MODEL))
    model.train(
        data=str(DATASET / "data.yaml"),
        epochs=8,
        imgsz=960,
        batch=12,
        device=0,
        workers=4,
        patience=4,
        project=str(PROJECT / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        cache=False,
        verbose=True,
    )


def copy_best_model() -> None:
    best = PROJECT / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, FINAL_MODEL)
    print(f"Saved rollback-safe model: {FINAL_MODEL}", flush=True)


def run_backtests() -> None:
    for video, output in BACKTESTS:
        if not video.exists():
            print(f"Skipping missing backtest video: {video}", flush=True)
            continue
        command = [
            sys.executable,
            str(PROJECT / "scripts" / "backtest_video.py"),
            "--video",
            str(video),
            "--model",
            str(FINAL_MODEL),
            "--output",
            str(output),
            "--start-seconds",
            "1.25",
            "--every-seconds",
            "5",
            "--paired-offset-seconds",
            "2",
            "--max-frames",
            "96",
            "--imgsz",
            "960",
            "--conf",
            "0.55",
            "--iou",
            "0.45",
        ]
        subprocess.run(command, cwd=str(PROJECT), check=True)


def write_round_manifest() -> None:
    manifest = PROJECT / "models" / "yolo" / "training_rounds.yaml"
    existing = yaml.safe_load(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    rounds = existing.get("rounds", []) if isinstance(existing, dict) else []
    rounds.append(
        {
            "name": "chinese_detection_v1",
            "model": str(FINAL_MODEL),
            "previous_model": str(PREVIOUS_MODEL),
            "run_dir": str(PROJECT / "runs" / "detect" / RUN_NAME),
            "dataset": str(DATASET),
            "backtests": [str(output) for _video, output in BACKTESTS],
            "rollback_note": "Use previous_model to roll back if Chinese Mahjong Detection fine-tuning hurts real-table behavior.",
        }
    )
    manifest.write_text(
        yaml.safe_dump({"rounds": rounds}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
