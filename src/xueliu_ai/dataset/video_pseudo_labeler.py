from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from xueliu_ai.mahjong.tiles import TILE_NAMES
from xueliu_ai.paths import resolve_path
from xueliu_ai.vision.yolo_detector import YoloDetector


@dataclass(frozen=True)
class VideoInfo:
    opened: bool
    frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class ExtractFramesResult:
    video: str
    output_dir: str
    fps: float
    duration_seconds: float
    saved: int


@dataclass(frozen=True)
class PseudoLabelResult:
    scanned: int
    kept: int
    skipped: int
    boxes: int
    dataset_dir: str


def probe_video(video_path: str | Path) -> VideoInfo:
    video = resolve_path(video_path)
    capture = cv2.VideoCapture(str(video))
    opened = capture.isOpened()
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    capture.release()
    duration = frames / fps if fps else 0.0
    return VideoInfo(
        opened=opened,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        duration_seconds=duration,
    )


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_seconds: float = 3.0,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    max_frames: int | None = None,
) -> ExtractFramesResult:
    video = resolve_path(video_path)
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0
    end = duration if end_seconds is None else min(end_seconds, duration)
    step = max(1, int(round(every_seconds * fps)))
    frame_index = max(0, int(round(start_seconds * fps)))
    end_frame = int(round(end * fps))

    saved = 0
    stem = video.stem
    while frame_index <= end_frame:
        if max_frames is not None and saved >= max_frames:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        seconds = frame_index / fps
        out_path = output / f"{stem}_{seconds:08.2f}s.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved += 1
        frame_index += step

    capture.release()
    return ExtractFramesResult(
        video=str(video),
        output_dir=str(output),
        fps=fps,
        duration_seconds=duration,
        saved=saved,
    )


def pseudo_label_frames(
    model_path: str | Path,
    image_dir: str | Path,
    dataset_dir: str | Path,
    conf: float = 0.85,
    iou: float = 0.5,
    min_detections: int = 5,
    val_ratio: float = 0.15,
) -> PseudoLabelResult:
    image_root = resolve_path(image_dir)
    dataset = resolve_path(dataset_dir)
    if dataset.exists():
        shutil.rmtree(dataset)
    _create_dataset_dirs(dataset)

    detector = YoloDetector(model_path)
    name_to_id = {name: index for index, name in enumerate(TILE_NAMES)}
    image_paths = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    scanned = kept = skipped = boxes = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]
        detections = detector.detect_image(image_path, conf=conf, iou=iou)
        detections = [det for det in detections if det.label in name_to_id]
        scanned += 1
        if len(detections) < min_detections:
            skipped += 1
            continue

        split = _split_for_name(image_path.name, val_ratio=val_ratio)
        target_image = dataset / "images" / split / image_path.name
        target_label = dataset / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text(
            "\n".join(_to_yolo_line(det, name_to_id[det.label], width, height) for det in detections)
            + "\n",
            encoding="utf-8",
        )
        kept += 1
        boxes += len(detections)

    _write_data_yaml(dataset)
    return PseudoLabelResult(
        scanned=scanned,
        kept=kept,
        skipped=skipped,
        boxes=boxes,
        dataset_dir=str(dataset),
    )


def _create_dataset_dirs(dataset: Path) -> None:
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_data_yaml(dataset: Path) -> None:
    data = {
        "path": str(dataset).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(TILE_NAMES)},
    }
    with (dataset / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def _to_yolo_line(det: object, class_id: int, width: int, height: int) -> str:
    x_center = ((det.x1 + det.x2) / 2) / width
    y_center = ((det.y1 + det.y2) / 2) / height
    box_width = (det.x2 - det.x1) / width
    box_height = (det.y2 - det.y1) / height
    values = (class_id, x_center, y_center, box_width, box_height)
    return f"{values[0]} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f} {values[4]:.6f}"


def _split_for_name(name: str, val_ratio: float) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"
