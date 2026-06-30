from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


PROJECT = Path(r"F:\xueliu-mahjong-ai")

CURRENT_MODEL = PROJECT / "models" / "yolo" / "xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7.pt"
CURRENT_BACKTEST_README = (
    PROJECT / "data" / "backtests" / "phone_landscape_post_training_yolo_mahjong_v7" / "README.txt"
)

MERGED_SOURCE = PROJECT / "datasets" / "xueliu_tiles_roboflow_mahjong_tiles_merged_v1"
MERGED_ROTATED = PROJECT / "datasets" / "xueliu_tiles_merged_dense_v1_rotated"

RUN_NAME = "merged_dense_after_yolo_mahjong_v7"
FINAL_MODEL = PROJECT / "models" / "yolo" / "xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7_merged_dense_v1.pt"
BACKTEST_ROOT = PROJECT / "data" / "backtests" / "phone_landscape_post_training_yolo_mahjong_v7_merged_dense_v1"
VIDEO = PROJECT / "data" / "input_videos" / "phone_landscape_20260629_154234.mp4"

ROTATIONS = {
    "r0": None,
    "r90": cv2.ROTATE_90_CLOCKWISE,
    "r180": cv2.ROTATE_180,
    "r270": cv2.ROTATE_90_COUNTERCLOCKWISE,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    wait_for_current_round()
    build_merged_rotated_dataset()
    train_merged_dense_model()
    copy_best_model()
    run_backtest()
    write_round_manifest()


def wait_for_current_round() -> None:
    print("Waiting for current yolo_mahjong_v7 round to finish...", flush=True)
    while True:
        if CURRENT_MODEL.exists() and CURRENT_BACKTEST_README.exists():
            print("Current round artifacts found. Continuing to merged dense training.", flush=True)
            return
        time.sleep(60)


def build_merged_rotated_dataset() -> None:
    if MERGED_ROTATED.exists():
        shutil.rmtree(MERGED_ROTATED)
    for split in ("train", "val", "test"):
        (MERGED_ROTATED / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_ROTATED / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = 0
    boxes = 0
    for split in ("train", "val", "test"):
        src_images = MERGED_SOURCE / "images" / split
        src_labels = MERGED_SOURCE / "labels" / split
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
                out_stem = f"merged_{image_path.stem}_{rotation_name}"
                out_image = MERGED_ROTATED / "images" / split / f"{out_stem}.jpg"
                out_label = MERGED_ROTATED / "labels" / split / f"{out_stem}.txt"
                cv2.imwrite(str(out_image), rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                converted = [
                    value
                    for value in (rotate_yolo_line(line, rotation_name) for line in raw_lines)
                    if value
                ]
                out_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
                images += 1
                boxes += len(converted)

    names = {index: f"{rank}{suit}" for index, (suit, rank) in enumerate((suit, rank) for suit in ("W", "T", "B") for rank in range(1, 10))}
    (MERGED_ROTATED / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(MERGED_ROTATED).replace("\\", "/"),
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
    (MERGED_ROTATED / "build_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "source": str(MERGED_SOURCE),
                "output": str(MERGED_ROTATED),
                "images": images,
                "boxes": boxes,
                "rotations": list(ROTATIONS),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print({"merged_rotated_images": images, "merged_rotated_boxes": boxes}, flush=True)


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


def train_merged_dense_model() -> None:
    model = YOLO(str(CURRENT_MODEL))
    model.train(
        data=str(MERGED_ROTATED / "data.yaml"),
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
    shutil.copy2(best, FINAL_MODEL)
    print(f"Saved rollback-safe model: {FINAL_MODEL}", flush=True)


def run_backtest() -> None:
    if BACKTEST_ROOT.exists():
        shutil.rmtree(BACKTEST_ROOT)
    images_dir = BACKTEST_ROOT / "images"
    pred_dir = BACKTEST_ROOT / "predictions_top_left_label"
    images_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(FINAL_MODEL))
    class_set = {f"{rank}{suit}" for suit in ("W", "T", "B") for rank in range(1, 10)}
    colors = {"W": (0, 220, 255), "T": (255, 120, 0), "B": (0, 220, 0)}

    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {VIDEO}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 60.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0
    stem = VIDEO.stem

    seconds = []
    second = 1.25
    while second <= duration and len(seconds) < 96:
        seconds.append(second)
        if second + 2.0 <= duration and len(seconds) < 96:
            seconds.append(second + 2.0)
        second += 5.0

    prediction_rows = []
    summary_rows = []
    for second in seconds:
        frame_index = int(round(second * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        image_name = f"{stem}_{second:08.2f}s.jpg"
        image_path = images_dir / image_name
        cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        detections = []
        results = model.predict(source=str(image_path), imgsz=960, conf=0.55, iou=0.45, verbose=False)
        height, width = frame.shape[:2]
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                label = str(names[int(box.cls[0].item())]).strip().upper()
                if label not in class_set:
                    continue
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                x1 = max(0.0, min(x1, width))
                x2 = max(0.0, min(x2, width))
                y1 = max(0.0, min(y1, height))
                y2 = max(0.0, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append((label, score, x1, y1, x2, y2))

        cv2.imwrite(
            str(pred_dir / image_name),
            draw_top_left_labels(frame, detections, colors),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        summary_rows.append(
            {
                "image": image_name,
                "seconds": f"{second:.2f}",
                "detections": len(detections),
                "high_conf_075": sum(1 for det in detections if det[1] >= 0.75),
            }
        )
        for label, score, x1, y1, x2, y2 in detections:
            prediction_rows.append(
                {
                    "image": image_name,
                    "seconds": f"{second:.2f}",
                    "label": label,
                    "confidence": f"{score:.6f}",
                    "x1": f"{x1:.1f}",
                    "y1": f"{y1:.1f}",
                    "x2": f"{x2:.1f}",
                    "y2": f"{y2:.1f}",
                }
            )

    capture.release()
    write_csv(BACKTEST_ROOT / "summary.csv", ["image", "seconds", "detections", "high_conf_075"], summary_rows)
    write_csv(
        BACKTEST_ROOT / "predictions.csv",
        ["image", "seconds", "label", "confidence", "x1", "y1", "x2", "y2"],
        prediction_rows,
    )
    (BACKTEST_ROOT / "README.txt").write_text(
        "Merged dense v1 post-training backtest. Use predictions_top_left_label for visual review.\n"
        "Labels are drawn only at the top-left of each box to reduce occlusion.\n"
        f"model={FINAL_MODEL}\n"
        f"previous_model={CURRENT_MODEL}\n"
        f"video={VIDEO}\n"
        f"frames={len(summary_rows)}\n"
        f"total_detections={len(prediction_rows)}\n"
        f"avg_detections_per_frame={len(prediction_rows) / max(1, len(summary_rows)):.2f}\n",
        encoding="utf-8",
    )


def draw_top_left_labels(image, detections, colors):
    canvas = image.copy()
    for label, score, x1, y1, x2, y2 in detections:
        color = colors.get(label[-1], (0, 255, 0))
        ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top = max(0, iy1 - text_height - 6)
        bottom = top + text_height + 6
        cv2.rectangle(canvas, (ix1, top), (ix1 + text_width + 6, bottom), color, -1)
        cv2.putText(
            canvas,
            label,
            (ix1 + 3, bottom - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_round_manifest() -> None:
    manifest = PROJECT / "models" / "yolo" / "training_rounds.yaml"
    existing = yaml.safe_load(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    rounds = existing.get("rounds", []) if isinstance(existing, dict) else []
    rounds.append(
        {
            "name": "merged_dense_v1",
            "model": str(FINAL_MODEL),
            "previous_model": str(CURRENT_MODEL),
            "run_dir": str(PROJECT / "runs" / "detect" / RUN_NAME),
            "dataset": str(MERGED_ROTATED),
            "backtest": str(BACKTEST_ROOT),
            "rollback_note": "Use previous_model to roll back if merged_dense_v1 hurts real-table behavior.",
        }
    )
    manifest.write_text(yaml.safe_dump({"rounds": rounds}, allow_unicode=True, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
