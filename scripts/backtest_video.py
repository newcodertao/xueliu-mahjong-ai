from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a visual YOLO backtest on sampled video frames.")
    parser.add_argument("--video", required=True, help="Video file to sample.")
    parser.add_argument("--model", required=True, help="YOLO model path.")
    parser.add_argument("--output", required=True, help="Output backtest directory.")
    parser.add_argument("--start-seconds", type=float, default=1.25)
    parser.add_argument("--every-seconds", type=float, default=5.0)
    parser.add_argument("--paired-offset-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = Path(args.video)
    model_path = Path(args.model)
    output = Path(args.output)

    if not video.exists():
        raise FileNotFoundError(video)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if output.exists() and not args.keep_existing:
        shutil.rmtree(output)

    images_dir = output / "images"
    pred_dir = output / "predictions_top_left_label"
    images_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    class_set = {f"{rank}{suit}" for suit in ("W", "T", "B") for rank in range(1, 10)}
    colors = {"W": (0, 220, 255), "T": (255, 120, 0), "B": (0, 220, 0)}

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 60.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0
    seconds = sample_seconds(duration, args.start_seconds, args.every_seconds, args.paired_offset_seconds, args.max_frames)

    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for second in seconds:
        frame_index = int(round(second * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        image_name = f"{video.stem}_{second:08.2f}s.jpg"
        image_path = images_dir / image_name
        cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        detections = predict_tiles(model, image_path, frame.shape[:2], class_set, args.imgsz, args.conf, args.iou)
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
    write_csv(output / "summary.csv", ["image", "seconds", "detections", "high_conf_075"], summary_rows)
    write_csv(
        output / "predictions.csv",
        ["image", "seconds", "label", "confidence", "x1", "y1", "x2", "y2"],
        prediction_rows,
    )
    (output / "README.txt").write_text(
        "Video backtest. Use predictions_top_left_label for visual review.\n"
        "Labels are drawn only at the top-left of each box to reduce occlusion.\n"
        f"model={model_path}\n"
        f"video={video}\n"
        f"frames={len(summary_rows)}\n"
        f"total_detections={len(prediction_rows)}\n"
        f"avg_detections_per_frame={len(prediction_rows) / max(1, len(summary_rows)):.2f}\n",
        encoding="utf-8",
    )
    print(f"backtest={output}")
    print(f"frames={len(summary_rows)}")
    print(f"total_detections={len(prediction_rows)}")
    print(f"avg_detections_per_frame={len(prediction_rows) / max(1, len(summary_rows)):.2f}")


def sample_seconds(
    duration: float,
    start_seconds: float,
    every_seconds: float,
    paired_offset_seconds: float,
    max_frames: int,
) -> list[float]:
    seconds: list[float] = []
    second = start_seconds
    while second <= duration and len(seconds) < max_frames:
        seconds.append(second)
        paired = second + paired_offset_seconds
        if paired <= duration and len(seconds) < max_frames:
            seconds.append(paired)
        second += every_seconds
    return seconds


def predict_tiles(model: YOLO, image_path: Path, shape: tuple[int, int], class_set: set[str], imgsz: int, conf: float, iou: float):
    height, width = shape
    detections = []
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
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
            if x2 > x1 and y2 > y1:
                detections.append((label, score, x1, y1, x2, y2))
    return detections


def draw_top_left_labels(image, detections, colors):
    canvas = image.copy()
    for label, _score, x1, y1, x2, y2 in detections:
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


if __name__ == "__main__":
    main()
