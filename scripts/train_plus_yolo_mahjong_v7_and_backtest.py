from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT = Path(r"F:\xueliu-mahjong-ai")
DATA = PROJECT / "datasets" / "xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7" / "data.yaml"
START_MODEL = PROJECT / "models" / "yolo" / "xueliu_tiles_rotated_plus_human_round1_640.pt"
FINAL_MODEL = PROJECT / "models" / "yolo" / "xueliu_tiles_rotated_plus_human_round1_yolo_mahjong_v7.pt"
VIDEO = PROJECT / "data" / "input_videos" / "phone_landscape_20260629_154234.mp4"
BACKTEST_ROOT = PROJECT / "data" / "backtests" / "phone_landscape_post_training_yolo_mahjong_v7"


def main() -> None:
    train_model()
    copy_best_model()
    run_backtest()


def train_model() -> None:
    model = YOLO(str(START_MODEL))
    model.train(
        data=str(DATA),
        epochs=12,
        imgsz=960,
        batch=12,
        device=0,
        workers=4,
        patience=5,
        project=str(PROJECT / "runs" / "detect"),
        name="rotated_plus_human_round1_yolo_mahjong_v7",
        exist_ok=True,
        pretrained=True,
        cache=False,
        verbose=True,
    )


def copy_best_model() -> None:
    best = PROJECT / "runs" / "detect" / "rotated_plus_human_round1_yolo_mahjong_v7" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    shutil.copy2(best, FINAL_MODEL)


def run_backtest() -> None:
    if BACKTEST_ROOT.exists():
        shutil.rmtree(BACKTEST_ROOT)
    images_dir = BACKTEST_ROOT / "images"
    pred_dir = BACKTEST_ROOT / "predictions_top_left_label"
    images_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(FINAL_MODEL))
    classes = [f"{rank}{suit}" for suit in ("W", "T", "B") for rank in range(1, 10)]
    class_set = set(classes)
    colors = {
        "W": (0, 220, 255),
        "T": (255, 120, 0),
        "B": (0, 220, 0),
    }

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

        annotated = draw_top_left_labels(frame, detections, colors)
        cv2.imwrite(str(pred_dir / image_name), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
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
        "Post-training backtest. Use predictions_top_left_label for visual review.\n"
        "Labels are drawn only at the top-left of each box to reduce occlusion.\n"
        f"model={FINAL_MODEL}\n"
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


if __name__ == "__main__":
    main()
