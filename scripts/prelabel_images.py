from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_NAMES = [f"{rank}{suit}" for suit in ("W", "T", "B") for rank in range(1, 10)]
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create editable X-AnyLabeling and YOLO prelabels.")
    parser.add_argument("--images", required=True, help="Source image directory.")
    parser.add_argument("--model", required=True, help="YOLO model path.")
    parser.add_argument("--output", required=True, help="Output directory for copied images and labels.")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--source-name", default="model_prelabeled")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.images)
    model_path = Path(args.model)
    output = Path(args.output)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    preview_dir = output / "_preview_boxes"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows = []
    total_boxes = 0
    for image_path in images:
        dst_image = output / image_path.name
        shutil.copy2(image_path, dst_image)
        image = cv2.imread(str(dst_image))
        if image is None:
            continue
        height, width = image.shape[:2]
        detections = predict(model, dst_image, width, height, args.imgsz, args.conf, args.iou)
        total_boxes += len(detections)
        write_yolo_label(output / f"{image_path.stem}.txt", detections, width, height)
        write_xany_json(
            output / f"{image_path.stem}.json",
            image_path.name,
            width,
            height,
            detections,
            args.source_name,
        )
        if args.preview:
            cv2.imwrite(str(preview_dir / image_path.name), draw_preview(image, detections))
        rows.append(
            {
                "image": image_path.name,
                "detections": len(detections),
                "high_conf_075": sum(1 for det in detections if det["score"] >= 0.75),
            }
        )

    (output / "classes.txt").write_text("\n".join(TARGET_NAMES) + "\n", encoding="utf-8")
    write_csv(output / "prelabel_summary.csv", ["image", "detections", "high_conf_075"], rows)
    (output / "README.txt").write_text(
        "Editable AI prelabels for X-AnyLabeling review.\n"
        "Review every image before using these labels for training.\n"
        f"source_images={image_dir}\n"
        f"model={model_path}\n"
        f"conf={args.conf}\n"
        f"iou={args.iou}\n"
        f"images={len(rows)}\n"
        f"total_boxes={total_boxes}\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"images={len(rows)}")
    print(f"total_boxes={total_boxes}")
    print(f"avg_boxes={total_boxes / max(1, len(rows)):.2f}")


def predict(model: YOLO, image_path: Path, width: int, height: int, imgsz: int, conf: float, iou: float):
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    detections = []
    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            label = str(names[int(box.cls[0].item())]).strip().upper()
            if label not in TARGET_INDEX:
                continue
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            x1 = max(0.0, min(x1, width))
            x2 = max(0.0, min(x2, width))
            y1 = max(0.0, min(y1, height))
            y2 = max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"label": label, "score": score, "xyxy": (x1, y1, x2, y2)})
    return detections


def write_yolo_label(path: Path, detections, width: int, height: int) -> None:
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{TARGET_INDEX[det['label']]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_xany_json(
    path: Path,
    image_name: str,
    width: int,
    height: int,
    detections,
    source_name: str,
) -> None:
    shapes = []
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        score = round(det["score"], 6)
        shapes.append(
            {
                "label": det["label"],
                "score": score,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": "AI prelabel; please review before training",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {"source": source_name, "confidence": score},
                "kie_linking": [],
            }
        )
    data = {
        "version": "3.3.10",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_preview(image, detections):
    canvas = image.copy()
    colors = {"W": (0, 220, 255), "T": (255, 120, 0), "B": (0, 220, 0)}
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = map(int, det["xyxy"])
        color = colors.get(label[-1], (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
