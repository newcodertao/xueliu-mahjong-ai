from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from xueliu_ai.paths import resolve_path
from xueliu_ai.vision.detection_types import Detection


class YoloDetector:
    def __init__(self, model_path: str | Path, image_size: int = 640) -> None:
        self.model_path = resolve_path(model_path)
        self.image_size = image_size
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}. Train or place a model there first."
            )
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

    def detect_image(
        self,
        image: str | Path | np.ndarray,
        conf: float = 0.6,
        iou: float = 0.5,
    ) -> list[Detection]:
        source = str(resolve_path(image)) if isinstance(image, (str, Path)) else image
        results = self.model.predict(source=source, imgsz=self.image_size, conf=conf, iou=iou, verbose=False)
        return list(_detections_from_results(results))


def _detections_from_results(results: Iterable[object]) -> Iterable[Detection]:
    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            yield Detection(
                label=str(names[cls_id]),
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = image.copy()
    for det in detections:
        cv2.rectangle(canvas, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{det.label} {det.confidence:.2f}",
            (int(det.x1), max(15, int(det.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas
